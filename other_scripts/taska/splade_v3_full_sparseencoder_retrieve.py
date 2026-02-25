#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
splade_v3_full_sparseencoder_retrieve.py

Standalone SPLADE retriever using Sentence-Transformers SparseEncoder
(recommended for SPLADE-v3 full).

Improvements over naive sparse dot-product:
- Optional IDF (log or BM25-style) weighting
- Optional L2 normalization (doc / query / both)
- Optional high-DF token suppression (df_cutoff_ratio)

Output format matches mt-rag-benchmark TaskA input:
{"task_id": ..., "contexts":[{"document_id":..., "score":...}, ...], "Collection": ...}
"""

import argparse, json, os, pickle, re, csv, math
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import heapq

import torch
from tqdm import tqdm

try:
    from sentence_transformers import SparseEncoder
except Exception as e:
    raise RuntimeError(
        "Failed to import SparseEncoder from sentence-transformers. "
        "Please upgrade/install: pip install -U sentence-transformers"
    ) from e


# -----------------------------
# Helpers
# -----------------------------
COLLECTIONS = {
    "clapnq": "mt-rag-clapnq-elser-512-100-20240503",
    "fiqa":   "mt-rag-fiqa-beir-elser-512-100-20240501",
    "govt":   "mt-rag-govt-elser-512-100-20240611",
    "cloud":  "mt-rag-ibmcloud-elser-512-100-20240502",
}

def load_corpus_map(corpus_path: Path) -> Tuple[List[str], List[str]]:
    doc_ids, doc_texts = [], []
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            did = obj.get("document_id") or obj.get("_id") or obj.get("id")
            if not did:
                continue
            title = obj.get("title", "")
            text  = obj.get("text", "") or obj.get("contents", "") or obj.get("content", "")
            full = (str(title) + " " + str(text)).strip()
            doc_ids.append(str(did))
            doc_texts.append(full)
    return doc_ids, doc_texts

def load_queries_map(query_path: Path) -> Dict[str, str]:
    q = {}
    with query_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            qid = obj.get("_id") or obj.get("task_id") or obj.get("query_id") or obj.get("id")
            if not qid:
                continue
            q[str(qid)] = str(obj.get("text", ""))
    return q

def load_split_qids(split_root: Path, task: str, dom: str, split_kind: str) -> Optional[set]:
    cand1 = split_root / dom / f"{split_kind}.tsv"
    cand2 = split_root / task / dom / f"{split_kind}.tsv"
    path = cand1 if cand1.exists() else cand2 if cand2.exists() else None
    if path is None:
        return None

    qids = set()
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if row[0].lower().startswith("query"):
                continue
            qids.add(str(row[0]))
    return qids

def slugify_model_name(model_name: str) -> str:
    s = model_name.strip().lower().replace("/", "__")
    s = re.sub(r"[^a-z0-9._-]+", "-", s)
    return s[:100]

def safe_pickle_load(path: Path):
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except (EOFError, pickle.UnpicklingError, OSError) as e:
        print(f"[WARN] cache corrupted: {path} ({type(e).__name__}: {e}) -> rebuild")
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return None

def atomic_pickle_dump(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.unlink()
    except FileNotFoundError:
        pass
    with tmp.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def cache_key(corpus_path: Path, model_name: str, max_length: int, doc_top_terms: int) -> str:
    st = corpus_path.stat()
    sig = (
        f"v2_postings_df|{corpus_path}|size={st.st_size}|mtime={int(st.st_mtime)}|"
        f"model={model_name}|ml={max_length}|docTop={doc_top_terms}"
    )
    import hashlib
    return hashlib.sha1(sig.encode("utf-8")).hexdigest()[:16]


# -----------------------------
# Sparse tensor utils
# -----------------------------
def sparse_tensor_rows_to_terms(sp: torch.Tensor) -> List[List[Tuple[int, float]]]:
    """
    Convert a 2D torch sparse tensor (COO/CSR) into per-row list[(col_id, value)].
    """
    if not torch.is_tensor(sp):
        raise TypeError(f"Expected torch.Tensor, got {type(sp)}")

    if sp.dim() == 1:
        sp = sp.unsqueeze(0)

    # Prefer CSR for fast row slicing
    if sp.layout != torch.sparse_csr:
        if sp.layout == torch.sparse_coo:
            sp = sp.coalesce()
        try:
            sp = sp.to_sparse_csr()
        except Exception:
            sp = sp.coalesce()
            idx = sp.indices()
            vals = sp.values()
            rows = idx[0].tolist()
            cols = idx[1].tolist()
            vals = vals.tolist()
            out = [[] for _ in range(sp.size(0))]
            for r, c, v in zip(rows, cols, vals):
                out[r].append((int(c), float(v)))
            return out

    crow = sp.crow_indices().cpu()
    col  = sp.col_indices().cpu()
    val  = sp.values().cpu()

    out: List[List[Tuple[int, float]]] = []
    n_rows = sp.size(0)
    for r in range(n_rows):
        s = int(crow[r].item())
        e = int(crow[r + 1].item())
        if e <= s:
            out.append([])
            continue
        rr = [(int(col[i].item()), float(val[i].item())) for i in range(s, e)]
        out.append(rr)
    return out


# -----------------------------
# Improved Inverted Index
# -----------------------------
def _idf_log(N: int, df: int) -> float:
    # log((N+1)/(df+1))  -- stable, always >=0
    return math.log((N + 1.0) / (df + 1.0))

def _idf_bm25(N: int, df: int) -> float:
    # BM25 idf: log((N - df + 0.5)/(df + 0.5))
    return math.log((N - df + 0.5) / (df + 0.5))

def _token_factor(idf: float, idf_mode: str) -> float:
    # factor used in dot product
    # none: 1
    # once: idf
    # both: idf^2  (equivalent to applying idf to both query and doc weights)
    if idf_mode == "none":
        return 1.0
    if idf_mode == "once":
        return idf
    if idf_mode == "both":
        return idf * idf
    raise ValueError(f"bad idf_mode={idf_mode}")

class SpladeInvertedIndex:
    """
    postings: token_id -> list[(doc_idx, weight)]
    df: token_id -> number of docs containing token_id
    """
    def __init__(self, doc_ids: List[str], postings: Dict[int, List[Tuple[int, float]]], df: Dict[int, int]):
        self.doc_ids = doc_ids
        self.postings = postings
        self.df = df
        self.N = len(doc_ids)

        # cached stats for a given scoring config
        self._cached_cfg = None
        self._idf = None
        self._doc_norm = None
        self._df_cutoff_mask = None  # optional: tokens to skip

    def prepare_stats(
        self,
        *,
        idf_method: str,
        idf_mode: str,
        idf_clip_zero: bool,
        df_cutoff_ratio: Optional[float],
        normalize: str,
        eps: float = 1e-12,
    ):
        """
        Precompute:
        - idf[token]
        - doc_norm[doc] used for normalization (if normalize includes doc/both)
        - optional df_cutoff token skip
        """
        cfg = (idf_method, idf_mode, bool(idf_clip_zero), df_cutoff_ratio, normalize)
        if self._cached_cfg == cfg and self._idf is not None and (normalize == "none" or self._doc_norm is not None):
            return

        # 1) idf
        idf: Dict[int, float] = {}
        for tid, dfi in self.df.items():
            if idf_method == "log":
                v = _idf_log(self.N, int(dfi))
            elif idf_method == "bm25":
                v = _idf_bm25(self.N, int(dfi))
            else:
                raise ValueError(f"bad idf_method={idf_method}")

            if idf_clip_zero and v < 0.0:
                v = 0.0
            idf[int(tid)] = float(v)

        # 2) df cutoff (optional)
        df_cut_mask = None
        if df_cutoff_ratio is not None:
            thr = float(df_cutoff_ratio)
            df_cut_mask = {}
            for tid, dfi in self.df.items():
                df_ratio = float(dfi) / float(self.N)
                df_cut_mask[int(tid)] = (df_ratio >= thr)

        # 3) doc norm (optional)
        doc_norm = None
        if normalize in ("doc", "both"):
            doc_norm2 = [0.0] * self.N

            # norm scale is sqrt(token_factor(idf))
            for tid, plist in self.postings.items():
                tid_i = int(tid)
                if df_cut_mask is not None and df_cut_mask.get(tid_i, False):
                    continue

                idf_v = idf.get(tid_i, 0.0)
                tfac = _token_factor(idf_v, idf_mode)
                nscale = math.sqrt(tfac) if tfac > 0.0 else 0.0
                if nscale == 0.0:
                    continue

                for doc_idx, dw in plist:
                    # accumulate (dw * nscale)^2
                    v = float(dw) * nscale
                    doc_norm2[int(doc_idx)] += v * v

            doc_norm = [math.sqrt(v) + eps for v in doc_norm2]

        self._idf = idf
        self._doc_norm = doc_norm
        self._df_cutoff_mask = df_cut_mask
        self._cached_cfg = cfg

    def score_query(
        self,
        q_terms: List[Tuple[int, float]],
        *,
        top_k: int,
        idf_method: str,
        idf_mode: str,
        idf_clip_zero: bool,
        df_cutoff_ratio: Optional[float],
        normalize: str,
        eps: float = 1e-12,
    ) -> List[Tuple[int, float]]:
        """
        Score with:
          raw = sum_t qw * dw * token_factor(idf(t), idf_mode)
        Optional normalization:
          - doc: raw / ||d||
          - query: raw / ||q||
          - both: raw / (||d|| * ||q||)
        """
        self.prepare_stats(
            idf_method=idf_method,
            idf_mode=idf_mode,
            idf_clip_zero=idf_clip_zero,
            df_cutoff_ratio=df_cutoff_ratio,
            normalize=normalize,
            eps=eps,
        )

        idf = self._idf or {}
        df_cut_mask = self._df_cutoff_mask
        doc_norm = self._doc_norm

        # query norm if needed
        q_norm = 1.0
        if normalize in ("query", "both"):
            acc = 0.0
            for tid, qw in q_terms:
                tid_i = int(tid)
                if df_cut_mask is not None and df_cut_mask.get(tid_i, False):
                    continue
                idf_v = idf.get(tid_i, 0.0)
                tfac = _token_factor(idf_v, idf_mode)
                nscale = math.sqrt(tfac) if tfac > 0.0 else 0.0
                v = float(qw) * nscale
                acc += v * v
            q_norm = math.sqrt(acc) + eps

        # dot accumulation
        scores = defaultdict(float)
        for tid, qw in q_terms:
            tid_i = int(tid)
            if df_cut_mask is not None and df_cut_mask.get(tid_i, False):
                continue

            plist = self.postings.get(tid_i)
            if not plist:
                continue

            idf_v = idf.get(tid_i, 0.0)
            tfac = _token_factor(idf_v, idf_mode)
            if tfac == 0.0:
                continue

            qw_f = float(qw)
            for doc_idx, dw in plist:
                scores[int(doc_idx)] += qw_f * float(dw) * tfac

        if not scores:
            return []

        # normalization
        if normalize in ("doc", "both"):
            assert doc_norm is not None
            if normalize == "doc":
                for d in list(scores.keys()):
                    scores[d] = scores[d] / doc_norm[d]
            else:
                # both
                for d in list(scores.keys()):
                    scores[d] = scores[d] / (doc_norm[d] * q_norm)
        elif normalize == "query":
            for d in list(scores.keys()):
                scores[d] = scores[d] / q_norm
        elif normalize == "none":
            pass
        else:
            raise ValueError(f"bad normalize={normalize}")

        return heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])


# -----------------------------
# Build or load index per domain
# -----------------------------
def build_or_load_index_for_domain(
    corpus_path: Path,
    cache_root: Path,
    model_name: str,
    model: "SparseEncoder",
    max_length: int,
    batch_size: int,
    doc_top_terms: int,
    block_size_docs: int,
    min_weight: float,
) -> Tuple[SpladeInvertedIndex, List[str], Path]:
    """
    Build postings + df only. (idf/norm computed on the fly per run config)
    """
    model_tag = slugify_model_name(model_name)
    dom_cache_dir = cache_root / model_tag / f"ml{max_length}_dt{doc_top_terms}"
    dom_cache_dir.mkdir(parents=True, exist_ok=True)

    key = cache_key(corpus_path, model_name, max_length, doc_top_terms)
    pkl = dom_cache_dir / f"index_{corpus_path.stem}_{key}.pkl"

    if pkl.exists():
        pack = safe_pickle_load(pkl)
        if pack is not None:
            doc_ids = pack["doc_ids"]
            postings = pack["postings"]
            df = pack.get("df")
            if df is None:
                # reconstruct df from postings if not stored
                df = {int(tid): len(plist) for tid, plist in postings.items()}
            idx = SpladeInvertedIndex(doc_ids, postings, df)
            return idx, doc_ids, pkl

    doc_ids, doc_texts = load_corpus_map(corpus_path)
    postings: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    df: Dict[int, int] = defaultdict(int)

    print(f"[INDEX] {corpus_path.name}: docs={len(doc_texts)} model={model_name} max_len={max_length} doc_top={doc_top_terms}")
    doc_idx_offset = 0

    for start in tqdm(range(0, len(doc_texts), block_size_docs), desc=f"encode_document({corpus_path.stem})"):
        block = doc_texts[start:start + block_size_docs]

        emb = model.encode_document(
            block,
            batch_size=batch_size,
            show_progress_bar=False,
            max_active_dims=doc_top_terms,
        )
        rows = sparse_tensor_rows_to_terms(emb)

        for local_i, terms in enumerate(rows):
            doc_idx = doc_idx_offset + local_i

            # terms in SparseEncoder output should already be unique per row
            for tid, w in terms:
                if w <= min_weight:
                    continue
                tid_i = int(tid)
                postings[tid_i].append((doc_idx, float(w)))
                df[tid_i] += 1

        doc_idx_offset += len(block)

    postings = dict(postings)
    df = dict(df)
    atomic_pickle_dump({"doc_ids": doc_ids, "postings": postings, "df": df}, pkl)

    idx = SpladeInvertedIndex(doc_ids, postings, df)
    return idx, doc_ids, pkl


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["lastturn", "questions", "rewrite", "rewrite_gpt", "rewrite_gpt_keywords", "concat_lastturn_rewrite_gpt"])
    ap.add_argument("--retrieval_tasks_root", default="human/retrieval_tasks")
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--top_k", type=int, default=100)

    ap.add_argument("--model_name", default="naver/splade-v3")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=256)

    ap.add_argument("--doc_top_terms", type=int, default=256)
    ap.add_argument("--query_top_terms", type=int, default=64)

    ap.add_argument("--cache_root", default="cache/splade_sparseencoder_index")
    ap.add_argument("--block_size_docs", type=int, default=4096, help="Docs per encode_document() block")
    ap.add_argument("--trust_remote_code", action="store_true", help="Pass trust_remote_code=True to SparseEncoder")
    ap.add_argument("--min_weight", type=float, default=0.0, help="Drop doc/query terms with weight <= min_weight")

    # --- scoring knobs ---
    ap.add_argument("--idf_method", choices=["log", "bm25"], default="log")
    ap.add_argument("--idf_mode", choices=["none", "once", "both"], default="none",
                    help='none: no idf; once: dot *= idf; both: dot *= idf^2 (tf-idf style)')
    ap.add_argument("--idf_clip_zero", action="store_true", help="Clip negative idf to 0 (useful for bm25 idf)")
    ap.add_argument("--normalize", choices=["none", "doc", "query", "both"], default="none",
                    help="L2 normalization mode. doc is usually the biggest win.")
    ap.add_argument("--df_cutoff_ratio", type=float, default=None,
                    help="If set, skip tokens with df/N >= ratio (e.g. 0.2). Helps suppress stopword-like tokens.")

    ap.add_argument("--split_root", default=None)
    ap.add_argument("--split_kind", default="valid", choices=["train", "valid"])

    args = ap.parse_args()

    root = Path(args.retrieval_tasks_root)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.cache_root)

    split_root = Path(args.split_root) if args.split_root else None

    print(f"[LOAD] SparseEncoder(model={args.model_name}, device={args.device})")
    model = SparseEncoder(args.model_name, device=args.device, trust_remote_code=args.trust_remote_code)

    supported = getattr(model, "max_seq_length", None)
    if supported is not None:
        eff_max_len = min(int(args.max_length), int(supported))
        if eff_max_len != int(args.max_length):
            print(f"[WARN] requested max_length={args.max_length} > model.max_seq_length={supported}; use {eff_max_len} instead")
        model.max_seq_length = eff_max_len
    else:
        eff_max_len = int(args.max_length)

    domains = ["clapnq", "fiqa", "govt", "cloud"]
    total_q = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for dom in domains:
            corpus_path = root / dom / f"{dom}.jsonl"
            query_path  = root / dom / f"{dom}_{args.task}.jsonl"
            if not corpus_path.exists():
                raise FileNotFoundError(f"Missing corpus: {corpus_path}")
            if not query_path.exists():
                raise FileNotFoundError(f"Missing queries: {query_path}")

            allow_qids = None
            if split_root is not None:
                allow_qids = load_split_qids(split_root, args.task, dom, args.split_kind)
                if allow_qids is None:
                    raise FileNotFoundError(f"Split qids not found under {split_root} for dom={dom}, task={args.task}, kind={args.split_kind}")

            queries = load_queries_map(query_path)
            if allow_qids is not None:
                queries = {qid: txt for qid, txt in queries.items() if qid in allow_qids}

            print(f"[{dom}] queries={len(queries)} corpus={corpus_path.name} task={args.task}")
            total_q += len(queries)

            index, doc_ids, cache_file = build_or_load_index_for_domain(
                corpus_path=corpus_path,
                cache_root=cache_root,
                model_name=args.model_name,
                model=model,
                max_length=eff_max_len,
                batch_size=args.batch_size,
                doc_top_terms=args.doc_top_terms,
                block_size_docs=args.block_size_docs,
                min_weight=float(args.min_weight),
            )
            print(f"[CACHE] {dom}: {cache_file}")

            # Encode queries in batch
            qids = list(queries.keys())
            qtexts = [queries[qid] for qid in qids]

            q_emb = model.encode_query(
                qtexts,
                batch_size=args.batch_size,
                show_progress_bar=False,
                max_active_dims=args.query_top_terms,
            )
            q_rows = sparse_tensor_rows_to_terms(q_emb)

            # optionally drop tiny query weights
            if args.min_weight > 0:
                q_rows = [[(tid, w) for (tid, w) in row if w > args.min_weight] for row in q_rows]

            for qid, q_terms in zip(qids, q_rows):
                top = index.score_query(
                    q_terms,
                    top_k=args.top_k,
                    idf_method=args.idf_method,
                    idf_mode=args.idf_mode,
                    idf_clip_zero=bool(args.idf_clip_zero),
                    df_cutoff_ratio=args.df_cutoff_ratio,
                    normalize=args.normalize,
                )
                ctxs = [{"document_id": doc_ids[i], "score": float(s)} for i, s in top]
                fout.write(json.dumps({
                    "task_id": qid,
                    "contexts": ctxs,
                    "Collection": COLLECTIONS[dom],
                }, ensure_ascii=False) + "\n")

    print(f"[DONE] wrote {out_path} total_queries={total_q}")


if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# splade_v3_full_sparseencoder_retrieve.py

# Standalone SPLADE retriever using Sentence-Transformers SparseEncoder
# (recommended for SPLADE-v3 full).

# - Builds a sparse inverted index from encode_document()
# - Encodes queries with encode_query()
# - Scores with sparse dot-product via postings lists
# - Atomic cache + auto model-specific cache path

# Output format matches mt-rag-benchmark TaskA input:
# {"task_id": ..., "contexts":[{"document_id":..., "score":...}, ...], "Collection": ...}
# """

# import argparse, json, os, pickle, re, csv, math
# from pathlib import Path
# from collections import defaultdict
# from typing import Dict, List, Tuple, Optional
# import heapq

# import torch
# from tqdm import tqdm

# # Sentence-Transformers SparseEncoder (official SPLADE-v3 inference path)
# try:
#     from sentence_transformers import SparseEncoder
# except Exception as e:
#     raise RuntimeError(
#         "Failed to import SparseEncoder from sentence-transformers. "
#         "Please upgrade/install: pip install -U sentence-transformers"
#     ) from e


# # -----------------------------
# # Helpers (same spirit as yours)
# # -----------------------------
# COLLECTIONS = {
#     "clapnq": "mt-rag-clapnq-elser-512-100-20240503",
#     "fiqa":   "mt-rag-fiqa-beir-elser-512-100-20240501",
#     "govt":   "mt-rag-govt-elser-512-100-20240611",
#     "cloud":  "mt-rag-ibmcloud-elser-512-100-20240502",
# }

# def load_corpus_map(corpus_path: Path) -> Tuple[List[str], List[str]]:
#     doc_ids, doc_texts = [], []
#     with corpus_path.open("r", encoding="utf-8") as f:
#         for line in f:
#             obj = json.loads(line)
#             did = obj.get("document_id") or obj.get("_id") or obj.get("id")
#             if not did:
#                 continue
#             title = obj.get("title", "")
#             text  = obj.get("text", "") or obj.get("contents", "") or obj.get("content", "")
#             full = (str(title) + " " + str(text)).strip()
#             doc_ids.append(str(did))
#             doc_texts.append(full)
#     return doc_ids, doc_texts

# def load_queries_map(query_path: Path) -> Dict[str, str]:
#     q = {}
#     with query_path.open("r", encoding="utf-8") as f:
#         for line in f:
#             obj = json.loads(line)
#             qid = obj.get("_id") or obj.get("task_id") or obj.get("query_id") or obj.get("id")
#             if not qid:
#                 continue
#             q[str(qid)] = str(obj.get("text", ""))
#     return q

# def load_split_qids(split_root: Path, task: str, dom: str, split_kind: str) -> Optional[set]:
#     cand1 = split_root / dom / f"{split_kind}.tsv"
#     cand2 = split_root / task / dom / f"{split_kind}.tsv"
#     path = cand1 if cand1.exists() else cand2 if cand2.exists() else None
#     if path is None:
#         return None

#     qids = set()
#     with path.open("r", encoding="utf-8") as f:
#         reader = csv.reader(f, delimiter="\t")
#         for row in reader:
#             if not row:
#                 continue
#             if row[0].lower().startswith("query"):
#                 continue
#             qids.add(str(row[0]))
#     return qids

# def slugify_model_name(model_name: str) -> str:
#     s = model_name.strip().lower().replace("/", "__")
#     s = re.sub(r"[^a-z0-9._-]+", "-", s)
#     return s[:100]

# def safe_pickle_load(path: Path):
#     try:
#         with path.open("rb") as f:
#             return pickle.load(f)
#     except (EOFError, pickle.UnpicklingError, OSError) as e:
#         print(f"[WARN] cache corrupted: {path} ({type(e).__name__}: {e}) -> rebuild")
#         try:
#             path.unlink()
#         except FileNotFoundError:
#             pass
#         return None

# def atomic_pickle_dump(obj, path: Path):
#     tmp = path.with_suffix(path.suffix + ".tmp")
#     try:
#         tmp.unlink()
#     except FileNotFoundError:
#         pass
#     with tmp.open("wb") as f:
#         pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
#         f.flush()
#         os.fsync(f.fileno())
#     os.replace(tmp, path)

# def cache_key(corpus_path: Path, model_name: str, max_length: int, doc_top_terms: int) -> str:
#     st = corpus_path.stat()
#     sig = (
#         f"v1_sparseencoder|{corpus_path}|size={st.st_size}|mtime={int(st.st_mtime)}|"
#         f"model={model_name}|ml={max_length}|docTop={doc_top_terms}"
#     )
#     import hashlib
#     return hashlib.sha1(sig.encode("utf-8")).hexdigest()[:16]


# # -----------------------------
# # Inverted index
# # -----------------------------
# class SpladeInvertedIndex:
#     def __init__(self, doc_ids: List[str], postings: Dict[int, List[Tuple[int, float]]]):
#         self.doc_ids = doc_ids
#         self.postings = postings  # token_id -> list[(doc_idx, weight)]
#         self.N = len(doc_ids)

#     def score_query(self, q_terms: List[Tuple[int, float]], top_k: int) -> List[Tuple[int, float]]:
#         scores = defaultdict(float)
#         for tid, qw in q_terms:
#             plist = self.postings.get(tid)
#             if not plist:
#                 continue
#             for doc_idx, dw in plist:
#                 scores[doc_idx] += qw * dw

#         if not scores:
#             return []
#         return heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])


# def sparse_tensor_rows_to_terms(sp: torch.Tensor) -> List[List[Tuple[int, float]]]:
#     """
#     Convert a 2D torch sparse tensor (COO/CSR) into per-row list[(col_id, value)].
#     """
#     if not torch.is_tensor(sp):
#         raise TypeError(f"Expected torch.Tensor, got {type(sp)}")

#     # Ensure 2D
#     if sp.dim() == 1:
#         sp = sp.unsqueeze(0)

#     # Prefer CSR for fast row slicing
#     if sp.layout != torch.sparse_csr:
#         # coalesce then convert
#         if sp.layout == torch.sparse_coo:
#             sp = sp.coalesce()
#         try:
#             sp = sp.to_sparse_csr()
#         except Exception:
#             # Fallback: COO scan
#             sp = sp.coalesce()
#             idx = sp.indices()
#             vals = sp.values()
#             rows = idx[0].tolist()
#             cols = idx[1].tolist()
#             vals = vals.tolist()
#             out = [[] for _ in range(sp.size(0))]
#             for r, c, v in zip(rows, cols, vals):
#                 out[r].append((int(c), float(v)))
#             return out

#     crow = sp.crow_indices().cpu()
#     col  = sp.col_indices().cpu()
#     val  = sp.values().cpu()

#     out: List[List[Tuple[int, float]]] = []
#     n_rows = sp.size(0)
#     for r in range(n_rows):
#         s = int(crow[r].item())
#         e = int(crow[r + 1].item())
#         if e <= s:
#             out.append([])
#             continue
#         rr = [(int(col[i].item()), float(val[i].item())) for i in range(s, e)]
#         out.append(rr)
#     return out


# def build_or_load_index_for_domain(
#     corpus_path: Path,
#     cache_root: Path,
#     model_name: str,
#     model: "SparseEncoder",
#     max_length: int,
#     batch_size: int,
#     doc_top_terms: int,
#     block_size_docs: int,
# ) -> Tuple[SpladeInvertedIndex, List[str], Path]:
#     """
#     Build postings index from SparseEncoder.encode_document().
#     """
#     model_tag = slugify_model_name(model_name)
#     dom_cache_dir = cache_root / model_tag / f"ml{max_length}_dt{doc_top_terms}"
#     dom_cache_dir.mkdir(parents=True, exist_ok=True)

#     key = cache_key(corpus_path, model_name, max_length, doc_top_terms)
#     pkl = dom_cache_dir / f"index_{corpus_path.stem}_{key}.pkl"

#     if pkl.exists():
#         pack = safe_pickle_load(pkl)
#         if pack is not None:
#             idx = SpladeInvertedIndex(pack["doc_ids"], pack["postings"])
#             return idx, pack["doc_ids"], pkl

#     doc_ids, doc_texts = load_corpus_map(corpus_path)

#     postings: Dict[int, List[Tuple[int, float]]] = defaultdict(list)

#     print(f"[INDEX] {corpus_path.name}: docs={len(doc_texts)} model={model_name} max_len={max_length} doc_top={doc_top_terms}")
#     doc_idx_offset = 0

#     # Encode in blocks to reduce peak memory
#     for start in tqdm(range(0, len(doc_texts), block_size_docs), desc=f"encode_document({corpus_path.stem})"):
#         block = doc_texts[start:start + block_size_docs]

#         emb = model.encode_document(
#             block,
#             batch_size=batch_size,
#             show_progress_bar=False,
#             max_active_dims=doc_top_terms,   # controls sparsity directly
#         )
#         # emb: 2D torch sparse tensor [B, V]
#         rows = sparse_tensor_rows_to_terms(emb)

#         for local_i, terms in enumerate(rows):
#             doc_idx = doc_idx_offset + local_i
#             for tid, w in terms:
#                 if w <= 0.0:
#                     continue
#                 postings[tid].append((doc_idx, float(w)))

#         doc_idx_offset += len(block)

#     postings = dict(postings)
#     atomic_pickle_dump({"doc_ids": doc_ids, "postings": postings}, pkl)

#     idx = SpladeInvertedIndex(doc_ids, postings)
#     return idx, doc_ids, pkl


# # -----------------------------
# # Main
# # -----------------------------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--task", required=True, choices=["lastturn", "questions", "rewrite", "rewrite_gpt"])
#     ap.add_argument("--retrieval_tasks_root", default="human/retrieval_tasks")
#     ap.add_argument("--out_jsonl", required=True)
#     ap.add_argument("--top_k", type=int, default=100)

#     ap.add_argument("--model_name", default="naver/splade-v3")
#     ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
#     ap.add_argument("--batch_size", type=int, default=8)
#     ap.add_argument("--max_length", type=int, default=256)

#     ap.add_argument("--doc_top_terms", type=int, default=256)
#     ap.add_argument("--query_top_terms", type=int, default=64)

#     ap.add_argument("--cache_root", default="cache/splade_sparseencoder_index")
#     ap.add_argument("--block_size_docs", type=int, default=4096, help="Docs per encode_document() block")

#     ap.add_argument("--trust_remote_code", action="store_true", help="Pass trust_remote_code=True to SparseEncoder")

#     ap.add_argument("--split_root", default=None)
#     ap.add_argument("--split_kind", default="valid", choices=["train", "valid"])

#     args = ap.parse_args()

#     root = Path(args.retrieval_tasks_root)
#     out_path = Path(args.out_jsonl)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     cache_root = Path(args.cache_root)

#     split_root = Path(args.split_root) if args.split_root else None

#     # Load SparseEncoder once (shared across domains)
#     print(f"[LOAD] SparseEncoder(model={args.model_name}, device={args.device})")
#     model = SparseEncoder(args.model_name, device=args.device, trust_remote_code=args.trust_remote_code)

#     # Respect model's supported max seq len (cannot increase above model limit)
#     supported = getattr(model, "max_seq_length", None)
#     if supported is not None:
#         eff_max_len = min(int(args.max_length), int(supported))
#         if eff_max_len != int(args.max_length):
#             print(f"[WARN] requested max_length={args.max_length} > model.max_seq_length={supported}; use {eff_max_len} instead")
#         model.max_seq_length = eff_max_len
#     else:
#         eff_max_len = int(args.max_length)

#     domains = ["clapnq", "fiqa", "govt", "cloud"]
#     total_q = 0

#     with out_path.open("w", encoding="utf-8") as fout:
#         for dom in domains:
#             corpus_path = root / dom / f"{dom}.jsonl"
#             query_path  = root / dom / f"{dom}_{args.task}.jsonl"
#             if not corpus_path.exists():
#                 raise FileNotFoundError(f"Missing corpus: {corpus_path}")
#             if not query_path.exists():
#                 raise FileNotFoundError(f"Missing queries: {query_path}")

#             allow_qids = None
#             if split_root is not None:
#                 allow_qids = load_split_qids(split_root, args.task, dom, args.split_kind)
#                 if allow_qids is None:
#                     raise FileNotFoundError(f"Split qids not found under {split_root} for dom={dom}, task={args.task}, kind={args.split_kind}")

#             queries = load_queries_map(query_path)
#             if allow_qids is not None:
#                 queries = {qid: txt for qid, txt in queries.items() if qid in allow_qids}

#             print(f"[{dom}] queries={len(queries)} corpus={corpus_path.name} task={args.task}")
#             total_q += len(queries)

#             index, doc_ids, cache_file = build_or_load_index_for_domain(
#                 corpus_path=corpus_path,
#                 cache_root=cache_root,
#                 model_name=args.model_name,
#                 model=model,
#                 max_length=eff_max_len,
#                 batch_size=args.batch_size,
#                 doc_top_terms=args.doc_top_terms,
#                 block_size_docs=args.block_size_docs,
#             )
#             print(f"[CACHE] {dom}: {cache_file}")

#             # Encode queries in batch
#             qids = list(queries.keys())
#             qtexts = [queries[qid] for qid in qids]

#             q_emb = model.encode_query(
#                 qtexts,
#                 batch_size=args.batch_size,
#                 show_progress_bar=False,
#                 max_active_dims=args.query_top_terms,
#             )
#             q_terms_rows = sparse_tensor_rows_to_terms(q_emb)

#             for qid, q_terms in zip(qids, q_terms_rows):
#                 top = index.score_query(q_terms, top_k=args.top_k)
#                 ctxs = [{"document_id": doc_ids[i], "score": float(s)} for i, s in top]
#                 fout.write(json.dumps({
#                     "task_id": qid,
#                     "contexts": ctxs,
#                     "Collection": COLLECTIONS[dom],
#                 }, ensure_ascii=False) + "\n")

#     print(f"[DONE] wrote {out_path} total_queries={total_q}")


# if __name__ == "__main__":
#     main()
