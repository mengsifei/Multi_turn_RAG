#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, os, pickle, re, csv, time, math
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import heapq
import sys, shlex, datetime

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


# -----------------------------
# Helpers (same as yours)
# -----------------------------
def guess_domain(collection_name: str) -> str:
    s = (collection_name or "").lower()
    for d in ["clapnq", "fiqa", "govt", "cloud"]:
        if d in s:
            return d
    raise ValueError(f"Cannot infer domain from Collection={collection_name}")

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


# -----------------------------
# SPLADEv2 encoder + sparse pooling
# -----------------------------
@torch.no_grad()
def splade_encode_to_sparse(
    tokenizer,
    model,
    texts: List[str],
    device: str,
    max_length: int,
    batch_size: int,
    top_terms: int,
    pooling: str = "max",              # SPLADEv2 typically max
    activation: str = "log1p_relu",    # common: log1p(relu(.))
) -> List[Dict[int, float]]:
    """
    Returns list of sparse dict: token_id -> weight (only kept top_terms per text).
    """
    assert pooling in ("max", "sum")
    assert activation in ("relu", "log1p_relu")

    model.eval()
    sparse_vecs: List[Dict[int, float]] = []

    use_cuda = device.startswith("cuda")
    # autocast_ctx = torch.cuda.amp.autocast if use_cuda else torch.cpu.amp.autocast  # cpu autocast exists in newer torch, harmless if not used
    # If cpu autocast unavailable, we won't use it.

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        attn = enc["attention_mask"]  # (B,L)

        # forward
        if use_cuda:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                out = model(**enc)
                logits = out.logits  # (B,L,V)
        else:
            out = model(**enc)
            logits = out.logits

        # activation
        if activation == "relu":
            x = torch.relu(logits)
        else:
            x = torch.log1p(torch.relu(logits))

        # mask padding tokens
        x = x * attn.unsqueeze(-1)

        # pooling over sequence length
        if pooling == "max":
            vec = x.max(dim=1).values  # (B,V)
        else:
            vec = x.sum(dim=1)         # (B,V)

        # to cpu for topk
        vec = vec.float().cpu()

        # pick top_terms
        k = min(top_terms, vec.shape[1])
        vals, idxs = torch.topk(vec, k=k, dim=1, largest=True, sorted=True)

        for row_vals, row_idxs in zip(vals, idxs):
            d = {}
            for v, i in zip(row_vals.tolist(), row_idxs.tolist()):
                if v <= 0.0:
                    break
                d[int(i)] = float(v)
            sparse_vecs.append(d)

    return sparse_vecs


# -----------------------------
# SPLADE inverted index (postings) + cache
# -----------------------------
class SpladeInvertedIndex:
    def __init__(self, doc_ids: List[str], postings: Dict[int, List[Tuple[int, float]]]):
        self.doc_ids = doc_ids
        self.postings = postings  # token_id -> list[(doc_idx, weight)]
        self.N = len(doc_ids)

    def score_query(self, q_sparse: Dict[int, float], top_k: int) -> List[Tuple[int, float]]:
        scores = defaultdict(float)
        for tid, qw in q_sparse.items():
            plist = self.postings.get(tid)
            if not plist:
                continue
            for doc_idx, dw in plist:
                scores[doc_idx] += qw * dw

        if not scores:
            return []
        return heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])


def cache_key(corpus_path: Path, model_name: str, max_length: int, doc_top_terms: int, activation: str) -> str:
    st = corpus_path.stat()
    sig = (
        f"{corpus_path}|size={st.st_size}|mtime={int(st.st_mtime)}|"
        f"model={model_name}|ml={max_length}|docTop={doc_top_terms}|act={activation}"
    )
    import hashlib
    return hashlib.sha1(sig.encode("utf-8")).hexdigest()[:16]


import pickle, os

def safe_pickle_load(path: Path):
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except (EOFError, pickle.UnpicklingError, OSError) as e:
        # corrupted/empty/partial cache
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


import re
def slugify_model_name(model_name: str) -> str:
    # naver/splade-v3 -> naver__splade-v3
    s = model_name.strip().lower().replace("/", "__")
    s = re.sub(r"[^a-z0-9._-]+", "-", s)   # 只保留安全字符
    return s[:80]  # 防止太长



def build_or_load_splade_index(
    corpus_path: Path,
    cache_dir: Path,
    model_name: str,
    device: str,
    max_length: int,
    batch_size: int,
    doc_top_terms: int,
    activation: str
) -> Tuple[SpladeInvertedIndex, List[str], AutoTokenizer, AutoModelForMaskedLM]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    # key = cache_key(corpus_path, model_name, max_length, doc_top_terms)
    key = cache_key(corpus_path, model_name, max_length, doc_top_terms, activation)

    # pkl = cache_dir / f"splade_v3_{corpus_path.stem}_{key}.pkl"
    model_tag = slugify_model_name(model_name)
    subdir = cache_dir / model_tag / f"ml{max_length}_doctop{doc_top_terms}_act{activation}"
    subdir.mkdir(parents=True, exist_ok=True)
    pkl = subdir / f"{corpus_path.stem}_{key}.pkl"



    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(device)

    if pkl.exists():
        pack = safe_pickle_load(pkl)
        if pack is not None:
            index = SpladeInvertedIndex(pack["doc_ids"], pack["postings"])
            return index, pack["doc_ids"], tokenizer, model

    doc_ids, doc_texts = load_corpus_map(corpus_path)

    print(f"[INDEX] Encoding corpus={corpus_path.name} docs={len(doc_texts)} model={model_name} device={device}")
    doc_sparse = splade_encode_to_sparse(
        tokenizer=tokenizer,
        model=model,
        texts=doc_texts,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
        top_terms=doc_top_terms,
        pooling="max",
        activation=activation,
    )

    postings: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for doc_idx, sp in enumerate(doc_sparse):
        for tid, w in sp.items():
            postings[tid].append((doc_idx, float(w)))

    # with pkl.open("wb") as f:
        # pickle.dump({"doc_ids": doc_ids, "postings": dict(postings)}, f)
    atomic_pickle_dump({"doc_ids": doc_ids, "postings": dict(postings)}, pkl)

    index = SpladeInvertedIndex(doc_ids, dict(postings))
    return index, doc_ids, tokenizer, model


# -----------------------------
# Main retrieval
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["lastturn", "questions", "rewrite", "rewrite_gpt"])
    ap.add_argument("--retrieval_tasks_root", default="human/retrieval_tasks")
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--top_k", type=int, default=100)

    ap.add_argument("--model_name", default="naver/splade_v2_distil")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=256)

    ap.add_argument("--doc_top_terms", type=int, default=256)
    ap.add_argument("--query_top_terms", type=int, default=64)

    ap.add_argument("--cache_dir", default="cache/splade_index")
    ap.add_argument("--activation", default="relu", choices=["relu","log1p_relu"])

    # split-only mode (optional)
    ap.add_argument("--split_root", default=None)
    ap.add_argument("--split_kind", default="valid", choices=["train", "valid"])

    args = ap.parse_args()

    root = Path(args.retrieval_tasks_root)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    split_root = Path(args.split_root) if args.split_root else None

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

            print(f"[{dom}] queries={len(queries)}  corpus={corpus_path.name}  task={args.task}")
            total_q += len(queries)

            index, doc_ids, tokenizer, model = build_or_load_splade_index(
                corpus_path=corpus_path,
                cache_dir=cache_dir,
                model_name=args.model_name,
                device=args.device,
                max_length=args.max_length,
                batch_size=args.batch_size,
                doc_top_terms=args.doc_top_terms,
                activation=args.activation,
            )


            # encode all queries in batches (much faster than one-by-one)
            qids = list(queries.keys())
            qtexts = [queries[qid] for qid in qids]
            q_sparse_list = splade_encode_to_sparse(
                tokenizer=tokenizer,
                model=model,
                texts=qtexts,
                device=args.device,
                max_length=min(args.max_length, 128),  # queries usually shorter; keep it tight
                batch_size=args.batch_size,
                top_terms=args.query_top_terms,
                pooling="max",
                activation=args.activation,
            )

            for qid, qsp in zip(qids, q_sparse_list):
                top = index.score_query(qsp, top_k=args.top_k)
                ctxs = [{"document_id": doc_ids[i], "score": float(s)} for i, s in top]
                fout.write(json.dumps({
                    "task_id": qid,
                    "contexts": ctxs,
                    "Collection": COLLECTIONS[dom],
                }, ensure_ascii=False) + "\n")

    print(f"[DONE] wrote {out_path}  total_queries={total_q}")


if __name__ == "__main__":
    cmd = " ".join(shlex.quote(x) for x in sys.argv)
    print(f"[CMD] {cmd}")
    print(f"[TIME] {datetime.datetime.now().isoformat(timespec='seconds')}")
    main()
