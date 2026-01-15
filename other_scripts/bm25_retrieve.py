#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, os, pickle, re, csv, time
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
import heapq
import math


# -----------------------------
# Helpers
# -----------------------------
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")  # 英文/数字够用了（clapnq/fiqa/govt/cloud 都是英文域）

def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall((text or "").lower())

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
    """
    returns: doc_ids, doc_texts
    corpus jsonl: either {"_id":..., "title":..., "text":...} or {"document_id":...}
    """
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
    """
    queries jsonl: {"_id":..., "text":...} or {"task_id":..., "text":...}
    """
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
    """
    支持两种结构：
      A) split_root/<dom>/<valid|train>.tsv
      B) split_root/<task>/<dom>/<valid|train>.tsv
    """
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
# BM25 index (inverted index + cache)
# -----------------------------
class BM25Index:
    def __init__(self, doc_ids: List[str], doc_tokens: List[List[str]], k1: float, b: float):
        self.doc_ids = doc_ids
        self.k1 = float(k1)
        self.b = float(b)

        self.N = len(doc_ids)
        self.doc_len = [len(toks) for toks in doc_tokens]
        self.avgdl = (sum(self.doc_len) / max(1, self.N))

        # df + postings(term -> list[(doc_idx, tf)])
        self.df = defaultdict(int)
        self.postings = defaultdict(list)

        for i, toks in enumerate(doc_tokens):
            tf = Counter(toks)
            for term, c in tf.items():
                self.df[term] += 1
                self.postings[term].append((i, c))

        # idf (BM25 okapi variant)
        self.idf = {}
        for term, df in self.df.items():
            self.idf[term] = math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def score_query(self, q_tokens: List[str], top_k: int) -> List[Tuple[int, float]]:
        """
        Returns list of (doc_idx, score) top_k
        Efficient: accum only docs appearing in postings of query terms.
        """
        scores = defaultdict(float)
        q_tf = Counter(q_tokens)

        for term, qf in q_tf.items():
            if term not in self.postings:
                continue
            idf = self.idf.get(term, 0.0)
            for doc_idx, tf in self.postings[term]:
                dl = self.doc_len[doc_idx]
                denom = tf + self.k1 * (1 - self.b + self.b * (dl / (self.avgdl + 1e-9)))
                scores[doc_idx] += idf * (tf * (self.k1 + 1) / (denom + 1e-9))

        if not scores:
            return []

        # top-k
        return heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])


def cache_key(corpus_path: Path, k1: float, b: float) -> str:
    st = corpus_path.stat()
    sig = f"{corpus_path}|size={st.st_size}|mtime={int(st.st_mtime)}|k1={k1}|b={b}"
    import hashlib
    return hashlib.sha1(sig.encode("utf-8")).hexdigest()[:16]

def build_or_load_index(corpus_path: Path, cache_dir: Path, k1: float, b: float) -> Tuple[BM25Index, List[str]]:
    """
    returns: index, doc_ids
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key(corpus_path, k1, b)
    pkl = cache_dir / f"bm25_{corpus_path.stem}_{key}.pkl"

    if pkl.exists():
        with pkl.open("rb") as f:
            pack = pickle.load(f)
        return pack["index"], pack["doc_ids"]

    doc_ids, doc_texts = load_corpus_map(corpus_path)
    doc_tokens = [tokenize(t) for t in doc_texts]
    index = BM25Index(doc_ids, doc_tokens, k1=k1, b=b)

    with pkl.open("wb") as f:
        pickle.dump({"index": index, "doc_ids": doc_ids}, f)

    return index, doc_ids


# -----------------------------
# Main retrieval
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["lastturn", "questions", "rewrite"])
    ap.add_argument("--retrieval_tasks_root", default="human/retrieval_tasks")
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--top_k", type=int, default=100)

    ap.add_argument("--k1", type=float, default=1.2)
    ap.add_argument("--b", type=float, default=0.75)

    ap.add_argument("--cache_dir", default="cache/bm25_index")

    # split-only mode (optional)
    ap.add_argument("--split_root", default=None,
                    help="If set, only retrieve for qids in split tsv. Accepts either <...>/<task>/<dom>/valid.tsv or <...>/<dom>/valid.tsv")
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

            # optional: filter queries by split
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

            index, doc_ids = build_or_load_index(corpus_path, cache_dir, args.k1, args.b)

            for qid, qtext in queries.items():
                q_tokens = tokenize(qtext)
                top = index.score_query(q_tokens, top_k=args.top_k)
                ctxs = [{"document_id": doc_ids[i], "score": float(s)} for i, s in top]
                fout.write(json.dumps({
                    "task_id": qid,
                    "contexts": ctxs,
                    "Collection": COLLECTIONS[dom],
                }, ensure_ascii=False) + "\n")

    print(f"[DONE] wrote {out_path}  total_queries={total_q}")


if __name__ == "__main__":
    main()
