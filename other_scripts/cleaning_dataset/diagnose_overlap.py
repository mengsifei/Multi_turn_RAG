#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, gzip
from collections import defaultdict

def open_maybe_gz(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, "rt", encoding="utf-8")

def load_qrels_tsv(path: str, min_rel: int = 1):
    qrels = defaultdict(set)
    with open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if line.lower().startswith("query-id") or line.lower().startswith("query_id"):
                continue
            qid, did, rel = line.rstrip("\n").split("\t")[:3]
            if int(rel) >= min_rel:
                qrels[str(qid)].add(str(did))
    return qrels

def parse_query_obj(o: dict):
    """
    Return (qid, ranked_docids_list) for one query-level json object.
    Supports:
      - {"task_id": ..., "contexts":[{"document_id":..., "score":...}, ...]}
      - {"query_id": ..., "results": {docid: score}}
      - {"query_id": ..., "documents":[{"doc_id"/"document_id":..., "score":...}, ...]}
    """
    qid = o.get("task_id") or o.get("query_id") or o.get("qid") or o.get("query-id") or o.get("id")
    if qid is None:
        return None, None
    qid = str(qid)

    # your format: contexts
    if isinstance(o.get("contexts"), list):
        tmp = []
        for x in o["contexts"]:
            did = x.get("document_id") or x.get("doc_id") or x.get("corpus_id")
            if did is None:
                continue
            tmp.append((str(did), float(x.get("score", 0.0))))
        tmp.sort(key=lambda x: x[1], reverse=True)
        return qid, [d for d, _ in tmp]

    # results dict
    if isinstance(o.get("results"), dict):
        items = [(str(d), float(s)) for d, s in o["results"].items()]
        items.sort(key=lambda x: x[1], reverse=True)
        return qid, [d for d, _ in items]

    # documents list
    if isinstance(o.get("documents"), list):
        tmp = []
        for x in o["documents"]:
            did = x.get("document_id") or x.get("doc_id") or x.get("id") or x.get("corpus_id")
            if did is None:
                continue
            tmp.append((str(did), float(x.get("score", 0.0))))
        tmp.sort(key=lambda x: x[1], reverse=True)
        return qid, [d for d, _ in tmp]

    return qid, []

def load_run_jsonl(path: str):
    run = {}
    with open_maybe_gz(path) as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            if not isinstance(o, dict):
                continue
            qid, ranked = parse_query_obj(o)
            if qid is not None:
                run[qid] = ranked
    return run

def topk(lst, k):
    return lst[:k] if lst else []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qrels_tsv", required=True)
    ap.add_argument("--dense_jsonl", required=True)
    ap.add_argument("--sparse_jsonl", required=True)
    ap.add_argument("--k", default="10,50,100,200,500,1000")
    ap.add_argument("--min_rel", type=int, default=1)
    ap.add_argument("--print_examples", action="store_true")
    args = ap.parse_args()

    Ks = [int(x) for x in args.k.split(",") if x.strip()]
    qrels = load_qrels_tsv(args.qrels_tsv, min_rel=args.min_rel)
    dense = load_run_jsonl(args.dense_jsonl)
    sparse = load_run_jsonl(args.sparse_jsonl)

    qids = sorted(set(qrels.keys()) | set(dense.keys()) | set(sparse.keys()))
    print(f"qrels={len(qrels)} dense={len(dense)} sparse={len(sparse)} union_qids={len(qids)}")

    if args.print_examples:
        if sparse:
            ex = next(iter(sparse))
            print("[example] sparse qid=", ex, "top3=", sparse[ex][:3])
        if dense:
            ex = next(iter(dense))
            print("[example] dense qid=", ex, "top3=", dense[ex][:3])

    for K in Ks:
        sum_dense = sum_sparse = sum_union = 0
        sum_dense_only = sum_sparse_only = 0
        sum_jacc = 0.0
        n = 0
        improve = 0

        for qid in qrels.keys():  # only evaluate where qrels exist
            rels = qrels[qid]
            A = set(topk(dense.get(qid, []), K))
            B = set(topk(sparse.get(qid, []), K))
            U = A | B

            hitA = len(rels & A)
            hitB = len(rels & B)
            hitU = len(rels & U)

            sum_dense += hitA
            sum_sparse += hitB
            sum_union += hitU
            sum_dense_only += len((rels & A) - (rels & B))
            sum_sparse_only += len((rels & B) - (rels & A))

            jacc = (len(A & B) / len(U)) if len(U) else 0.0
            sum_jacc += jacc

            if hitU > hitA:
                improve += 1
            n += 1

        print(f"\n[K={K}] over {n} queries")
        print(f"  avg rel hits: dense={sum_dense/n:.3f}  bm25={sum_sparse/n:.3f}  union={sum_union/n:.3f}")
        print(f"  avg unique contribution: dense_only={sum_dense_only/n:.3f}  bm25_only={sum_sparse_only/n:.3f}")
        print(f"  avg topK overlap jaccard≈{sum_jacc/n:.3f}")
        print(f"  %queries where BM25 adds new rels vs dense: {improve/n:.1%}")

if __name__ == "__main__":
    main()
