#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json
from collections import defaultdict

def read_run(path):
    # returns: dict[qid] -> (collection, {docid: (rank, score)})
    mp = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            qid = str(j["task_id"])
            col = j.get("Collection")
            ctxs = j.get("contexts", [])
            d = {}
            for r, c in enumerate(ctxs, start=1):
                did = str(c["document_id"])
                d[did] = (r, float(c.get("score", 0.0)))
            mp[qid] = (col, d)
    return mp

def minmax_norm(scores_dict):
    # dict[docid] -> score  => dict[docid] -> norm_score in [0,1]
    vals = [s for (_, s) in scores_dict.values()]
    if not vals:
        return {}
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {did: 0.0 for did in scores_dict}
    return {did: (scores_dict[did][1] - lo) / (hi - lo) for did in scores_dict}

def fuse_rrf(a, b, k=60):
    # a/b: dict[docid] -> (rank, score)
    fused = defaultdict(float)
    for did, (r, _) in a.items():
        fused[did] += 1.0 / (k + r)
    for did, (r, _) in b.items():
        fused[did] += 1.0 / (k + r)
    return fused

def fuse_wsum(a, b, wa=0.5, wb=0.5):
    # min-max normalize per query
    an = minmax_norm(a)
    bn = minmax_norm(b)
    fused = defaultdict(float)
    for did, s in an.items():
        fused[did] += wa * s
    for did, s in bn.items():
        fused[did] += wb * s
    return fused

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_a", required=True, help="dense run jsonl")
    ap.add_argument("--run_b", required=True, help="sparse run jsonl (splade)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", choices=["rrf", "wsum"], default="rrf")
    ap.add_argument("--k_rrf", type=int, default=60)
    ap.add_argument("--wa", type=float, default=0.5, help="weight for run_a (wsum)")
    ap.add_argument("--wb", type=float, default=0.5, help="weight for run_b (wsum)")
    ap.add_argument("--top_k_out", type=int, default=100)
    args = ap.parse_args()

    A = read_run(args.run_a)
    B = read_run(args.run_b)

    qids = sorted(set(A.keys()) | set(B.keys()))
    with open(args.out, "w", encoding="utf-8") as fout:
        for qid in qids:
            colA, a = A.get(qid, (None, {}))
            colB, b = B.get(qid, (None, {}))
            col = colA or colB

            if args.method == "rrf":
                fused = fuse_rrf(a, b, k=args.k_rrf)
            else:
                fused = fuse_wsum(a, b, wa=args.wa, wb=args.wb)

            # sort + cut
            items = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:args.top_k_out]
            ctxs = [{"document_id": did, "score": float(sc)} for did, sc in items]

            fout.write(json.dumps({
                "task_id": qid,
                "contexts": ctxs,
                "Collection": col,
            }, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
