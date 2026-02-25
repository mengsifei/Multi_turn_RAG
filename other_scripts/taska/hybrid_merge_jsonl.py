#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def load_run(path: Path) -> Dict[str, dict]:
    """
    Returns dict[qid] = {
        "Collection": str,
        "contexts": List[{"document_id": str, "score": float, ...}]
    }
    """
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = str(obj["task_id"])
            out[qid] = obj
    return out


def sort_contexts(ctxs: List[dict]) -> List[dict]:
    # ensure descending
    return sorted(ctxs, key=lambda x: float(x.get("score", 0.0)), reverse=True)


def rrf_fuse(
    dense_ctxs: List[dict],
    sparse_ctxs: List[dict],
    *,
    rrf_k: int,
    w_dense: float,
    w_sparse: float,
) -> Dict[str, float]:
    """
    RRF score for doc = sum_i w_i / (rrf_k + rank_i)
    rank is 1-based
    """
    scores: Dict[str, float] = {}

    dense_ctxs = sort_contexts(dense_ctxs)
    sparse_ctxs = sort_contexts(sparse_ctxs)

    for rank, c in enumerate(dense_ctxs, start=1):
        did = str(c["document_id"])
        scores[did] = scores.get(did, 0.0) + (w_dense / (rrf_k + rank))

    for rank, c in enumerate(sparse_ctxs, start=1):
        did = str(c["document_id"])
        scores[did] = scores.get(did, 0.0) + (w_sparse / (rrf_k + rank))

    return scores


def minmax_norm(ctxs: List[dict]) -> Dict[str, float]:
    ctxs = sort_contexts(ctxs)
    if not ctxs:
        return {}
    vals = [float(c.get("score", 0.0)) for c in ctxs]
    lo, hi = min(vals), max(vals)
    denom = (hi - lo) if (hi - lo) > 1e-12 else 1.0
    return {str(c["document_id"]): (float(c.get("score", 0.0)) - lo) / denom for c in ctxs}


def linear_fuse(
    dense_ctxs: List[dict],
    sparse_ctxs: List[dict],
    *,
    w_dense: float,
    w_sparse: float,
) -> Dict[str, float]:
    """
    score = w_dense * norm(dense_score) + w_sparse * norm(sparse_score)
    Using per-query min-max normalization.
    """
    d = minmax_norm(dense_ctxs)
    s = minmax_norm(sparse_ctxs)
    doc_ids = set(d.keys()) | set(s.keys())
    out = {}
    for did in doc_ids:
        out[did] = w_dense * d.get(did, 0.0) + w_sparse * s.get(did, 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense_jsonl", required=True, help="Jasper retrieval jsonl")
    ap.add_argument("--sparse_jsonl", required=True, help="BM25/BGE-sparse retrieval jsonl")
    ap.add_argument("--out_jsonl", required=True)

    ap.add_argument("--method", choices=["rrf", "linear"], default="rrf")
    ap.add_argument("--rrf_k", type=int, default=60, help="RRF constant (typical: 60)")
    ap.add_argument("--w_dense", type=float, default=1.0)
    ap.add_argument("--w_sparse", type=float, default=1.0)

    ap.add_argument("--topk", type=int, default=200, help="keep topk candidates per query after fusion")
    ap.add_argument("--keep_orig_scores", action="store_true",
                    help="add dense_score/sparse_score fields for debugging")

    args = ap.parse_args()

    dense = load_run(Path(args.dense_jsonl))
    sparse = load_run(Path(args.sparse_jsonl))

    qids = sorted(set(dense.keys()) | set(sparse.keys()))
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fout:
        for qid in qids:
            d_obj = dense.get(qid)
            s_obj = sparse.get(qid)

            # collection name: prefer dense, fallback sparse
            collection = None
            if d_obj is not None:
                collection = d_obj.get("Collection")
            if collection is None and s_obj is not None:
                collection = s_obj.get("Collection")

            # sanity: if both exist and differ -> raise
            if d_obj is not None and s_obj is not None:
                cd = d_obj.get("Collection")
                cs = s_obj.get("Collection")
                if cd and cs and cd != cs:
                    raise ValueError(f"Collection mismatch for qid={qid}: dense={cd} sparse={cs}")

            d_ctxs = d_obj.get("contexts", []) if d_obj is not None else []
            s_ctxs = s_obj.get("contexts", []) if s_obj is not None else []

            if args.method == "rrf":
                fused = rrf_fuse(
                    d_ctxs, s_ctxs,
                    rrf_k=args.rrf_k,
                    w_dense=args.w_dense,
                    w_sparse=args.w_sparse,
                )
            else:
                fused = linear_fuse(
                    d_ctxs, s_ctxs,
                    w_dense=args.w_dense,
                    w_sparse=args.w_sparse,
                )

            # optional: attach original scores for debugging
            dense_map = {str(c["document_id"]): float(c.get("score", 0.0)) for c in d_ctxs}
            sparse_map = {str(c["document_id"]): float(c.get("score", 0.0)) for c in s_ctxs}

            # sort by fused score desc
            items = sorted(fused.items(), key=lambda x: x[1], reverse=True)[: args.topk]

            ctxs_out = []
            for did, sc in items:
                row = {"document_id": did, "score": float(sc)}
                if args.keep_orig_scores:
                    if did in dense_map:
                        row["dense_score"] = dense_map[did]
                    if did in sparse_map:
                        row["sparse_score"] = sparse_map[did]
                ctxs_out.append(row)

            fout.write(json.dumps(
                {"task_id": qid, "contexts": ctxs_out, "Collection": collection},
                ensure_ascii=False
            ) + "\n")

    print(f"[DONE] wrote -> {out_path} (queries={len(qids)})")


if __name__ == "__main__":
    main()
