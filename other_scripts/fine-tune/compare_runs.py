#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare two retrieval/rerank runs in MTRAG jsonl format by per-query metrics
using pytrec_eval (same family as official eval).

Example:
python compare_runs.py \
  --qrels_tsv human/retrieval_tasks/fiqa/qrels/dev.tsv \
  --run_a outputs/OLD.jsonl \
  --run_b outputs/NEW.jsonl \
  --k 5 \
  --topn 30
"""

import argparse
import csv
import json
from collections import defaultdict
from typing import Dict, Tuple, List

import pytrec_eval


def load_qrels_tsv(path: str) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        # expected header: query-id corpus-id score
        for row in reader:
            if not row or len(row) < 3:
                continue
            qid, docid, score = row[0], row[1], row[2]
            try:
                score_i = int(float(score))
            except Exception:
                score_i = 0
            qrels[qid][docid] = score_i
    return qrels


def load_run_jsonl(path: str) -> Dict[str, Dict[str, float]]:
    """
    MTRAG jsonl: {"task_id": "...", "contexts": [{"document_id": "...", "score": ...}, ...], "Collection": "..."}
    Return results dict in pytrec_eval format: {qid: {docid: score}}
    """
    results: Dict[str, Dict[str, float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            qid = o.get("task_id") or o.get("query_id")
            if not qid:
                continue
            ctxs = o.get("contexts") or []
            d: Dict[str, float] = {}
            for c in ctxs:
                did = c.get("document_id") or c.get("doc_id") or c.get("corpus_id")
                if not did:
                    continue
                try:
                    s = float(c.get("score", 0.0))
                except Exception:
                    s = 0.0
                # if duplicates appear, keep max score
                if did not in d or s > d[did]:
                    d[did] = s
            results[qid] = d
    return results


def eval_per_query(qrels: Dict[str, Dict[str, int]],
                   results: Dict[str, Dict[str, float]],
                   k: int) -> Dict[str, Dict[str, float]]:
    """
    Return per-query metrics dict from pytrec_eval
    """
    measures = {
        f"ndcg_cut_{k}",
        f"recall_{k}",
        f"map_cut_{k}",
        f"P_{k}",
    }
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
    return evaluator.evaluate(results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qrels_tsv", required=True)
    ap.add_argument("--run_a", required=True)
    ap.add_argument("--run_b", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--topn", type=int, default=30, help="Show top-N biggest drops and gains.")
    args = ap.parse_args()

    qrels = load_qrels_tsv(args.qrels_tsv)
    run_a = load_run_jsonl(args.run_a)
    run_b = load_run_jsonl(args.run_b)

    eva = eval_per_query(qrels, run_a, args.k)
    evb = eval_per_query(qrels, run_b, args.k)

    # intersect only queries present in qrels (pytrec_eval evaluates those)
    qids = sorted(set(qrels.keys()))
    rows: List[Tuple[float, str, float, float, float, float]] = []
    # (delta_ndcg, qid, ndcg_a, ndcg_b, recall_a, recall_b)
    for qid in qids:
        a = eva.get(qid, {})
        b = evb.get(qid, {})
        ndcg_a = float(a.get(f"ndcg_cut_{args.k}", 0.0))
        ndcg_b = float(b.get(f"ndcg_cut_{args.k}", 0.0))
        rec_a = float(a.get(f"recall_{args.k}", 0.0))
        rec_b = float(b.get(f"recall_{args.k}", 0.0))
        rows.append((ndcg_b - ndcg_a, qid, ndcg_a, ndcg_b, rec_a, rec_b))

    rows_sorted = sorted(rows, key=lambda x: x[0])  # ascending: biggest drops first

    def show(title: str, subset: List[Tuple[float, str, float, float, float, float]]):
        print("\n" + title)
        print(f"{'delta':>9}  {'qid':<40}  {'ndcg_a':>7}  {'ndcg_b':>7}  {'rec_a':>7}  {'rec_b':>7}")
        for d, qid, na, nb, ra, rb in subset:
            print(f"{d:9.4f}  {qid:<40}  {na:7.4f}  {nb:7.4f}  {ra:7.4f}  {rb:7.4f}")

    show(f"Top {args.topn} DROPS (B - A) for nDCG@{args.k}",
         rows_sorted[: args.topn])

    show(f"Top {args.topn} GAINS (B - A) for nDCG@{args.k}",
         list(reversed(rows_sorted[-args.topn:])))

    # summary
    avg_delta = sum(r[0] for r in rows) / max(1, len(rows))
    print(f"\nSummary over {len(rows)} qrels queries: mean delta nDCG@{args.k} = {avg_delta:.6f}")


if __name__ == "__main__":
    main()
