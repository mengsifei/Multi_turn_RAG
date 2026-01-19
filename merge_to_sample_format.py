#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge a run/submission jsonl (task_id + contexts) back into the official
sample-like format (conversation_id, task_id, Collection, input, contexts),
and validate that context scores are monotonically non-increasing per query.

Usage:
  python3 merge_to_sample_format.py \
    --orig_jsonl eval_data/rag_taskAC_rewrite_gpt.jsonl \
    --run_jsonl  outputs/my_submission.jsonl \
    --out_jsonl  outputs/my_submission_sample_format.jsonl \
    --score_key  score \
    --monotonic_mode non_increasing \
    --allow_equal

Notes:
- `orig_jsonl` must contain fields: conversation_id, task_id, Collection, input
- `run_jsonl` must contain fields: task_id, contexts (list of dicts)
- contexts item must have document_id and the score key you choose (default: score)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error in {path} at line {i}: {e}") from e
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_orig_map(orig_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    mp: Dict[str, Dict[str, Any]] = {}
    for r in orig_rows:
        tid = r.get("task_id")
        if not tid:
            raise KeyError(f"orig_jsonl row missing task_id: keys={list(r.keys())}")
        # Keep the whole record; we'll copy conversation_id, Collection, input
        mp[tid] = r
    return mp


def build_run_map(run_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    mp: Dict[str, Dict[str, Any]] = {}
    for r in run_rows:
        tid = r.get("task_id")
        if not tid:
            raise KeyError(f"run_jsonl row missing task_id: keys={list(r.keys())}")
        mp[tid] = r
    return mp


def extract_score(ctx: Dict[str, Any], score_key: str) -> Optional[float]:
    v = ctx.get(score_key, None)
    if v is None:
        return None
    # allow int/float or numeric string
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


def is_monotonic(scores: List[float], mode: str, allow_equal: bool) -> bool:
    # mode: "non_increasing" (default) or "decreasing"
    if len(scores) <= 1:
        return True
    for a, b in zip(scores, scores[1:]):
        if mode == "non_increasing":
            if allow_equal:
                if b > a:
                    return False
            else:
                # strictly non-increasing => b < a
                if b >= a:
                    return False
        elif mode == "increasing":
            if allow_equal:
                if b < a:
                    return False
            else:
                if b <= a:
                    return False
        else:
            raise ValueError(f"Unknown monotonic mode: {mode}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_jsonl", type=Path, required=True,
                    help="Official Task A/C input jsonl (has conversation_id/task_id/Collection/input)")
    ap.add_argument("--run_jsonl", type=Path, required=True,
                    help="Your submission/run jsonl (has task_id + contexts)")
    ap.add_argument("--out_jsonl", type=Path, required=True,
                    help="Output jsonl in sample-like format")
    ap.add_argument("--report_path", type=Path, default=None,
                    help="Optional path to write a JSON report about monotonicity violations")
    ap.add_argument("--score_key", type=str, default="score",
                    help="Which key inside each context to check for monotonicity (default: score)")
    ap.add_argument("--monotonic_mode", type=str, default="non_increasing",
                    choices=["non_increasing", "increasing"],
                    help="Check monotonicity direction. Usually non_increasing.")
    ap.add_argument("--allow_equal", action="store_true",
                    help="Allow equal adjacent scores (non-increasing with ties). If not set, requires strict.")
    ap.add_argument("--keep_extra_context_fields", action="store_true",
                    help="Keep extra fields in contexts (e.g., orig_score/rerank_score). Default keeps them anyway.")
    ap.add_argument("--topk", type=int, default=None,
                    help="Optionally truncate contexts to top-k after reading (keeps order).")
    args = ap.parse_args()

    orig_rows = read_jsonl(args.orig_jsonl)
    run_rows = read_jsonl(args.run_jsonl)

    orig_map = build_orig_map(orig_rows)
    run_map = build_run_map(run_rows)

    merged: List[Dict[str, Any]] = []
    violations: List[Dict[str, Any]] = []
    missing_in_run: List[str] = []
    missing_in_orig: List[str] = []

    # Merge in the original order (orig_rows order)
    for o in orig_rows:
        tid = o["task_id"]
        r = run_map.get(tid)
        if r is None:
            missing_in_run.append(tid)
            # still output sample-like structure but with empty contexts
            contexts = []
        else:
            contexts = r.get("contexts", [])
            if contexts is None:
                contexts = []
            if not isinstance(contexts, list):
                raise TypeError(f"run contexts must be a list for task_id={tid}, got {type(contexts)}")
            if args.topk is not None:
                contexts = contexts[: args.topk]

        out_obj = {
            "conversation_id": o.get("conversation_id"),
            "task_id": tid,
            "Collection": o.get("Collection"),
            "input": o.get("input", []),
            "contexts": contexts,
        }
        merged.append(out_obj)

        # Monotonicity check
        extracted: List[float] = []
        missing_scores = 0
        for c in contexts:
            sc = extract_score(c, args.score_key)
            if sc is None:
                missing_scores += 1
            else:
                extracted.append(sc)

        # Only check if we have at least 2 extracted scores AND they match contexts length (otherwise ambiguous)
        # If you prefer: check monotonicity on available scores only; we do that, but report missing.
        ok = is_monotonic(extracted, args.monotonic_mode, args.allow_equal) if len(extracted) >= 2 else True
        if not ok:
            violations.append({
                "task_id": tid,
                "Collection": o.get("Collection"),
                "score_key": args.score_key,
                "mode": args.monotonic_mode,
                "allow_equal": args.allow_equal,
                "scores": extracted[:50],  # cap for readability
                "num_contexts": len(contexts),
                "num_scores_missing": missing_scores,
            })

    # Check run rows not present in orig (extra)
    for r in run_rows:
        tid = r.get("task_id")
        if tid and tid not in orig_map:
            missing_in_orig.append(tid)

    write_jsonl(args.out_jsonl, merged)

    report = {
        "orig_jsonl": str(args.orig_jsonl),
        "run_jsonl": str(args.run_jsonl),
        "out_jsonl": str(args.out_jsonl),
        "score_key": args.score_key,
        "monotonic_mode": args.monotonic_mode,
        "allow_equal": args.allow_equal,
        "num_orig": len(orig_rows),
        "num_run": len(run_rows),
        "missing_in_run": {
            "count": len(missing_in_run),
            "task_ids_preview": missing_in_run[:20],
        },
        "extra_in_run_not_in_orig": {
            "count": len(missing_in_orig),
            "task_ids_preview": missing_in_orig[:20],
        },
        "monotonicity_violations": {
            "count": len(violations),
            "preview": violations[:10],
        },
    }

    if args.report_path is None:
        # default: alongside out_jsonl
        args.report_path = args.out_jsonl.with_suffix(args.out_jsonl.suffix + ".report.json")

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] wrote merged sample-format jsonl: {args.out_jsonl}")
    print(f"[DONE] wrote report: {args.report_path}")
    print(f"[STATS] missing_in_run={len(missing_in_run)} extra_in_run_not_in_orig={len(missing_in_orig)}")
    print(f"[STATS] monotonicity_violations={len(violations)} (score_key={args.score_key})")


if __name__ == "__main__":
    main()
