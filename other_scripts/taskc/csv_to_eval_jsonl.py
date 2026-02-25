#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
import pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_jsonl", required=True, help="test100_taska.jsonl (must contain contexts+targets+input etc.)")
    ap.add_argument("--ans_csv", required=True, help="ans_official.csv")
    ap.add_argument("--answer_col", required=True, help="column name in csv, e.g. raw_answer_official")
    ap.add_argument("--out_jsonl", required=True, help="output jsonl for evaluation")
    ap.add_argument("--overwrite_preds", action="store_true", help="overwrite predictions even if already present")
    args = ap.parse_args()

    df = pd.read_csv(args.ans_csv, dtype={"task_id": str})
    if "task_id" not in df.columns:
        raise ValueError("ans_csv must contain column: task_id")
    if args.answer_col not in df.columns:
        raise ValueError(f"ans_csv missing answer_col={args.answer_col}. cols={list(df.columns)}")

    ans_map = {}
    for _, r in df.iterrows():
        tid = str(r["task_id"])
        ans = "" if pd.isna(r[args.answer_col]) else str(r[args.answer_col])
        ans_map[tid] = ans

    base_path = Path(args.base_jsonl)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    missing_ans = 0

    with base_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            tid = str(j.get("task_id", ""))

            if not tid or tid not in ans_map:
                missing_ans += 1
                continue

            ans = ans_map[tid].strip()
            if (not args.overwrite_preds) and j.get("predictions"):
                # already has predictions
                pass
            else:
                j["predictions"] = [{"text": ans}]

            fout.write(json.dumps(j, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[DONE] wrote={out_path} kept={kept} missing_answer_for={missing_ans}")

if __name__ == "__main__":
    main()
