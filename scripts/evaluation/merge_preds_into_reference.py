#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="human/generation_tasks/reference.jsonl")
    ap.add_argument("--preds", required=True, help="your generated file: outputs/taskc_generation_all_domains.jsonl")
    ap.add_argument("--out", required=True, help="merged output jsonl (reference + predictions)")
    ap.add_argument("--strict", action="store_true", help="fail if a prediction is missing for a reference task_id")
    args = ap.parse_args()

    # load predictions: task_id -> predictions
    pred_map = {}
    for j in read_jsonl(Path(args.preds)):
        tid = j.get("task_id") or j.get("_id")
        if tid is None:
            continue
        pred_map[tid] = j.get("predictions", [{"text": ""}])

    missing = 0
    written = 0

    with Path(args.out).open("w", encoding="utf-8") as fout:
        for ref in read_jsonl(Path(args.reference)):
            tid = ref.get("task_id") or ref.get("_id")
            if tid is None:
                continue

            if tid in pred_map:
                ref["predictions"] = pred_map[tid]
            else:
                missing += 1
                if args.strict:
                    raise KeyError(f"Missing prediction for task_id={tid}")
                # 不中断评测：留空预测
                ref["predictions"] = [{"text": ""}]

            fout.write(json.dumps(ref, ensure_ascii=False) + "\n")
            written += 1

    print(f"[OK] wrote={written}, missing_predictions={missing}")

if __name__ == "__main__":
    main()
