#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taska_jsonl", required=True)
    ap.add_argument("--rag_gold_jsonl", required=True)
    ap.add_argument("--answers_csv", required=True)
    ap.add_argument("--answer_col", required=True)
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    # gold index by task_id
    gold = {}
    for j in read_jsonl(Path(args.rag_gold_jsonl)):
        tid = j.get("task_id") or j.get("_id") or j.get("id")
        if tid:
            gold[str(tid)] = j
    print(f"[GOLD] loaded={len(gold)} from {args.rag_gold_jsonl}")

    # answers index
    df = pd.read_csv(args.answers_csv)
    if "task_id" not in df.columns:
        raise ValueError("answers_csv must contain column: task_id")
    if args.answer_col not in df.columns:
        raise ValueError(f"answers_csv missing answer_col={args.answer_col!r}. columns={list(df.columns)}")

    ans_map = {}
    for _, r in df.iterrows():
        tid = str(r["task_id"])
        a = r[args.answer_col]
        if pd.isna(a):
            continue
        a = str(a).strip()
        if a:
            ans_map[tid] = a
    print(f"[ANS] loaded={len(ans_map)} from {args.answers_csv} col={args.answer_col}")

    kept = 0
    skip_no_ans = 0
    skip_no_gold = 0
    skip_gold_no_targets = 0

    outp = Path(args.out_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)

    with outp.open("w", encoding="utf-8") as w:
        for rec in tqdm(read_jsonl(Path(args.taska_jsonl)), desc="merge->eval_input"):
            tid = str(rec.get("task_id") or "")
            if not tid:
                continue

            if tid not in ans_map:
                skip_no_ans += 1
                continue

            g = gold.get(tid)
            if g is None:
                skip_no_gold += 1
                continue

            targets = g.get("targets", None)
            if not isinstance(targets, list) or len(targets) == 0:
                skip_gold_no_targets += 1
                continue

            out = dict(rec)
            out["targets"] = targets
            if "input" in g and g["input"] is not None:
                out["input"] = g["input"]
            if "answerability" in g:
                out["answerability"] = g["answerability"]
            if "Question Type" in g:
                out["Question Type"] = g["Question Type"]

            out["predictions"] = [{"text": ans_map[tid]}]

            w.write(json.dumps(out, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[DONE] kept={kept} skip_no_ans={skip_no_ans} skip_no_gold={skip_no_gold} skip_gold_no_targets={skip_gold_no_targets}")
    print(f"[OUT] {outp}")

if __name__ == "__main__":
    main()
