#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import pandas as pd


def safe_str(x):
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def parse_answerability(x):
    """
    Supports:
    - '["ANSWERABLE"]'
    - '["PARTIAL"]'
    - '["UNANSWERABLE"]'
    - 'ANSWERABLE'
    """
    if isinstance(x, list):
        vals = x
    else:
        s = safe_str(x).strip()
        try:
            vals = json.loads(s)
            if not isinstance(vals, list):
                vals = [vals]
        except Exception:
            vals = [s]

    vals = [safe_str(v).strip().upper() for v in vals if safe_str(v).strip()]
    return vals[0] if vals else ""


def normalize_idk_label(x):
    s = safe_str(x).strip().lower()
    s = re.sub(r"[^a-z]", "", s)

    if s.startswith("yes"):
        return "yes"
    if s.startswith("no"):
        return "no"
    if s.startswith("partial"):
        return "partial"
    return "parse_error"


def idk_judge_score(answerability, idk_label):
    """
    Judge-quality score against gold answerability.

    ANSWERABLE/PARTIAL:
        no or partial -> 1
        yes -> 0

    UNANSWERABLE/CONVERSATIONAL:
        yes -> 1
        no or partial -> 0

    parse_error -> 0
    """
    a = parse_answerability(answerability)
    y = normalize_idk_label(idk_label)

    if y == "parse_error":
        return 0

    if a in {"UNANSWERABLE", "CONVERSATIONAL"}:
        return 1 if y == "yes" else 0

    if a in {"ANSWERABLE", "PARTIAL"}:
        return 1 if y in {"no", "partial"} else 0

    # Unknown gold label: count as wrong
    return 0


def load_jsonl(path, judge_name, limit=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            if not line.strip():
                continue

            obj = json.loads(line)

            answerability = obj.get("answerability", "")
            idk_label = obj.get("idk_label", obj.get("idk_raw", ""))

            a_norm = parse_answerability(answerability)
            y_norm = normalize_idk_label(idk_label)
            score = idk_judge_score(answerability, idk_label)

            rows.append({
                "row_index": obj.get("row_index", i),
                "task_id": obj.get("task_id", ""),
                "answerability": answerability,
                "answerability_norm": a_norm,
                f"{judge_name}_raw": obj.get("idk_raw", ""),
                f"{judge_name}_label": y_norm,
                f"{judge_name}_score": score,
                "question": obj.get("question", ""),
                "raw_answer": obj.get("raw_answer", ""),
            })

    return pd.DataFrame(rows)


def summarize(df, judge_name):
    score_col = f"{judge_name}_score"
    label_col = f"{judge_name}_label"

    print(f"\n===== {judge_name} =====")
    print("rows:", len(df))
    print("mean_score:", df[score_col].mean())
    print("\nLabel counts:")
    print(df[label_col].value_counts(dropna=False).to_string())
    print("\nGold x judge:")
    print(pd.crosstab(df["answerability_norm"], df[label_col]).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemma_jsonl", required=True)
    ap.add_argument("--deepseek_jsonl", required=True)
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    gemma = load_jsonl(args.gemma_jsonl, "gemma", limit=args.limit)
    deepseek = load_jsonl(args.deepseek_jsonl, "deepseek", limit=args.limit)

    summarize(gemma, "gemma")
    summarize(deepseek, "deepseek")

    merged = gemma.merge(
        deepseek[[
            "row_index",
            "task_id",
            "deepseek_raw",
            "deepseek_label",
            "deepseek_score",
        ]],
        on=["row_index", "task_id"],
        how="outer"
    )

    print("\n===== comparison =====")
    print("rows:", len(merged))
    print("gemma_mean:", merged["gemma_score"].mean())
    print("deepseek_mean:", merged["deepseek_score"].mean())

    diff = merged[merged["gemma_label"] != merged["deepseek_label"]]
    print("label_differences:", len(diff))

    if len(diff) > 0:
        print("\nDifferent labels:")
        cols = [
            "row_index",
            "task_id",
            "answerability_norm",
            "gemma_label",
            "gemma_score",
            "deepseek_label",
            "deepseek_score",
            "question",
            "raw_answer",
        ]
        print(diff[cols].to_string(index=False))

    if args.out_csv:
        merged.to_csv(args.out_csv, index=False)
        print(f"\n[DONE] wrote {args.out_csv}")


if __name__ == "__main__":
    main()