#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CSV-only algorithmic scorer for ans_df-style files.

Input CSV default columns:
  - raw_answer: model answer / prediction
  - target_text: reference answer

Outputs:
  1) JSONL with original row fields plus:
       predictions, targets, metrics
  2) optional flat summary CSV via --summary_csv

Metrics:
  - RougeL_stemFalse
  - Recall: lexical token recall after normalization
  - BertscoreP
  - BertscoreR
  - BertscoreF1
  - BertKPrec: same as BertscoreP for answer-vs-target CSV evaluation
  - RB_agg: harmonic mean of rescaled BertscoreR, RougeL_stemFalse, rescaled BertKPrec
"""

from __future__ import annotations
import sys, shlex, datetime
import argparse
import csv
import html
import json
import os
import re
import string
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import evaluate
import pandas as pd
import torch
from bs4 import BeautifulSoup
from tqdm import tqdm


BERTSCORE_MODEL_TYPE = os.environ.get("BERTSCORE_MODEL_TYPE", "microsoft/deberta-v3-large")


def increase_csv_field_limit() -> None:
    max_size = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_size)
            return
        except OverflowError:
            max_size = int(max_size / 10)


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def remove_articles(text: str) -> str:
    return re.sub(r"\b(a|an|the)\b", " ", text)


def remove_punc(text: str) -> str:
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)


def normalize_white_spaces(text: str) -> str:
    return " ".join([x for x in text.split() if x])


def normalize_text(text: str) -> str:
    return normalize_white_spaces(remove_articles(remove_punc(safe_str(text).lower())))


def lexical_recall(prediction: str, target: str) -> float:
    prediction_tokens = normalize_text(prediction).split()
    target_tokens = normalize_text(target).split()
    if not target_tokens:
        return 0.0
    common_token = Counter(prediction_tokens) & Counter(target_tokens)
    num_common_tokens = sum(common_token.values())
    if num_common_tokens == 0:
        return 0.0
    return num_common_tokens / len(target_tokens)


def rb_agg(rouge_l: float, bert_r: float, bert_kprec: float) -> float:
    """
    Same formula as run_algorithmic.py rb_agg:
      recall = (BertscoreR + 1) / 2
      rouge = RougeL_stemFalse
      extractiveness = (max(BertKPrec) + 1) / 2
    Here each row has one target, so max(BertKPrec) == BertKPrec.
    """
    recall = (bert_r + 1.0) / 2.0
    rouge = float(rouge_l or 0.0)
    extractiveness = (bert_kprec + 1.0) / 2.0

    if recall == 0 or rouge == 0 or extractiveness == 0:
        return 0.0

    return 3 * recall * rouge * extractiveness / (
        recall * rouge + recall * extractiveness + rouge * extractiveness
    )


def load_metrics():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    rouge_metric = evaluate.load("rouge", experiment_id=f"rme.{timestamp}")
    bertscore_metric = evaluate.load("bertscore", experiment_id=f"bs.{timestamp}")
    return rouge_metric, bertscore_metric


def compute_scores(
    df: pd.DataFrame,
    prediction_col: str,
    target_col: str,
    batch_size: int,
) -> pd.DataFrame:
    if prediction_col not in df.columns:
        raise ValueError(f"Missing prediction column {prediction_col!r}. Columns: {list(df.columns)}")
    if target_col not in df.columns:
        raise ValueError(f"Missing target column {target_col!r}. Columns: {list(df.columns)}")

    preds = df[prediction_col].map(safe_str).tolist()
    refs = df[target_col].map(safe_str).tolist()

    rouge_metric, bertscore_metric = load_metrics()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] BERTSCORE_MODEL_TYPE={BERTSCORE_MODEL_TYPE}")
    print(f"[INFO] device={device}")
    print("[INFO] computing RougeL...")

    rouge_out = rouge_metric.compute(
        predictions=preds,
        references=refs,
        rouge_types=["rougeL"],
        use_aggregator=False,
        use_stemmer=False,
    )
    rouge_l_scores = rouge_out["rougeL"]

    print("[INFO] computing lexical Recall...")
    recall_scores = [lexical_recall(p, r) for p, r in zip(preds, refs)]

    print("[INFO] computing BERTScore...")
    bert_p: List[float] = []
    bert_r: List[float] = []
    bert_f1: List[float] = []

    for start in tqdm(range(0, len(preds), batch_size), desc="BERTScore batches"):
        batch_preds = preds[start:start + batch_size]
        batch_refs = refs[start:start + batch_size]
        score = bertscore_metric.compute(
            predictions=batch_preds,
            references=batch_refs,
            lang="en",
            rescale_with_baseline=False,
            model_type=BERTSCORE_MODEL_TYPE,
            device=device,
            batch_size=batch_size,
        )
        bert_p.extend(score["precision"])
        bert_r.extend(score["recall"])
        bert_f1.extend(score["f1"])

    out = df.copy()
    out["RougeL_stemFalse"] = rouge_l_scores
    out["Recall"] = recall_scores
    out["BertscoreP"] = bert_p
    out["BertscoreR"] = bert_r
    out["BertscoreF1"] = bert_f1
    out["BertKPrec"] = bert_p
    out["RB_agg"] = [
        rb_agg(rl, br, bp)
        for rl, br, bp in zip(rouge_l_scores, bert_r, bert_p)
    ]
    return out


def row_to_jsonl_obj(row: pd.Series, prediction_col: str, target_col: str) -> Dict[str, Any]:
    task_id = safe_str(
        row.get("task_id")
        or row.get("_id")
        or row.get("id")
        or row.get("question_id")
    )
    conversation_id = safe_str(row.get("conversation_id") or (task_id.split("<::>")[0] if task_id else ""))
    prediction = safe_str(row.get(prediction_col))
    target = safe_str(row.get(target_col))

    obj: Dict[str, Any] = {}
    for k, v in row.to_dict().items():
        # make JSON serialization safe for numpy / pandas scalars
        if pd.isna(v) if not isinstance(v, (list, dict)) else False:
            obj[k] = None
        else:
            try:
                obj[k] = v.item()  # numpy scalar
            except Exception:
                obj[k] = v

    obj["task_id"] = task_id
    obj["conversation_id"] = conversation_id
    obj["predictions"] = [{"text": prediction}]
    obj["targets"] = [{"text": target}]
    obj["metrics"] = {
        "RougeL_stemFalse": [float(row["RougeL_stemFalse"])],
        "Recall": [float(row["Recall"])],
        "BertscoreP": [float(row["BertscoreP"])],
        "BertscoreR": [float(row["BertscoreR"])],
        "BertscoreF1": [float(row["BertscoreF1"])],
        "BertKPrec": [float(row["BertKPrec"])],
        "RB_agg": [float(row["RB_agg"])],
    }
    return obj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input ans_df CSV")
    parser.add_argument("-o", "--output", required=True, help="Output JSONL with metrics")
    parser.add_argument("--prediction_col", default="raw_answer", help="Prediction column. Default: raw_answer")
    parser.add_argument("--target_col", default="target_text", help="Target/reference column. Default: target_text")
    parser.add_argument("--batch_size", type=int, default=64, help="BERTScore batch size")
    parser.add_argument("--summary_csv", default=None, help="Optional flat CSV with metric columns")
    args = parser.parse_args()

    increase_csv_field_limit()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] reading {in_path}")
    df = pd.read_csv(in_path, dtype=str, keep_default_na=False)
    print(f"[INFO] rows={len(df)}")

    cmd = " ".join(shlex.quote(x) for x in sys.argv)
    print(f"[CMD] {cmd}")
    print(f"[TIME] {datetime.now().isoformat(timespec='seconds')}")

    scored_df = compute_scores(
        df=df,
        prediction_col=args.prediction_col,
        target_col=args.target_col,
        batch_size=args.batch_size,
    )

    print(f"[INFO] writing JSONL metrics to {out_path}")
    with out_path.open("w", encoding="utf-8") as f:
        for _, row in scored_df.iterrows():
            obj = row_to_jsonl_obj(row, args.prediction_col, args.target_col)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    if args.summary_csv:
        summary_path = Path(args.summary_csv)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        scored_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] wrote summary CSV to {summary_path}")

    print("========== SUMMARY ==========")
    for metric in ["RougeL_stemFalse", "Recall", "BertscoreP", "BertscoreR", "BertscoreF1", "BertKPrec", "RB_agg"]:
        print(f"{metric}: mean={scored_df[metric].mean():.6f}")
    if "domain" in scored_df.columns:
        print("\nBy domain RB_agg:")
        print(scored_df.groupby("domain")["RB_agg"].mean().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
