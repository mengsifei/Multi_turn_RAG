#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def dedup_last(df: pd.DataFrame, key: str = "task_id") -> pd.DataFrame:
    if key not in df.columns:
        raise ValueError(f"Missing key column {key!r}. Columns={list(df.columns)}")
    return df.drop_duplicates(subset=[key], keep="last").copy()


def safe_float(x: Any) -> float:
    try:
        if x is None:
            return np.nan
        s = str(x).strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            return np.nan
        return float(s)
    except Exception:
        return np.nan


def parse_answerability(x: Any) -> str:
    s = "" if x is None else str(x).strip()
    if not s:
        return ""
    try:
        obj = json.loads(s)
    except Exception:
        try:
            obj = ast.literal_eval(s)
        except Exception:
            return s.upper()
    if isinstance(obj, list) and obj:
        return str(obj[0]).upper()
    return str(obj).upper()


def get_eval_group(answerability_norm: Any) -> str:
    ans = str(answerability_norm).upper()
    if "CONVERSATIONAL" in ans:
        return "conversational"
    if "UNANSWERABLE" in ans:
        return "unanswerable"
    if "ANSWERABLE" in ans or "PARTIAL" in ans:
        return "answerable_partial"
    return "unknown"


def hm(values: List[float]) -> float:
    vals = []
    for x in values:
        try:
            x = float(x)
        except Exception:
            return np.nan
        if math.isnan(x):
            return np.nan
        if x <= 0:
            return 0.0
        vals.append(x)
    if not vals:
        return np.nan
    return len(vals) / sum(1.0 / x for x in vals)


def normalize_idk_label(x: Any) -> str:
    s = str(x).strip().lower()
    if s in {"yes", "y", "1", "true"}:
        return "yes"
    if s in {"no", "n", "0", "false"}:
        return "no"
    if s in {"partial", "partially", "part", "0.5"}:
        return "partial"
    if not s or s in {"nan", "none", "null"}:
        return "missing"
    return s


def idk_correct(row: pd.Series, partial_as_answer: bool = True) -> float:
    group = row["eval_group"]
    label = row["idk_label_norm"]
    answer_labels = {"no", "partial"} if partial_as_answer else {"no"}

    if group == "answerable_partial":
        return 1.0 if label in answer_labels else 0.0
    if group == "unanswerable":
        return 1.0 if label == "yes" else 0.0
    return np.nan


def condition_metric(row: pd.Series, raw_col: str, partial_as_answer: bool = True) -> float:
    group = row["eval_group"]
    label = row["idk_label_norm"]
    answer_labels = {"no", "partial"} if partial_as_answer else {"no"}

    if group == "answerable_partial":
        if label in answer_labels:
            return row[raw_col]
        if label == "yes":
            return 0.0
        return np.nan

    if group == "unanswerable":
        if label == "yes":
            return 1.0
        if label in {"no", "partial"}:
            return 0.0
        return np.nan

    return np.nan


def mean_or_nan(s: pd.Series) -> float:
    vals = pd.to_numeric(s, errors="coerce")
    return float(vals.mean()) if vals.notna().any() else float("nan")


def build_summary(df: pd.DataFrame, main_df: pd.DataFrame) -> Dict[str, Any]:
    metric_pairs = {
        "alg_RB_agg": "alg_RB_agg_idk_cond",
        "faith_RL_F": "faith_RL_F_idk_cond",
        "rbllm_RB_llm": "rbllm_RB_llm_idk_cond",
        "HM_alg_faith_rbllm_raw": "HM_alg_faith_rbllm_idk_cond",
    }

    summary: Dict[str, Any] = {
        "counts": {
            "all_rows": int(len(df)),
            "main_rows": int(len(main_df)),
            "conversational_rows": int((df["eval_group"] == "conversational").sum()),
            "unknown_rows": int((df["eval_group"] == "unknown").sum()),
        },
        "raw_all": {},
        "raw_main": {},
        "idk_conditioned_main": {},
        "by_eval_group": {},
    }

    for raw_col, cond_col in metric_pairs.items():
        summary["raw_all"][raw_col] = mean_or_nan(df[raw_col])
        summary["raw_main"][raw_col] = mean_or_nan(main_df[raw_col])
        summary["idk_conditioned_main"][cond_col] = mean_or_nan(main_df[cond_col])

    summary["idk_score_main"] = mean_or_nan(main_df["idk_correct"])

    ct = pd.crosstab(df["eval_group"], df["idk_label_norm"], margins=True)
    summary["idk_distribution"] = json.loads(ct.to_json())

    for group, gdf in df.groupby("eval_group"):
        gd: Dict[str, Any] = {
            "n": int(len(gdf)),
            "idk_score": mean_or_nan(gdf["idk_correct"]),
        }
        for raw_col, cond_col in metric_pairs.items():
            gd[f"raw_{raw_col}"] = mean_or_nan(gdf[raw_col])
            gd[f"cond_{cond_col}"] = mean_or_nan(gdf[cond_col])
        summary["by_eval_group"][group] = gd

    return summary


def flatten_summary(summary: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for scope in ("raw_all", "raw_main", "idk_conditioned_main"):
        for metric, value in summary.get(scope, {}).items():
            rows.append({"scope": scope, "metric": metric, "value": value})

    rows.append({"scope": "idk", "metric": "idk_score_main", "value": summary.get("idk_score_main")})

    for group, gd in summary.get("by_eval_group", {}).items():
        rows.append({"scope": f"group:{group}", "metric": "n", "value": gd.get("n")})
        rows.append({"scope": f"group:{group}", "metric": "idk_score", "value": gd.get("idk_score")})
        for k, v in gd.items():
            if k in {"n", "idk_score"}:
                continue
            rows.append({"scope": f"group:{group}", "metric": k, "value": v})

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--base_csv", required=True)
    ap.add_argument("--alg_csv", required=True)
    ap.add_argument("--faith_csv", required=True)
    ap.add_argument("--rbllm_csv", required=True)

    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--summary_json", required=True)
    ap.add_argument("--summary_csv", required=True)

    ap.add_argument("--task_id_col", default="task_id")
    ap.add_argument("--answerability_col", default="answerability")
    ap.add_argument("--idk_col", default="idk_label")

    ap.add_argument("--alg_col", default="RB_agg")
    ap.add_argument("--faith_col", default="RL_F")
    ap.add_argument("--rbllm_col", default="RB_llm")

    ap.add_argument(
        "--partial_as_idk",
        action="store_true",
        help="Treat idk_label=partial as IDK. Default treats partial as attempted answer.",
    )
    ap.add_argument(
        "--include_conversational_in_main",
        action="store_true",
        help="By default conversational rows are excluded from main average.",
    )

    args = ap.parse_args()
    partial_as_answer = not bool(args.partial_as_idk)

    base = dedup_last(read_csv(args.base_csv), key=args.task_id_col)
    alg = dedup_last(read_csv(args.alg_csv), key=args.task_id_col)
    faith = dedup_last(read_csv(args.faith_csv), key=args.task_id_col)
    rbllm = dedup_last(read_csv(args.rbllm_csv), key=args.task_id_col)

    if args.answerability_col not in base.columns:
        raise ValueError(f"Base CSV missing answerability column {args.answerability_col!r}. Columns={list(base.columns)}")
    if args.idk_col not in base.columns:
        raise ValueError(f"Base CSV missing IDK column {args.idk_col!r}. Columns={list(base.columns)}")

    for df_name, mdf, col in [
        ("alg_csv", alg, args.alg_col),
        ("faith_csv", faith, args.faith_col),
        ("rbllm_csv", rbllm, args.rbllm_col),
    ]:
        if col not in mdf.columns:
            raise ValueError(f"{df_name} missing metric column {col!r}. Columns={list(mdf.columns)}")

    base = base.rename(columns={args.task_id_col: "task_id"}).copy()
    alg_small = alg[[args.task_id_col, args.alg_col]].rename(columns={args.task_id_col: "task_id", args.alg_col: "alg_RB_agg"})
    faith_small = faith[[args.task_id_col, args.faith_col]].rename(columns={args.task_id_col: "task_id", args.faith_col: "faith_RL_F"})
    rbllm_small = rbllm[[args.task_id_col, args.rbllm_col]].rename(columns={args.task_id_col: "task_id", args.rbllm_col: "rbllm_RB_llm"})

    df = (
        base
        .merge(alg_small, on="task_id", how="left")
        .merge(faith_small, on="task_id", how="left")
        .merge(rbllm_small, on="task_id", how="left")
    )

    for col in ["alg_RB_agg", "faith_RL_F", "rbllm_RB_llm"]:
        df[col] = df[col].apply(safe_float)

    df["answerability_norm"] = df[args.answerability_col].apply(parse_answerability)
    df["eval_group"] = df["answerability_norm"].apply(get_eval_group)
    df["idk_label_norm"] = df[args.idk_col].apply(normalize_idk_label)

    df["HM_alg_faith_rbllm_raw"] = [
        hm([a, f, r])
        for a, f, r in zip(df["alg_RB_agg"], df["faith_RL_F"], df["rbllm_RB_llm"])
    ]

    df["idk_correct"] = df.apply(lambda row: idk_correct(row, partial_as_answer=partial_as_answer), axis=1)

    for raw_col, cond_col in [
        ("alg_RB_agg", "alg_RB_agg_idk_cond"),
        ("faith_RL_F", "faith_RL_F_idk_cond"),
        ("rbllm_RB_llm", "rbllm_RB_llm_idk_cond"),
    ]:
        df[cond_col] = df.apply(lambda row: condition_metric(row, raw_col, partial_as_answer=partial_as_answer), axis=1)

    df["HM_alg_faith_rbllm_idk_cond"] = [
        hm([a, f, r])
        for a, f, r in zip(df["alg_RB_agg_idk_cond"], df["faith_RL_F_idk_cond"], df["rbllm_RB_llm_idk_cond"])
    ]

    if args.include_conversational_in_main:
        main_df = df[df["eval_group"].isin(["answerable_partial", "unanswerable", "conversational"])].copy()
    else:
        main_df = df[df["eval_group"].isin(["answerable_partial", "unanswerable"])].copy()

    summary = build_summary(df, main_df)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    summary_csv = Path(args.summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    flatten_summary(summary).to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print("========== Task C Metric Summary ==========")
    print(f"all rows: {summary['counts']['all_rows']}")
    print(f"main rows: {summary['counts']['main_rows']}")
    print(f"conversational rows: {summary['counts']['conversational_rows']}")
    print()
    print("Raw main:")
    print(f"  ALG RB_agg: {summary['raw_main']['alg_RB_agg']:.6f}")
    print(f"  Faith RL_F: {summary['raw_main']['faith_RL_F']:.6f}")
    print(f"  RB_llm:     {summary['raw_main']['rbllm_RB_llm']:.6f}")
    print(f"  HM raw:     {summary['raw_main']['HM_alg_faith_rbllm_raw']:.6f}")
    print()
    print("IDK:")
    print(f"  IDK score main: {summary['idk_score_main']:.6f}")
    print()
    print("IDK-conditioned main:")
    print(f"  ALG cond:    {summary['idk_conditioned_main']['alg_RB_agg_idk_cond']:.6f}")
    print(f"  Faith cond:  {summary['idk_conditioned_main']['faith_RL_F_idk_cond']:.6f}")
    print(f"  RB_llm cond: {summary['idk_conditioned_main']['rbllm_RB_llm_idk_cond']:.6f}")
    print(f"  HM cond:     {summary['idk_conditioned_main']['HM_alg_faith_rbllm_idk_cond']:.6f}")
    print()
    print(f"wrote merged CSV: {out_csv}")
    print(f"wrote summary JSON: {summary_json}")
    print(f"wrote summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
