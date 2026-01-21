#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import argparse
from typing import Tuple, Any, Dict
import sys, shlex, datetime

import requests
import pandas as pd


URL = "https://api.deepseek.com/chat/completions"


def call_deepseek(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    retries: int,
) -> Tuple[str, Dict[str, Any]]:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY env var. e.g. export DEEPSEEK_API_KEY='...'")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": False,
    }

    last_err = None
    for attempt in range(retries):
        try:
            r = requests.post(URL, headers=headers, json=payload, timeout=timeout)
            r.raise_for_status()
            j = r.json()
            return j["choices"][0]["message"]["content"], j
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)

    raise RuntimeError(f"DeepSeek call failed after {retries} retries: {last_err}")


def safe_save(df: pd.DataFrame, path: str):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def is_empty(x) -> bool:
    if pd.isna(x):
        return True
    s = str(x).strip()
    return len(s) == 0


def ensure_cols(df: pd.DataFrame, answer_col: str) -> pd.DataFrame:
    # answer_col 可以自定义（比如 raw_answer_official）
    defaults = {
        answer_col: "",
        "raw_response_json": "",
        "model": "",
        "temperature": "",
        "error": "",
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="Input CSV (prompts)")
    ap.add_argument("--out_csv", required=True, help="Output CSV (will resume if exists)")
    ap.add_argument("--prompt_col", required=True, help="Which column contains the prompt")
    ap.add_argument("--answer_col", default="raw_answer", help="Column name to store the answer (default: raw_answer)")

    ap.add_argument("--model", default="deepseek-chat", help="DeepSeek model name")
    ap.add_argument("--temp", type=float, default=0.0, help="Temperature (default: 0.0)")
    ap.add_argument("--max_tokens", type=int, default=1200, help="max_tokens per request")
    ap.add_argument("--timeout", type=int, default=180, help="HTTP timeout seconds")
    ap.add_argument("--retries", type=int, default=3, help="Retries per request")
    ap.add_argument("--save_every", type=int, default=5, help="Save progress every N new answers")
    args = ap.parse_args()

    cmd = " ".join(shlex.quote(x) for x in sys.argv)
    print(f"[CMD] {cmd}")
    print(f"[TIME] {datetime.datetime.now().isoformat(timespec='seconds')}")

    # resume logic: if out_csv exists, load it; else load in_csv
    if os.path.exists(args.out_csv):
        df = pd.read_csv(args.out_csv)
        print(f"[RESUME] loaded existing out_csv={args.out_csv} rows={len(df)}")
    else:
        df = pd.read_csv(args.in_csv)
        print(f"[START] loaded in_csv={args.in_csv} rows={len(df)}")

    if args.prompt_col not in df.columns:
        raise KeyError(f"--prompt_col {args.prompt_col!r} not found in CSV columns={list(df.columns)}")

    df = ensure_cols(df, answer_col=args.answer_col)

    # todo = answer_col empty
    todo_mask = df[args.answer_col].apply(is_empty)
    todo_indices = df.index[todo_mask].tolist()
    print(f"[TODO] remaining={len(todo_indices)} / total={len(df)} (answer_col={args.answer_col})")

    newly_done = 0
    for n, idx in enumerate(todo_indices, 1):
        prompt = df.at[idx, args.prompt_col]

        # prompt 本身也可能为空，直接跳过并记录 error
        if is_empty(prompt):
            df.at[idx, "error"] = f"empty prompt in col={args.prompt_col}"
            continue

        try:
            ans, raw = call_deepseek(
                prompt=str(prompt),
                model=args.model,
                temperature=args.temp,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                retries=args.retries,
            )
            df.at[idx, args.answer_col] = ans
            df.at[idx, "raw_response_json"] = json.dumps(raw, ensure_ascii=False)
            df.at[idx, "model"] = args.model
            df.at[idx, "temperature"] = args.temp
            df.at[idx, "error"] = ""
            newly_done += 1
        except Exception as e:
            # 网络/DNS炸了：先落盘，再退出，下次继续补空行
            df.at[idx, "error"] = str(e)
            safe_save(df, args.out_csv)
            print(f"[ERROR] idx={idx} saved progress -> {args.out_csv}\n{e}")
            print("[EXIT] fix network/DNS then rerun; it will resume automatically.")
            raise SystemExit(1)

        if newly_done % args.save_every == 0:
            safe_save(df, args.out_csv)
            remaining = len(todo_indices) - n
            print(f"[SAVE] newly_done={newly_done} remaining={remaining} -> {args.out_csv}")

    safe_save(df, args.out_csv)
    print(f"[DONE] wrote -> {args.out_csv} (newly_done={newly_done})")


if __name__ == "__main__":
    main()
