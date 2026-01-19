#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import requests
import pandas as pd

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise RuntimeError("Missing DEEPSEEK_API_KEY env var. e.g. export DEEPSEEK_API_KEY='...'")

URL = "https://api.deepseek.com/chat/completions"
HEADERS = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}

def call_deepseek(
    prompt: str,
    model: str = "deepseek-chat",
    temperature: float = 0.0,
    max_tokens: int = 1200,
    timeout: int = 180,
    retries: int = 3,
):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.post(URL, headers=HEADERS, json=payload, timeout=timeout)
            r.raise_for_status()
            j = r.json()
            return j["choices"][0]["message"]["content"], j
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"DeepSeek call failed after {retries} retries: {last_err}")


# ----------------- main -----------------

IN_CSV = "prompts_train.csv"
OUT_CSV = "prompts_train_with_answers_deepseek.csv"

MODEL = "deepseek-chat"
TEMPERATURE = 0.0
MAX_TOKENS = 1200
SAVE_EVERY = 5

def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col, default in [
        ("raw_answer", ""),
        ("raw_response_json", ""),
        ("model", ""),
        ("temperature", ""),
        ("error", ""),   # ✅ 新增：记录失败原因，方便resume
    ]:
        if col not in df.columns:
            df[col] = default
    return df

def safe_save(df: pd.DataFrame, path: str):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

# ✅ 关键：如果 OUT_CSV 已存在，就从它继续；否则从 IN_CSV 开始
if os.path.exists(OUT_CSV):
    df = pd.read_csv(OUT_CSV)
    print(f"[RESUME] loaded existing {OUT_CSV} rows={len(df)}")
else:
    df = pd.read_csv(IN_CSV)
    print(f"[START] loaded {IN_CSV} rows={len(df)}")

df = ensure_cols(df)

def is_empty(x) -> bool:
    if pd.isna(x):
        return True
    s = str(x).strip()
    return len(s) == 0

todo_mask = df["raw_answer"].apply(is_empty)
todo_indices = df.index[todo_mask].tolist()
print(f"[TODO] remaining={len(todo_indices)} / total={len(df)}")

done = 0
for n, idx in enumerate(todo_indices, 1):
    prompt = df.at[idx, "prompt"]

    try:
        ans, raw = call_deepseek(
            prompt,
            model=MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        df.at[idx, "raw_answer"] = ans
        df.at[idx, "raw_response_json"] = json.dumps(raw, ensure_ascii=False)
        df.at[idx, "model"] = MODEL
        df.at[idx, "temperature"] = TEMPERATURE
        df.at[idx, "error"] = ""
        done += 1
    except Exception as e:
        # ✅ 网络/DNS炸了：先落盘，再退出，下次继续补空行
        df.at[idx, "error"] = str(e)
        safe_save(df, OUT_CSV)
        print(f"[ERROR] idx={idx} saved progress -> {OUT_CSV}\n{e}")
        print("[EXIT] fix network/DNS then rerun; it will resume automatically.")
        raise SystemExit(1)

    if done % SAVE_EVERY == 0:
        safe_save(df, OUT_CSV)
        print(f"[SAVE] newly_done={done} remaining={len(todo_indices)-n} -> {OUT_CSV}")

safe_save(df, OUT_CSV)
print(f"[DONE] wrote -> {OUT_CSV}")

# import os
# import json
# import time
# import requests
# import pandas as pd

# DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
# if not DEEPSEEK_API_KEY:
#     raise RuntimeError("Missing DEEPSEEK_API_KEY env var. e.g. export DEEPSEEK_API_KEY='...'")

# URL = "https://api.deepseek.com/chat/completions"
# HEADERS = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}

# def call_deepseek(
#     prompt: str,
#     model: str = "deepseek-chat",
#     temperature: float = 0.0,
#     max_tokens: int = 1200,
#     timeout: int = 180,
#     retries: int = 3,
# ):
#     payload = {
#         "model": model,
#         "messages": [{"role": "user", "content": prompt}],
#         "temperature": temperature,   # ✅ 你原来没传进去
#         "max_tokens": max_tokens,
#         "stream": False,
#     }
#     last_err = None
#     for attempt in range(retries):
#         try:
#             r = requests.post(URL, headers=HEADERS, json=payload, timeout=timeout)
#             r.raise_for_status()
#             j = r.json()
#             return j["choices"][0]["message"]["content"], j
#         except Exception as e:
#             last_err = e
#             time.sleep(2 ** attempt)
#     raise RuntimeError(f"DeepSeek call failed after {retries} retries: {last_err}")


# # ----------------- main -----------------

# IN_CSV = "prompts_train.csv"
# OUT_CSV = "prompts_train_with_answers_deepseek.csv"  # ✅ 新文件名（你可以改）

# df = pd.read_csv(IN_CSV)

# # 新增输出列（不会影响原 prompts_train.csv）
# if "raw_answer" not in df.columns:
#     df["raw_answer"] = ""
# if "raw_response_json" not in df.columns:
#     df["raw_response_json"] = ""
# if "model" not in df.columns:
#     df["model"] = ""
# if "temperature" not in df.columns:
#     df["temperature"] = ""

# MODEL = "deepseek-chat"
# TEMPERATURE = 0.0
# MAX_TOKENS = 1200

# SAVE_EVERY = 5  # 每 5 条落盘一次，防止中途断掉全没了

# for i, idx in enumerate(df.index, 1):
#     prompt = df.at[idx, "prompt"]

#     ans, raw = call_deepseek(
#         prompt,
#         model=MODEL,
#         temperature=TEMPERATURE,
#         max_tokens=MAX_TOKENS,
#     )

#     df.at[idx, "raw_answer"] = ans
#     df.at[idx, "raw_response_json"] = json.dumps(raw, ensure_ascii=False)
#     df.at[idx, "model"] = MODEL
#     df.at[idx, "temperature"] = TEMPERATURE

#     if i % SAVE_EVERY == 0:
#         df.to_csv(OUT_CSV, index=False)
#         print(f"[SAVE] {i}/{len(df)} -> {OUT_CSV}")

# # 最后再保存一次
# df.to_csv(OUT_CSV, index=False)
# print(f"[DONE] wrote -> {OUT_CSV}")
