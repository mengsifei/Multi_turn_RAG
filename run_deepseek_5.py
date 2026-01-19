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

def call_deepseek(prompt: str, model="deepseek-chat", temperature=0.0, max_tokens=1200, timeout=180, retries=3):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
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

import os

out_path = "prompts_for_taskc_with_answers.csv"

# 如果已经跑过，继续用带答案的文件；否则从原始 prompts_for_taskc.csv 开始
if os.path.exists(out_path):
    df = pd.read_csv(out_path)
else:
    df = pd.read_csv("prompts_for_taskc.csv")


if "raw_answer" not in df.columns:
    df["raw_answer"] = ""
if "raw_response_json" not in df.columns:
    df["raw_response_json"] = ""

out_path = "prompts_for_taskc_with_answers.csv"
n = 5  # 前5条你已经跑过

for idx in df.index[n:]:
    if pd.notna(df.at[idx, "raw_answer"]) and str(df.at[idx, "raw_answer"]).strip():
        continue

    prompt = df.at[idx, "prompt"]
    ans, raw = call_deepseek(prompt)

    df.at[idx, "raw_answer"] = ans
    df.at[idx, "raw_response_json"] = json.dumps(raw, ensure_ascii=False)

    df.to_csv(out_path, index=False)


print(f"[DONE] updated -> {out_path}")

