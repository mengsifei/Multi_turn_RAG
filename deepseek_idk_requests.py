#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import re
import sys
import time
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import requests


URL = "https://api.deepseek.com/chat/completions"

PROMPT_TEMPLATE = """You are an IDK judge.

Classify whether the response refuses to answer the user's exact inquiry.

Return exactly one label: yes, no, or partial.

Definitions:
- yes: The response ultimately says the exact inquiry cannot be answered, or that there is not enough information to answer it.
- no: The response gives a concrete answer or inference to the exact inquiry. Do not judge whether the answer is factually correct.
- partial: The inquiry asks for multiple distinct requested items, and the response directly answers at least one requested item but explicitly says another requested item cannot be answered.

Strict rules:
- Judge only the response's answerability behavior.
- Do NOT use outside knowledge.
- Do NOT judge factual correctness.
- Do NOT compare against the gold answerability label.
- If the response gives an answer to the exact inquiry, output no, even if the answer may be unsupported or wrong.
- If the response only gives related background but says the exact inquiry cannot be answered, output yes.
- Use partial only when the response both answers one requested part and refuses another requested part.
- Do not output partial just because the response contains caveats, limitations, "however", "not", "different", or "does not explicitly say".
- If unsure between yes and partial, choose yes unless there is a clearly answered subquestion.
- If unsure between no and partial, choose no if the exact inquiry is answered.

Examples:
[Inquiry] What time does the museum close today?
[Response] The document says the museum usually closes at 6 PM, but it does not provide today's closing time.
[Output] yes

[Inquiry] Where do the Arizona Cardinals play this week?
[Response] The documents say the Cardinals played the Rams in London in 2017, but they do not specify where the Cardinals play this week.
[Output] yes

[Inquiry] Why doesn't the government prevent people from living in areas prone to flooding?
[Response] The government does not outright prevent people from living in flood-prone areas; instead, it promotes mitigation, preparedness, building standards, and flood insurance.
[Output] no

[Inquiry] What are sheltered rooms designated for use?
[Response] A safe room is designated for protection from tornadoes, while a shelter-in-place room is used during events involving airborne contaminants.
[Output] no

[Inquiry] What items should I keep in a designated sheltered room?
[Response] You should keep duct tape, plastic sheeting, and disaster supplies in the designated sheltered room.
[Output] no

[Inquiry] Are sheltered room designations and recommended items the same for earthquakes?
[Response] I do not have specific information. The documents discuss tornadoes, windstorms, and shelter-in-place supplies, but they do not address earthquake preparedness, safe room designations, or recommended items for earthquakes.
[Output] yes

[Inquiry] What is NLP and how can I build a sentiment analysis model using Python and NLTK?
[Response] NLP is a subfield of AI focused on human language. However, the documents do not describe how to build a sentiment analysis model using Python and NLTK.
[Output] partial

[Inquiry] Which state has more wildfires?
[Response] The documents only discuss wildfires in California and do not provide data comparing wildfire frequency or size with other states.
[Output] yes

[Inquiry]
{question}

[Response]
{answer}

[Output]"""

SYSTEM_PROMPT = "You are an IDK judge. Output exactly one lowercase label: yes, no, or partial. Do not explain."


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def extract_between(text: Any, start_marker: str, end_marker: str) -> str:
    text = safe_str(text)
    m = re.search(
        re.escape(start_marker) + r"\s*(.*?)\s*" + re.escape(end_marker),
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return m.group(1).strip() if m else ""


def extract_question_from_prompt(prompt: Any) -> str:
    q = extract_between(prompt, "[Question]", "[Answer]")
    if q:
        return q
    q = extract_between(prompt, "[Question]", "[Final Answer]")
    if q:
        return q
    q = extract_between(prompt, "<Question>:", "</Question>")
    if q:
        return q
    return ""


def _strip_user_prefix(q: str) -> str:
    return re.sub(r"\|user\|:\s*", "", q).strip()


def clean_question(q: Any) -> str:
    """Light cleanup for chat-formatted or accidentally duplicated questions."""
    q = _strip_user_prefix(safe_str(q))
    q = re.sub(r"\s+", " ", q).strip()

    # Remove exact duplicate adjacent sentences, e.g. "Q? Q?"
    parts = re.split(r"(?<=[?.!])\s+", q)
    cleaned = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if cleaned and p.lower() == cleaned[-1].lower():
            continue
        cleaned.append(p)
    return " ".join(cleaned).strip()


def normalize_label(text: Any) -> str:
    s = safe_str(text).strip().lower()
    s = re.sub(r"[^a-z]", "", s)

    if s.startswith("yes"):
        return "yes"
    if s.startswith("no"):
        return "no"
    if s.startswith("partial"):
        return "partial"
    return "parse_error"


def idk_to_score(label: str) -> float:
    if label == "yes":
        return 1.0
    if label == "no":
        return 0.0
    if label == "partial":
        return 0.5
    return 0.5


def extract_content(raw: Dict[str, Any]) -> str:
    """Extract assistant final content from DeepSeek/OpenAI-compatible response JSON."""
    try:
        msg = raw["choices"][0]["message"]
    except Exception:
        return ""

    # Normal chat-completions path.
    content = msg.get("content", "")
    if isinstance(content, str) and content.strip():
        return content.strip()

    # Some clients/models may return content blocks rather than a plain string.
    if isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, dict):
                pieces.append(safe_str(item.get("text") or item.get("content") or ""))
            else:
                pieces.append(safe_str(item))
        joined = "".join(pieces).strip()
        if joined:
            return joined

    # Do not use reasoning_content as the answer, but returning it is useful to avoid a silent blank.
    # The caller will still normalize it, usually to parse_error if it is not exactly a label.
    reasoning = msg.get("reasoning_content", "")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()

    return ""


def call_deepseek(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    retries: int,
) -> Tuple[str, Dict[str, Any], Optional[str]]:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY env var. e.g. export DEEPSEEK_API_KEY='...'")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": False,
    }

    last_err = None
    for attempt in range(retries):
        try:
            r = requests.post(URL, headers=headers, json=payload, timeout=timeout)
            # Keep response body in the exception message when possible.
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:1000]}")
            raw = r.json()
            ans = extract_content(raw)
            err = None if ans else "empty message.content returned by API; inspect raw_response_json"
            return ans, raw, err
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(min(30.0, (2 ** attempt) + random.random()))

    raise RuntimeError(f"DeepSeek call failed after {retries} retries: {last_err}")


def main() -> None:
    ap = argparse.ArgumentParser()
    # Accept both names so it works with your earlier command style and my previous script style.
    ap.add_argument("--input_csv", "--in_csv", dest="input_csv", required=True)
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--answer_col", default="raw_answer")
    ap.add_argument("--prompt_col", default="prompt")
    ap.add_argument("--question_col", default=None)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--out_jsonl", default="/tmp/deepseek_idk_5.jsonl")
    ap.add_argument("--max_tokens", type=int, default=32)
    ap.add_argument("--temperature", "--temp", dest="temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--no_clean_question", action="store_true")
    ap.add_argument("--save_raw", action="store_true", help="Store full raw DeepSeek response JSON per row.")
    args = ap.parse_args()

    print(f"[INFO] model={args.model}")
    print(f"[INFO] endpoint={URL}")
    print(f"[INFO] input_csv={args.input_csv}")

    df = pd.read_csv(args.input_csv, dtype=str, keep_default_na=False)
    if args.limit is not None and args.limit > 0:
        df = df.head(args.limit)

    rows = []
    for i, row in df.iterrows():
        answer = safe_str(row.get(args.answer_col))

        if args.question_col and args.question_col in row:
            question = safe_str(row.get(args.question_col))
        else:
            question = extract_question_from_prompt(row.get(args.prompt_col))

        if not args.no_clean_question:
            question = clean_question(question)

        judge_prompt = PROMPT_TEMPLATE.format(question=question, answer=answer)

        try:
            gen, raw, err = call_deepseek(
                prompt=judge_prompt,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                retries=args.retries,
            )
        except Exception as e:
            gen, raw, err = "", {}, str(e)

        label = normalize_label(gen)
        obj = {
            "row_index": int(i),
            "task_id": row.get("task_id", ""),
            "answerability": row.get("answerability", ""),
            "question": question,
            "raw_answer": answer,
            "idk_raw": gen,
            "idk_label": label,
            "idk_score": idk_to_score(label),
            "judge_model": args.model,
            "judge_provider": "deepseek_requests",
        }
        if err:
            obj["error"] = err
        if args.save_raw:
            obj["raw_response_json"] = raw
        rows.append(obj)

        # print("\n==============================")
        # print("task_id:", obj["task_id"])
        # print("answerability:", obj["answerability"])
        # print("question:", question)
        # print("raw_answer:", answer)
        # if err:
        #     print("error:", err, file=sys.stderr)
        # print("idk_raw:", gen)
        # print("idk_label:", label)
        # print("idk_score:", obj["idk_score"])

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[DONE] wrote {args.out_jsonl}")


if __name__ == "__main__":
    main()
