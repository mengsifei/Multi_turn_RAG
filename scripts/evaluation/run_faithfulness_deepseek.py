#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import argparse
import re
from typing import Any, Dict, List, Optional
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import requests
from deepseek_client import DeepSeekClient

# -----------------------------
# Helpers
# -----------------------------
def read_jsonl_df(path: str) -> pd.DataFrame:
    return pd.read_json(path, lines=path.endswith(".jsonl"), dtype={"task_id": str, "conversation_id": str})

def last_user_question(conv: Any) -> str:
    # conv is list of {"speaker","text"} (or string repr)
    if isinstance(conv, str):
        # best-effort parse
        try:
            conv = eval(conv, {"__builtins__": {}})
        except Exception:
            return ""
    if not isinstance(conv, list):
        return ""
    for t in reversed(conv):
        if isinstance(t, dict) and t.get("speaker") == "user":
            return str(t.get("text", ""))
    return ""

def get_answer(predictions: Any) -> str:
    if isinstance(predictions, str):
        try:
            predictions = eval(predictions, {"__builtins__": {}})
        except Exception:
            return ""
    if isinstance(predictions, list) and predictions:
        if isinstance(predictions[0], dict):
            return str(predictions[0].get("text", ""))
    return ""

def contexts_to_text(contexts: Any, max_docs: int = 5, max_chars_per_doc: int = 1400) -> str:
    if isinstance(contexts, str):
        try:
            contexts = eval(contexts, {"__builtins__": {}})
        except Exception:
            contexts = []
    if not isinstance(contexts, list):
        contexts = []
    chunks = []
    for i, c in enumerate(contexts[:max_docs]):
        if not isinstance(c, dict):
            continue
        txt = str(c.get("text", ""))
        txt = re.sub(r"\s+", " ", txt).strip()
        if max_chars_per_doc and len(txt) > max_chars_per_doc:
            txt = txt[:max_chars_per_doc] + " ..."
        doc_id = c.get("document_id", f"doc{i+1}")
        chunks.append(f"[{doc_id}] {txt}")
    return "\n".join(chunks).strip()

def safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    # strip code fences if any
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    # try direct
    try:
        return json.loads(s)
    except Exception:
        pass
    # try find first {...} block
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


# -----------------------------
# Prompt (RAGAS-like)
# -----------------------------
def build_faithfulness_prompt(question: str, answer: str, context: str) -> str:
    return f"""You are evaluating **faithfulness** of an assistant answer given only the provided context.
Faithfulness means: every factual claim in the ANSWER must be supported by the CONTEXT. If the answer adds facts not in the context, it is unfaithful.

Instructions:
1) Split the ANSWER into a small list of atomic factual claims (max 12). Ignore purely conversational fluff.
2) For each claim, label it as:
   - "supported" (clearly supported by context)
   - "not_supported" (contradicted or not found in context)
   - "unverifiable" (too vague to verify or depends on missing details)
3) Compute RL_F score in [0,1] using this rule (approximate RAGAS):
   - Start at 1.0
   - Subtract 1.0 * (not_supported_claims / total_claims)
   - Subtract 0.5 * (unverifiable_claims / total_claims)
   - Clamp to [0,1]
4) Return STRICT JSON only, in this schema:

{{
  "score": <float 0..1>,
  "claims": [
    {{"claim": "...", "label": "supported|not_supported|unverifiable", "evidence": "short quote or doc_id if supported"}}
  ]
}}

Now evaluate:

[QUESTION]
{question}

[ANSWER]
{answer}

[CONTEXT]
{context}
"""


def compute_score_from_claims(claims: List[Dict[str, Any]]) -> float:
    if not claims:
        return 0.0
    total = 0
    ns = 0
    uv = 0
    for c in claims:
        lab = str(c.get("label", "")).strip().lower()
        if not lab:
            continue
        total += 1
        if lab == "not_supported":
            ns += 1
        elif lab == "unverifiable":
            uv += 1
    if total == 0:
        return 0.0
    score = 1.0 - 1.0 * (ns / total) - 0.5 * (uv / total)
    if score < 0:
        score = 0.0
    if score > 1:
        score = 1.0
    return float(score)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="jsonl file with predictions/contexts/metrics")
    ap.add_argument("-o", "--output", required=True, help="output jsonl (written via tmp + atomic replace)")
    ap.add_argument("--max_docs", type=int, default=5)
    ap.add_argument("--max_chars_per_doc", type=int, default=1400)
    ap.add_argument("--max_tokens", type=int, default=700)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--resume", action="store_true", help="skip items that already have metrics.RL_F")
    args = ap.parse_args()

    client = DeepSeekClient(
        model=os.environ.get("DEEPSEEK_JUDGE_MODEL", "deepseek-chat"),
        timeout=180,
        retries=6,
    )

    df = read_jsonl_df(args.input)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ALWAYS write to tmp then atomically replace output
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    skipped = 0
    computed = 0
    failed_parse = 0

    with open(tmp_path, "w", encoding="utf-8") as w:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="DeepSeek RL_F"):
            rec = row.to_dict()
            metrics = rec.get("metrics") or {}
            if not isinstance(metrics, dict):
                metrics = {}

            # RESUME: if RL_F already exists and is non-empty -> keep as-is
            if args.resume:
                rl_f = metrics.get("RL_F", None)
                if isinstance(rl_f, list) and len(rl_f) > 0 and rl_f[0] is not None:
                    w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    skipped += 1
                    continue

            q = last_user_question(rec.get("input"))
            ans = get_answer(rec.get("predictions"))
            ctx = contexts_to_text(
                rec.get("contexts"),
                max_docs=args.max_docs,
                max_chars_per_doc=args.max_chars_per_doc,
            )

            # empty answer -> define RL_F=0.0
            if not ans.strip():
                metrics["RL_F"] = [0.0]
                rec["metrics"] = metrics
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                computed += 1
                continue

            prompt = build_faithfulness_prompt(q, ans, ctx)

            try:
                raw = client.generate_response(
                    prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
            except Exception as e:
                # API failure -> store fallback and error message
                metrics["RL_F"] = [0.5]
                metrics["RL_F_raw"] = [f"[DEEPSEEK_ERROR] {type(e).__name__}: {str(e)[:500]}"]
                rec["metrics"] = metrics
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                computed += 1
                continue

            j = safe_parse_json(raw)

            if not isinstance(j, dict):
                # parse fail -> fallback score
                metrics["RL_F"] = [0.5]
                metrics["RL_F_raw"] = [raw[:2000]]
                rec["metrics"] = metrics
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                computed += 1
                failed_parse += 1
                continue

            claims = j.get("claims") if isinstance(j.get("claims"), list) else []

            # prefer model-provided score, else recompute
            score = j.get("score")
            try:
                score = float(score)
                if score < 0 or score > 1:
                    raise ValueError()
            except Exception:
                score = compute_score_from_claims(claims)

            metrics["RL_F"] = [float(score)]
            metrics["RL_F_claims"] = [claims[:12]]  # optional audit
            rec["metrics"] = metrics

            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            computed += 1

    os.replace(tmp_path, out_path)

    print(f"[DONE] wrote -> {args.output}")
    if args.resume:
        print(f"[RESUME] skipped={skipped}, computed={computed}, failed_parse={failed_parse}")


if __name__ == "__main__":
    main()
