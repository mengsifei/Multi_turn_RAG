#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSV-only DeepSeek faithfulness evaluator.

Input CSV columns normally include:
  task_id, conversation_id, Collection, domain, prompt, raw_answer, n_contexts, target_text

Default behavior:
  - answer:   --prediction_col raw_answer
  - question: extracted from [Question] ... [Answer] or [Question] ... [Final Answer] in prompt
  - context:  extracted from [Documents] ... [Question] in prompt

Output:
  A separate JSONL file, e.g. *_metrics_faith.jsonl.
  The input CSV is never overwritten.

Example:
  PYTHONPATH=. python scripts/evaluation/run_faithfulness_deepseek_csv_only.py \
    -i outputs/taskc/prompt_official_lastturn_rewrite_gpt_ans_with_targets.csv \
    -o outputs/taskc/prompt_official_lastturn_rewrite_gpt_ans_with_targets_metrics_faith.jsonl \
    --prediction_col raw_answer \
    --prompt_col prompt \
    --summary_csv outputs/taskc/prompt_official_lastturn_rewrite_gpt_ans_with_targets_metrics_faith_scores.csv \
    --resume
"""

from __future__ import annotations
import sys, shlex, datetime
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm
from deepseek_client import DeepSeekClient


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


def read_csv_df(path: str) -> pd.DataFrame:
    increase_csv_field_limit()
    return pd.read_csv(path, dtype=str, keep_default_na=False, engine="python")


def strip_user_prefix(text: Any) -> str:
    lines: List[str] = []
    for line in safe_str(text).splitlines():
        line = line.strip()
        if line.startswith("|user|:"):
            line = line[len("|user|:"):].strip()
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def extract_between(text: Any, start_marker: str, end_marker: str) -> str:
    s = safe_str(text)
    pattern = re.compile(
        re.escape(start_marker) + r"\s*(.*?)\s*" + re.escape(end_marker),
        flags=re.DOTALL | re.IGNORECASE,
    )
    m = pattern.search(s)
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


def extract_context_from_prompt(prompt: Any) -> str:
    ctx = extract_between(prompt, "[Documents]", "[Question]")
    if ctx:
        return ctx
    ctx = extract_between(prompt, "<Contexts>", "</Contexts>")
    if ctx:
        return ctx
    return ""


def parse_jsonish(value: Any, default: Any):
    s = safe_str(value).strip()
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default


def contexts_json_to_text(contexts: Any, max_docs: int, max_chars_per_doc: int) -> str:
    parsed = parse_jsonish(contexts, default=[])
    if not isinstance(parsed, list):
        return ""

    chunks: List[str] = []
    for i, c in enumerate(parsed[:max_docs]):
        if isinstance(c, dict):
            doc_id = safe_str(c.get("document_id") or c.get("id") or f"doc{i+1}")
            txt = safe_str(c.get("text") or c.get("content"))
        else:
            doc_id = f"doc{i+1}"
            txt = safe_str(c)

        txt = re.sub(r"\s+", " ", txt).strip()
        if not txt:
            continue
        if max_chars_per_doc and len(txt) > max_chars_per_doc:
            txt = txt[:max_chars_per_doc] + " ..."
        chunks.append(f"[{doc_id}] {txt}")

    return "\n".join(chunks).strip()


def truncate_prompt_context(context: Any, max_docs: int, max_chars_per_doc: int) -> str:
    context = safe_str(context).strip()
    if not context:
        return ""

    context = re.sub(r"^\s*Documents:\s*", "", context, flags=re.IGNORECASE).strip()
    docs = [p.strip() for p in re.split(r"(?=\[Document\s+\d+\])", context, flags=re.IGNORECASE) if p.strip()]

    if not docs:
        return context[: max_docs * max_chars_per_doc]

    out: List[str] = []
    for doc in docs[:max_docs]:
        m = re.match(r"(\[Document\s+\d+\][^\n]*)(.*)", doc, flags=re.DOTALL | re.IGNORECASE)
        if m:
            header = m.group(1).strip()
            body = re.sub(r"\s+", " ", m.group(2)).strip()
            if max_chars_per_doc and len(body) > max_chars_per_doc:
                body = body[:max_chars_per_doc] + " ..."
            out.append(f"{header}\n{body}".strip())
        else:
            body = re.sub(r"\s+", " ", doc).strip()
            if max_chars_per_doc and len(body) > max_chars_per_doc:
                body = body[:max_chars_per_doc] + " ..."
            out.append(body)
    return "\n".join(out).strip()


def get_task_id(row: Dict[str, Any], idx: int) -> str:
    for k in ("task_id", "_id", "id", "question_id"):
        v = safe_str(row.get(k)).strip()
        if v:
            return v
    return f"row::{idx}"


def get_conversation_id(row: Dict[str, Any], task_id: str) -> str:
    for k in ("conversation_id", "conversationId"):
        v = safe_str(row.get(k)).strip()
        if v:
            return v
    return task_id.split("<::>", 1)[0] if "<::>" in task_id else task_id


def get_question(row: Dict[str, Any], args: argparse.Namespace) -> str:
    if args.question_col and args.question_col in row:
        q = strip_user_prefix(row.get(args.question_col))
        if q:
            return q

    for col in ("question", "rewrite_query", "lastturn_query", "questions"):
        if col in row:
            q = strip_user_prefix(row.get(col))
            if q:
                return q

    if args.prompt_col and args.prompt_col in row:
        q = extract_question_from_prompt(row.get(args.prompt_col))
        if q:
            return q.strip()

    return ""


def get_context(row: Dict[str, Any], args: argparse.Namespace) -> str:
    if args.contexts_col and args.contexts_col in row and safe_str(row.get(args.contexts_col)).strip():
        ctx = contexts_json_to_text(row.get(args.contexts_col), args.max_docs, args.max_chars_per_doc)
        if ctx:
            return ctx

    if args.prompt_col and args.prompt_col in row:
        ctx = extract_context_from_prompt(row.get(args.prompt_col))
        if ctx:
            return truncate_prompt_context(ctx, args.max_docs, args.max_chars_per_doc)

    return ""


def safe_parse_json(text: Any) -> Optional[Dict[str, Any]]:
    s = safe_str(text).strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except Exception:
        pass

    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return None

    try:
        return json.loads(m.group(0))
    except Exception:
        return None


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

    total = ns = uv = 0
    for c in claims:
        lab = safe_str(c.get("label")).strip().lower()
        if not lab:
            continue
        total += 1
        if lab == "not_supported":
            ns += 1
        elif lab == "unverifiable":
            uv += 1

    if total == 0:
        return 0.0

    return float(max(0.0, min(1.0, 1.0 - 1.0 * (ns / total) - 0.5 * (uv / total))))


def has_rl_f(rec: Dict[str, Any]) -> bool:
    metrics = rec.get("metrics") or {}
    if not isinstance(metrics, dict):
        return False
    value = metrics.get("RL_F")
    return isinstance(value, list) and len(value) > 0 and value[0] is not None


def load_existing_output(path: Path) -> Dict[str, Dict[str, Any]]:
    done: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            task_id = safe_str(rec.get("task_id"))
            if task_id and has_rl_f(rec):
                done[task_id] = rec
    return done


def write_summary_csv(metrics_jsonl: Path, summary_csv: Path) -> None:
    """
    Write a flat CSV summary.

    If the JSONL has duplicate task_id entries from previous interrupted/resumed runs,
    keep the last valid record for each task_id.
    """
    by_task: Dict[str, Dict[str, Any]] = {}

    with metrics_jsonl.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception as e:
                print(f"[WARN] skip malformed JSONL line {line_no} in summary: {e}")
                continue

            task_id = safe_str(rec.get("task_id"))
            if not task_id:
                task_id = f"row::{line_no}"

            by_task[task_id] = rec

    rows: List[Dict[str, Any]] = []
    for task_id, rec in by_task.items():
        metrics = rec.get("metrics") or {}
        rl_f = metrics.get("RL_F", [None])
        rows.append(
            {
                "task_id": rec.get("task_id"),
                "conversation_id": rec.get("conversation_id"),
                "Collection": rec.get("Collection", rec.get("collection")),
                "domain": rec.get("domain"),
                "n_contexts": rec.get("n_contexts"),
                "RL_F": rl_f[0] if isinstance(rl_f, list) and rl_f else None,
                "RL_F_error": (
                    metrics.get("RL_F_error", [""])[0]
                    if isinstance(metrics.get("RL_F_error"), list)
                    else ""
                ),
            }
        )

    df = pd.DataFrame(rows)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    if len(df):
        print(f"[SUMMARY] rows={len(df)} RL_F mean={pd.to_numeric(df['RL_F'], errors='coerce').mean():.6f}")
    print(f"[SUMMARY] wrote -> {summary_csv}")


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("-i", "--input", required=True, help="input CSV with raw_answer and prompt/contexts")
    ap.add_argument("-o", "--output", required=True, help="output JSONL, e.g. *_metrics_faith.jsonl")

    ap.add_argument("--prediction_col", default="raw_answer")
    ap.add_argument("--prompt_col", default="prompt")
    ap.add_argument("--question_col", default=None)
    ap.add_argument("--contexts_col", default=None)

    ap.add_argument("--max_docs", type=int, default=5)
    ap.add_argument("--max_chars_per_doc", type=int, default=1400)
    ap.add_argument("--max_tokens", type=int, default=700)
    ap.add_argument("--temperature", type=float, default=0.0)

    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summary_csv", default=None)

    ap.add_argument("--api_error_score", type=float, default=0.5)
    ap.add_argument("--parse_error_score", type=float, default=0.5)
    ap.add_argument("--empty_answer_score", type=float, default=0.0)
    ap.add_argument("--empty_context_score", type=float, default=0.0)

    args = ap.parse_args()
    cmd = " ".join(shlex.quote(x) for x in sys.argv)
    print(f"[CMD] {cmd}")
    print(f"[TIME] {datetime.datetime.now().isoformat(timespec='seconds')}")

    df = read_csv_df(args.input)

    if args.prediction_col not in df.columns:
        raise ValueError(f"Missing prediction_col={args.prediction_col!r}. Columns={list(df.columns)}")
    if args.contexts_col and args.contexts_col not in df.columns:
        raise ValueError(f"Missing contexts_col={args.contexts_col!r}. Columns={list(df.columns)}")
    if args.prompt_col and args.prompt_col not in df.columns and not args.contexts_col:
        raise ValueError(
            f"Missing prompt_col={args.prompt_col!r}, and no contexts_col was provided. "
            f"Columns={list(df.columns)}"
        )

    client = DeepSeekClient(
        model=os.environ.get("DEEPSEEK_JUDGE_MODEL", "deepseek-chat"),
        timeout=180,
        retries=6,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # True resume mode:
    # - if --resume, read existing completed records from the output JSONL
    # - append new records directly to the same JSONL
    # - if the job is killed, already written records remain available for the next resume
    existing_done = load_existing_output(out_path) if args.resume else {}

    skipped = computed = failed_parse = api_failed = empty_answer = empty_context = 0
    printed_first_prompt = False

    write_mode = "a" if args.resume and out_path.exists() else "w"
    if args.resume:
        print(f"[RESUME] existing completed records: {len(existing_done)}")
        print(f"[RESUME] write_mode={write_mode} output={out_path}")

    with out_path.open(write_mode, encoding="utf-8") as w:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="DeepSeek faithfulness CSV"):
            row_dict = row.to_dict()

            task_id = get_task_id(row_dict, idx)
            conv_id = get_conversation_id(row_dict, task_id)

            if args.resume and task_id in existing_done:
                w.write(json.dumps(existing_done[task_id], ensure_ascii=False) + "\n")
                skipped += 1
                continue

            answer = safe_str(row_dict.get(args.prediction_col)).strip()
            question = get_question(row_dict, args)
            context = get_context(row_dict, args)

            rec: Dict[str, Any] = dict(row_dict)
            rec["task_id"] = task_id
            rec["conversation_id"] = conv_id
            rec["predictions"] = [{"text": answer}]
            rec["faithfulness_question"] = question
            rec["faithfulness_context"] = context
            rec["metrics"] = {}

            metrics = rec["metrics"]

            if not answer:
                metrics["RL_F"] = [float(args.empty_answer_score)]
                metrics["RL_F_error"] = ["EMPTY_ANSWER"]
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                computed += 1
                empty_answer += 1
                continue

            if not context:
                metrics["RL_F"] = [float(args.empty_context_score)]
                metrics["RL_F_error"] = ["EMPTY_CONTEXT"]
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                computed += 1
                empty_context += 1
                continue

            judge_prompt = build_faithfulness_prompt(question, answer, context)
            if not printed_first_prompt:
                print("\n========== FIRST FAITHFULNESS PROMPT ==========")
                print(judge_prompt)
                print("========== END FIRST FAITHFULNESS PROMPT ==========\n")
                printed_first_prompt = True

            try:
                raw = client.generate_response(
                    judge_prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
            except Exception as e:
                metrics["RL_F"] = [float(args.api_error_score)]
                metrics["RL_F_raw"] = [f"[DEEPSEEK_ERROR] {type(e).__name__}: {str(e)[:500]}"]
                metrics["RL_F_error"] = ["API_ERROR"]
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                computed += 1
                api_failed += 1
                continue

            parsed = safe_parse_json(raw)
            if not isinstance(parsed, dict):
                metrics["RL_F"] = [float(args.parse_error_score)]
                metrics["RL_F_raw"] = [safe_str(raw)[:2000]]
                metrics["RL_F_error"] = ["PARSE_ERROR"]
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                computed += 1
                failed_parse += 1
                continue

            claims = parsed.get("claims") if isinstance(parsed.get("claims"), list) else []

            try:
                score = float(parsed.get("score"))
                if score < 0 or score > 1:
                    raise ValueError()
            except Exception:
                score = compute_score_from_claims(claims)

            metrics["RL_F"] = [float(score)]
            metrics["RL_F_claims"] = [claims[:12]]
            metrics["RL_F_raw"] = [parsed]
            metrics["RL_F_error"] = [""]

            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            computed += 1

    print(f"[DONE] wrote/appended -> {out_path}")
    print(
        f"[STATS] skipped={skipped}, computed={computed}, "
        f"empty_answer={empty_answer}, empty_context={empty_context}, "
        f"api_failed={api_failed}, failed_parse={failed_parse}"
    )

    if args.summary_csv:
        write_summary_csv(out_path, Path(args.summary_csv))


if __name__ == "__main__":
    main()
