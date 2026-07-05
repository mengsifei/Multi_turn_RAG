#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSV-only RadBench / RBllm judge using DeepSeek/OpenAI/local HF client.

Designed for CSV files like:
  outputs/taskc/prompt_official_lastturn_rewrite_gpt_hybrid842_top5_ans_with_targets_idk_deepseek_judge.csv

Expected useful columns:
  - task_id
  - conversation_id
  - raw_answer
  - target_text
  - prompt
  - domain
  - Collection / collection
  - n_contexts

Default behavior:
  - answer:          --prediction_col raw_answer
  - reference:       --target_col target_text
  - question/context extracted from prompt:
      [Documents] ... [Question]
      [Question]  ... [Answer] / [Final Answer]

Output:
  - separate JSONL with metrics.RB_llm
  - optional flat summary CSV

This script never overwrites the input CSV and does not touch alg/faith files.

It supports true resume:
  --resume reads existing output JSONL and skips task_id entries that already have metrics.RB_llm.
  New records are appended directly to output JSONL, so interrupted runs keep completed lines.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import shlex
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.evaluation.deepseek_client import DeepSeekClient
from scripts.evaluation.azure_openai_client import AzureOpenAIClient

try:
    from scripts.evaluation.judge_utils import extract_rating
except Exception:
    extract_rating = None


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

    # Common columns.
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


def parse_rating_fallback(text: Any) -> Optional[float]:
    """
    Fallback parser if judge_utils.extract_rating is unavailable or returns None.

    Looks for common forms:
      Rating: 0.7
      Score: 4
      [[3]]
      3/5
    If score looks like 1..5, normalize to 0..1.
    If score already in 0..1, keep as is.
    """
    s = safe_str(text).strip()

    patterns = [
        r"(?:rating|score)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
        r"\[\[\s*([0-9]+(?:\.[0-9]+)?)\s*\]\]",
        r"\b([0-9]+(?:\.[0-9]+)?)\s*/\s*5\b",
        r"\b([0-9]+(?:\.[0-9]+)?)\b",
    ]

    for pat in patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if not m:
            continue
        try:
            x = float(m.group(1))
        except Exception:
            continue

        if 0.0 <= x <= 1.0:
            return x
        if 1.0 <= x <= 5.0:
            return x / 5.0
        if 0.0 <= x <= 100.0:
            return x / 100.0

    return None


def parse_rating(text: Any) -> Optional[float]:
    if extract_rating is not None:
        try:
            val = extract_rating(text)
            if val is not None and not pd.isna(val):
                return float(val)
        except Exception:
            pass
    return parse_rating_fallback(text)


def build_radbench_prompt(question: str, answer: str, reference_answer: str, context: str) -> str:
    """
    CSV-friendly RadBench-style judge prompt.

    Scores answer quality against reference answer and context.
    Returns a 0..1 rating.
    """
    return f"""You are a strict evaluator for a retrieval-augmented question answering system.

Evaluate the assistant answer for the user question using the reference answer and the provided documents.

Criteria:
- Correctness: Does the assistant answer match the reference answer?
- Grounding: Is the assistant answer supported by the provided documents?
- Completeness: Does the assistant answer address the user question?
- Penalize hallucinated or unsupported claims.
- If the assistant says it cannot answer when the reference answer contains an answer, give a low score.
- If the reference answer says the question cannot be answered and the assistant also refuses to answer, give a high score.

Return STRICT JSON only:
{{"rating": <float from 0.0 to 1.0>, "reason": "brief explanation"}}

[Question]
{question}

[Reference Answer]
{reference_answer}

[Assistant Answer]
{answer}

[Documents]
{context}

[JSON]
"""


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


def has_rb_llm(rec: Dict[str, Any]) -> bool:
    metrics = rec.get("metrics") or {}
    if not isinstance(metrics, dict):
        return False
    value = metrics.get("RB_llm")
    return isinstance(value, list) and len(value) > 0 and value[0] is not None


def load_existing_output(path: Path) -> Dict[str, Dict[str, Any]]:
    done: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return done

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception as e:
                print(f"[WARN] skip malformed existing line {line_no}: {e}")
                continue
            task_id = safe_str(rec.get("task_id"))
            if task_id and has_rb_llm(rec):
                done[task_id] = rec
    return done


def write_summary_csv(metrics_jsonl: Path, summary_csv: Path) -> None:
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

            task_id = safe_str(rec.get("task_id")) or f"row::{line_no}"
            by_task[task_id] = rec

    rows: List[Dict[str, Any]] = []
    for task_id, rec in by_task.items():
        metrics = rec.get("metrics") or {}
        rb = metrics.get("RB_llm", [None])
        rows.append(
            {
                "task_id": rec.get("task_id"),
                "conversation_id": rec.get("conversation_id"),
                "Collection": rec.get("Collection", rec.get("collection")),
                "domain": rec.get("domain"),
                "n_contexts": rec.get("n_contexts"),
                "RB_llm": rb[0] if isinstance(rb, list) and rb else None,
                "RB_llm_error": (
                    metrics.get("RB_llm_error", [""])[0]
                    if isinstance(metrics.get("RB_llm_error"), list)
                    else ""
                ),
            }
        )

    df = pd.DataFrame(rows)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    if len(df):
        print(f"[SUMMARY] rows={len(df)} RB_llm mean={pd.to_numeric(df['RB_llm'], errors='coerce').mean():.6f}")
    print(f"[SUMMARY] wrote -> {summary_csv}")


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("-i", "--input", required=True, help="input CSV with raw_answer, target_text, prompt/contexts")
    ap.add_argument("-o", "--output", required=True, help="output JSONL, e.g. *_metrics_rbllm_raw.jsonl")

    ap.add_argument("--judge_model", default="deepseek", help="deepseek, openai, or local HF model name/path")
    ap.add_argument("--prediction_col", default="raw_answer")
    ap.add_argument("--target_col", default="target_text")
    ap.add_argument("--prompt_col", default="prompt")
    ap.add_argument("--question_col", default=None)
    ap.add_argument("--contexts_col", default=None)

    ap.add_argument("--max_docs", type=int, default=5)
    ap.add_argument("--max_chars_per_doc", type=int, default=1400)
    ap.add_argument("--max_tokens", type=int, default=800)
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
    if args.target_col not in df.columns:
        raise ValueError(f"Missing target_col={args.target_col!r}. Columns={list(df.columns)}")
    if args.contexts_col and args.contexts_col not in df.columns:
        raise ValueError(f"Missing contexts_col={args.contexts_col!r}. Columns={list(df.columns)}")
    if args.prompt_col and args.prompt_col not in df.columns and not args.contexts_col:
        raise ValueError(
            f"Missing prompt_col={args.prompt_col!r}, and no contexts_col was provided. "
            f"Columns={list(df.columns)}"
        )

    if args.judge_model == "deepseek":
        client = DeepSeekClient(model=os.environ.get("DEEPSEEK_JUDGE_MODEL", "deepseek-chat"))
    elif args.judge_model == "openai":
        client = AzureOpenAIClient("gpt-4o-mini-2024-07-18")
    else:
        # Reuse local HF client from original wrapper when needed.
        from scripts.evaluation.judge_wrapper import HuggingFaceLLMClient, clear_cuda
        clear_cuda()
        client = HuggingFaceLLMClient(args.judge_model)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_done = load_existing_output(out_path) if args.resume else {}

    skipped = computed = failed_parse = api_failed = empty_answer = empty_context = 0
    printed_first_prompt = False

    write_mode = "a" if args.resume and out_path.exists() else "w"
    if args.resume:
        print(f"[RESUME] existing completed records: {len(existing_done)}")
        print(f"[RESUME] write_mode={write_mode} output={out_path}")

    with out_path.open(write_mode, encoding="utf-8") as w:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="DeepSeek RBllm CSV"):
            row_dict = row.to_dict()

            task_id = get_task_id(row_dict, idx)
            conv_id = get_conversation_id(row_dict, task_id)

            if args.resume and task_id in existing_done:
                skipped += 1
                continue

            answer = safe_str(row_dict.get(args.prediction_col)).strip()
            reference = safe_str(row_dict.get(args.target_col)).strip()
            question = get_question(row_dict, args)
            context = get_context(row_dict, args)

            rec: Dict[str, Any] = dict(row_dict)
            rec["task_id"] = task_id
            rec["conversation_id"] = conv_id
            rec["predictions"] = [{"text": answer}]
            rec["targets"] = [{"text": reference}]
            rec["rbllm_question"] = question
            rec["rbllm_context"] = context
            rec["metrics"] = {}

            metrics = rec["metrics"]

            if not answer:
                metrics["RB_llm"] = [float(args.empty_answer_score)]
                metrics["RB_llm_error"] = ["EMPTY_ANSWER"]
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                w.flush()
                computed += 1
                empty_answer += 1
                continue

            if not context:
                metrics["RB_llm"] = [float(args.empty_context_score)]
                metrics["RB_llm_error"] = ["EMPTY_CONTEXT"]
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                w.flush()
                computed += 1
                empty_context += 1
                continue

            judge_prompt = build_radbench_prompt(question, answer, reference, context)

            if not printed_first_prompt:
                print("\n========== FIRST RBLLM PROMPT ==========")
                print(judge_prompt)
                print("========== END FIRST RBLLM PROMPT ==========\n")
                printed_first_prompt = True

            try:
                if args.judge_model == "deepseek":
                    raw = client.generate_response(
                        judge_prompt,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                    )
                else:
                    raw = client.generate_response(judge_prompt)
            except Exception as e:
                metrics["RB_llm"] = [float(args.api_error_score)]
                metrics["RB_llm_raw"] = [f"[JUDGE_ERROR] {type(e).__name__}: {str(e)[:500]}"]
                metrics["RB_llm_error"] = ["API_ERROR"]
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                w.flush()
                computed += 1
                api_failed += 1
                continue

            parsed = safe_parse_json(raw)

            rating: Optional[float] = None
            if isinstance(parsed, dict):
                for key in ("rating", "score", "RB_llm", "rb_llm"):
                    if key in parsed:
                        try:
                            rating = float(parsed[key])
                            break
                        except Exception:
                            pass

            if rating is None:
                rating = parse_rating(raw)

            if rating is None or pd.isna(rating):
                rating = float(args.parse_error_score)
                metrics["RB_llm_error"] = ["PARSE_ERROR"]
            else:
                metrics["RB_llm_error"] = [""]

            # Clamp to [0, 1].
            rating = float(max(0.0, min(1.0, rating)))

            metrics["RB_llm"] = [rating]
            metrics["RB_llm_raw"] = [parsed if parsed is not None else safe_str(raw)[:2000]]
            metrics["RB_llm_judge_model"] = [args.judge_model]

            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            w.flush()
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
