#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, List

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


THINK_END_TOKEN_ID = 151668  # Qwen3 official: </think>


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_user_lines(text: str) -> List[str]:
    out = []
    for ln in text.splitlines():
        ln = ln.strip()
        if ln.lower().startswith("|user|:"):
            out.append(ln)
    return out


def strip_user_prefix(line: str) -> str:
    return re.sub(r"^\|user\|\s*:\s*", "", line, flags=re.IGNORECASE).strip()


def truncate_by_token_budget(tok, user_lines: List[str], max_tokens: int, keep_first: bool = False) -> List[str]:
    if not user_lines:
        return user_lines

    kept_rev = []
    total = 0
    for ln in reversed(user_lines):
        txt = strip_user_prefix(ln)
        n = len(tok.encode(txt, add_special_tokens=False)) + 1
        if kept_rev and (total + n) > max_tokens:
            break
        if not kept_rev and n > max_tokens:
            ids = tok.encode(txt, add_special_tokens=False)[:max_tokens]
            txt = tok.decode(ids, skip_special_tokens=True)
            kept_rev.append(txt)
            total = max_tokens
            break
        kept_rev.append(txt)
        total += n

    kept = list(reversed(kept_rev))

    if keep_first:
        first = strip_user_prefix(user_lines[0])
        if first not in kept:
            n_first = len(tok.encode(first, add_special_tokens=False)) + 1
            if (total + n_first) <= max_tokens:
                kept = [first] + kept

    return kept


def build_messages(benchmark: str, domain: str, turns_str: str) -> List[Dict[str, str]]:
    # 少量 few-shot（可保留）
    fewshot_turns = (
        "Turns:\n"
        "[1] What is finance?\n"
        "[2] Who is the most famous person in this area?\n"
        "Output:\n"
        "|user|: Who is the most famous person in finance?\n"
        "-----\n"
    )

    system = (
        f"You are a query rewriter for the {benchmark} multi-turn retrieval benchmark.\n"
        f"Domain: {domain if domain else 'unknown'}.\n\n"
        "Task: Transform the last user turn into a self-contained, standalone search query.\n\n"
        "Critical Instructions:\n"
        "1) Focus on the LAST turn ONLY. Rewrite it to be self-contained - resolve all pronouns, "
        "references, and implicit context using information from earlier turns if needed.\n"
        "2) DO NOT simply copy the last turn verbatim. You must REWRITE it to make it standalone.\n"
        "   - Replace pronouns (e.g., 'he', 'it', 'they') with the actual entities from earlier turns.\n"
        "   - Expand references (e.g., 'this area', 'that movement') to their explicit meanings.\n"
        "   - Incorporate necessary context to make the query meaningful without prior conversation.\n"
        "3) Do NOT combine or merge multiple earlier questions - only use them for disambiguation.\n"
        "4) Preserve all critical information: entities, constraints, time ranges, locations, negations.\n"
        "5) Do NOT answer the question. Do NOT add facts not present in the conversation.\n"
        "6) Output EXACTLY ONE line starting with: |user|:\n"
        "7) No extra lines, no explanations, no thinking process.\n"
    )

    user = (
        # fewshot_turns +
        "Turns:\n"
        f"{turns_str}\n"
        "Output:\n"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def clean_output(s: str) -> str:
    s = s.strip()
    # 找第一行以 |user|: 开头
    for ln in s.splitlines():
        ln = ln.strip()
        if ln.lower().startswith("|user|:"):
            # 避免把 think 标签当输出
            if "<think>" in ln.lower():
                continue
            return ln
    # fallback
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    ln = lines[0] if lines else ""
    if not ln.lower().startswith("|user|:"):
        ln = "|user|: " + ln.lstrip(": ").strip()
    return ln


def split_thinking_and_content(tok, output_ids: List[int]) -> (str, str):
    """
    Official-style splitting by token id 151668 (</think>).
    output_ids: generated token ids ONLY (prompt removed).
    Returns: (thinking_text, content_text)
    """
    try:
        # index = position AFTER last </think>
        idx = len(output_ids) - output_ids[::-1].index(THINK_END_TOKEN_ID)
    except ValueError:
        idx = 0

    thinking = tok.decode(output_ids[:idx], skip_special_tokens=True).strip("\n")
    content = tok.decode(output_ids[idx:], skip_special_tokens=True).strip("\n")
    return thinking, content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_queries_jsonl", required=True)
    ap.add_argument("--out_rewrite_jsonl", required=True)
    ap.add_argument("--model", required=True)

    ap.add_argument("--benchmark", default="IBM MT-RAG")
    ap.add_argument("--domain", default="")

    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=128)

    ap.add_argument("--max_turn_tokens", type=int, default=256)
    ap.add_argument("--keep_first_turn", action="store_true")

    ap.add_argument("--temperature", type=float, default=0.4)

    # ✅ official switch
    ap.add_argument("--enable_thinking", action="store_true",
                    help="Enable Qwen3 thinking mode (default: False).")

    ap.add_argument("--log_first_n", type=int, default=5)
    ap.add_argument("--log_max_chars", type=int, default=2000)
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # decoder-only batch: left padding
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if args.device == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()

    rows = load_jsonl(Path(args.in_queries_jsonl))
    out_rows: List[Dict[str, Any]] = []

    printed = 0
    bs = args.batch_size

    with torch.no_grad():
        for i in tqdm(range(0, len(rows), bs), desc="rewrite (Qwen3 official)", file=sys.stdout):
            batch = rows[i:i + bs]

            qids: List[str] = []
            texts_for_log: List[str] = []
            turns_for_log: List[str] = []
            model_inputs = []

            for r in batch:
                qid = r.get("_id") or r.get("id") or r.get("query_id")
                if qid is None:
                    raise ValueError("Missing _id/id/query_id.")
                qids.append(str(qid))

                raw_text = r.get("text") or r.get("query") or ""
                texts_for_log.append(raw_text)

                user_lines = parse_user_lines(raw_text)
                turns = truncate_by_token_budget(tok, user_lines, args.max_turn_tokens, args.keep_first_turn)
                turns_str = "\n".join([f"[{k+1}] {t}" for k, t in enumerate(turns)]) if turns else raw_text.strip()
                turns_for_log.append(turns_str)

                messages = build_messages(args.benchmark, args.domain, turns_str)

                # ✅ official: apply_chat_template with enable_thinking switch
                text = tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=bool(args.enable_thinking),
                )
                model_inputs.append(text)

            enc = tok(model_inputs, return_tensors="pt", padding=True, truncation=True)
            if args.device == "cuda":
                enc = {k: v.to(model.device) for k, v in enc.items()}

            gen_kwargs = dict(
                max_new_tokens=args.max_new_tokens,
                do_sample=(args.temperature > 0),
                temperature=(args.temperature if args.temperature > 0 else None),
                eos_token_id=tok.eos_token_id,
                pad_token_id=(tok.pad_token_id or tok.eos_token_id),
            )
            gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

            generated = model.generate(**enc, **gen_kwargs)

            # ✅ official slicing: remove the prompt tokens by length
            # Note: with padding, each sample has same seq_len after padding, but prompt length differs.
            # We use attention_mask sum to get each sample's true prompt token count, then slice output ids.
            # This works because we slice in *token space*, not decoded string space.
            for j, qid in enumerate(qids):
                T = enc["input_ids"].shape[1]          # ✅ batch padded input length
                output_ids = generated[j][T:].tolist() # ✅ generated continuation only
                thinking, content = split_thinking_and_content(tok, output_ids)

                rewritten = clean_output(content)

                out_rows.append({"_id": qid, "text": rewritten})

                if printed < args.log_first_n:
                    orig = texts_for_log[j].strip()
                    turns_used = turns_for_log[j].strip()
                    if len(orig) > args.log_max_chars:
                        orig = orig[:args.log_max_chars] + "\n...[truncated]..."
                    if len(turns_used) > args.log_max_chars:
                        turns_used = turns_used[:args.log_max_chars] + "\n...[truncated]..."

                    print("\n" + "=" * 80, flush=True)
                    print(f"[REWRITE SAMPLE {printed+1}/{args.log_first_n}] _id = {qid}", flush=True)
                    print("-" * 80, flush=True)
                    print("[ORIGINAL QUERY HISTORY (raw text field)]", flush=True)
                    print(orig, flush=True)
                    print("-" * 80, flush=True)
                    print("[TURNS USED (numbered)]", flush=True)
                    print(turns_used, flush=True)
                    print("-" * 80, flush=True)
                    print("[THINKING CONTENT]", flush=True)
                    print(thinking, flush=True)
                    print("-" * 80, flush=True)
                    print("[MODEL CONTENT]", flush=True)
                    print(content, flush=True)
                    print("-" * 80, flush=True)
                    print("[REWRITTEN QUERY]", flush=True)
                    print(rewritten, flush=True)
                    print("=" * 80 + "\n", flush=True)

                    printed += 1

    write_jsonl(Path(args.out_rewrite_jsonl), out_rows)
    print(f"[OK] wrote: {args.out_rewrite_jsonl} (n={len(out_rows)})")


if __name__ == "__main__":
    main()
