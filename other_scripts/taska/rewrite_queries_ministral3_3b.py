#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM-only query rewrite for IBM MT-RAG using Ministral 3 (3B Instruct 2512).

- Keeps same _id for qrels/dev.tsv alignment
- No overlap heuristics; only token-budget truncation (recency-first)
- Strong prompt: rewrite ONLY the latest intent; history only for disambiguation
- Uses Transformers recommended stack for Mistral3: AutoProcessor + AutoModelForImageTextToText
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, List

import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText

try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
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


def truncate_by_token_budget(tokenizer, user_lines: List[str], max_tokens: int, keep_first: bool) -> List[str]:
    """Recency-first truncation by token budget. No semantic filtering."""
    if not user_lines:
        return user_lines

    kept_rev = []
    total = 0

    for ln in reversed(user_lines):
        txt = strip_user_prefix(ln)
        n = len(tokenizer.encode(txt, add_special_tokens=False)) + 1
        if kept_rev and (total + n) > max_tokens:
            break
        if not kept_rev and n > max_tokens:
            ids = tokenizer.encode(txt, add_special_tokens=False)[:max_tokens]
            txt = tokenizer.decode(ids, skip_special_tokens=True)
            kept_rev.append(txt)
            total = max_tokens
            break
        kept_rev.append(txt)
        total += n

    kept = list(reversed(kept_rev))

    if keep_first and user_lines:
        first = strip_user_prefix(user_lines[0])
        if first not in kept:
            n_first = len(tokenizer.encode(first, add_special_tokens=False)) + 1
            if (total + n_first) <= max_tokens:
                kept = [first] + kept

    return kept


def build_messages(benchmark: str, domain: str, turns_str: str) -> List[Dict[str, str]]:
    system = (
        f"You are a query rewriter for the {benchmark} multi-turn retrieval benchmark.\n"
        f"Domain: {domain if domain else 'unknown'}.\n\n"
        "Task: Rewrite into ONE self-contained search query for document retrieval.\n\n"
        "Rules (must follow):\n"
        "1) Rewrite ONLY the latest user intent (the last turn). Use earlier turns ONLY to resolve references "
        "(entities/pronouns/constraints).\n"
        "2) Do NOT merge unrelated earlier topics/questions.\n"
        "3) Keep key entities, constraints, time ranges, locations, negations needed for retrieval.\n"
        "4) Do NOT answer. Do NOT add new facts.\n"
        "5) Output EXACTLY ONE line starting with: |user|:\n"
        "6) No extra lines, no explanations.\n"
    )

    user = (
        "Input turns (chronological):\n"
        f"{turns_str}\n\n"
        "Output (one line only):"
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def clean_output(s: str) -> str:
    s = s.strip()

    for ln in s.splitlines():
        ln = ln.strip()
        if ln.lower().startswith("|user|:"):
            return ln

    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    ln = lines[0] if lines else ""
    ln = ln.strip("\"'“”")
    if not ln.lower().startswith("|user|:"):
        ln = "|user|: " + ln.lstrip(": ").strip()
    return ln


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_queries_jsonl", required=True)
    ap.add_argument("--out_rewrite_jsonl", required=True)
    ap.add_argument("--model", required=True, help="Local path, e.g. ./Ministral-3-3B-Instruct-2512-BF16")

    ap.add_argument("--benchmark", default="IBM MT-RAG")
    ap.add_argument("--domain", default="")

    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dtype", default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=96)

    ap.add_argument("--max_turn_tokens", type=int, default=320)
    ap.add_argument("--keep_first_turn", action="store_true")

    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)

    ap.add_argument("--load_in_4bit", action="store_true", help="Optional: bitsandbytes 4bit load")
    ap.add_argument("--log_first_n", type=int, default=5)
    ap.add_argument("--log_max_chars", type=int, default=2000)
    args = ap.parse_args()

    # Normalize model path to absolute path to avoid HF validation errors
    model_path = str(Path(args.model).resolve())

    # ---- load processor/tokenizer ----
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tok = getattr(processor, "tokenizer", None) or processor

    # decoder-only batch safety
    if hasattr(tok, "padding_side"):
        tok.padding_side = "left"
    if hasattr(tok, "pad_token_id") and tok.pad_token_id is None and hasattr(tok, "eos_token"):
        tok.pad_token = tok.eos_token

    # ---- dtype ----
    torch_dtype = None
    if args.dtype == "auto":
        torch_dtype = "auto"
    elif args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # ---- quant ----
    quantization_config = None
    if args.load_in_4bit:
        if not _HAS_BNB:
            raise RuntimeError("BitsAndBytesConfig not available. Install bitsandbytes or run without --load_in_4bit.")
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    # ---- load model ----
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        device_map="auto" if args.device == "cuda" else None,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        trust_remote_code=True,
        local_files_only=True
    )
    model.eval()

    rows = load_jsonl(Path(args.in_queries_jsonl))
    out_rows: List[Dict[str, Any]] = []

    printed = 0
    bs = args.batch_size

    with torch.no_grad():
        for i in tqdm(range(0, len(rows), bs), desc="rewrite (Ministral3-3B)", file=sys.stdout):
            batch = rows[i:i + bs]

            qids: List[str] = []
            texts_for_processor: List[str] = []
            debug_infos: List[Dict[str, str]] = []

            for r in batch:
                qid = r.get("_id") or r.get("id") or r.get("query_id")
                if qid is None:
                    raise ValueError("Missing _id/id/query_id in input.")
                qid = str(qid)

                raw_text = r.get("text") or r.get("query") or ""
                user_lines = parse_user_lines(raw_text)
                turns = truncate_by_token_budget(tok, user_lines, args.max_turn_tokens, args.keep_first_turn)
                turns_str = "\n".join([f"[{k+1}] {t}" for k, t in enumerate(turns)]) if turns else raw_text.strip()

                messages = build_messages(args.benchmark, args.domain, turns_str)

                # chat template -> string
                chat_text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                qids.append(qid)
                texts_for_processor.append(chat_text)
                debug_infos.append({"qid": qid, "orig_text": raw_text, "turns_str": turns_str})

            inputs = processor(
                text=texts_for_processor,
                return_tensors="pt",
                padding=True,
            )

            # move tensors
            if args.device == "cuda":
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            # some models like dtype for inputs
            if args.device == "cuda" and isinstance(torch_dtype, torch.dtype):
                for k, v in inputs.items():
                    if v.dtype.is_floating_point:
                        inputs[k] = v.to(dtype=torch_dtype)

            gen_kwargs = dict(
                max_new_tokens=args.max_new_tokens,
                do_sample=(args.temperature > 0),
                temperature=(args.temperature if args.temperature > 0 else None),
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
            gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

            generated = model.generate(**inputs, **gen_kwargs)

            T = inputs["input_ids"].shape[1]
            raw_out = processor.batch_decode(generated[:, T:], skip_special_tokens=True)

            for j, qid in enumerate(qids):
                raw = raw_out[j]
                rewritten = clean_output(raw)
                out_rows.append({"_id": qid, "text": rewritten})

                if printed < args.log_first_n:
                    info = debug_infos[j]
                    orig = info["orig_text"].strip()
                    turns_used = info["turns_str"].strip()
                    if len(orig) > args.log_max_chars:
                        orig = orig[:args.log_max_chars] + "\n...[truncated]..."
                    if len(turns_used) > args.log_max_chars:
                        turns_used = turns_used[:args.log_max_chars] + "\n...[truncated]..."

                    print("\n" + "=" * 80, flush=True)
                    print(f"[REWRITE SAMPLE {printed+1}/{args.log_first_n}] _id = {info['qid']}", flush=True)
                    print("-" * 80, flush=True)
                    print("[ORIGINAL QUERY HISTORY (raw text field)]", flush=True)
                    print(orig, flush=True)
                    print("-" * 80, flush=True)
                    print("[TURNS USED (numbered)]", flush=True)
                    print(turns_used, flush=True)
                    print("-" * 80, flush=True)
                    print("[MODEL RAW GENERATION (continuation only)]", flush=True)
                    print(raw, flush=True)
                    print("-" * 80, flush=True)
                    print("[REWRITTEN QUERY]", flush=True)
                    print(rewritten, flush=True)
                    print("=" * 80 + "\n", flush=True)
                    printed += 1

    write_jsonl(Path(args.out_rewrite_jsonl), out_rows)
    print(f"[OK] wrote: {args.out_rewrite_jsonl} (n={len(out_rows)})")


if __name__ == "__main__":
    main()
