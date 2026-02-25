#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_keywords_qwen3.py

Keyword extraction for rewritten standalone queries using a local instruct LLM (Qwen3-4B-Instruct-2507).

Input JSONL format (typical MTRAG):
  {"_id": "...", "text": "|user|: ..."}  or {"_id": "...", "text": "..."}

Output JSONL:
  adds fields:
    - "keywords": [ ... ]
    - "keyword_query": "kw1; kw2; ..."

Example:
  python extract_keywords_qwen3.py \
    --model /home/smen/mt-rag/mt-rag-benchmark/Qwen3-4B-Instruct-2507 \
    --in_jsonl /home/smen/mt-rag/mt-rag-benchmark/cleaned_dataset/clapnq/clapnq_rewrite_gpt.jsonl \
    --out_jsonl /home/smen/mt-rag/mt-rag-benchmark/cleaned_dataset/clapnq/clapnq_rewrite_gpt_with_keywords.jsonl \
    --device cuda --dtype bf16 --batch_size 16
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# Fallback keyword extraction (cheap & robust)
# -----------------------------
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-\.\/]*")
STOP = {
    "the","a","an","and","or","but","if","then","than","to","of","in","on","at","for","from","with","without",
    "is","are","was","were","be","been","being","do","does","did","doing","have","has","had",
    "i","you","he","she","it","we","they","them","this","that","these","those","there","here",
    "what","which","who","whom","when","where","why","how","can","could","would","should","may","might","will",
    "about","into","over","under","between","within","across","also","too",
}

def fallback_keywords(q: str, k_min: int = 8, k_max: int = 15) -> List[str]:
    toks = [t.lower() for t in TOKEN_RE.findall(q or "")]
    # keep numbers/codes + longer tokens
    scored = []
    for t in toks:
        if t in STOP:
            continue
        score = 0
        if any(ch.isdigit() for ch in t):
            score += 3
        if len(t) >= 8:
            score += 2
        if "-" in t or "_" in t or "/" in t:
            score += 2
        if len(t) >= 4:
            score += 1
        scored.append((score, t))
    scored.sort(key=lambda x: (-x[0], x[1]))
    uniq = []
    seen = set()
    for _, t in scored:
        if t in seen:
            continue
        seen.add(t)
        uniq.append(t)
        if len(uniq) >= k_max:
            break
    # pad if too few
    if len(uniq) < k_min:
        # add remaining non-stop tokens
        for t in toks:
            if t in STOP or t in seen:
                continue
            seen.add(t)
            uniq.append(t)
            if len(uniq) >= k_min:
                break
    return uniq[:k_max]

# -----------------------------
# Prompting & parsing
# -----------------------------
SYS_PROMPT = (
    """
    You extract retrieval keywords from the query.

    CRITICAL RULES:
    - DO NOT invent new facts, entities, examples, or specific states/companies not explicitly mentioned in the query.
    - Keywords must be exact spans copied from the query, OR very small morphological variants (singular/plural), OR obvious acronym expansions ONLY when the acronym appears in the query.
    - Prefer concrete noun phrases, named entities, technical terms, and numbers.
    - Avoid generic paraphrases and broad concepts not present in the query.

    Return ONLY valid JSON: {"keywords":[...]} with 1-5 items.
    Each keyword is 1-4 words. No duplicates.

    
    > Example:

    Original query:

    ```text
    When was the GDPR enacted, and what does it regulate?
    ```

    Expected output (JSON only):

    ```json
    {"keywords":["GDPR","enacted","regulate"]}
    ```

    > Example:

    Original query:

    ```text
    Does OpenSSL 3.0 support TLS 1.3 by default?
    ```

    Expected output (JSON only):

    ```json
    {"keywords":["OpenSSL 3.0","TLS 1.3","by default"]}
    ```
    """
)

def clean_query_text(text: str) -> str:
    s = (text or "").strip()
    # common MTRAG format: "|user|: ..."
    s = re.sub(r"^\|user\|\s*:\s*", "", s).strip()
    return s

def extract_first_json_obj(s: str) -> Optional[str]:
    # find first {...} block
    if not s:
        return None
    start = s.find("{")
    if start < 0:
        return None
    # naive brace matching
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start:i+1]
    return None

def parse_keywords(model_out: str) -> Optional[List[str]]:
    js = extract_first_json_obj(model_out)
    if not js:
        return None
    try:
        obj = json.loads(js)
        kws = obj.get("keywords", None)
        if not isinstance(kws, list):
            return None
        # normalize and filter empties
        norm = []
        seen = set()
        for x in kws:
            if not isinstance(x, str):
                continue
            t = x.strip()
            if not t:
                continue
            key = t.lower()
            if key in seen:
                continue
            seen.add(key)
            norm.append(t)
        if 1 <= len(norm) <= 64:
            return norm
        return None
    except Exception:
        return None

# -----------------------------
# Model inference
# -----------------------------
@torch.inference_mode()
def generate_keywords_batch(
    tokenizer,
    model,
    queries: List[str],
    max_input_tokens: int,
    max_new_tokens: int,
    temperature: float,
) -> List[str]:
    messages_list = []
    for q in queries:
        user = f"Query:\n{q}\n\nReturn JSON now."
        messages_list.append([
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": user},
        ])

    # Use chat template if available; fallback to plain formatting
    if hasattr(tokenizer, "apply_chat_template"):
        texts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_list
        ]
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        )
    else:
        texts = []
        for m in messages_list:
            texts.append(f"[SYSTEM]\n{m[0]['content']}\n[USER]\n{m[1]['content']}\n[ASSISTANT]\n")
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        )

    enc = {k: v.to(model.device) for k, v in enc.items()}
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature if temperature > 0 else None,
        top_p=0.95 if temperature > 0 else None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    out = model.generate(**enc, **gen_kwargs)
    # only decode generated tail for each sample
    res = []
    for i in range(out.size(0)):
        # decode full then strip prompt by length
        prompt_len = enc["input_ids"][i].size(0)
        gen_ids = out[i, prompt_len:]
        res.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Local HF model path")
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--text_key", default="text", help="Field name containing query text")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_input_tokens", type=int, default=256)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.0, help="0.0 for deterministic")
    ap.add_argument("--k_min", type=int, default=8)
    ap.add_argument("--k_max", type=int, default=15)
    args = ap.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # left padding tends to be safer for batched causal LM
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto" if args.device == "cuda" else None,
    )
    model.eval()

    in_path = Path(args.in_jsonl)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    # Prepare cleaned queries
    cleaned: List[str] = []
    for r in rows:
        q = clean_query_text(str(r.get(args.text_key, "")))
        cleaned.append(q)

    # Batch inference
    outputs: List[Optional[List[str]]] = [None] * len(rows)
    for start in range(0, len(rows), args.batch_size):
        end = min(len(rows), start + args.batch_size)
        batch_q = cleaned[start:end]

        gen_texts = generate_keywords_batch(
            tokenizer=tokenizer,
            model=model,
            queries=batch_q,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        for j, gen in enumerate(gen_texts):
            kws = parse_keywords(gen)
            if kws is None:
                kws = fallback_keywords(batch_q[j], k_min=args.k_min, k_max=args.k_max)
            # clamp count
            kws = kws[: args.k_max]
            if len(kws) < args.k_min:
                # pad with fallback if needed
                extra = fallback_keywords(batch_q[j], k_min=args.k_min, k_max=args.k_max)
                seen = {x.lower() for x in kws}
                for x in extra:
                    if x.lower() in seen:
                        continue
                    kws.append(x)
                    seen.add(x.lower())
                    if len(kws) >= args.k_min:
                        break
                kws = kws[: args.k_max]
            outputs[start + j] = kws

    # Write output
    with out_path.open("w", encoding="utf-8") as wf:
        for r, q, kws in zip(rows, cleaned, outputs):
            assert kws is not None
            r["keywords"] = kws
            r["keyword_query"] = "; ".join(kws)
            # also keep a convenient field for downstream if you want
            r["clean_query"] = q
            wf.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[OK] wrote: {out_path}  (n={len(rows)})")

if __name__ == "__main__":
    main()
