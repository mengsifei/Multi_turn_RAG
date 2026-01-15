#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gemma2-2B query rewrite for IBM MT-RAG.

Goal (IMPORTANT):
- For each history (turn 1..k), produce ONE standalone query representing the user's
  information need at turn k (often requires combining relevant earlier turns).
- Keep same _id to align with qrels/dev.tsv.

Key fixes:
- NO stop-on-newline (Gemma often emits leading newline -> empty generations).
- Two-pass generation:
  1) normal rewrite
  2) if output collapses to last-turn, force inclusion of anchor keywords from context
- Strong postprocess + fallback heuristic.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------- IO ----------------
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


# ---------------- Turns ----------------
def parse_user_lines(text: str) -> List[str]:
    out = []
    for ln in text.splitlines():
        ln = ln.strip()
        if ln.lower().startswith("|user|:"):
            out.append(ln)
    return out


def strip_user_prefix(line: str) -> str:
    return re.sub(r"^\|user\|\s*:\s*", "", line, flags=re.IGNORECASE).strip()


def truncate_by_token_budget(tok, turns: List[str], max_tokens: int, keep_first: bool) -> List[str]:
    """Recency-first truncation by token budget (no semantic filtering)."""
    if not turns:
        return turns

    kept_rev = []
    total = 0
    for t in reversed(turns):
        n = len(tok.encode(t, add_special_tokens=False)) + 1
        if kept_rev and (total + n) > max_tokens:
            break
        if not kept_rev and n > max_tokens:
            ids = tok.encode(t, add_special_tokens=False)[:max_tokens]
            kept_rev.append(tok.decode(ids, skip_special_tokens=True))
            total = max_tokens
            break
        kept_rev.append(t)
        total += n

    kept = list(reversed(kept_rev))
    if keep_first and turns:
        first = turns[0]
        if first not in kept:
            n_first = len(tok.encode(first, add_special_tokens=False)) + 1
            if (total + n_first) <= max_tokens:
                kept = [first] + kept
    return kept


# ---------------- Prompt ----------------
FEWSHOT = (
    "Example:\n"
    "Turns:\n"
    "[1] where do the arizona cardinals play this week\n"
    "[2] Do the Arizona Cardinals play outside the US?\n"
    "Rewrite:\n"
    "Where do the Arizona Cardinals play this week, including whether the game is outside the United States?\n"
    "----\n"
    "Turns:\n"
    "[1] tell me about the chicago cardinals\n"
    "[2] are they the same as the arizona cardinals?\n"
    "Rewrite:\n"
    "Are the Chicago Cardinals the same team as the Arizona Cardinals?\n"
    "----\n"
)
def build_prompt(
    turns_str: str,
    benchmark: str,
    domain: str,
    required_keywords: List[str] | None = None,
) -> str:
    req = ""
    if required_keywords:
        req = (
            "\nREQUIRED KEYWORDS:\n"
            + ", ".join(required_keywords)
            + "\nYour rewrite MUST include at least one REQUIRED KEYWORD.\n"
        )

    return (
        "Purpose:\n"
        "Rewrite a multi-turn conversational query into a single, clean, standalone question, "
        "following the style of official CLAPNQ rewrites.\n\n"

        "Input:\n"
        "- A conversation history consisting of multiple user turns.\n"
        "- The LAST line is the current user question.\n"
        "- Earlier lines provide context and disambiguation.\n\n"

        "Task:\n"
        "Rewrite ONLY the last user question into ONE standalone question that can be fully "
        "understood without seeing the conversation history.\n\n"

        "Requirements:\n"
        "- Output ONE complete English question.\n"
        "- Preserve the original question intent and structure "
        "(who / what / where / when / why / how).\n"
        "- Use earlier turns ONLY when necessary to resolve ambiguity "
        "(pronouns, ellipsis, vague references).\n"
        "- Remove conversational fillers such as: "
        "\"No, I meant...\", \"That is too bad.\", \"Thanks\", \"I see\".\n"
        "- Do NOT introduce new concepts, categories, or explanations.\n"
        "- Do NOT merge multiple questions.\n"
        "- Do NOT copy or concatenate the conversation history.\n"
        "- Do NOT answer the question.\n\n"

        "What NOT to do:\n"
        "- Do not paraphrase the whole conversation.\n"
        "- Do not expand the topic.\n"
        "- Do not add examples or lists.\n"
        "- Do not change a question into a statement.\n\n"

        f"{req}"
        f"{FEWSHOT}"
        "Conversation history (chronological):\n"
        f"{turns_str}\n"
        "Rewrite (one question only): "
    )



# ---------------- Output cleaning & quality ----------------
_STOPWORDS = {
    "the","and","are","was","were","how","many","what","where","when","which","who",
    "does","do","did","is","in","on","at","to","of","a","an","for","with","this","that",
    "it","they","them","their","as","by","from","or","be","been","being","about"
}

def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s'-]", "", s)
    return s

def clean_one_line(gen: str) -> str:
    # take first non-empty line
    lines = [ln.strip() for ln in gen.strip().splitlines() if ln.strip()]
    s = lines[0] if lines else ""
    # strip common wrappers
    s = s.replace("</OUTPUT>", "").replace("</output>", "").strip()
    s = re.sub(r"^\s*(final\s*:|output\s*:)\s*", "", s, flags=re.IGNORECASE).strip()
    s = s.strip("\"'“”")
    if s.lower().startswith("|user|:"):
        s = re.sub(r"^\|user\|\s*:\s*", "", s, flags=re.IGNORECASE).strip()
    return s

def looks_garbage(s: str) -> bool:
    t = s.strip()
    if len(t) < 6:
        return True
    if re.fullmatch(r"\d+", t):
        return True
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", t):
        return True
    low = t.lower()
    if "|document|" in low or "|rank|" in low or "|score|" in low or "|relevance|" in low:
        return True
    return False

def is_standalone_question(q: str) -> bool:
    q = q.strip()
    if not q.endswith("?"):
        return False
    # no pronouns / vague refs
    if re.search(r"\b(it|they|this|that|these|those|he|she|them|him|her|back then)\b", q, re.I):
        return False
    # long enough and has content
    return len(q.split()) >= 4


def extract_anchors(context_turns: List[str], latest_turn: str, k: int = 3) -> List[str]:
    """Pick a few anchor words from context that are not present in latest."""
    ctx = " ".join(context_turns)
    latest = set(re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", latest_turn.lower()))
    words = re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", ctx)
    cand = []
    for w in words:
        wl = w.lower()
        if wl in _STOPWORDS:
            continue
        if wl in latest:
            continue
        cand.append(w)

    # frequency + length
    freq = {}
    for w in cand:
        wl = w.lower()
        freq[wl] = freq.get(wl, 0) + 1
    scored = sorted(freq.items(), key=lambda x: (x[1], len(x[0])), reverse=True)
    anchors = [w for w, _ in scored[:k]]
    return anchors

def heuristic_combine(turns: List[str]) -> str:
    """
    Very simple fallback: if model fails, combine last 2 turns:
    - base: previous turn
    - refine: last turn
    """
    if not turns:
        return ""
    if len(turns) == 1:
        return turns[0]
    prev, last = turns[-2], turns[-1]
    # if last is yes/no, attach as constraint
    if re.match(r"^(do|does|did|is|are|was|were|can|could|will|would|should|has|have|had)\b", last.strip().lower()):
        return f"{prev.rstrip('?').strip()}, including whether {last.rstrip('?').strip().lower()}."
    return f"{last.strip()} (context: {prev.strip()})"


# ---------------- Generation helpers ----------------
def generate_batch(model, tok, prompts: List[str], *, max_new_tokens: int, temperature: float, top_p: float,
                   repetition_penalty: float, no_repeat_ngram_size: int) -> List[str]:
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=(temperature if temperature > 0 else None),
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    out = model.generate(**enc, **gen_kwargs)
    T = enc["input_ids"].shape[1]
    return [tok.decode(out[i, T:], skip_special_tokens=True) for i in range(out.shape[0])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_queries_jsonl", required=True)
    ap.add_argument("--out_rewrite_jsonl", required=True)
    ap.add_argument("--model", required=True, help='Local path, e.g. "./gemma-2b"')
    ap.add_argument("--benchmark", default="IBM MT-RAG")
    ap.add_argument("--domain", default="")

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--max_turn_tokens", type=int, default=320)
    ap.add_argument("--keep_first_turn", action="store_true")

    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=6)

    ap.add_argument("--log_first_n", type=int, default=5)
    ap.add_argument("--log_max_chars", type=int, default=2000)
    args = ap.parse_args()

    model_path = str(Path(args.model).resolve())
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()

    rows = load_jsonl(Path(args.in_queries_jsonl))
    out_rows: List[Dict[str, Any]] = []

    printed = 0
    bs = args.batch_size

    with torch.no_grad():
        for i in tqdm(range(0, len(rows), bs), desc="rewrite (Gemma2-2B v3)", file=sys.stdout):
            batch = rows[i:i + bs]

            qids: List[str] = []
            turns_list: List[List[str]] = []
            prompts1: List[str] = []
            anchors_list: List[List[str]] = []
            debug: List[Dict[str, str]] = []

            # ---------- build pass1 ----------
            for r in batch:
                qid = r.get("_id") or r.get("id") or r.get("query_id")
                if qid is None:
                    raise ValueError("Missing _id/id/query_id in input.")
                qid = str(qid)

                raw_text = r.get("text") or r.get("query") or ""
                user_lines = parse_user_lines(raw_text)
                turns = [strip_user_prefix(x) for x in user_lines] if user_lines else [raw_text.strip()]
                turns = truncate_by_token_budget(tok, turns, args.max_turn_tokens, args.keep_first_turn)

                turns_str = "\n".join([f"[{k+1}] {t}" for k, t in enumerate(turns)])
                prompt = build_prompt(turns_str, args.benchmark, args.domain, required_keywords=None)

                context = turns[:-1] if len(turns) > 1 else []
                latest = turns[-1] if turns else ""
                anchors = extract_anchors(context, latest, k=3)

                qids.append(qid)
                turns_list.append(turns)
                anchors_list.append(anchors)
                prompts1.append(prompt)
                debug.append({"qid": qid, "orig": raw_text, "turns_str": turns_str, "latest": latest, "anchors": ", ".join(anchors)})

            raw1 = generate_batch(
                model, tok, prompts1,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )

            out1 = [clean_one_line(x) for x in raw1]

            # decide which need pass2
            need2_idx = []
            for j, turns in enumerate(turns_list):
                latest = turns[-1] if turns else ""
                anchors = anchors_list[j]
                if len(turns) <= 1:
                    continue
                if not anchors:
                    continue
                if looks_garbage(out1[j]):
                    need2_idx.append(j)
                    continue
                # if output basically equals the latest -> likely missed context refinement
                if normalize(out1[j]) == normalize(latest):
                    if not is_standalone_question(latest):
                        need2_idx.append(j)

            # ---------- pass2 (force anchors) ----------
            out_final = out1[:]
            raw2_map = {}

            if need2_idx:
                prompts2 = []
                idx2 = []
                for j in need2_idx:
                    turns = turns_list[j]
                    turns_str = "\n".join([f"[{k+1}] {t}" for k, t in enumerate(turns)])
                    anchors = anchors_list[j]
                    prompts2.append(build_prompt(turns_str, args.benchmark, args.domain, required_keywords=anchors))
                    idx2.append(j)

                raw2 = generate_batch(
                    model, tok, prompts2,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                )

                for k, j in enumerate(idx2):
                    raw2_map[j] = raw2[k]
                    cand = clean_one_line(raw2[k])
                    # accept cand if not garbage and contains at least one anchor
                    if not looks_garbage(cand):
                        if any(a.lower() in cand.lower() for a in anchors_list[j]):
                            out_final[j] = cand

            # ---------- finalize + fallback ----------
            for j, qid in enumerate(qids):
                turns = turns_list[j]
                latest = turns[-1] if turns else ""

                s = out_final[j].strip()
                if looks_garbage(s):
                    s = heuristic_combine(turns) or latest

                # one-line + prefix
                s = s.strip().splitlines()[0].strip()
                out_rows.append({"_id": qid, "text": "|user|: " + s})

                # logs
                if printed < args.log_first_n:
                    info = debug[j]
                    orig = info["orig"].strip()
                    if len(orig) > args.log_max_chars:
                        orig = orig[:args.log_max_chars] + "\n...[truncated]..."

                    print("\n" + "=" * 80, flush=True)
                    print(f"[REWRITE SAMPLE {printed+1}/{args.log_first_n}] _id = {qid}", flush=True)
                    print("-" * 80, flush=True)
                    print("[ORIGINAL QUERY HISTORY]", flush=True)
                    print(orig, flush=True)
                    print("-" * 80, flush=True)
                    print("[TURNS USED]", flush=True)
                    print(info["turns_str"], flush=True)
                    print("-" * 80, flush=True)
                    print("[LATEST]", flush=True)
                    print(info["latest"], flush=True)
                    print("-" * 80, flush=True)
                    print("[ANCHORS]", flush=True)
                    print(info["anchors"] if info["anchors"] else "(none)", flush=True)
                    print("-" * 80, flush=True)
                    print("[MODEL RAW pass1]", flush=True)
                    print(raw1[j], flush=True)
                    if j in raw2_map:
                        print("-" * 80, flush=True)
                        print("[MODEL RAW pass2]", flush=True)
                        print(raw2_map[j], flush=True)
                    print("-" * 80, flush=True)
                    print("[REWRITTEN]", flush=True)
                    print("|user|: " + s, flush=True)
                    print("=" * 80 + "\n", flush=True)
                    printed += 1

    write_jsonl(Path(args.out_rewrite_jsonl), out_rows)
    print(f"[OK] wrote: {args.out_rewrite_jsonl} (n={len(out_rows)})")


if __name__ == "__main__":
    main()
