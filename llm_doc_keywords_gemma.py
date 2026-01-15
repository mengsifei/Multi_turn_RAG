#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, re, math
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# -------------------------
# Basic tokenization utils (for post-filter)
# -------------------------
STOPWORDS = set("""
a an the and or but if then else when while for from to of in on at by with without
is are was were be been being do does did doing have has had having
this that these those it its it's i you he she we they them their our your my
as not no yes can could may might will would should must
about into over under above below up down out off
""".split())

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")

def norm_kw(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def is_bad_kw(s: str) -> bool:
    s = norm_kw(s)
    if not s:
        return True
    # 太短
    if len(s) < 3:
        return True
    # stopword 单词
    if s.lower() in STOPWORDS:
        return True
    # 纯数字
    if s.isdigit():
        return True
    return False


# -------------------------
# FIQA parent parser
# -------------------------
def fiqa_parent(pid: str) -> str:
    # 10171-0-2129 -> 10171
    parts = (pid or "").split("-")
    return parts[0] if parts else pid


def build_doc_texts_from_passages(
    in_jsonl: str,
    max_chars_per_doc: int = 6000,
) -> Dict[str, str]:
    """
    Group passages by parent docid, concat them (order by start offset if possible),
    then truncate by max_chars_per_doc.
    """
    buckets = defaultdict(list)  # parent -> list[(start, text)]
    with open(in_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            pid = o.get("id") or o.get("_id")
            if not pid:
                continue
            parent = fiqa_parent(pid)
            text = (o.get("title","") + "\n" + o.get("text","")).strip()

            # parse start if possible
            parts = pid.split("-")
            start = int(parts[-2]) if len(parts) >= 3 and parts[-2].isdigit() else 0
            buckets[parent].append((start, text))

    out = {}
    for parent, items in buckets.items():
        items.sort(key=lambda x: x[0])
        doc = "\n\n".join(t for _, t in items).strip()
        if len(doc) > max_chars_per_doc:
            doc = doc[:max_chars_per_doc]
        out[parent] = doc
    return out


# -------------------------
# Gemma prompting
# -------------------------
def build_prompt(doc_text: str) -> str:
    # 强约束：只输出 JSON，不要解释；关键词要利于检索（区分性/术语/实体）
    return (
        "You are an information retrieval expert.\n"
        "Task: Given a document, produce EXACTLY 5 retrieval keywords (or short keyphrases).\n"
        "Rules:\n"
        "- Output MUST be valid JSON ONLY, no extra text.\n"
        "- JSON schema: {\"keywords\": [\"k1\",\"k2\",\"k3\",\"k4\",\"k5\"]}\n"
        "- Each keyword: 1-3 words, specific and discriminative.\n"
        "- Prefer entities, technical terms, product names, institutions, metrics.\n"
        "- Avoid generic words (e.g., 'time', 'people', 'good', 'thing').\n"
        "- Avoid stopwords-only phrases.\n"
        "- Do NOT include quotes from the document.\n\n"
        "DOCUMENT:\n"
        f"{doc_text}\n\n"
        "JSON:\n"
    )


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

def extract_json_obj(s: str) -> Optional[dict]:
    """
    Gemma 有时会多吐一点空格/换行；我们用正则捞出第一个 {...} 再 json.loads
    """
    if not s:
        return None
    m = _JSON_OBJ_RE.search(s)
    if not m:
        return None
    chunk = m.group(0)
    try:
        return json.loads(chunk)
    except Exception:
        return None


@torch.no_grad()
def generate_keywords_batch(
    model,
    tokenizer,
    prompts: List[str],
    *,
    max_new_tokens: int = 128,
    batch_size: int = 8,
) -> List[List[str]]:
    out_kws = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        tok = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        tok = {k: v.to(model.device) for k, v in tok.items()}

        gen = model.generate(
            **tok,
            max_new_tokens=max_new_tokens,
            do_sample=True,          # 关键：确定性输出
            temperature=0.4,
            top_p=1.0,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
        )
        texts = tokenizer.batch_decode(gen, skip_special_tokens=True)

        # 只取最后一段：模型输出可能包含 prompt 原文
        for full in texts:
            # 尝试从末尾提取 JSON
            obj = extract_json_obj(full)
            kws = []
            if obj and isinstance(obj.get("keywords"), list):
                kws = [norm_kw(x) for x in obj["keywords"] if isinstance(x, str)]
            out_kws.append(kws)
    return out_kws


def post_filter_keywords(
    kws: List[str],
    *,
    k: int = 5,
) -> List[str]:
    seen = set()
    out = []
    for w in kws:
        w = norm_kw(w)
        if is_bad_kw(w):
            continue
        wl = w.lower()
        if wl in seen:
            continue
        seen.add(wl)
        out.append(w)
        if len(out) >= k:
            break
    # 兜底：不够 5 个就补空（或者你也可以补频次词）
    while len(out) < k:
        out.append("UNKNOWN")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True, help="cleaned passage-level corpus, e.g. cleaned_dataset/fiqa/fiqa.jsonl")
    ap.add_argument("--gemma_path", default="./gemma-2b", help="local path to Gemma 2 2B, e.g. ./gemma-2b")
    ap.add_argument("--out_doc_ctx", required=True, help="output jsonl: one line per parent with doc_keywords")
    ap.add_argument("--max_chars_per_doc", type=int, default=6000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--limit_docs", type=int, default=0, help="0 = no limit; >0 = process only first N docs (debug)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.gemma_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.gemma_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else None,
        device_map="auto" if device == "cuda" else None,
    ).eval()

    # Build docs (parent -> doc_text)
    doc_texts = build_doc_texts_from_passages(args.in_jsonl, max_chars_per_doc=args.max_chars_per_doc)
    parents = sorted(doc_texts.keys())
    if args.limit_docs and args.limit_docs > 0:
        parents = parents[:args.limit_docs]

    prompts = [build_prompt(doc_texts[p]) for p in parents]
    raw_kws = generate_keywords_batch(
        model, tokenizer, prompts,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )

    out_path = Path(args.out_doc_ctx)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for parent, kws in zip(parents, raw_kws):
            kws = post_filter_keywords(kws, k=5)
            f.write(json.dumps({
                "domain": "fiqa",
                "parent_id": parent,
                "doc_keywords": kws,
            }, ensure_ascii=False) + "\n")

    print(f"[OK] wrote {len(parents)} docs -> {out_path}")

if __name__ == "__main__":
    main()
