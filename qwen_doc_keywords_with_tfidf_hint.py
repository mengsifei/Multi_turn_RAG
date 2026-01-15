#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_doc_keywords_with_tfidf_hint.py (FIXED)

Generate retrieval-friendly doc keywords using Qwen/Qwen3-4B-Instruct-2507:
- Build doc text by concatenating passage-level entries per parent_id.
- Compute TF-IDF top-N as "hints" (NOT a hard constraint).
- Ask Qwen to produce:
    * 3 bullet-point summary (high information density)
    * 5 retrieval keywords/keyphrases (1-4 words, query-like)
  Output strict JSON only.

Output JSONL (one per parent):
{
  "domain": "...",
  "parent_id": "...",
  "n_passages": ...,
  "bullets": [...],
  "doc_keywords": [...],
  "tfidf_hints": [...]
}
"""

import argparse, json, re, math
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# -------------------------
# Tokenization / filters
# -------------------------
STOPWORDS = set("""
a an the and or but if then else when while for from to of in on at by with without
is are was were be been being do does did doing have has had having
this that these those it its it's i you he she we they them their our your my
as not no yes can could may might will would should must
about into over under above below up down out off
""".split())

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")

BAD_EXACT = set([
    "http", "https", "www", "com", "org", "net", "html", "php",
    "amp", "quot", "reddit", "autotldr"
])

def tokenize_for_tfidf(text: str) -> List[str]:
    toks = TOKEN_RE.findall((text or "").lower())
    out = []
    for t in toks:
        if len(t) < 3:
            continue
        if t in STOPWORDS:
            continue
        if t.isdigit():
            continue
        if t in BAD_EXACT:
            continue
        out.append(t)
    return out

def norm_phrase(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def looks_like_urlish(s: str) -> bool:
    sl = s.lower()
    if "http" in sl or "www" in sl:
        return True
    if any(x in sl for x in [".com", ".org", ".net", "/"]):
        return True
    return False

def post_filter_keywords(kws: List[str], k: int = 5) -> List[str]:
    out, seen = [], set()
    for w in kws:
        w = norm_phrase(w)
        if not w:
            continue
        if looks_like_urlish(w):
            continue

        wl = w.lower()
        if wl in BAD_EXACT:
            continue
        if wl.isdigit():
            continue

        toks = [t for t in tokenize_for_tfidf(w) if t not in STOPWORDS and t not in BAD_EXACT]
        if not toks:
            continue

        key = " ".join(toks)
        if key in seen:
            continue
        seen.add(key)

        # limit 1~4 words
        if len(w.split()) > 4:
            w = " ".join(w.split()[:4])

        out.append(w)
        if len(out) >= k:
            break

    while len(out) < k:
        out.append("UNKNOWN")
    return out


# -------------------------
# Parent parsing (reuse your heuristics)
# -------------------------
_RE_CLAP4 = re.compile(r"^(?P<doc>.+)_(?P<s>\d+)-(?P<e>\d+)-(?P<a>\d+)-(?P<b>\d+)$")
_RE_2NUM  = re.compile(r"^(?P<doc>.+)-(?P<s>\d+)-(?P<b>\d+)$")

def parse_parent_start(passage_id: str) -> Tuple[str, int]:
    """
    clapnq: 822086267_22716-22948-0-232 -> parent like 822086267_22716-22948 (stable within doc)
    fiqa/govt/cloud: 10171-0-2129 -> parent 10171
    """
    m = _RE_CLAP4.match(passage_id)
    if m:
        parent = m.group("doc") + "_" + m.group("s") + "-" + m.group("e")
        return parent, int(m.group("s"))
    m = _RE_2NUM.match(passage_id)
    if m:
        return m.group("doc"), int(m.group("s"))
    return passage_id, 0


# -------------------------
# Build doc texts from passage-level
# -------------------------
def build_docs_from_passages(in_jsonl: str, max_chars_per_doc: int) -> Tuple[Dict[str, str], Dict[str, int]]:
    buckets = defaultdict(list)  # parent -> [(start, text)]
    counts = Counter()

    with open(in_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            pid = o.get("id") or o.get("_id") or o.get("document_id")
            if not pid:
                continue
            parent, start = parse_parent_start(str(pid))
            txt = (o.get("title","") + "\n" + o.get("text","")).strip()
            buckets[parent].append((start, txt))
            counts[parent] += 1

    doc_texts = {}
    for parent, items in buckets.items():
        items.sort(key=lambda x: x[0])
        doc = "\n\n".join(t for _, t in items).strip()
        if len(doc) > max_chars_per_doc:
            doc = doc[:max_chars_per_doc]
        doc_texts[parent] = doc

    return doc_texts, dict(counts)


# -------------------------
# Tiny TF-IDF hints
# -------------------------
def compute_df(docs_tokens: Dict[str, List[str]]) -> Counter:
    df = Counter()
    for toks in docs_tokens.values():
        df.update(set(toks))
    return df

def tfidf_topn(toks: List[str], df: Counter, N: int, topn: int) -> List[str]:
    tf = Counter(toks)
    scored = []
    for w, f in tf.items():
        dfi = df.get(w, 0)
        idf = math.log((N + 1) / (dfi + 1)) + 1.0
        scored.append((f * idf, w))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [w for _, w in scored[:topn]]


# -------------------------
# Qwen prompt + JSON extraction (more robust)
# -------------------------
def build_prompt(doc_snippet: str, tfidf_hints: List[str]) -> str:
    hint_str = ", ".join(tfidf_hints[:30])
    return (
        "You are an expert at information retrieval.\n"
        "Your job: create a compact, retrieval-oriented representation of the document.\n\n"
        "Return STRICT JSON ONLY (no markdown, no extra text).\n"
        "Schema:\n"
        "{\n"
        '  "bullets": ["b1", "b2", "b3"],\n'
        '  "keywords": ["k1","k2","k3","k4","k5"]\n'
        "}\n\n"
        "Rules for bullets:\n"
        "- EXACTLY 3 bullets.\n"
        "- Each bullet <= 16 words.\n"
        "- Each bullet must include at least one concrete term/entity from the document.\n"
        "- NO placeholders like '...' or 'N/A'.\n\n"
        "Rules for keywords:\n"
        "- EXACTLY 5 keywords or short keyphrases.\n"
        "- 1-4 words each.\n"
        "- Should be terms a user would type to retrieve this document.\n"
        "- Prefer entities/technical terms/metrics.\n"
        "- Avoid URL fragments, markup tokens, or generic filler words.\n\n"
        "TF-IDF hints (optional; you may ignore bad ones):\n"
        f"{hint_str}\n\n"
        "DOCUMENT:\n"
        f"{doc_snippet}\n\n"
        "JSON:\n"
    )

_JSON_OBJ_RE = re.compile(r"\{.*?\}", re.DOTALL)

def extract_json_obj(s: str) -> Optional[dict]:
    if not s:
        return None
    matches = list(_JSON_OBJ_RE.finditer(s))
    if not matches:
        return None

    # from end to start: pick first JSON that contains bullets/keywords
    for m in reversed(matches):
        chunk = m.group(0)
        if ("\"keywords\"" not in chunk) and ("\"bullets\"" not in chunk):
            continue
        try:
            return json.loads(chunk)
        except Exception:
            continue
    return None


def build_chat_inputs(tokenizer, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "Output strict JSON only. No markdown. No extra text."},
            {"role": "user", "content": prompt},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def bullets_fallback_from_hints(hints: List[str]) -> List[str]:
    # 超简单兜底：保证不是 "..."
    # 用 hints 的前几个拼成“定位句”
    h = [x for x in hints if x and x not in BAD_EXACT][:12]
    if not h:
        return ["", "", ""]
    b1 = f"Topic: {', '.join(h[:4])}."
    b2 = f"Key terms: {', '.join(h[4:8])}."
    b3 = f"Related: {', '.join(h[8:12])}."
    return [b1, b2, b3]


@torch.no_grad()
def generate_batch(model, tokenizer, prompts: List[str], max_new_tokens: int, batch_size: int) -> List[dict]:
    outs = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Qwen generate"):
        batch = prompts[i:i+batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}

        in_len = enc["input_ids"].shape[1]

        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,  # ✅
        )

        # ✅ decode only generated part (avoid parsing prompt schema)
        gen_only = gen[:, in_len:]
        texts = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

        for t in texts:
            outs.append(extract_json_obj(t) or {})
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["fiqa","clapnq","govt","cloud"])
    ap.add_argument("--in_jsonl", required=True, help="cleaned passage-level corpus jsonl")
    ap.add_argument("--qwen_path", default="Qwen3-4B-Instruct-2507", help="HF id or local path")
    ap.add_argument("--out_doc_ctx", required=True, help="output jsonl: one line per parent")

    ap.add_argument("--max_chars_per_doc", type=int, default=8000)
    ap.add_argument("--snippet_chars", type=int, default=2500)
    ap.add_argument("--tfidf_topn", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=180)
    ap.add_argument("--limit_docs", type=int, default=0, help="debug limit, 0=no limit")
    ap.add_argument("--local_files_only", action="store_true")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ✅ left-padding for decoder-only
    tokenizer = AutoTokenizer.from_pretrained(
        args.qwen_path,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # ✅ double保险

    model = AutoModelForCausalLM.from_pretrained(
        args.qwen_path,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else None,
        device_map="auto" if device == "cuda" else None,
    ).eval()

    # Build docs
    doc_texts, n_passages = build_docs_from_passages(args.in_jsonl, max_chars_per_doc=args.max_chars_per_doc)
    parents = sorted(doc_texts.keys())
    if args.limit_docs and args.limit_docs > 0:
        parents = parents[:args.limit_docs]

    # TF-IDF hints
    docs_tokens = {}
    for p in tqdm(parents, desc="Tokenize for TF-IDF"):
        docs_tokens[p] = tokenize_for_tfidf(doc_texts[p])
    df = compute_df(docs_tokens)
    N = len(parents)

    prompts = []
    tfidf_map = {}
    for p in tqdm(parents, desc="Build prompts"):
        hints = tfidf_topn(docs_tokens[p], df, N, topn=args.tfidf_topn)
        tfidf_map[p] = hints
        snippet = doc_texts[p][: args.snippet_chars]
        prompts.append(build_chat_inputs(tokenizer, build_prompt(snippet, hints)))

    # Generate
    objs = generate_batch(model, tokenizer, prompts, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)

    # Write
    out_path = Path(args.out_doc_ctx)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for p, obj in zip(parents, objs):
            hints = tfidf_map.get(p, [])[:args.tfidf_topn]

            bullets = obj.get("bullets") if isinstance(obj.get("bullets"), list) else []
            bullets = [norm_phrase(x) for x in bullets if isinstance(x, str) and norm_phrase(x)]
            bullets = bullets[:3]
            while len(bullets) < 3:
                bullets.append("")

            # ✅ reject placeholder bullets
            if all((not b) or (b.strip() == "...") for b in bullets):
                bullets = bullets_fallback_from_hints(hints)

            kws_raw = obj.get("keywords") if isinstance(obj.get("keywords"), list) else []
            kws_raw = [str(x) for x in kws_raw if isinstance(x, str)]
            kws = post_filter_keywords(kws_raw, k=5)

            # ✅ fallback: if still all UNKNOWN, use TF-IDF hints
            if all(x == "UNKNOWN" for x in kws):
                kws = post_filter_keywords(hints, k=5)

            out_obj = {
                "domain": args.domain,
                "parent_id": p,
                "n_passages": int(n_passages.get(p, 0)),
                "bullets": bullets,
                "doc_keywords": kws,
                "tfidf_hints": hints,
            }
            f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    print(f"[OK] wrote {len(parents)} docs -> {out_path}")


if __name__ == "__main__":
    main()
