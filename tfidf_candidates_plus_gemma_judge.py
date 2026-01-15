#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tfidf_candidates_plus_gemma_judge.py

方案 1：TF-IDF 产候选 + Gemma 2 2B 只从候选里选 5 个（强烈推荐）

Input:  cleaned passage-level corpus jsonl (FIQA/GOVT/CLOUD/CLAPNQ 都可；按 id 规则分 parent)
Output: doc_ctx jsonl: one line per parent:
  {"domain": "...", "parent_id": "...", "doc_keywords": ["..."]*5, "n_passages": N, "tfidf_candidates": [...]}

Key features:
- TF-IDF 先给每个 doc 生成 topN 候选（默认 30）
- Gemma 作为“裁判”，只能从候选中挑 EXACT 5 个（JSON-only）
- 严格校验：不在 candidates 里的词直接丢弃；不足 5 个按 candidates 顺序回填
- 生成进度条（tqdm）
- 默认确定性生成（do_sample=False, temperature=0）
"""

import argparse, json, re, math
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# -------------------------
# Stopwords / tokenization
# -------------------------
STOPWORDS = set("""
a an the and or but if then else when while for from to of in on at by with without
is are was were be been being do does did doing have has had having
this that these those it its it's i you he she we they them their our your my
as not no yes can could may might will would should must
about into over under above below up down out off
""".split())

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")

def tokenize(text: str) -> List[str]:
    toks = TOKEN_RE.findall((text or "").lower())
    out = []
    for t in toks:
        if t in STOPWORDS:
            continue
        if t.isdigit():
            continue
        if len(t) < 3:
            continue
        out.append(t)
    return out


# -------------------------
# ID parsing (parent + start)
# Reuse your eval heuristics:
# - clapnq: 822086267_22716-22948-0-232
# - fiqa/govt/cloud: 10171-0-2129 / ibmcld_00422-0-387 / <hash>-2-2092
# -------------------------
_RE_CLAP4 = re.compile(r"^(?P<doc>.+)_(?P<s>\d+)-(?P<e>\d+)-(?P<a>\d+)-(?P<b>\d+)$")
_RE_2NUM  = re.compile(r"^(?P<doc>.+)-(?P<s>\d+)-(?P<b>\d+)$")

def parse_parent_start(passage_id: str) -> Tuple[str, int]:
    m = _RE_CLAP4.match(passage_id)
    if m:
        return m.group("doc") + "_" + m.group("s") + "-" + m.group("e"), int(m.group("s"))
    m = _RE_2NUM.match(passage_id)
    if m:
        return m.group("doc"), int(m.group("s"))
    # fallback
    return passage_id, 0


# -------------------------
# Build doc texts (concat passages per parent)
# -------------------------
def build_docs_from_passages(
    in_jsonl: str,
    *,
    max_chars_per_doc: int = 6000,
) -> Tuple[Dict[str, str], Dict[str, int]]:
    """
    Returns:
      doc_texts[parent] = concatenated (title+text) ordered by start
      n_passages[parent] = count
    """
    buckets = defaultdict(list)   # parent -> list[(start, text)]
    counts  = Counter()

    with open(in_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            pid = o.get("id") or o.get("_id") or o.get("document_id")
            if not pid:
                continue
            parent, start = parse_parent_start(pid)
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
# Tiny TF-IDF candidates
# -------------------------
def compute_df(docs_tokens: Dict[str, List[str]]) -> Counter:
    df = Counter()
    for toks in docs_tokens.values():
        df.update(set(toks))
    return df

def tfidf_topn_for_doc(toks: List[str], df: Counter, N_docs: int, topn: int) -> List[str]:
    tf = Counter(toks)
    scored = []
    for w, f in tf.items():
        dfi = df.get(w, 0)
        idf = math.log((N_docs + 1) / (dfi + 1)) + 1.0
        scored.append((f * idf, w))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [w for _, w in scored[:topn]]


# -------------------------
# Gemma prompt: choose ONLY from candidates
# -------------------------
def build_prompt(doc_snippet: str, candidates: List[str]) -> str:
    cand_str = ", ".join(candidates)
    return (
        "You are an information retrieval expert.\n"
        "Given a document snippet and candidate keywords produced by TF-IDF, select EXACTLY 5 keywords about this document snippet.\n"
        "HARD RULES:\n"
        "- You MUST choose ONLY from the provided candidate list.\n"
        "- Prefer nouns/terms/entities. Avoid style words and adverbs (e.g., words ending with 'ly').\n"
        "- Output JSON ONLY with schema: {\"keywords\": [\"k1\",\"k2\",\"k3\",\"k4\",\"k5\"]}\n\n"
        "DOCUMENT SNIPPET:\n"
        f"{doc_snippet}\n\n"
        "CANDIDATES:\n"
        f"{cand_str}\n\n"
        "JSON:\n"
    )


_JSON_OBJ_RE = re.compile(r"\{.*?\}", re.DOTALL)

def extract_json_obj(s: str) -> Optional[dict]:
    if not s:
        return None
    # 取最后一个 {...} 更像最终答案
    matches = list(_JSON_OBJ_RE.finditer(s))
    if not matches:
        return None
    chunk = matches[-1].group(0)
    try:
        return json.loads(chunk)
    except Exception:
        return None


def norm_kw(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def choose_5_with_fallback(llm_kws: List[str], candidates: List[str]) -> List[str]:
    cand_norm = [norm_kw(c) for c in candidates]
    cand_set = set(cand_norm)

    out = []
    seen = set()

    # 1) keep only kws that are in candidates
    for k in llm_kws:
        kn = norm_kw(k)
        if not kn:
            continue
        if kn not in cand_set:
            continue
        if kn in seen:
            continue
        seen.add(kn)
        out.append(candidates[cand_norm.index(kn)])  # keep original candidate form
        if len(out) == 5:
            return out

    # 2) fallback fill from candidates in order
    for c in candidates:
        cn = norm_kw(c)
        if cn in seen:
            continue
        seen.add(cn)
        out.append(c)
        if len(out) == 5:
            break

    # 3) last resort
    while len(out) < 5:
        out.append("UNKNOWN")
    return out[:5]


@torch.no_grad()
def gemma_judge_batch(
    model,
    tokenizer,
    prompts: List[str],
    *,
    max_new_tokens: int = 96,
    batch_size: int = 8,
    desc: str = "Gemma judge",
) -> List[List[str]]:
    outs: List[List[str]] = []
    for i in tqdm(range(0, len(prompts), batch_size), desc=desc):
        batch = prompts[i:i+batch_size]
        tok = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        tok = {k: v.to(model.device) for k, v in tok.items()}

        gen = model.generate(
            **tok,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # ✅ 强烈建议：确定性
            temperature=0.0,
            top_p=1.0,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
        )
        texts = tokenizer.batch_decode(gen, skip_special_tokens=True)

        for full in texts:
            obj = extract_json_obj(full)
            kws = []
            if obj and isinstance(obj.get("keywords"), list):
                kws = [str(x) for x in obj["keywords"] if isinstance(x, str)]
            outs.append(kws)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="fiqa", help="Just stored in output. (fiqa/clapnq/govt/cloud)")
    ap.add_argument("--in_jsonl", required=True, help="cleaned passage-level corpus jsonl")
    ap.add_argument("--gemma_path", default="./gemma-2b", help="local path, e.g. ./gemma-2b")
    ap.add_argument("--out_doc_ctx", required=True, help="output jsonl: one line per parent")

    ap.add_argument("--max_chars_per_doc", type=int, default=6000)
    ap.add_argument("--snippet_chars", type=int, default=2500, help="how many chars to show Gemma (snippet from doc start)")
    ap.add_argument("--cand_topn", type=int, default=30, help="TF-IDF candidate count per doc")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--limit_docs", type=int, default=0, help="debug limit; 0=no limit")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Gemma
    tokenizer = AutoTokenizer.from_pretrained(args.gemma_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.gemma_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if (device == "cuda") else None,
        device_map="auto" if (device == "cuda") else None,
    ).eval()

    # Build docs
    doc_texts, n_passages = build_docs_from_passages(
        args.in_jsonl, max_chars_per_doc=args.max_chars_per_doc
    )
    parents = sorted(doc_texts.keys())
    if args.limit_docs and args.limit_docs > 0:
        parents = parents[:args.limit_docs]

    # Tokenize docs for DF/TF-IDF
    docs_tokens = {}
    for p in tqdm(parents, desc="Tokenize docs"):
        docs_tokens[p] = tokenize(doc_texts[p])

    df = compute_df(docs_tokens)
    N_docs = len(parents)

    # Build candidates and prompts
    candidates_map: Dict[str, List[str]] = {}
    prompts: List[str] = []
    for p in tqdm(parents, desc="Build prompts"):
        cands = tfidf_topn_for_doc(docs_tokens[p], df, N_docs, topn=args.cand_topn)
        if not cands:
            cands = ["unknown"] * min(args.cand_topn, 5)
        candidates_map[p] = cands
        snippet = doc_texts[p][: args.snippet_chars]
        prompts.append(build_prompt(snippet, cands))

    # Gemma judge
    llm_kws = gemma_judge_batch(
        model, tokenizer, prompts,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        desc="Gemma judge (choose 5)",
    )

    # Write output
    out_path = Path(args.out_doc_ctx)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for p, kws_raw in zip(parents, llm_kws):
            cands = candidates_map[p]
            final5 = choose_5_with_fallback(kws_raw, cands)
            obj = {
                "domain": args.domain,
                "parent_id": p,
                "doc_keywords": final5,
                "n_passages": int(n_passages.get(p, 0)),
                "tfidf_candidates": cands,  # 方便你 debug/审计
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[OK] wrote {len(parents)} docs -> {out_path}")


if __name__ == "__main__":
    main()
