#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
contextualize_passage_level.py

Step 2: Build doc-level context (keywords, optional extractive summary) from
passage_level corpus, then attach to each passage as ctx_text.

Designed for FIQA passage ids like: <docid>-<start>-<end>
Example: 10171-0-2129  -> parent_id=10171

Input:  cleaned_dataset/<domain>/<domain>.jsonl  (or .gz)
Output:
  1) cleaned_dataset/<domain>/<domain>.ctx.jsonl        (or .gz)
  2) cache/doc_ctx/<domain>.doc_ctx.jsonl               (or .gz)
  3) stats json

This DOES NOT change passage boundaries. It only adds context fields.

No external deps by default. If you want TF-IDF weighting, install scikit-learn
and pass --keyword_method tfidf.
"""

import argparse
import json
import gzip
import math
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple, Optional

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

# Minimal English stopwords (small but effective). Extend if needed.
STOPWORDS = set("""
a an the and or but if then else when while for from to of in on at by with without
is are was were be been being do does did doing have has had having
this that these those it its it's i you he she we they them their our your my
as not no yes can could may might will would should must
about into over under above below up down out off
""".split())


def open_text(path: str):
    p = Path(path)
    if p.suffix == ".gz":
        return gzip.open(p, "rt", encoding="utf-8")
    return open(p, "rt", encoding="utf-8")


def write_text(path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == ".gz":
        return gzip.open(p, "wt", encoding="utf-8")
    return open(p, "wt", encoding="utf-8")


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")

def tokenize(text: str, *, keep_case: bool = True) -> List[str]:
    toks = TOKEN_RE.findall(text or "")
    if not keep_case:
        toks = [t.lower() for t in toks]
    # stopwords 用 lower 去对齐
    out = []
    for t in toks:
        tl = t.lower()
        if tl in STOPWORDS:
            continue
        # 过滤纯数字
        if tl.isdigit():
            continue
        out.append(t if keep_case else tl)
    return out


from collections import Counter

def compute_df(docs_tokens: Dict[str, List[str]]) -> Counter:
    df = Counter()
    for toks in docs_tokens.values():
        df.update(set(t.lower() for t in toks))
    return df


import re

_HAS_ALPHA = re.compile(r"[A-Za-z]")
_HAS_DIGIT = re.compile(r"\d")

def terminess(token: str) -> float:
    t = token
    tl = t.lower()

    score = 0.0
    # 长度：3~20 范围内越长越像术语（但太长也不一定）
    L = len(tl)
    score += min(2.0, (L - 3) * 0.15)  # L=3 ->0, L=10 ->~1.05, cap 2.0

    # 含数字/连字符/字母数字混合
    has_a = bool(_HAS_ALPHA.search(t))
    has_d = bool(_HAS_DIGIT.search(t))
    if has_d:
        score += 0.6
    if "-" in t:
        score += 0.5
    if has_a and has_d:
        score += 0.6

    # 大写信息（如果你保留原case）
    if any(ch.isupper() for ch in t):
        score += 0.4

    return score


import math
from typing import List, Dict

def select_doc_keywords(
    doc_tokens: List[str],
    df: Counter,
    N_docs: int,
    *,
    topk: int = 12,
    min_len: int = 3,
    min_tf: int = 2,          # 你说“可选 >=2”，默认开启；想关就传 1
    df_max_ratio: float = 0.02,  # 过滤“过高频词”：出现覆盖率 >2% 的词丢掉（可调 0.01~0.05）
) -> List[str]:
    # doc 内 tf（lower 聚合，返回时我们用 lower 版本即可）
    tf = Counter(t.lower() for t in doc_tokens)

    # 过滤 + 打分
    cand = []
    for w, f in tf.items():
        if len(w) < min_len:
            continue
        if f < min_tf:
            continue

        # DF 过滤
        dfi = df.get(w, 0)
        if N_docs > 0 and (dfi / N_docs) > df_max_ratio:
            continue

        # idf 平滑
        idf = math.log((N_docs + 1) / (dfi + 1)) + 1.0
        base = f * idf

        # terminess：用原 token 的形态信息会更好，但我们这里 lower 了
        # 简化：用 w 本身
        tscore = terminess(w)

        score = base + 0.7 * tscore
        cand.append((score, w))

    cand.sort(reverse=True, key=lambda x: x[0])
    out = [w for _, w in cand[:topk]]

    # 如果过滤太狠导致太少：放宽 min_tf 回填
    if len(out) < min(8, topk):
        # 回填：允许 min_tf=1，但仍保留 df 过滤 + min_len
        cand2 = []
        for w, f in tf.items():
            if len(w) < min_len:
                continue
            dfi = df.get(w, 0)
            if N_docs > 0 and (dfi / N_docs) > df_max_ratio:
                continue
            idf = math.log((N_docs + 1) / (dfi + 1)) + 1.0
            score = f * idf + 0.7 * terminess(w)
            cand2.append((score, w))
        cand2.sort(reverse=True, key=lambda x: x[0])
        out = []
        seen = set()
        for _, w in cand2:
            if w in seen:
                continue
            seen.add(w)
            out.append(w)
            if len(out) >= topk:
                break

    return out


def parse_fiqa_parent_start_end(pid: str) -> Tuple[str, int, int]:
    # Expect: <docid>-<start>-<end>
    parts = pid.split("-")
    if len(parts) < 3 or not parts[-1].isdigit() or not parts[-2].isdigit():
        # fallback
        return pid, -1, -1
    parent = parts[0]
    start = int(parts[-2])
    end = int(parts[-1])
    return parent, start, end


def split_sentences(text: str, max_sents: int = 30) -> List[str]:
    sents = [s.strip() for s in SENT_SPLIT_RE.split(text or "") if s.strip()]
    if len(sents) > max_sents:
        sents = sents[:max_sents]
    return sents


def pick_gist_sentence(passage_text: str, doc_keywords: List[str], max_chars: int = 200) -> str:
    """
    Extractive gist:
      - score each sentence by keyword hits + token length (light)
      - choose best, truncate
    """
    sents = split_sentences(passage_text, max_sents=25)
    if not sents:
        return (passage_text or "").strip()[:max_chars]

    kw_set = set(doc_keywords)
    best = sents[0]
    best_score = -1.0
    for s in sents[:25]:
        toks = tokenize(s)
        if not toks:
            continue
        hit = sum(1 for t in toks if t in kw_set)
        # prefer informative sentences but avoid super long
        score = hit * 3.0 + math.log(1 + len(toks))
        if score > best_score:
            best_score = score
            best = s
    best = best.strip()
    if len(best) > max_chars:
        best = best[:max_chars].rsplit(" ", 1)[0]
    return best


def keywords_freq(doc_text: str, topk: int) -> List[str]:
    c = Counter(tokenize(doc_text))
    return [w for w, _ in c.most_common(topk)]


def keywords_tfidf(docs_tokens: Dict[str, List[str]], topk: int) -> Dict[str, List[str]]:
    """
    Tiny TF-IDF without sklearn:
      - docs_tokens: parent_id -> tokens list
    Returns parent_id -> topk keywords by tfidf
    """
    df = Counter()
    for toks in docs_tokens.values():
        df.update(set(toks))

    N = max(1, len(docs_tokens))
    out = {}
    for pid, toks in docs_tokens.items():
        tf = Counter(toks)
        scores = {}
        for w, f in tf.items():
            # smooth idf
            idf = math.log((N + 1) / (df[w] + 1)) + 1.0
            scores[w] = f * idf
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]
        out[pid] = [w for w, _ in top]
    return out


def build_prefix(parent_id: str,
                 doc_keywords: List[str],
                 passage_gist: str,
                 max_keywords: int,
                 max_prefix_tokens: int) -> str:
    kws = doc_keywords[:max_keywords]
    kw_str = "; ".join(kws)

    prefix = []
    prefix.append(f"[DOC] {parent_id}")
    if kw_str:
        prefix.append(f"[DOC_KW] {kw_str}")
    if passage_gist:
        prefix.append(f"[GIST] {passage_gist}")

    # crude token budget control for prefix
    joined = "\n".join(prefix)
    toks = joined.split()
    if len(toks) > max_prefix_tokens:
        # drop gist first, then trim keywords
        if passage_gist:
            prefix = [p for p in prefix if not p.startswith("[GIST]")]
        joined = "\n".join(prefix)
        toks = joined.split()
        if len(toks) > max_prefix_tokens and kws:
            # trim keywords
            keep = max(3, int(len(kws) * max_prefix_tokens / max(1, len(toks))))
            kw_str = "; ".join(kws[:keep])
            prefix = [f"[DOC] {parent_id}", f"[DOC_KW] {kw_str}"]
    return "\n".join(prefix)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, help="fiqa (this script assumes FIQA-style ids).")
    ap.add_argument("--in_jsonl", required=True, help="cleaned passage_level jsonl(.gz ok)")
    ap.add_argument("--out_ctx_jsonl", required=True, help="output jsonl with ctx_text(.gz ok)")
    ap.add_argument("--out_doc_ctx", required=True, help="output doc context cache jsonl(.gz ok)")
    ap.add_argument("--out_stats", required=True, help="output stats json")
    ap.add_argument("--keyword_method", choices=["freq", "tfidf"], default="tfidf")
    ap.add_argument("--doc_kw_topk", type=int, default=16, help="topK doc keywords to compute/store")
    ap.add_argument("--max_keywords_in_prefix", type=int, default=12, help="how many kws to include in ctx prefix")
    ap.add_argument("--max_prefix_tokens", type=int, default=120, help="hard cap for prefix token count (space split)")
    ap.add_argument("--replace_text", action="store_true",
                    help="If set, replace field 'text' with ctx_text. Default: keep original text, add ctx_text.")
    args = ap.parse_args()

    domain = args.domain.lower()
    assert domain == "fiqa", "This script currently assumes FIQA id format <docid>-<start>-<end>."

    # Pass 1: read all passages, group by parent, keep objects in memory
    parents_passages: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    parents_doc_text: Dict[str, List[str]] = defaultdict(list)

    total = 0
    with open_text(args.in_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pid = obj.get("id") or obj.get("_id")
            parent, start, end = parse_fiqa_parent_start_end(pid)

            text = obj.get("text", "") or ""
            total += 1
            # store minimal for later write
            obj["_parent_id"] = parent
            obj["_start"] = start
            obj["_end"] = end
            parents_passages[parent].append(obj)
            parents_doc_text[parent].append(text)

    # Sort passages per parent by start/end when available
    for parent, lst in parents_passages.items():
        lst.sort(key=lambda o: (o.get("_start", -1), o.get("_end", -1)))


    doc_tokens: Dict[str, List[str]] = {}
    for parent, parts in parents_doc_text.items():
        doc_text = "\n".join(parts)
        doc_tokens[parent] = tokenize(doc_text)

     # Build doc tokens for TF-IDF
    df = compute_df(doc_tokens)
    N_docs = len(doc_tokens)

    doc_kws = {}
    for parent, toks in doc_tokens.items():
        doc_kws[parent] = select_doc_keywords(
            toks, df, N_docs,
            topk=12,        # 你要 8~12，这里给 12
            min_len=3,
            min_tf=2,       # 开启“>=2”
            df_max_ratio=0.02,
        )

    # Compute doc keywords
    # if args.keyword_method == "tfidf":
    #     doc_kws = keywords_tfidf(doc_tokens, topk=args.doc_kw_topk)
    # else:
    #     doc_kws = {}
    #     for parent, toks in doc_tokens.items():
    #         c = Counter(toks)
    #         doc_kws[parent] = [w for w, _ in c.most_common(args.doc_kw_topk)]

    # Write doc_ctx cache
    n_parents = 0
    with write_text(args.out_doc_ctx) as out:
        for parent in doc_kws:
            n_parents += 1
            out.write(json.dumps({
                "domain": domain,
                "parent_id": parent,
                "doc_keywords": doc_kws[parent],
                "n_passages": len(parents_passages[parent]),
            }, ensure_ascii=False) + "\n")

    # Write ctx corpus
    with write_text(args.out_ctx_jsonl) as out:
        for parent, plist in parents_passages.items():
            kws = doc_kws.get(parent, [])
            for obj in plist:
                orig_text = obj.get("text", "") or ""
                gist = pick_gist_sentence(orig_text, kws, max_chars=220)
                prefix = build_prefix(
                    parent_id=parent,
                    doc_keywords=kws,
                    passage_gist=gist,
                    max_keywords=args.max_keywords_in_prefix,
                    max_prefix_tokens=args.max_prefix_tokens,
                )
                ctx_text = prefix + "\n[TEXT]\n" + orig_text

                # cleanup temp fields
                obj.pop("_parent_id", None)
                obj.pop("_start", None)
                obj.pop("_end", None)

                obj["doc_parent"] = parent
                obj["doc_keywords"] = kws[:args.doc_kw_topk]
                obj["passage_gist"] = gist
                obj["ctx_text"] = ctx_text

                if args.replace_text:
                    obj["text"] = ctx_text

                out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    stats = {
        "domain": domain,
        "in_jsonl": args.in_jsonl,
        "out_ctx_jsonl": args.out_ctx_jsonl,
        "out_doc_ctx": args.out_doc_ctx,
        "keyword_method": args.keyword_method,
        "doc_kw_topk": args.doc_kw_topk,
        "max_keywords_in_prefix": args.max_keywords_in_prefix,
        "max_prefix_tokens": args.max_prefix_tokens,
        "n_passages": total,
        "n_parents": n_parents,
        "notes": "ctx_text adds [DOC]/[DOC_KW]/[GIST] prefix then [TEXT] original passage. Passage boundaries unchanged.",
    }
    Path(args.out_stats).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("[OK] passages:", total)
    print("[OK] parents :", n_parents)
    print("[OK] wrote   :", args.out_ctx_jsonl)
    print("[OK] wrote   :", args.out_doc_ctx)
    print("[OK] wrote   :", args.out_stats)


if __name__ == "__main__":
    main()
