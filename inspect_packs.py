#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, gzip, random
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from transformers import AutoTokenizer

# ---- copy your helpers (parse/build packs) ----
import re
_NUM_RE = re.compile(r"\d+")

def _parse_docid_start(passage_id: str):
    if "_" not in passage_id:
        return None
    docid, rest = passage_id.split("_", 1)
    nums = _NUM_RE.findall(rest)
    if not nums:
        return None
    return docid, int(nums[0])

def open_maybe_gz(path: str, mode="rt"):
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8", errors="ignore")
    return open(path, mode, encoding="utf-8", errors="ignore")

def load_corpus(corpus_path: str) -> Dict[str, Dict[str, str]]:
    corpus = {}
    with open_maybe_gz(corpus_path, "rt") as f:
        for line in f:
            o = json.loads(line)
            did = str(o.get("document_id") or o.get("_id") or o.get("id") or "")
            if not did:
                continue
            corpus[did] = {"title": o.get("title",""), "text": o.get("text","")}
    return corpus

def build_packs_from_passage_corpus(
    corpus: Dict[str, Dict[str, str]],
    tok,
    *,
    max_tokens: int = 512,
    overlap_tokens: int = 0,
    sep: str = "\n\n",
):
    passage_text = {}
    items = []
    for pid, o in corpus.items():
        full = ((o.get("title", "") + " " + o.get("text", "")).strip())
        passage_text[pid] = full
        key = _parse_docid_start(pid)
        if key is None:
            docid, start = ("__UNKNOWN__", 0)
        else:
            docid, start = key
        tlen = len(tok(full, add_special_tokens=False)["input_ids"])
        items.append((docid, start, pid, tlen))

    buckets = defaultdict(list)
    for docid, start, pid, tlen in items:
        buckets[docid].append((start, pid, tlen))
    for docid in buckets:
        buckets[docid].sort(key=lambda x: x[0])

    pack_ids, pack_texts = [], []
    pack2child = {}

    def _make_pack_id(docid: str, k: int) -> str:
        return f"{docid}__pack{k:05d}"

    for docid, arr in buckets.items():
        i, pack_k, n = 0, 0, len(arr)
        while i < n:
            cur_toks, child, parts = 0, [], []
            j = i
            while j < n:
                _, pid, tlen = arr[j]
                add = tlen + (1 if parts else 0)
                if child and (cur_toks + add) > max_tokens:
                    break
                child.append(pid)
                parts.append(passage_text[pid])
                cur_toks += add
                j += 1

            pack_id = _make_pack_id(docid, pack_k); pack_k += 1
            pack_text = sep.join(parts)
            pack_ids.append(pack_id)
            pack_texts.append(pack_text)
            pack2child[pack_id] = child

            if j >= n:
                break
            if overlap_tokens <= 0:
                i = j
            else:
                keep, k = 0, j - 1
                while k >= i and keep < overlap_tokens:
                    keep += arr[k][2]
                    k -= 1
                i = max(k + 1, i)
                if i >= j:
                    i = j

    return pack_ids, pack_texts, pack2child, passage_text

# ---- inspection ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="e.g. human/retrieval_tasks_derived/clapnq.cleaned.jsonl")
    ap.add_argument("--tokenizer", required=True, help="HF tokenizer path/id (use same as embedder)")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--overlap_tokens", type=int, default=0)
    ap.add_argument("--mode", choices=["random", "passage", "pack"], default="random")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--passage_id", default=None)
    ap.add_argument("--pack_id", default=None)
    ap.add_argument("--snip", type=int, default=800)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    corpus = load_corpus(args.corpus)

    pack_ids, pack_texts, pack2child, passage_text = build_packs_from_passage_corpus(
        corpus, tok, max_tokens=args.max_tokens, overlap_tokens=args.overlap_tokens
    )
    packid2text = {pid: t for pid, t in zip(pack_ids, pack_texts)}

    # build reverse index: passage -> pack
    child2packs = defaultdict(list)
    for pid, childs in pack2child.items():
        for c in childs:
            child2packs[c].append(pid)

    def show_pack(pid: str):
        childs = pack2child.get(pid, [])
        text = packid2text.get(pid, "")
        print("="*100)
        print(f"PACK: {pid}")
        print(f"num_children={len(childs)}")
        print("children:")
        for c in childs:
            tl = len(tok(passage_text[c], add_special_tokens=False)["input_ids"])
            print(f"  - {c}  (tok={tl})")
        print("-"*100)
        print(text[:args.snip])
        if len(text) > args.snip:
            print("...")

    if args.mode == "random":
        for pid in random.sample(pack_ids, k=min(args.n, len(pack_ids))):
            show_pack(pid)

    elif args.mode == "pack":
        if not args.pack_id:
            raise SystemExit("--pack_id required for mode=pack")
        show_pack(args.pack_id)

    elif args.mode == "passage":
        if not args.passage_id:
            raise SystemExit("--passage_id required for mode=passage")
        packs = child2packs.get(args.passage_id, [])
        print(f"passage {args.passage_id} appears in {len(packs)} packs:")
        for pid in packs[:args.n]:
            show_pack(pid)

if __name__ == "__main__":
    main()
