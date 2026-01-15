#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FAST/ROBUST version:
- Stream docs from zip (no huge RAM)
- DO NOT full-doc tokenize with offsets (the main "stuck at 0%" cause)
- Build packs using token-length estimates from substrings (anchors + small gaps)
- Final STRICT 512/overlap windowing uses offsets on pack_text only (small, fast)
"""

import argparse
import json
import gzip
import zipfile
import os
import faulthandler
import signal
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from transformers import AutoTokenizer
from tqdm import tqdm


# -----------------------------
# Args
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc_zip", required=True)
    ap.add_argument("--doc_member", required=True)
    ap.add_argument("--in_chunk_gz", required=True)

    ap.add_argument("--out_chunk_gz", required=True)
    ap.add_argument("--out_map_gz", required=True)

    ap.add_argument("--hf_tokenizer", required=True)

    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--overlap_tokens", type=int, default=100)

    ap.add_argument("--max_gap_tokens", type=int, default=64,
                    help="Do not concat across gaps larger than this (token gap), unless current pack is too short.")
    ap.add_argument("--min_fill_tokens", type=int, default=384,
                    help="If current pack token_len < this, we may ignore max_gap_tokens to fill up.")

    ap.add_argument("--pad_to_max", action="store_true",
                    help="Pad each pack to ~max_tokens by expanding left/right in doc CHAR space guided by token counts.")
    ap.add_argument("--pad_strategy", type=str, default="center", choices=["center", "left", "right"])
    ap.add_argument("--pad_min_tokens", type=int, default=0,
                    help="Only pad packs whose token_len < this. 0 means always pad if < max_tokens.")

    ap.add_argument("--max_doc_chars", type=int, default=5_000_000,
                    help="Skip docs whose raw char length exceeds this.")

    # performance guards for tokenizing huge gaps / padding search
    ap.add_argument("--gap_char_cap", type=int, default=40_000,
                    help="If a char-gap between anchors is larger than this, treat as 'huge gap' and do not tokenize it.")
    ap.add_argument("--pad_max_chars", type=int, default=120_000,
                    help="Max chars to expand when padding (prevents pack_text from becoming huge).")

    ap.add_argument("--gzip_level", type=int, default=1)
    return ap.parse_args()


# -----------------------------
# IDs / loaders
# -----------------------------
def get_base_id(obj: dict) -> Optional[str]:
    for k in ("_id", "id", "document_id", "doc_id", "docid"):
        if k not in obj:
            continue
        v = obj.get(k)
        if v is None:
            continue
        if isinstance(v, str):
            v2 = v.strip()
            if v2:
                return v2
            continue
        try:
            return str(v)
        except Exception:
            continue
    return None


def load_chunk_spans(in_chunk_gz: str) -> Dict[str, List[Tuple[int, int]]]:
    """
    returns: parent_id -> list[(start,end)]  (char offsets in doc)
    """
    by_parent: Dict[str, List[Tuple[int, int]]] = {}
    with gzip.open(in_chunk_gz, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc="load-chunks"):
            o = json.loads(line)
            pid = str(o["parent_id"])
            s = int(o["start"])
            e = int(o["end"])
            if e > s:
                by_parent.setdefault(pid, []).append((s, e))
    return by_parent


def union_intervals(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: (x[0], x[1]))
    out = []
    cs, ce = spans[0]
    for s, e in spans[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            out.append((cs, ce))
            cs, ce = s, e
    out.append((cs, ce))
    return out


# -----------------------------
# STRICT token windows (final guarantee: each chunk <= max_tokens)
# Requires FAST tokenizer offsets mapping.
# -----------------------------
def safe_tokenize_with_offsets(tokenizer, text: str):
    out = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return out["input_ids"], out["offset_mapping"]


def token_windows_char_offsets(tokenizer, text: str, max_tokens: int, overlap_tokens: int) -> List[Tuple[int, int]]:
    ids, offsets = safe_tokenize_with_offsets(tokenizer, text)
    n = len(ids)
    if n == 0:
        return []

    step = max(1, max_tokens - overlap_tokens)
    starts = [s for (s, e) in offsets]
    ends = [e for (s, e) in offsets]

    spans: List[Tuple[int, int]] = []
    for st in range(0, n, step):
        ed = min(st + max_tokens, n)

        s_idx = st
        while s_idx < ed and ends[s_idx] <= starts[s_idx]:
            s_idx += 1
        e_idx = ed - 1
        while e_idx >= st and ends[e_idx] <= starts[e_idx]:
            e_idx -= 1

        if s_idx >= ed or e_idx < st:
            if ed >= n:
                break
            continue

        c0 = starts[s_idx]
        c1 = ends[e_idx]
        if c1 <= c0:
            c1 = max(c0 + 1, c1)

        spans.append((c0, c1))
        if ed >= n:
            break

    return spans


# -----------------------------
# Per-doc token length helper (substring only, no offsets)
# -----------------------------
class DocTok:
    def __init__(self, tokenizer, text: str):
        self.tok = tokenizer
        self.text = text
        self.cache: Dict[Tuple[int, int], int] = {}

    def toklen(self, s: int, e: int) -> int:
        if e <= s:
            return 0
        key = (s, e)
        if key in self.cache:
            return self.cache[key]
        # encode substring only
        L = len(self.tok.encode(self.text[s:e], add_special_tokens=False))
        self.cache[key] = L
        return L


def rightmost_end_len_leq(dt: DocTok, s: int, e0: int, e1: int, limit: int) -> int:
    """
    Find maximum end in [e0, e1] such that toklen(s, end) <= limit.
    Assumes toklen mostly non-decreasing with end (good enough here).
    """
    e0 = max(e0, s + 1)
    e1 = max(e1, e0)
    if dt.toklen(s, e0) > limit:
        return e0
    if dt.toklen(s, e1) <= limit:
        return e1

    lo, hi = e0, e1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if dt.toklen(s, mid) <= limit:
            lo = mid
        else:
            hi = mid - 1
    return lo


def leftmost_start_len_leq(dt: DocTok, s0: int, s1: int, e: int, limit: int) -> int:
    """
    Find minimum start in [s0, s1] such that toklen(start, e) <= limit.
    Moving start left increases toklen; we want 'furthest left' without exceeding.
    """
    s0 = max(0, s0)
    s1 = max(0, s1)
    if s0 > s1:
        s0, s1 = s1, s0

    # if even the furthest-left is OK, take it
    if dt.toklen(s0, e) <= limit:
        return s0
    # if current start already exceeds, cannot expand
    if dt.toklen(s1, e) > limit:
        return s1

    lo, hi = s0, s1
    while lo < hi:
        mid = (lo + hi) // 2
        if dt.toklen(mid, e) <= limit:
            hi = mid
        else:
            lo = mid + 1
    return lo


def pad_span_to_max(
    dt: DocTok,
    s: int,
    e: int,
    *,
    target_tokens: int,
    strategy: str,
    pad_max_chars: int,
) -> Tuple[int, int]:
    cur = dt.toklen(s, e)
    if cur >= target_tokens:
        return s, e

    need = target_tokens - cur
    if strategy == "left":
        add_l, add_r = need, 0
    elif strategy == "right":
        add_l, add_r = 0, need
    else:
        add_l = need // 2
        add_r = need - add_l

    text_len = len(dt.text)

    # bound expansions by chars so pack_text never becomes gigantic
    s0 = max(0, s - pad_max_chars)
    e1 = min(text_len, e + pad_max_chars)

    # expand left but keep <= (target - add_r) so we leave "budget" for right
    left_limit = max(cur, target_tokens - add_r)
    new_s = leftmost_start_len_leq(dt, s0, s, e, left_limit)

    # then expand right up to target_tokens
    new_e = rightmost_end_len_leq(dt, new_s, e, e1, target_tokens)

    return new_s, new_e


# -----------------------------
# Budget packs from anchor spans (char) with token-length estimates
# -----------------------------
def build_budget_packs_for_doc_fast(
    *,
    spans_char: List[Tuple[int, int]],
    doc_text: str,
    tokenizer,
    max_tokens: int,
    overlap_tokens: int,
    max_gap_tokens: int,
    min_fill_tokens: int,
    pad_to_max: bool,
    pad_strategy: str,
    pad_min_tokens: int,
    gap_char_cap: int,
    pad_max_chars: int,
) -> List[Tuple[int, int]]:
    if not spans_char:
        return []

    spans_char = union_intervals(spans_char)
    dt = DocTok(tokenizer, doc_text)

    # anchors: (s,e,toklen)
    anchors: List[Tuple[int, int, int]] = []
    for s, e in spans_char:
        if e > s:
            anchors.append((s, e, dt.toklen(s, e)))
    if not anchors:
        return []

    packs: List[Tuple[int, int]] = []
    i = 0
    while i < len(anchors):
        pack_s, pack_e, pack_tok = anchors[i]
        j = i + 1

        while j < len(anchors):
            ns, ne, ntok = anchors[j]
            gap_chars = max(0, ns - pack_e)

            # huge gap -> treat as too large, don't tokenize it
            if gap_chars > gap_char_cap:
                gap_tok = max_gap_tokens + 1
            else:
                gap_tok = dt.toklen(pack_e, ns)

            # stop at huge gap if already sufficiently filled
            if gap_tok > max_gap_tokens and pack_tok >= min_fill_tokens:
                break

            cand_tok = pack_tok + gap_tok + ntok
            if cand_tok > max_tokens:
                break

            pack_e = max(pack_e, ne)
            pack_tok = cand_tok
            j += 1

        # optional pad to ~max_tokens (still keeps pack toklen <= max_tokens)
        if pad_to_max and (pad_min_tokens <= 0 or pack_tok < pad_min_tokens):
            pack_s2, pack_e2 = pad_span_to_max(
                dt, pack_s, pack_e,
                target_tokens=max_tokens,
                strategy=pad_strategy,
                pad_max_chars=pad_max_chars
            )
            pack_s, pack_e = pack_s2, pack_e2
            pack_tok = dt.toklen(pack_s, pack_e)

        packs.append((pack_s, pack_e))

        # approximate overlap in CHAR space to avoid "no overlap between packs"
        # (strict windowing inside packs already has overlap_tokens)
        if overlap_tokens > 0 and pack_tok > 0 and j < len(anchors):
            avg_cpt = max(1.0, (pack_e - pack_s) / max(1, pack_tok))
            tail_chars = int(overlap_tokens * avg_cpt)
            ov_char = max(pack_s + 1, pack_e - max(1, tail_chars))

            # advance i to first anchor whose end passes ov_char (ensures progress)
            k = i + 1
            while k < len(anchors) and anchors[k][1] <= ov_char:
                k += 1
            i = k if k > i else (i + 1)
        else:
            i = j

    # de-dup / sort
    packs = sorted(set((s, e) for (s, e) in packs if e > s), key=lambda x: (x[0], x[1]))
    return packs


# -----------------------------
# Main (stream docs)
# -----------------------------
def main():
    args = parse_args()

    # debug stack dump if needed
    faulthandler.register(signal.SIGUSR1)
    # print("PID:", os.getpid())

    tokenizer = AutoTokenizer.from_pretrained(args.hf_tokenizer, trust_remote_code=True, use_fast=True)

    # offsets mapping REQUIRED for strict windowing (start/end char offsets output)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            "Tokenizer is NOT fast (tokenizer.is_fast == False). "
            "This pipeline requires offsets mapping for strict windowing + char start/end. "
            "Please use a fast tokenizer or fix the model repo to provide a fast tokenizer."
        )

    by_parent = load_chunk_spans(args.in_chunk_gz)
    needed = set(by_parent.keys())

    out_chunk = Path(args.out_chunk_gz)
    out_map = Path(args.out_map_gz)
    out_chunk.parent.mkdir(parents=True, exist_ok=True)
    out_map.parent.mkdir(parents=True, exist_ok=True)

    total_in_parents = len(needed)
    seen = set()
    total_out_chunks = 0
    total_out_packs = 0
    empty_text_docs = 0
    skipped_big_docs = 0

    with gzip.open(out_chunk, "wt", encoding="utf-8", compresslevel=args.gzip_level) as fcor, \
         gzip.open(out_map, "wt", encoding="utf-8", compresslevel=args.gzip_level) as fmap, \
         zipfile.ZipFile(args.doc_zip, "r") as zf, \
         zf.open(args.doc_member, "r") as fin:

        pbar = tqdm(total=total_in_parents, desc="budget-concat", dynamic_ncols=True)
        for raw in fin:
            obj = json.loads(raw.decode("utf-8"))
            pid = get_base_id(obj)
            if not pid or pid not in needed:
                continue

            seen.add(pid)
            pbar.update(1)

            title = obj.get("title") or ""
            text = obj.get("text") or ""
            if not text.strip():
                empty_text_docs += 1
                continue
            if len(text) > args.max_doc_chars:
                skipped_big_docs += 1
                continue

            spans = union_intervals(by_parent.get(pid, []))
            if not spans:
                continue

            try:
                packs = build_budget_packs_for_doc_fast(
                    spans_char=spans,
                    doc_text=text,
                    tokenizer=tokenizer,
                    max_tokens=args.max_tokens,
                    overlap_tokens=args.overlap_tokens,
                    max_gap_tokens=args.max_gap_tokens,
                    min_fill_tokens=args.min_fill_tokens,
                    pad_to_max=args.pad_to_max,
                    pad_strategy=args.pad_strategy,
                    pad_min_tokens=args.pad_min_tokens,
                    gap_char_cap=args.gap_char_cap,
                    pad_max_chars=args.pad_max_chars,
                )
            except Exception:
                skipped_big_docs += 1
                continue

            if not packs:
                continue
            total_out_packs += len(packs)

            # STRICT 512/overlap windowing for final chunks (offsets on pack_text only)
            for pack_idx, (span_start, span_end) in enumerate(packs):
                if span_end <= span_start:
                    continue
                pack_text = text[span_start:span_end]
                if not pack_text.strip():
                    continue

                wins = token_windows_char_offsets(
                    tokenizer,
                    pack_text,
                    max_tokens=args.max_tokens,
                    overlap_tokens=args.overlap_tokens,
                )
                if not wins:
                    continue

                for win_idx, (w0, w1) in enumerate(wins):
                    ca = span_start + w0
                    cb = span_start + w1
                    if cb <= ca:
                        continue
                    chunk_text = text[ca:cb]
                    if not chunk_text.strip():
                        continue

                    cid = f"{pid}__c{pack_idx:06d}__w{win_idx:03d}"
                    rec = {
                        "id": cid,
                        "parent_id": pid,
                        "start": int(ca),
                        "end": int(cb),
                        "title": title,
                        "text": chunk_text,
                    }
                    fcor.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fmap.write(json.dumps(
                        {"chunk_id": cid, "parent_id": pid, "start": int(ca), "end": int(cb)},
                        ensure_ascii=False
                    ) + "\n")
                    total_out_chunks += 1

        pbar.close()

    missing_docs = total_in_parents - len(seen)

    print("\n[DONE]")
    print(" parents_in_chunks :", total_in_parents)
    print(" parents_seen_docs :", len(seen))
    print(" missing_docs      :", missing_docs)
    print(" empty_text_docs   :", empty_text_docs)
    print(" skipped_big_docs  :", skipped_big_docs)
    print(" out_packs         :", total_out_packs)
    print(" out_chunks        :", total_out_chunks)
    print(" wrote:", args.out_chunk_gz)
    print(" wrote:", args.out_map_gz)


if __name__ == "__main__":
    main()
