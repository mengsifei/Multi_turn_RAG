#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Doc-level semantic chunking (char-offset aligned)
Input : corpora/document_level/<domain>.jsonl.zip (member: <domain>.jsonl)
Output: gz jsonl chunk corpus with fields:
  - id: chunk_id
  - parent_id: base passage id (e.g., 822086267_7384-8758)
  - start, end: char offsets within parent passage text (0-based, end-exclusive)
  - title: optional
  - text: exact slice of parent passage text [start:end]
Also outputs map jsonl.gz: {chunk_id, parent_id, start, end}
"""

import argparse, json, re, zipfile, gzip
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer


# Sentence/paragraph boundary: punctuation OR newline OR semicolon/colon.
# Govt often lacks .!? so we rely on \n and ; :
_BOUNDARY_RE = re.compile(r"(?<=[\.\!\?])\s+|[\r\n]+|[;:]\s+")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_zip", required=True)
    ap.add_argument("--jsonl_name", required=True)
    ap.add_argument("--out_corpus", required=True)
    ap.add_argument("--out_map", required=True)

    ap.add_argument("--st_model", required=True)
    ap.add_argument("--hf_tokenizer", required=True)

    # final chunking target (match official)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--overlap_tokens", type=int, default=100)

    # semantic boundary threshold (cosine on adjacent "units")
    ap.add_argument("--sim_threshold", type=float, default=0.55)

    # embedding safety (prevents Govt OOM)
    ap.add_argument("--sent_max_tokens", type=int, default=128,
                    help="max tokens per semantic unit for embedding (hard split long units)")
    ap.add_argument("--sent_overlap_tokens", type=int, default=16,
                    help="overlap tokens when splitting long semantic units")

    # batching
    ap.add_argument("--embed_bs", type=int, default=64, help="mini-batch for SentenceTransformer.encode")
    ap.add_argument("--buf_units", type=int, default=4096, help="buffer units across docs before a GPU encode flush")
    ap.add_argument("--gzip_level", type=int, default=1)

    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    return ap.parse_args()


# def get_base_id(obj: dict) -> str:
#     # Prefer id/_id as-is; doc_level typically already uses base_id like docid_start-end
#     base = obj.get("id") or obj.get("_id")
#     if base is None:
#         raise ValueError("missing id/_id")

#     # If only doc id but has start/end fields, construct.
#     if isinstance(base, str) and re.match(r"^.+?_\d+-\d+$", base):
#         return base

#     for doc_key in ["doc_id", "docid", "document_id"]:
#         if doc_key in obj and "start" in obj and "end" in obj:
#             return f"{obj[doc_key]}_{int(obj['start'])}-{int(obj['end'])}"

#     # last resort: return base
#     return str(base)


def get_base_id(obj: dict, *, debug: bool = False) -> str | None:
    for key in ("_id", "id", "document_id", "doc_id", "docid"):
        if key not in obj:
            continue
        v: Any = obj.get(key)

        # explicitly reject None
        if v is None:
            continue

        # normalize strings
        if isinstance(v, str):
            v2 = v.strip()
            if v2 == "":
                continue
            return v2

        # numbers / other types: accept (including 0)
        try:
            return str(v)
        except Exception:
            # extremely rare: non-stringifiable object
            continue

    return None

def safe_tokenize_with_offsets(tokenizer, text: str):
    """
    Returns (input_ids, offsets) where offsets are char spans in 'text'.
    If tokenizer doesn't support offsets, returns (input_ids, None).
    """
    try:
        out = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids = out["input_ids"]
        offsets = out["offset_mapping"]
        return input_ids, offsets
    except Exception:
        # fall back
        input_ids = tokenizer.encode(text, add_special_tokens=False)
        return input_ids, None


def token_windows_char_offsets(tokenizer, text: str, max_tokens: int, overlap_tokens: int) -> List[Tuple[int, int]]:
    """
    Produce token windows mapped to (char_start, char_end) within 'text' (end-exclusive).
    Best-effort: uses offsets_mapping when available; else falls back to coarse char slicing.
    """
    ids, offsets = safe_tokenize_with_offsets(tokenizer, text)
    n = len(ids)
    if n == 0:
        return []

    if offsets is None:
        # coarse fallback: char slicing by ratio (rare; try to keep stable)
        # use approx chars-per-token
        approx_cpt = max(1.0, len(text) / max(1, n))
        step = max(1, max_tokens - overlap_tokens)
        spans = []
        for st in range(0, n, step):
            ed = min(st + max_tokens, n)
            c0 = int(st * approx_cpt)
            c1 = int(ed * approx_cpt)
            c0 = max(0, min(c0, len(text)))
            c1 = max(c0 + 1, min(c1, len(text)))
            spans.append((c0, c1))
            if ed >= n:
                break
        return spans

    # offsets: list[(s,e)] per token
    # normalize to ensure non-decreasing
    starts = [s for (s, e) in offsets]
    ends = [e for (s, e) in offsets]

    step = max(1, max_tokens - overlap_tokens)
    spans = []
    for st in range(0, n, step):
        ed = min(st + max_tokens, n)
        # find a non-empty token for boundaries
        s_idx = st
        while s_idx < ed and ends[s_idx] <= starts[s_idx]:
            s_idx += 1
        e_idx = ed - 1
        while e_idx >= st and ends[e_idx] <= starts[e_idx]:
            e_idx -= 1

        if s_idx >= ed or e_idx < st:
            # all empty, skip
            if ed >= n:
                break
            continue

        c0 = starts[s_idx]
        c1 = ends[e_idx]
        if c1 <= c0:
            # safety
            c0 = max(0, min(c0, len(text)))
            c1 = max(c0 + 1, min(c1, len(text)))
        spans.append((c0, c1))
        if ed >= n:
            break
    return spans


def rough_split_with_offsets(text: str) -> List[Tuple[int, int]]:
    """
    Split by boundaries into spans [start,end) in original 'text'.
    Keeps offsets aligned to original text.
    """
    spans = []
    prev = 0
    for m in _BOUNDARY_RE.finditer(text):
        cut = m.start()
        if cut > prev:
            spans.append((prev, cut))
        prev = m.end()
    if prev < len(text):
        spans.append((prev, len(text)))

    # trim whitespace-only spans
    out = []
    for a, b in spans:
        seg = text[a:b]
        # keep offsets but trim leading/trailing spaces by shifting a/b
        l = len(seg) - len(seg.lstrip())
        r = len(seg.rstrip())
        a2 = a + l
        b2 = a + r
        if b2 > a2:
            out.append((a2, b2))
    return out


def build_units(text: str, tokenizer, sent_max_tokens: int, sent_overlap_tokens: int) -> List[Tuple[int, int]]:
    """
    Return list of semantic units as char spans within text.
    Each unit is guaranteed to be <= sent_max_tokens tokens (best-effort, via token windows).
    """
    base_spans = rough_split_with_offsets(text)
    units: List[Tuple[int, int]] = []

    for a, b in base_spans:
        seg = text[a:b]
        if not seg.strip():
            continue
        ids, _ = safe_tokenize_with_offsets(tokenizer, seg)
        if len(ids) <= sent_max_tokens:
            units.append((a, b))
        else:
            # split long span into token windows, map to char offsets within seg then shift by a
            win = token_windows_char_offsets(tokenizer, seg, sent_max_tokens, sent_overlap_tokens)
            for c0, c1 in win:
                ua = a + c0
                ub = a + c1
                if ub > ua:
                    units.append((ua, ub))

    # merge duplicates / ensure increasing
    units = sorted(set(units), key=lambda x: (x[0], x[1]))
    # remove zero/overlaps that are identical
    cleaned = []
    last = (-1, -1)
    for u in units:
        if u != last and u[1] > u[0]:
            cleaned.append(u)
        last = u
    return cleaned


def cosine_adj_boundaries(emb: np.ndarray, threshold: float) -> List[int]:
    """
    emb: [n, d] float32
    returns boundary indices i where we cut before i (i in 1..n-1)
    """
    if emb.shape[0] <= 1:
        return []
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    sims = (emb[:-1] * emb[1:]).sum(axis=1)
    return [i + 1 for i, s in enumerate(sims) if s < threshold]


def segments_from_units(units: List[Tuple[int, int]], boundaries: List[int]) -> List[Tuple[int, int]]:
    idxs = [0] + boundaries + [len(units)]
    segs = []
    for a, b in zip(idxs[:-1], idxs[1:]):
        sa = units[a][0]
        sb = units[b - 1][1]
        if sb > sa:
            segs.append((sa, sb))
    return segs


def main():
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.hf_tokenizer, trust_remote_code=True)
    model = SentenceTransformer(args.st_model, trust_remote_code=True, device=device)

    out_corpus = Path(args.out_corpus)
    out_map = Path(args.out_map)
    out_corpus.parent.mkdir(parents=True, exist_ok=True)
    out_map.parent.mkdir(parents=True, exist_ok=True)

    # buffering units across docs for one big encode
    buf_unit_texts: List[str] = []
    buf_docs: List[dict] = []  # {base_id,title,text,units,unit_range:(s,e)}

    def flush(fcor, fmap):
        nonlocal buf_unit_texts, buf_docs
        if not buf_docs:
            return

        # encode all units in buffer
        emb_all = model.encode(
            buf_unit_texts,
            batch_size=args.embed_bs,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        ).astype(np.float32)

        # per-doc process
        offset = 0
        for d in buf_docs:
            units = d["units"]
            n = len(units)
            emb = emb_all[offset:offset + n]
            offset += n

            text = d["text"]
            base_id = d["base_id"]
            title = d["title"]

            # semantic segments
            bnds = cosine_adj_boundaries(emb, args.sim_threshold)
            segs = segments_from_units(units, bnds)
            if not segs:
                segs = [(0, len(text))]

            chunk_idx = 0
            for sa, sb in segs:
                seg_text = text[sa:sb]
                # final windows: 512/100 token windows mapped to char offsets in seg_text
                wins = token_windows_char_offsets(tokenizer, seg_text, args.max_tokens, args.overlap_tokens)
                if not wins:
                    continue
                for c0, c1 in wins:
                    ca = sa + c0
                    cb = sa + c1
                    if cb <= ca:
                        continue
                    chunk_text = text[ca:cb]
                    cid = f"{base_id}__c{chunk_idx:06d}"
                    rec = {
                        "id": cid,
                        "parent_id": base_id,
                        "start": int(ca),
                        "end": int(cb),
                        "title": title,
                        "text": chunk_text,
                    }
                    fcor.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fmap.write(json.dumps({"chunk_id": cid, "parent_id": base_id, "start": int(ca), "end": int(cb)},
                                          ensure_ascii=False) + "\n")
                    chunk_idx += 1

        buf_unit_texts = []
        buf_docs = []

    with zipfile.ZipFile(args.in_zip, "r") as zf, \
        zf.open(args.jsonl_name, "r") as fin, \
        gzip.open(out_corpus, "wt", encoding="utf-8", compresslevel=args.gzip_level) as fcor, \
        gzip.open(out_map, "wt", encoding="utf-8", compresslevel=args.gzip_level) as fmap:
        skipped = 0
        skipped_text = 0
        used_title_fallback = 0

        for raw in tqdm(fin, desc="doc-semchunk"):
            obj = json.loads(raw.decode("utf-8"))
            base_id = get_base_id(obj)
            text = (obj.get("text") or "")
            title = obj.get("title") or ""

            if base_id is None:
                skipped_id += 1
                if skipped_id <= 5:
                    print("[SKIP:id-missing] keys=", list(obj.keys())[:20], " sample=", {k: obj.get(k) for k in ("_id","id","document_id")})
                continue

            if text is None or (isinstance(text, str) and not text.strip()):
                # 如果 title 有内容，用 title 顶上（推荐：别丢 queryable 的 doc）
                if title:
                    text = title
                    used_title_fallback += 1
                else:
                    skipped_text += 1
                    if skipped_text <= 5:
                        tid = base_id
                        print(f"[SKIP:text-empty] id={tid} keys=", list(obj.keys())[:20])
                    continue
            text = str(text)
            units = build_units(text, tokenizer, args.sent_max_tokens, args.sent_overlap_tokens)
            if len(units) <= 1:
                # no semantic units; just do 512/100 windows on whole text
                wins = token_windows_char_offsets(tokenizer, text, args.max_tokens, args.overlap_tokens)
                chunk_idx = 0
                for a, b in wins:
                    if b <= a:
                        continue
                    cid = f"{base_id}__c{chunk_idx:06d}"
                    rec = {
                        "id": cid,
                        "parent_id": base_id,
                        "start": int(a),
                        "end": int(b),
                        "title": title,
                        "text": text[a:b],
                    }
                    fcor.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fmap.write(json.dumps({"chunk_id": cid, "parent_id": base_id, "start": int(a), "end": int(b)},
                                          ensure_ascii=False) + "\n")
                    chunk_idx += 1
                continue

            # buffer this doc units for a GPU flush
            unit_texts = [text[a:b] for (a, b) in units]
            # if adding would exceed buffer, flush first
            if len(buf_unit_texts) + len(unit_texts) >= args.buf_units:
                flush(fcor, fmap)

            start_off = len(buf_unit_texts)
            buf_unit_texts.extend(unit_texts)
            buf_docs.append({
                "base_id": base_id,
                "title": title,
                "text": text,
                "units": units,
                "range": (start_off, start_off + len(unit_texts)),
            })
        print(f"[DONE] skipped_id={skipped} skipped_text={skipped_text} title_fallback={used_title_fallback}")
        # final flush
        flush(fcor, fmap)


if __name__ == "__main__":
    main()
