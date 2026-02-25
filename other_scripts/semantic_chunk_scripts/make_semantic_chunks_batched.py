#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, re, zipfile, gzip
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer


_SENT_SPLIT = re.compile(r'(?<=[\.\!\?])\s+')

def split_sentences(text: str) -> List[str]:
    text = text.replace("\r\n", "\n").strip()
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    sents: List[str] = []
    for p in paras:
        ss = [s.strip() for s in _SENT_SPLIT.split(p) if s.strip()]
        sents.extend(ss if ss else [p])
    return sents

def tok_len(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

def semantic_boundaries(sent_emb: np.ndarray, threshold: float = 0.55) -> List[int]:
    sent_emb = sent_emb / (np.linalg.norm(sent_emb, axis=1, keepdims=True) + 1e-12)
    sims = (sent_emb[:-1] * sent_emb[1:]).sum(axis=1)
    return [i + 1 for i, s in enumerate(sims) if s < threshold]

def pack_segments_by_tokens(sents: List[str], boundaries: List[int]) -> List[List[str]]:
    idxs = [0] + boundaries + [len(sents)]
    return [sents[a:b] for a, b in zip(idxs[:-1], idxs[1:])]

def force_max_tokens(tokenizer, seg_sents: List[str], max_tokens: int, overlap_tokens: int) -> List[str]:
    text = " ".join(seg_sents).strip()
    if tok_len(tokenizer, text) <= max_tokens:
        return [text]

    ids = tokenizer.encode(text, add_special_tokens=False)
    out = []
    step = max(1, max_tokens - overlap_tokens)
    for st in range(0, len(ids), step):
        ed = min(st + max_tokens, len(ids))
        out.append(tokenizer.decode(ids[st:ed]).strip())
        if ed >= len(ids):
            break
    return out

def write_chunks(doc_id: str, sents: List[str], sent_emb: np.ndarray,
                 tokenizer, max_tokens: int, overlap_tokens: int, sim_threshold: float,
                 fcor, fmap):
    if len(sents) == 1:
        chunks = force_max_tokens(tokenizer, sents, max_tokens, overlap_tokens)
        for j, ch in enumerate(chunks):
            cid = f"{doc_id}__c{j:04d}"
            fcor.write(json.dumps({"id": cid, "text": ch, "parent_id": doc_id}, ensure_ascii=False) + "\n")
            fmap.write(json.dumps({"chunk_id": cid, "parent_id": doc_id}, ensure_ascii=False) + "\n")
        return

    bnds = semantic_boundaries(sent_emb, threshold=sim_threshold)
    segs = pack_segments_by_tokens(sents, bnds)

    chunk_idx = 0
    for seg in segs:
        chunk_texts = force_max_tokens(tokenizer, seg, max_tokens, overlap_tokens)
        for ch in chunk_texts:
            cid = f"{doc_id}__c{chunk_idx:04d}"
            fcor.write(json.dumps({"id": cid, "text": ch, "parent_id": doc_id}, ensure_ascii=False) + "\n")
            fmap.write(json.dumps({"chunk_id": cid, "parent_id": doc_id}, ensure_ascii=False) + "\n")
            chunk_idx += 1

def split_long_sentence_for_embedding(tokenizer, sent: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    sent = sent.strip()
    if not sent:
        return []
    ids = tokenizer.encode(sent, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return [sent]

    step = max(1, max_tokens - overlap_tokens)
    out = []
    for st in range(0, len(ids), step):
        ed = min(st + max_tokens, len(ids))
        out.append(tokenizer.decode(ids[st:ed]).strip())
        if ed >= len(ids):
            break
    return [x for x in out if x]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_zip", required=True)
    ap.add_argument("--jsonl_name", required=True)
    ap.add_argument("--out_corpus", required=True)
    ap.add_argument("--out_map", required=True)
    ap.add_argument("--st_model", required=True)
    ap.add_argument("--hf_tokenizer", required=True)
    ap.add_argument("--max_tokens", type=int, default=384)
    ap.add_argument("--overlap_tokens", type=int, default=64)
    ap.add_argument("--sim_threshold", type=float, default=0.55)

    ap.add_argument("--embed_bs", type=int, default=512, help="mini-batch inside model.encode")
    ap.add_argument("--buf_sents", type=int, default=16384, help="buffer sentences across docs before encoding")
    ap.add_argument("--gzip_level", type=int, default=1, help="gzip compresslevel (1 fastest)")
    
    ap.add_argument("--sent_max_tokens", type=int, default=128, help="max tokens per 'sentence' for embedding")
    ap.add_argument("--sent_overlap_tokens", type=int, default=16, help="overlap when splitting long sentences for embedding")


    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.hf_tokenizer, trust_remote_code=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.st_model, trust_remote_code=True, device=device)

    out_corpus = Path(args.out_corpus)
    out_map = Path(args.out_map)
    out_corpus.parent.mkdir(parents=True, exist_ok=True)
    out_map.parent.mkdir(parents=True, exist_ok=True)

    # buffer
    buf_sents: List[str] = []
    buf_docs: List[Tuple[str, List[str], int]] = []  # (doc_id, sents, n_sents)

    def flush(fcor, fmap):
        nonlocal buf_sents, buf_docs
        if not buf_docs:
            return

        sent_emb_all = model.encode(
            buf_sents,
            batch_size=args.embed_bs,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )

        off = 0
        for doc_id, sents, n in buf_docs:
            emb = sent_emb_all[off:off+n]
            off += n
            write_chunks(
                doc_id, sents, emb,
                tokenizer, args.max_tokens, args.overlap_tokens, args.sim_threshold,
                fcor, fmap
            )

        buf_sents = []
        buf_docs = []

    with zipfile.ZipFile(args.in_zip, "r") as zf, \
         zf.open(args.jsonl_name, "r") as fin, \
         gzip.open(out_corpus, "wt", encoding="utf-8", compresslevel=args.gzip_level) as fcor, \
         gzip.open(out_map, "wt", encoding="utf-8", compresslevel=args.gzip_level) as fmap:

        for raw in tqdm(fin, desc="chunking"):
            obj = json.loads(raw.decode("utf-8"))
            doc_id = obj.get("id") or obj.get("_id")
            text = (obj.get("text") or "").strip()
            if not doc_id or not text:
                continue

            sents = split_sentences(text)
            # --- IMPORTANT: cap sentence length for embedding (especially Govt) ---
            norm_sents: List[str] = []
            for s in sents:
                norm_sents.extend(split_long_sentence_for_embedding(
                    tokenizer, s, args.sent_max_tokens, args.sent_overlap_tokens
                ))
            sents = norm_sents if norm_sents else [text[:2000]]  # fallback

            if len(sents) <= 1:
                # 1-sent doc 直接写（不走 GPU）
                write_chunks(
                    doc_id, sents, np.zeros((len(sents), 1), dtype=np.float32),
                    tokenizer, args.max_tokens, args.overlap_tokens, args.sim_threshold,
                    fcor, fmap
                )
                continue

            # buffer accumulate
            buf_docs.append((doc_id, sents, len(sents)))
            buf_sents.extend(sents)

            if len(buf_sents) >= args.buf_sents:
                flush(fcor, fmap)

        # last flush
        flush(fcor, fmap)

if __name__ == "__main__":
    main()
