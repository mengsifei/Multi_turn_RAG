#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, re, zipfile, gzip
from pathlib import Path
from typing import List, Dict, Iterable, Tuple

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer


_SENT_SPLIT = re.compile(r'(?<=[\.\!\?])\s+')

def split_sentences(text: str) -> List[str]:
    # 够用的轻量句切；你也可以换成 spacy/nltk
    text = text.replace("\r\n", "\n").strip()
    # 先按段落粗切，减少句切误差
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    sents: List[str] = []
    for p in paras:
        ss = [s.strip() for s in _SENT_SPLIT.split(p) if s.strip()]
        sents.extend(ss if ss else [p])
    return sents

def tok_len(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

def semantic_boundaries(
    sent_emb: np.ndarray,
    threshold: float = 0.55,
) -> List[int]:
    """
    返回 boundary indices：在 i 处切开，意味着 [prev, i) 是一个段。
    用相邻句 cosine 相似度低于阈值作为“话题跳变”。
    """
    # normalize
    sent_emb = sent_emb / (np.linalg.norm(sent_emb, axis=1, keepdims=True) + 1e-12)
    sims = (sent_emb[:-1] * sent_emb[1:]).sum(axis=1)  # (n-1,)
    bnds = [i+1 for i, s in enumerate(sims) if s < threshold]
    return bnds

def pack_segments_by_tokens(
    sents: List[str],
    boundaries: List[int],
) -> List[List[str]]:
    idxs = [0] + boundaries + [len(sents)]
    segs = []
    for a, b in zip(idxs[:-1], idxs[1:]):
        segs.append(sents[a:b])
    return segs

def force_max_tokens(
    tokenizer,
    seg_sents: List[str],
    max_tokens: int,
    overlap_tokens: int,
) -> List[str]:
    """
    把一个 segment（句子列表）压到 max_tokens 内。
    超长则退化成 sliding window（按 token）。
    返回 chunk_text 列表。
    """
    text = " ".join(seg_sents).strip()
    if tok_len(tokenizer, text) <= max_tokens:
        return [text]

    # token sliding window fallback
    ids = tokenizer.encode(text, add_special_tokens=False)
    out = []
    step = max(1, max_tokens - overlap_tokens)
    for st in range(0, len(ids), step):
        ed = min(st + max_tokens, len(ids))
        chunk = tokenizer.decode(ids[st:ed])
        out.append(chunk.strip())
        if ed >= len(ids):
            break
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_zip", required=True, help=".../clapnq.jsonl.zip")
    ap.add_argument("--jsonl_name", default="clapnq.jsonl", help="member name inside zip")
    ap.add_argument("--out_corpus", required=True, help="chunked corpus jsonl.gz")
    ap.add_argument("--out_map", required=True, help="chunk_id -> parent_id map jsonl.gz")
    ap.add_argument("--st_model", required=True, help="SentenceTransformer path (e.g., Jasper base dir)")
    ap.add_argument("--hf_tokenizer", required=True, help="Tokenizer path (usually same as Jasper base)")
    ap.add_argument("--max_tokens", type=int, default=384)
    ap.add_argument("--overlap_tokens", type=int, default=64)
    ap.add_argument("--sim_threshold", type=float, default=0.55)
    ap.add_argument("--embed_bs", type=int, default=512)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.hf_tokenizer, trust_remote_code=True)
    model = SentenceTransformer(args.st_model, trust_remote_code=True)

    out_corpus = Path(args.out_corpus)
    out_map = Path(args.out_map)
    out_corpus.parent.mkdir(parents=True, exist_ok=True)
    out_map.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.in_zip, "r") as zf, \
         zf.open(args.jsonl_name, "r") as fin, \
         gzip.open(out_corpus, "wt", encoding="utf-8") as fcor, \
         gzip.open(out_map, "wt", encoding="utf-8") as fmap:

        for raw in tqdm(fin, desc="chunking"):
            obj = json.loads(raw.decode("utf-8"))
            doc_id = obj.get("id") or obj.get("_id")
            text = obj.get("text", "").strip()
            if not doc_id or not text:
                continue

            sents = split_sentences(text)
            if len(sents) == 1:
                # 仍然做 max_tokens 控制
                chunks = force_max_tokens(tokenizer, sents, args.max_tokens, args.overlap_tokens)
                for j, ch in enumerate(chunks):
                    cid = f"{doc_id}__c{j:04d}"
                    fcor.write(json.dumps({"id": cid, "text": ch, "parent_id": doc_id}, ensure_ascii=False) + "\n")
                    fmap.write(json.dumps({"chunk_id": cid, "parent_id": doc_id}, ensure_ascii=False) + "\n")
                continue

            # sentence embeddings
            sent_emb = model.encode(
                sents,
                batch_size=args.embed_bs,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )

            bnds = semantic_boundaries(sent_emb, threshold=args.sim_threshold)
            segs = pack_segments_by_tokens(sents, bnds)

            chunk_idx = 0
            for seg in segs:
                chunk_texts = force_max_tokens(tokenizer, seg, args.max_tokens, args.overlap_tokens)
                for ch in chunk_texts:
                    cid = f"{doc_id}__c{chunk_idx:04d}"
                    fcor.write(json.dumps({"id": cid, "text": ch, "parent_id": doc_id}, ensure_ascii=False) + "\n")
                    fmap.write(json.dumps({"chunk_id": cid, "parent_id": doc_id}, ensure_ascii=False) + "\n")
                    chunk_idx += 1

if __name__ == "__main__":
    main()
