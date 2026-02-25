#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import hashlib
import shutil
import argparse
import subprocess
import gzip
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# -----------------------------
# Utils: Jasper remote-code sync
# -----------------------------
def sync_remote_code_if_missing(ft_dir: str, base_dir: str | None):
    """
    Jasper uses HF remote code (e.g. modeling_qwen3_jasper.py).
    If your FT output dir doesn't contain these *.py files, ST/Transformers will fail to load.
    """
    ft = Path(ft_dir)
    need = ft / "modeling_qwen3_jasper.py"
    if need.exists():
        return

    if base_dir is None:
        raise OSError(
            f"{ft_dir} missing {need.name}. "
            f"Provide --base_dir to copy remote-code files automatically."
        )

    base = Path(base_dir)
    if not base.exists():
        raise OSError(f"base_dir not found: {base_dir}")

    print(f"[SYNC] {need.name} missing in {ft_dir}. Copying remote-code files from {base_dir} ...")
    for pat in ["modeling_*.py", "configuration_*.py", "tokenization_*.py"]:
        for src in base.glob(pat):
            dst = ft / src.name
            shutil.copy2(src, dst)
            print(f"[SYNC] {src} -> {dst}")


# -----------------------------
# Cache helpers
# -----------------------------
def _safe_tag(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-+" else "_" for ch in s)

def _cache_key(model_tag: str, collection_name: str, max_len: int, compression_ratio: float, corpus_path: str) -> str:
    st = os.stat(corpus_path)
    sig = f"{model_tag}|{collection_name}|len={max_len}|cr={compression_ratio}|size={st.st_size}|mtime={int(st.st_mtime)}"
    h = hashlib.sha1(sig.encode("utf-8")).hexdigest()[:16]
    return f"{_safe_tag(collection_name)}__{_safe_tag(model_tag)}__len{max_len}__cr{compression_ratio:.4f}__{h}"

def get_or_build_doc_embeddings(
    *,
    model,
    doc_texts: List[str],
    doc_ids: List[str],
    cache_dir: str,
    model_tag: str,
    collection_name: str,
    max_len: int,
    compression_ratio: float,
    corpus_path: str,
    batch_size: int = 256,
    force_recompute: bool = False,
) -> Tuple[List[str], torch.Tensor]:
    """
    Returns:
      doc_ids_cached: List[str]
      doc_emb_cpu: torch.Tensor on CPU, dtype float16, shape [N, D]
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    key = _cache_key(model_tag, collection_name, max_len, compression_ratio, corpus_path)
    cache_path = os.path.join(cache_dir, key + ".pt")

    if (not force_recompute) and os.path.exists(cache_path):
        pack = torch.load(cache_path, map_location="cpu")
        if pack.get("doc_ids") == doc_ids:
            print(f"[CACHE HIT] {cache_path}")
            return pack["doc_ids"], pack["doc_emb"]
        print("[CACHE STALE] doc_ids mismatch -> recompute")

    print(f"[CACHE MISS] Encoding docs for {collection_name} ...")
    doc_emb = model.encode_cuda(
        doc_texts,
        batch_size=batch_size,
        prompt_name=None,                  # docs: no prompt
        compression_ratio=compression_ratio,
    )  # CPU tensor (because encode_cuda appends emb.cpu())

    doc_emb = doc_emb.to(torch.float16).contiguous()
    pack = {
        "doc_ids": doc_ids,
        "doc_emb": doc_emb,
        "meta": {
            "model_tag": model_tag,
            "collection_name": collection_name,
            "max_len": max_len,
            "compression_ratio": compression_ratio,
            "corpus_path": corpus_path,
            "saved_time": time.time(),
        },
    }
    torch.save(pack, cache_path)
    print(f"[CACHE SAVE] {cache_path} | shape={tuple(doc_emb.shape)}")
    return doc_ids, doc_emb


# -----------------------------
# IO helpers
# -----------------------------
def _open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


# -----------------------------
# Data loading
# -----------------------------
def load_corpus(corpus_path: str) -> Dict[str, Dict[str, str]]:
    """
    Supports:
      - official corpus jsonl: {"_id": "...", "title": "...", "text": "..."} or {"document_id": "...", ...}
      - semantic chunk corpus jsonl.gz: {"id": "...__c0000", "text": "...", "parent_id": "..."}
    """
    corpus: Dict[str, Dict[str, str]] = {}
    with _open_text(corpus_path) as f:
        for line in f:
            item = json.loads(line)

            doc_id = item.get("document_id") or item.get("_id") or item.get("id")
            if doc_id is None:
                continue

            corpus[doc_id] = {
                "title": item.get("title", "") or "",
                "text": item.get("text", "") or "",
                "parent_id": item.get("parent_id", "") or "",
            }
    return corpus

def load_queries(query_path: str) -> Dict[str, str]:
    queries = {}
    with open(query_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            qid = item.get("_id") or item.get("query_id") or item.get("task_id")
            if qid is None:
                continue
            queries[qid] = item["text"]
    return queries


# -----------------------------
# Jasper loader (with encode_cuda)
# -----------------------------
def load_jasper_st(
    model_name: str,
    device: str,
    default_compression_ratio: float = 0.3333,
):
    print("Using device:", device)
    device_map = "cuda" if device == "cuda" else None

    model = SentenceTransformer(
        model_name,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
            "trust_remote_code": True,
            "device_map": device_map,
        },
        tokenizer_kwargs={
            "padding_side": "left",
            "trust_remote_code": True,
            "fix_mistral_regex": True
        },
        trust_remote_code=True,
        local_files_only=True,
        device=device,
    )
    model.max_seq_length = 512

    def encode_cuda(texts, batch_size=32, prompt_name=None, compression_ratio=None):
        if compression_ratio is None:
            compression_ratio = default_compression_ratio

        all_emb = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding", leave=False):
            batch = texts[i:i + batch_size]
            emb = model.encode(
                batch,
                batch_size=batch_size,
                convert_to_tensor=True,
                show_progress_bar=False,
                normalize_embeddings=True,
                prompt_name=prompt_name,
                compression_ratio=compression_ratio,
            )
            all_emb.append(emb.cpu())
        return torch.cat(all_emb, dim=0)

    model.encode_cuda = encode_cuda
    return model


# -----------------------------
# Exact chunked top-k (no big sims matrix)
# -----------------------------
@torch.no_grad()
def batched_topk_stream(
    q_emb_cpu: torch.Tensor,      # [Q, D] on CPU (float32/float16 ok)
    doc_emb_cpu: torch.Tensor,    # [N, D] on CPU (float16 cache)
    k: int,
    chunk_size: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exact global top-k over all docs, computed in doc chunks.
    Scoring uses float32 to avoid precision drift.
    Returns:
      top_vals_cpu: [Q, k] float32 on CPU
      top_idx_cpu : [Q, k] int64 on CPU (indices into doc list)
    """
    q = q_emb_cpu.to(device, non_blocking=True)
    q_f = q.float()

    Q = q_f.size(0)
    top_vals = torch.full((Q, k), -1e9, device=device, dtype=torch.float32)
    top_idx = torch.full((Q, k), -1, device=device, dtype=torch.int64)

    N = doc_emb_cpu.size(0)
    for start in range(0, N, chunk_size):
        chunk = doc_emb_cpu[start:start + chunk_size].to(device, non_blocking=True)
        sims = q_f @ chunk.float().T
        vals, idx = torch.topk(sims, k, dim=1)
        idx = idx + start

        merged_vals = torch.cat([top_vals, vals], dim=1)
        merged_idx  = torch.cat([top_idx,  idx],  dim=1)
        new_vals, pos = torch.topk(merged_vals, k, dim=1)
        new_idx = torch.gather(merged_idx, 1, pos)
        top_vals, top_idx = new_vals, new_idx

    return top_vals.cpu(), top_idx.cpu()


def chunk_id_to_parent_id(chunk_id: str) -> str:
    # semantic chunk ids look like: {parent_id}__c0000
    if "__c" in chunk_id:
        return chunk_id.rsplit("__c", 1)[0]
    return chunk_id


# -----------------------------
# Retrieval runner (with cache + chunk-topk + aggregation)
# -----------------------------
def run_retrieval_for_collection(
    name: str,
    cfg: dict,
    model,
    top_k_parent: int,
    top_k_chunks: int,
    cache_dir: str,
    model_tag: str,
    compression_ratio: float,
    doc_bs: int,
    query_bs: int,
    force_recompute_docs: bool,
    chunk_size: int,
    use_semantic_chunks: bool,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    root = cfg["root"]
    official_corpus_path = os.path.join(root, cfg["corpus_file"])
    query_path = os.path.join(root, cfg["query_file"])

    corpus_path = official_corpus_path
    use_sem = False
    if use_semantic_chunks:
        sem_path = cfg.get("semchunk_corpus", "")
        if sem_path and os.path.exists(sem_path):
            corpus_path = sem_path
            use_sem = True
        else:
            print(f"[WARN] {name}: --use_semantic_chunks on, but semchunk corpus not found: {sem_path}. Fallback to official corpus.")

    print(f"\n========== Collection: {name} ==========")
    print("Corpus:", corpus_path, ("(SEMCHUNK)" if use_sem else "(OFFICIAL)"))
    print("Queries:", query_path)

    corpus = load_corpus(corpus_path)
    queries = load_queries(query_path)

    doc_ids = list(corpus.keys())
    doc_texts = [
        (corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip()
        for d in doc_ids
    ]

    doc_ids_cached, doc_emb_cpu = get_or_build_doc_embeddings(
        model=model,
        doc_texts=doc_texts,
        doc_ids=doc_ids,
        cache_dir=cache_dir,
        model_tag=model_tag,
        collection_name=cfg["collection_name"] + ("__semchunk" if use_sem else "__official"),
        max_len=getattr(model, "max_seq_length", 512),
        compression_ratio=compression_ratio,
        corpus_path=corpus_path,
        batch_size=doc_bs,
        force_recompute=force_recompute_docs,
    )

    q_ids = list(queries.keys())
    q_texts = [queries[q] for q in q_ids]

    # queries: prompt_name="query"
    q_emb_cpu = model.encode_cuda(
        q_texts,
        batch_size=query_bs,
        prompt_name="query",
        compression_ratio=compression_ratio,
    )  # CPU

    k_retrieve = top_k_chunks if use_sem else top_k_parent

    top_vals, top_idx = batched_topk_stream(
        q_emb_cpu=q_emb_cpu,
        doc_emb_cpu=doc_emb_cpu,
        k=k_retrieve,
        chunk_size=chunk_size,
        device=device,
    )

    results = []
    for qi, qid in enumerate(tqdm(q_ids, desc="Building results")):
        idx_row = top_idx[qi].tolist()
        val_row = top_vals[qi].tolist()

        if not use_sem:
            ctxs = [{"document_id": doc_ids_cached[j], "score": float(v)}
                    for j, v in zip(idx_row, val_row) if j >= 0]
        else:
            # aggregate chunk scores -> parent passage id scores
            best: Dict[str, float] = {}
            for j, v in zip(idx_row, val_row):
                if j < 0:
                    continue
                cid = doc_ids_cached[j]
                pid = chunk_id_to_parent_id(cid)  # official qrels id space
                vv = float(v)
                if (pid not in best) or (vv > best[pid]):
                    best[pid] = vv

            items = sorted(best.items(), key=lambda x: x[1], reverse=True)[:top_k_parent]
            ctxs = [{"document_id": pid, "score": sc} for pid, sc in items]

        results.append({"task_id": qid, "contexts": ctxs, "Collection": cfg["collection_name"]})
    return results


def run_official_eval(input_file: str, output_file: str, model_name: str, task_name: str):
    cmd = [
        "python3",
        "scripts/evaluation/run_retrieval_eval.py",
        "--input_file", input_file,
        "--output_file", output_file,
        "--model_name", model_name,
        "--task_name", task_name,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Done, scored file:", output_file)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, default="lastturn", choices=["lastturn", "questions", "rewrite"])
    ap.add_argument("--model_dir", type=str, default="jasper-ft-lastturn", help="FT model folder (or base model folder)")
    ap.add_argument("--base_dir", type=str, default=None, help="Base Jasper folder to sync remote-code *.py if missing")
    ap.add_argument("--model_name", type=str, default="jasper_semchunk", help="Name used in output filenames / official eval")
    ap.add_argument("--cache_dir", type=str, default="cache/doc_emb_jasper_semchunk")
    ap.add_argument("--model_tag", type=str, default=None, help="Cache tag; default uses model_dir name")
    ap.add_argument("--compression_ratio", type=float, default=0.3333)

    ap.add_argument("--top_k", type=int, default=50, help="Final top-k PARENTS to output (matches eval k range needs).")
    ap.add_argument("--top_k_chunks", type=int, default=500, help="If semchunk enabled: retrieve top-k CHUNKS then aggregate.")
    ap.add_argument("--doc_bs", type=int, default=256)
    ap.add_argument("--query_bs", type=int, default=256)
    ap.add_argument("--chunk_size", type=int, default=50000, help="Doc chunk size for topk streaming (tune for GPU mem)")
    ap.add_argument("--force_recompute_docs", action="store_true")

    ap.add_argument("--use_semantic_chunks", action="store_true",
                    help="Use semantic sub-chunk corpora (*.jsonl.gz) and aggregate back to parent passage ids for official eval.")
    ap.add_argument("--chunk_corpus_dir", type=str, default="corpora/chunk_level",
                    help="Directory containing {domain}_passage_semantic.jsonl.gz")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Ensure remote-code exists in FT dir
    sync_remote_code_if_missing(args.model_dir, args.base_dir)

    # Load Jasper (FT or base)
    model = load_jasper_st(
        model_name=args.model_dir,
        device=device,
        default_compression_ratio=args.compression_ratio,
    )

    model_tag = args.model_tag or Path(args.model_dir).name

    COLLECTIONS = {
        "clapnq": {
            "collection_name": "mt-rag-clapnq-elser-512-100-20240503",
            "root": "human/retrieval_tasks/clapnq",
            "corpus_file": "clapnq.jsonl",
            "query_file": f"clapnq_{args.task}.jsonl",
            "semchunk_corpus": os.path.join(args.chunk_corpus_dir, "clapnq_passage_semantic.jsonl.gz"),
        },
        "fiqa": {
            "collection_name": "mt-rag-fiqa-beir-elser-512-100-20240501",
            "root": "human/retrieval_tasks/fiqa",
            "corpus_file": "fiqa.jsonl",
            "query_file": f"fiqa_{args.task}.jsonl",
            "semchunk_corpus": os.path.join(args.chunk_corpus_dir, "fiqa_passage_semantic.jsonl.gz"),
        },
        "govt": {
            "collection_name": "mt-rag-govt-elser-512-100-20240611",
            "root": "human/retrieval_tasks/govt",
            "corpus_file": "govt.jsonl",
            "query_file": f"govt_{args.task}.jsonl",
            "semchunk_corpus": os.path.join(args.chunk_corpus_dir, "govt_passage_semantic.jsonl.gz"),
        },
        "cloud": {
            "collection_name": "mt-rag-ibmcloud-elser-512-100-20240502",
            "root": "human/retrieval_tasks/cloud",
            "corpus_file": "cloud.jsonl",
            "query_file": f"cloud_{args.task}.jsonl",
            "semchunk_corpus": os.path.join(args.chunk_corpus_dir, "cloud_passage_semantic.jsonl.gz"),
        },
    }

    os.makedirs("outputs", exist_ok=True)
    sub_path = f"outputs/{args.model_name}_{args.task}.jsonl"
    scored_path = f"outputs/{args.model_name}_{args.task}_score.jsonl"

    all_results = []
    print("Start!")
    print("Current task:", args.task)
    print("Semantic chunks:", args.use_semantic_chunks)
    if args.use_semantic_chunks:
        print("Chunk corpus dir:", args.chunk_corpus_dir)
        print("top_k_chunks:", args.top_k_chunks, "| top_k_parent:", args.top_k)

    for name, cfg in COLLECTIONS.items():
        res = run_retrieval_for_collection(
            name=name,
            cfg=cfg,
            model=model,
            top_k_parent=args.top_k,
            top_k_chunks=args.top_k_chunks,
            cache_dir=args.cache_dir,
            model_tag=model_tag,
            compression_ratio=args.compression_ratio,
            doc_bs=args.doc_bs,
            query_bs=args.query_bs,
            force_recompute_docs=args.force_recompute_docs,
            chunk_size=args.chunk_size,
            use_semantic_chunks=args.use_semantic_chunks,
        )
        all_results.extend(res)

    with open(sub_path, "w", encoding="utf-8") as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Saved submission to:", sub_path)
    run_official_eval(sub_path, scored_path, model_name=args.model_name, task_name=args.task)


if __name__ == "__main__":
    main()
