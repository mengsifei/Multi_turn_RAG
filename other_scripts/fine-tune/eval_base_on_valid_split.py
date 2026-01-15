#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import atexit
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# -------------------------
# Tee stdout/stderr to file
# -------------------------
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except Exception:
                pass
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


def setup_tee(output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(output_dir) / "run.log"
    f = open(log_path, "a", encoding="utf-8")
    atexit.register(lambda: f.close())
    sys.stdout = Tee(sys.__stdout__, f)
    sys.stderr = Tee(sys.__stderr__, f)
    print(f"[LOG] tee -> {log_path}")


# -----------------------------
# Utils: Jasper remote-code sync
# -----------------------------
def sync_remote_code_if_missing(model_dir: str, base_dir: Optional[str]):
    md = Path(model_dir)
    need = md / "modeling_qwen3_jasper.py"
    if need.exists():
        return
    if base_dir is None:
        raise OSError(f"{model_dir} missing {need.name}. Provide --base_dir.")
    base = Path(base_dir)
    if not base.exists():
        raise OSError(f"base_dir not found: {base_dir}")
    print(f"[SYNC] missing remote-code in {model_dir}, copy from {base_dir}")
    for pat in ["modeling_*.py", "configuration_*.py", "tokenization_*.py", "custom_st.py"]:
        for src in base.glob(pat):
            dst = md / src.name
            try:
                shutil.copy2(src, dst)
                print(f"[SYNC] {src} -> {dst}")
            except Exception:
                pass


# -----------------------------
# Data loading
# -----------------------------
def load_corpus(corpus_path: str) -> Dict[str, Dict[str, str]]:
    corpus = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            doc_id = item["document_id"] if "document_id" in item else item["_id"]
            corpus[str(doc_id)] = {"title": item.get("title", ""), "text": item.get("text", "")}
    return corpus


def load_queries(query_path: str) -> Dict[str, str]:
    queries = {}
    with open(query_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            qid = item.get("_id") or item.get("query_id") or item["task_id"]
            queries[str(qid)] = item["text"]
    return queries


def load_qrels_pos(tspath: str) -> Dict[str, Set[str]]:
    qrels: Dict[str, Set[str]] = defaultdict(set)
    with open(tspath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            qid = str(parts[0])
            did = str(parts[1])
            rel = 1
            if len(parts) >= 3:
                try:
                    rel = int(float(parts[2]))
                except Exception:
                    rel = 1
            if rel > 0:
                qrels[qid].add(did)
    return dict(qrels)


def load_valid_split_from_dir(split_dir: str, task: str, domain: str) -> Dict[str, Set[str]]:
    """
    Required path:
      <split_dir>/<task>/<domain>/valid.tsv
    """
    p = Path(split_dir) / task / domain / "valid.tsv"
    if not p.exists():
        raise FileNotFoundError(f"[split_dir] missing valid.tsv: {p}")
    return load_qrels_pos(str(p))


# -----------------------------
# Jasper loader (with encode_cuda)
# -----------------------------
def load_jasper_st(model_name: str, device: str, default_compression_ratio: float = 0.3333):
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
            "fix_mistral_regex": True,
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
        prompt_name=None,
        compression_ratio=compression_ratio,
    )
    doc_emb = doc_emb.to(torch.float16).contiguous()
    pack = {"doc_ids": doc_ids, "doc_emb": doc_emb, "meta": {"time": time.time()}}
    torch.save(pack, cache_path)
    print(f"[CACHE SAVE] {cache_path} | shape={tuple(doc_emb.shape)}")
    return doc_ids, doc_emb


# -----------------------------
# FP32 top-k streaming (exact)
# -----------------------------
@torch.no_grad()
def batched_topk_stream(
    q_emb_cpu: torch.Tensor,
    doc_emb_cpu: torch.Tensor,
    k: int,
    chunk_size: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
        merged_idx = torch.cat([top_idx, idx], dim=1)
        new_vals, pos = torch.topk(merged_vals, k, dim=1)
        new_idx = torch.gather(merged_idx, 1, pos)
        top_vals, top_idx = new_vals, new_idx

    return top_vals.cpu(), top_idx.cpu()


# -----------------------------
# Collections
# -----------------------------
@dataclass
class CollectionCfg:
    key: str
    collection_name: str
    root: str
    corpus_file: str
    query_file: str


def build_collections(task: str) -> Dict[str, CollectionCfg]:
    return {
        "clapnq": CollectionCfg(
            key="clapnq",
            collection_name="mt-rag-clapnq-elser-512-100-20240503",
            root="human/retrieval_tasks/clapnq",
            corpus_file="clapnq.jsonl",
            query_file=f"clapnq_{task}.jsonl",
        ),
        "fiqa": CollectionCfg(
            key="fiqa",
            collection_name="mt-rag-fiqa-beir-elser-512-100-20240501",
            root="human/retrieval_tasks/fiqa",
            corpus_file="fiqa.jsonl",
            query_file=f"fiqa_{task}.jsonl",
        ),
        "govt": CollectionCfg(
            key="govt",
            collection_name="mt-rag-govt-elser-512-100-20240611",
            root="human/retrieval_tasks/govt",
            corpus_file="govt.jsonl",
            query_file=f"govt_{task}.jsonl",
        ),
        "cloud": CollectionCfg(
            key="cloud",
            collection_name="mt-rag-ibmcloud-elser-512-100-20240502",
            root="human/retrieval_tasks/cloud",
            corpus_file="cloud.jsonl",
            query_file=f"cloud_{task}.jsonl",
        ),
    }


# -----------------------------
# Retrieval for VALID qids only
# (NO metrics here; official eval does scoring)
# -----------------------------
@torch.no_grad()
def run_retrieval_for_valid_qids(
    *,
    dom: str,
    cfg: CollectionCfg,
    model,
    qids: List[str],
    compression_ratio: float,
    cache_dir: str,
    model_tag: str,
    doc_bs: int,
    query_bs: int,
    chunk_size: int,
    top_k: int,
    force_recompute_docs: bool,
    device: str,
) -> List[dict]:
    corpus_path = os.path.join(cfg.root, cfg.corpus_file)
    query_path = os.path.join(cfg.root, cfg.query_file)

    corpus = load_corpus(corpus_path)
    queries = load_queries(query_path)

    qids = [q for q in qids if q in queries]
    if not qids:
        print(f"[RETRIEVE] domain={dom} valid_q=0 (skip)")
        return []

    print(f"\n[RETRIEVE] domain={dom} valid_q={len(qids)}")

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
        collection_name=cfg.collection_name,
        max_len=getattr(model, "max_seq_length", 512),
        compression_ratio=compression_ratio,
        corpus_path=corpus_path,
        batch_size=doc_bs,
        force_recompute=force_recompute_docs,
    )

    q_texts = [queries[qid] for qid in qids]
    q_emb_cpu = model.encode_cuda(
        q_texts,
        batch_size=query_bs,
        prompt_name="query",
        compression_ratio=compression_ratio,
    )

    top_vals, top_idx = batched_topk_stream(
        q_emb_cpu=q_emb_cpu,
        doc_emb_cpu=doc_emb_cpu,
        k=top_k,
        chunk_size=chunk_size,
        device=device,
    )

    results = []
    for i, qid in enumerate(tqdm(qids, desc=f"Build contexts ({dom})")):
        idx_row = top_idx[i].tolist()
        val_row = top_vals[i].tolist()
        ctxs = [{"document_id": doc_ids_cached[j], "score": float(v)} for j, v in zip(idx_row, val_row)]
        results.append({"task_id": qid, "contexts": ctxs, "Collection": cfg.collection_name})

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def run_official_eval(eval_script: str, input_file: str, output_file: str, model_name: str, task_name: str):
    cmd = [
        "python3",
        eval_script,
        "--input_file", input_file,
        "--output_file", output_file,
        "--model_name", model_name,
        "--task_name", task_name,
    ]
    print("[OFFICIAL EVAL] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[OFFICIAL EVAL] Done, scored file:", output_file)


# -----------------------------
# Main (split_dir ONLY)
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--task", type=str, choices=["lastturn", "questions", "rewrite"], required=True)
    ap.add_argument("--model_dir", type=str, required=True, help="Base model folder or FT folder")
    ap.add_argument("--base_dir", type=str, default=None, help="Base Jasper folder to sync remote-code if missing")

    ap.add_argument("--split_dir", type=str, required=True,
                    help="MUST be prepared split root, e.g. splits/human_s42_r01 ; expects <split_dir>/<task>/<dom>/valid.tsv")

    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--model_name", type=str, default=None, help="Name used by official eval (defaults to model_dir name)")

    ap.add_argument("--compression_ratio", type=float, default=0.3333)
    ap.add_argument("--cache_dir", type=str, default="cache/doc_emb_eval_base")
    ap.add_argument("--model_tag", type=str, default=None)

    ap.add_argument("--doc_bs", type=int, default=256)
    ap.add_argument("--query_bs", type=int, default=256)
    ap.add_argument("--chunk_size", type=int, default=50000)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--force_recompute_docs", action="store_true")

    ap.add_argument("--eval_script", type=str, default="scripts/evaluation/run_retrieval_eval.py")
    ap.add_argument("--no_tee", action="store_true")
    args = ap.parse_args()

    if not args.no_tee:
        setup_tee(args.output)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sync_remote_code_if_missing(args.model_dir, args.base_dir)
    model = load_jasper_st(args.model_dir, device=device, default_compression_ratio=args.compression_ratio)

    cfgs = build_collections(args.task)

    model_tag = args.model_tag or Path(args.model_dir).name
    model_name = args.model_name or Path(args.model_dir).name
    split_tag = Path(args.split_dir).name

    sub_path = out_dir / f"{model_name}_{args.task}_{split_tag}.jsonl"
    scored_path = out_dir / f"{model_name}_{args.task}_{split_tag}_score.jsonl"
    used_qids_path = out_dir / f"valid_qids_{args.task}_{split_tag}.json"

    # ---- load VALID qids from split_dir ONLY ----
    qids_by_dom: Dict[str, List[str]] = {}
    for dom in cfgs.keys():
        qrels_valid = load_valid_split_from_dir(args.split_dir, args.task, dom)
        valid_qids = [qid for qid, rel in qrels_valid.items() if len(rel) > 0]
        qids_by_dom[dom] = sorted(valid_qids)
        print(f"[SPLIT:{dom}] valid_q={len(valid_qids)} from {args.split_dir}/{args.task}/{dom}/valid.tsv")

    with open(used_qids_path, "w", encoding="utf-8") as f:
        json.dump(qids_by_dom, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] used valid qids -> {used_qids_path}")

    # ---- retrieve only these VALID qids ----
    all_results: List[dict] = []
    for dom, cfg in cfgs.items():
        valid_qids = qids_by_dom.get(dom, [])
        if not valid_qids:
            print(f"[RETRIEVE] domain={dom} valid_q=0 (skip)")
            continue

        results = run_retrieval_for_valid_qids(
            dom=dom,
            cfg=cfg,
            model=model,
            qids=valid_qids,
            compression_ratio=args.compression_ratio,
            cache_dir=args.cache_dir,
            model_tag=model_tag,
            doc_bs=args.doc_bs,
            query_bs=args.query_bs,
            chunk_size=args.chunk_size,
            top_k=args.top_k,
            force_recompute_docs=args.force_recompute_docs,
            device=device,
        )
        all_results.extend(results)

    # write submission
    with open(sub_path, "w", encoding="utf-8") as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[SAVE] submission -> {sub_path}  (#lines={len(all_results)})")

    # ---- OFFICIAL EVAL ONLY ----
    run_official_eval(
        eval_script=args.eval_script,
        input_file=str(sub_path),
        output_file=str(scored_path),
        model_name=model_name,
        task_name=args.task,
    )


if __name__ == "__main__":
    main()
