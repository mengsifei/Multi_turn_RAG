#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import atexit
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
import pytrec_eval


# ============================================================
# Reproducibility
# ============================================================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Tee stdout/stderr to file
# ============================================================
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


# ============================================================
# Jasper remote-code sync (for saved checkpoints)
# ============================================================
def sync_remote_code_if_missing(ft_dir: str, base_dir: Optional[str]):
    ft = Path(ft_dir)
    need = ft / "modeling_qwen3_jasper.py"
    if need.exists():
        return
    if base_dir is None:
        raise OSError(f"{ft_dir} missing {need.name}. Provide --base_dir.")
    base = Path(base_dir)
    if not base.exists():
        raise OSError(f"base_dir not found: {base_dir}")

    print(f"[SYNC] {need.name} missing in {ft_dir}. Copying remote-code files from {base_dir} ...")
    for pat in ["modeling_*.py", "configuration_*.py", "tokenization_*.py", "custom_st.py"]:
        for src in base.glob(pat):
            dst = ft / src.name
            try:
                shutil.copy2(src, dst)
                print(f"[SYNC] {src} -> {dst}")
            except Exception:
                pass


# ============================================================
# Query prompt prefix (so training text matches prompt_name='query' behavior)
# ============================================================
def get_query_prompt_prefix(st_model: SentenceTransformer, prompt_name: str = "query") -> str:
    prompts = getattr(st_model, "prompts", None)
    if isinstance(prompts, dict) and prompt_name in prompts and isinstance(prompts[prompt_name], str):
        return prompts[prompt_name]
    try:
        first = st_model._first_module()
        prompts2 = getattr(first, "prompts", None)
        if isinstance(prompts2, dict) and prompt_name in prompts2 and isinstance(prompts2[prompt_name], str):
            return prompts2[prompt_name]
    except Exception:
        pass
    return ""


# ============================================================
# Compression ratio alignment (train forward uses config)
# ============================================================
def set_default_compression_ratio(st_model: SentenceTransformer, ratio: float):
    first = st_model._first_module()
    auto_model = getattr(first, "auto_model", None)
    if auto_model is None:
        print("[CR] WARNING: auto_model not found; cannot set compression ratio.")
        return
    cfg = getattr(auto_model, "config", None)
    if cfg is None:
        print("[CR] WARNING: auto_model.config not found; cannot set compression ratio.")
        return
    names = ["compression_ratio", "default_compression_ratio", "token_compression_ratio", "jasper_compression_ratio"]
    touched = []
    for n in names:
        try:
            setattr(cfg, n, float(ratio))
            touched.append(n)
        except Exception:
            pass
    if touched:
        print(f"[CR] Set config fields: {touched} = {ratio}")
    else:
        try:
            setattr(cfg, "compression_ratio", float(ratio))
            print(f"[CR] Set config field: compression_ratio = {ratio}")
        except Exception:
            print("[CR] WARNING: failed to set any compression ratio field.")


# ============================================================
# Freeze encoder except last layer (optional)
# ============================================================
def freeze_encoder_except_last_layer(st_model: SentenceTransformer, unfreeze_final_norm: bool = True):
    first = st_model._first_module()
    auto_model = getattr(first, "auto_model", None)
    if auto_model is None:
        raise RuntimeError("Cannot find underlying auto_model in SentenceTransformer first module.")

    for p in auto_model.parameters():
        p.requires_grad = False

    if hasattr(auto_model, "model") and hasattr(auto_model.model, "layers"):
        layers = auto_model.model.layers
        for p in layers[-1].parameters():
            p.requires_grad = True
        if unfreeze_final_norm and hasattr(auto_model.model, "norm"):
            for p in auto_model.model.norm.parameters():
                p.requires_grad = True
        print("[FREEZE] train last transformer layer (+norm)")
        return

    if hasattr(auto_model, "encoder") and hasattr(auto_model.encoder, "layer"):
        layers = auto_model.encoder.layer
        for p in layers[-1].parameters():
            p.requires_grad = True
        print("[FREEZE] train last encoder.layer")
        return

    raise RuntimeError("Unknown transformer layout; cannot locate layers to unfreeze.")


# ============================================================
# OFFICIAL: load corpus/queries
# ============================================================
def load_corpus(corpus_path: str) -> Dict[str, Dict[str, str]]:
    corpus = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            doc_id = item["document_id"] if "document_id" in item else item["_id"]
            doc_id = str(doc_id)
            corpus[doc_id] = {"title": item.get("title", ""), "text": item.get("text", "")}
    return corpus


def load_queries(query_path: str) -> Dict[str, str]:
    queries = {}
    with open(query_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            qid = item.get("_id") or item.get("query_id") or item["task_id"]
            qid = str(qid)
            queries[qid] = item["text"]
    return queries


# ============================================================
# Split qrels loading (your split format: qid \t did \t rel?)
# ============================================================
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


def load_split_qrels(split_dir: str, task: str, domain: str) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    sd = Path(split_dir)
    train_p = sd / task / domain / "train.tsv"
    valid_p = sd / task / domain / "valid.tsv"
    if not train_p.exists():
        raise FileNotFoundError(f"missing: {train_p}")
    if not valid_p.exists():
        raise FileNotFoundError(f"missing: {valid_p}")
    return load_qrels_pos(str(train_p)), load_qrels_pos(str(valid_p))


def dump_split_snapshot(out_dir: Path, split_dir: str, task: str, domain: str):
    src_train = Path(split_dir) / task / domain / "train.tsv"
    src_valid = Path(split_dir) / task / domain / "valid.tsv"
    dst_root = out_dir / "splits_used" / task / domain
    dst_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_train, dst_root / "train.tsv")
    shutil.copy2(src_valid, dst_root / "valid.tsv")


# ============================================================
# OFFICIAL: Jasper loader with encode_cuda
# ============================================================
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


# ============================================================
# OFFICIAL: Cache helpers for doc embeddings
# ============================================================
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
    pack = {"doc_ids": doc_ids, "doc_emb": doc_emb, "meta": {"saved_time": time.time()}}
    torch.save(pack, cache_path)
    print(f"[CACHE SAVE] {cache_path} | shape={tuple(doc_emb.shape)}")
    return doc_ids, doc_emb


import re

def cleanup_epoch_caches(cache_dir: str, run_tag: str, kind: str, current_epoch: int, keep_last: int):
    """
    Delete doc-embedding cache files (.pt) for THIS run older than (current_epoch - keep_last).
    We only delete files whose filename contains:
      f"__{run_tag}__{kind}_ep<NUM>__"
    where kind in {"mine", "eval"}.

    cache filename format (from _cache_key):
      <collection>__<model_tag>__len...__cr...__hash.pt
    and model_tag includes:
      f"{run_tag}__{kind}_ep{epoch}"
    """
    if keep_last <= 0:
        return
    cache_p = Path(cache_dir)
    if not cache_p.exists():
        return

    # delete epochs <= threshold
    threshold = current_epoch - keep_last
    if threshold <= 0:
        return

    # run_tag is already safe_tag(out_dir.name)
    # match "...__<run_tag>__mine_ep12__..." or "...__<run_tag>__eval_ep3__..."
    pat = re.compile(rf"__{re.escape(run_tag)}__{kind}_ep(\d+)__")

    removed = 0
    kept = 0
    for fp in cache_p.glob("*.pt"):
        m = pat.search(fp.name)
        if not m:
            continue  # not this run/kind
        ep = int(m.group(1))
        if ep <= threshold:
            try:
                fp.unlink()
                removed += 1
            except Exception:
                pass
        else:
            kept += 1

    if removed > 0:
        print(f"[CACHE CLEAN] {kind}: removed={removed} kept={kept} (keep_last={keep_last}, current_epoch={current_epoch})")


# ============================================================
# Exact chunked top-k streaming (FP32 scoring)
# ============================================================
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


# ============================================================
# Training dataset: triplets built from mined hard-negs
# ============================================================
class TripletDataset(Dataset):
    def __init__(self, triplets: List[Tuple[str, str, str]], seed: int):
        self.triplets = triplets
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx: int) -> InputExample:
        q, p, n = self.triplets[idx]
        return InputExample(texts=[q, p, n], label=0.0)


def batch_to_device(sentence_features, labels, device: str):
    for sf in sentence_features:
        for k, v in sf.items():
            if isinstance(v, torch.Tensor):
                sf[k] = v.to(device, non_blocking=True)
    if isinstance(labels, torch.Tensor):
        labels = labels.to(device, non_blocking=True)
    return sentence_features, labels


# ============================================================
# Collection config
# ============================================================
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


# ============================================================
# Hard-negative mining (per collection)
# ============================================================
@torch.no_grad()
def mine_hard_negs_for_collection(
    *,
    model,
    device: str,
    cfg: CollectionCfg,
    queries: Dict[str, str],
    corpus: Dict[str, Dict[str, str]],
    qrels_train: Dict[str, Set[str]],
    query_prefix: str,
    compression_ratio: float,
    cache_dir: str,
    model_tag: str,
    doc_bs: int,
    query_bs: int,
    chunk_size: int,
    mine_topk: int,
    max_pos_per_query: int,
    neg_per_pos: int,
    seed: int,
    force_recompute_docs: bool,
) -> List[Tuple[str, str, str]]:
    corpus_path = os.path.join(cfg.root, cfg.corpus_file)

    doc_ids = list(corpus.keys())
    doc_texts = [(corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip() for d in doc_ids]

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

    train_qids = [qid for qid in qrels_train.keys() if qid in queries and len(qrels_train[qid]) > 0]
    if not train_qids:
        return []

    q_texts = [queries[qid] for qid in train_qids]
    q_emb_cpu = model.encode_cuda(
        q_texts,
        batch_size=query_bs,
        prompt_name="query",
        compression_ratio=compression_ratio,
    )

    _, top_idx = batched_topk_stream(
        q_emb_cpu=q_emb_cpu,
        doc_emb_cpu=doc_emb_cpu,
        k=mine_topk,
        chunk_size=chunk_size,
        device=device,
    )

    rng = random.Random(seed)
    triplets: List[Tuple[str, str, str]] = []

    def doc_full_text(did: str) -> str:
        return (corpus[did].get("title", "") + " " + corpus[did].get("text", "")).strip()

    for i, qid in enumerate(train_qids):
        rel = qrels_train[qid]
        pos_list = [d for d in rel if d in corpus]
        if not pos_list:
            continue
        rng.shuffle(pos_list)
        if max_pos_per_query > 0:
            pos_list = pos_list[:max_pos_per_query]

        ranked_ids = [doc_ids_cached[j] for j in top_idx[i].tolist()]
        neg_candidates = [d for d in ranked_ids if (d not in rel) and (d in corpus)]
        if not neg_candidates:
            continue

        q_text = f"{query_prefix}{queries[qid].strip()}"

        for pd in pos_list:
            p_text = doc_full_text(pd)
            if not p_text:
                continue

            if len(neg_candidates) >= neg_per_pos:
                negs = rng.sample(neg_candidates, k=neg_per_pos)
            else:
                negs = [rng.choice(neg_candidates) for _ in range(neg_per_pos)]

            for nd in negs:
                n_text = doc_full_text(nd)
                if n_text:
                    triplets.append((q_text, p_text, n_text))

    return triplets


# ============================================================
# Build OFFICIAL submission jsonl but ONLY for valid qids
# ============================================================
def build_valid_submission_jsonl(
    *,
    out_path: str,
    model,
    device: str,
    cfgs: Dict[str, CollectionCfg],
    qrels_valid_by_domain: Dict[str, Dict[str, Set[str]]],
    compression_ratio: float,
    cache_dir: str,
    model_tag: str,
    doc_bs: int,
    query_bs: int,
    chunk_size: int,
    top_k: int,
    force_recompute_docs: bool,
):
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    all_results = []
    for dom, cfg in cfgs.items():
        corpus_path = os.path.join(cfg.root, cfg.corpus_file)
        query_path = os.path.join(cfg.root, cfg.query_file)

        corpus = load_corpus(corpus_path)
        queries = load_queries(query_path)
        qrels_valid = qrels_valid_by_domain.get(dom, {})

        valid_qids = [qid for qid in qrels_valid.keys() if qid in queries and len(qrels_valid[qid]) > 0]
        if not valid_qids:
            continue

        doc_ids = list(corpus.keys())
        doc_texts = [(corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip() for d in doc_ids]

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

        q_texts = [queries[qid] for qid in valid_qids]
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

        for i, qid in enumerate(valid_qids):
            idx_row = top_idx[i].tolist()
            val_row = top_vals[i].tolist()
            ctxs = [{"document_id": doc_ids_cached[j], "score": float(v)} for j, v in zip(idx_row, val_row)]
            all_results.append({"task_id": qid, "contexts": ctxs, "Collection": cfg.collection_name})

    with open(out_path, "w", encoding="utf-8") as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[SAVE] valid submission -> {out_path}  (rows={len(all_results)})")


# ============================================================
# Patch official qrels/dev.tsv to use our split_dir valid.tsv
# ============================================================
def write_qrels_tsv(path: str, qrels: Dict[str, Set[str]]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid, dids in qrels.items():
            for did in dids:
                # official script expects TREC style: qid \t did \t rel
                f.write(f"{qid}\t{did}\t1\n")


class PatchDevQrels:
    """
    Temporarily replace human/retrieval_tasks/<dom>/qrels/dev.tsv with our valid qrels.
    Always restore on exit.
    """
    def __init__(self, cfgs: Dict[str, CollectionCfg], qrels_valid_by_domain: Dict[str, Dict[str, Set[str]]]):
        self.cfgs = cfgs
        self.qrels_valid_by_domain = qrels_valid_by_domain
        self.backups: List[Tuple[str, str]] = []  # (orig, bak)

    def __enter__(self):
        stamp = f"{os.getpid()}_{int(time.time())}"
        for dom, cfg in self.cfgs.items():
            dev_path = os.path.join(cfg.root, "qrels", "dev.tsv")
            if not os.path.exists(dev_path):
                raise FileNotFoundError(f"official dev.tsv not found: {dev_path}")

            bak_path = dev_path + f".bak_{stamp}"
            shutil.copy2(dev_path, bak_path)
            self.backups.append((dev_path, bak_path))

            qrels_valid = self.qrels_valid_by_domain.get(dom, {})
            write_qrels_tsv(dev_path, qrels_valid)
            print(f"[PATCH] {dev_path} <= valid(q={len(qrels_valid)})  (backup={bak_path})")
        return self

    def __exit__(self, exc_type, exc, tb):
        for dev_path, bak_path in self.backups:
            try:
                shutil.move(bak_path, dev_path)
                print(f"[RESTORE] {dev_path} <= {bak_path}")
            except Exception as e:
                print(f"[RESTORE-ERR] {dev_path} from {bak_path}: {e}")
        self.backups.clear()
        return False


# ============================================================
# Run official eval (writes scored jsonl)
# ============================================================
def run_official_eval(eval_script: str, input_file: str, output_file: str, model_name: str, task_name: str):
    cmd = [
        sys.executable,
        eval_script,
        "--input_file", input_file,
        "--output_file", output_file,
        "--model_name", model_name,
        "--task_name", task_name,
    ]
    print("[OFFICIAL EVAL] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[OFFICIAL EVAL] Done ->", output_file)


# ============================================================
# Compute weighted metrics from scored jsonl using OUR valid qrels (no dev.tsv dependency)
# ============================================================
def score_jsonl_to_results_by_collection(score_jsonl: Path):
    # returns: {collection_name: {qid: {docid: score}}}
    out = defaultdict(dict)
    with score_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            coll = obj["Collection"]
            qid = str(obj["task_id"])  # keep exactly as in split/qrels
            ctxs = obj.get("contexts", [])
            out[coll][qid] = {c["document_id"]: float(c["score"]) for c in ctxs}
    return out


def qrels_sets_to_trec(qrels_sets: Dict[str, Set[str]]):
    # {qid: set(docid)} -> {qid: {docid: 1}}
    return {str(qid): {str(did): 1 for did in dids} for qid, dids in qrels_sets.items()}


def compute_weighted_metrics_from_score_jsonl(
    score_jsonl: str,
    *,
    cfgs: Dict[str, CollectionCfg],
    qrels_by_domain: Dict[str, Dict[str, Set[str]]],
    k_list=(1, 3, 5, 10),
):
    score_jsonl = Path(score_jsonl)
    results_by_coll = score_jsonl_to_results_by_collection(score_jsonl)

    weighted_ndcg = [0.0 for _ in k_list]
    weighted_recall = [0.0 for _ in k_list]
    total_q = 0
    by_domain = {}

    measures = set()
    for k in k_list:
        measures.add(f"ndcg_cut.{k}")
        measures.add(f"recall.{k}")

    for dom, cfg in cfgs.items():
        coll = cfg.collection_name
        results = results_by_coll.get(coll, {})
        if not results:
            continue

        qrels = qrels_sets_to_trec(qrels_by_domain.get(dom, {}))
        qrels = {qid: qrels.get(qid, {}) for qid in results.keys()}  # restrict to scored qids

        evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
        scores = evaluator.evaluate(results)

        n_q = len(scores)
        if n_q == 0:
            continue

        dom_ndcg = []
        dom_recall = []
        for k in k_list:
            nd_key = f"ndcg_cut_{k}"
            rc_key = f"recall_{k}"
            dom_ndcg.append(sum(v.get(nd_key, 0.0) for v in scores.values()) / n_q)
            dom_recall.append(sum(v.get(rc_key, 0.0) for v in scores.values()) / n_q)

        by_domain[dom] = {"n_q": n_q, "ndcg": dom_ndcg, "recall": dom_recall}

        for i in range(len(k_list)):
            weighted_ndcg[i] += dom_ndcg[i] * n_q
            weighted_recall[i] += dom_recall[i] * n_q
        total_q += n_q

    if total_q > 0:
        weighted_ndcg = [x / total_q for x in weighted_ndcg]
        weighted_recall = [x / total_q for x in weighted_recall]

    return {
        "k_list": list(k_list),
        "weighted_ndcg": weighted_ndcg,
        "weighted_recall": weighted_recall,
        "by_domain": by_domain,
        "total_q": total_q,
        "source_score_jsonl": str(score_jsonl),
    }


def get_weighted_ndcg_at_5(metrics: Dict) -> float:
    # {"weighted_ndcg":[@1,@3,@5,@10], "k_list":[1,3,5,10]}
    if "weighted_ndcg" in metrics and isinstance(metrics["weighted_ndcg"], list):
        k_list = metrics.get("k_list", [1, 3, 5, 10])
        if isinstance(k_list, list) and 5 in k_list:
            return float(metrics["weighted_ndcg"][k_list.index(5)])
        if len(metrics["weighted_ndcg"]) >= 3:
            return float(metrics["weighted_ndcg"][2])
    raise KeyError(f"Cannot find weighted nDCG@5 in keys={list(metrics.keys())[:50]}")


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--task", type=str, choices=["lastturn", "questions", "rewrite"], required=True)
    ap.add_argument("--model", type=str, default="Jasper-Token-Compression-600M")
    ap.add_argument("--base_dir", type=str, default=None)
    ap.add_argument("--output", type=str, required=True)

    ap.add_argument("--split_dir", type=str, required=True)
    ap.add_argument("--allow_overlap", action="store_true")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--loss_margin", type=float, default=0.2)

    ap.add_argument("--compression_ratio", type=float, default=0.3333)
    ap.add_argument("--freeze_last_layer_only", action="store_true")

    # mining
    ap.add_argument("--mine_cache_dir", type=str, default="cache/mine_doc_emb")
    ap.add_argument("--mine_doc_bs", type=int, default=256)
    ap.add_argument("--mine_query_bs", type=int, default=256)
    ap.add_argument("--mine_chunk_size", type=int, default=50000)
    ap.add_argument("--mine_topk", type=int, default=50)
    ap.add_argument("--max_pos_per_query", type=int, default=2)
    ap.add_argument("--neg_per_pos", type=int, default=2)
    ap.add_argument("--force_recompute_mine_docs", action="store_true")

    # eval submission build (valid only)
    ap.add_argument("--eval_cache_dir", type=str, default="cache/doc_emb_eval")
    ap.add_argument("--eval_doc_bs", type=int, default=256)
    ap.add_argument("--eval_query_bs", type=int, default=256)
    ap.add_argument("--eval_chunk_size", type=int, default=50000)
    ap.add_argument("--eval_topk", type=int, default=50)
    ap.add_argument("--force_recompute_eval_docs", action="store_true")

    # official eval
    ap.add_argument("--official_eval_script", type=str, default="scripts/evaluation/run_retrieval_eval.py")
    ap.add_argument("--official_model_name", type=str, default=None)

    # new: remine
    ap.add_argument("--remine_each_epoch", action="store_true",
                    help="Re-mine hard negatives at the start of every epoch using current model weights.")

    ap.add_argument("--no_tee", action="store_true")
    
    ap.add_argument(
        "--keep_cache_epochs",
        type=int,
        default=2,
        help="Keep only the most recent N epochs of THIS run's doc-embedding cache (mine/eval). 0 disables cleanup."
    )

    args = ap.parse_args()

    if not args.no_tee:
        setup_tee(args.output)

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    metrics_path = out_dir / "eval_metrics.jsonl"
    print(f"[LOG] eval metrics -> {metrics_path}")

    model = load_jasper_st(args.model, device=device, default_compression_ratio=args.compression_ratio)
    set_default_compression_ratio(model, args.compression_ratio)

    query_prefix = get_query_prompt_prefix(model, "query")
    print(f"[PROMPT] query_prefix_len={len(query_prefix)} head={repr(query_prefix[:80])}" if query_prefix else
          "[PROMPT] WARNING: No query prefix found.")

    if args.freeze_last_layer_only:
        freeze_encoder_except_last_layer(model, unfreeze_final_norm=True)

    cfgs = build_collections(args.task)

    # -------------------------
    # Load split qrels (train/valid), strict check, snapshot
    # -------------------------
    print("\n==== Load TRAIN/VALID split from split_dir (snapshot) ====")

    qrels_valid_by_domain: Dict[str, Dict[str, Set[str]]] = {}
    split_meta = {"split_dir": args.split_dir, "task": args.task, "domains": {}}

    domain_state = {}  # dom -> {cfg, qrels_train, corpus, queries}

    for dom, cfg in cfgs.items():
        qrels_train, qrels_valid = load_split_qrels(args.split_dir, args.task, dom)

        inter = set(qrels_train.keys()).intersection(set(qrels_valid.keys()))
        if inter and (not args.allow_overlap):
            some = sorted(list(inter))[:20]
            raise RuntimeError(
                f"[LEAKAGE] split_dir has train/valid qid overlap for domain={dom}: "
                f"{len(inter)} overlaps. examples={some}. Fix split files or pass --allow_overlap."
            )
        if inter:
            print(f"[WARN] domain={dom} train/valid qid overlap={len(inter)} (allowed by --allow_overlap)")

        qrels_valid_by_domain[dom] = qrels_valid
        dump_split_snapshot(out_dir, args.split_dir, args.task, dom)

        split_meta["domains"][dom] = {"train_q": len(qrels_train), "valid_q": len(qrels_valid), "overlap_q": len(inter)}
        print(f"[SPLIT:{dom}] train_q={len(qrels_train)} valid_q={len(qrels_valid)}")

        corpus_path = os.path.join(cfg.root, cfg.corpus_file)
        query_path = os.path.join(cfg.root, cfg.query_file)
        corpus = load_corpus(corpus_path)
        queries = load_queries(query_path)

        domain_state[dom] = {
            "cfg": cfg,
            "qrels_train": qrels_train,
            "corpus": corpus,
            "queries": queries,
        }

    (out_dir / "split_meta.json").write_text(json.dumps(split_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAVE] split snapshot -> {out_dir/'splits_used'}")
    print(f"[SAVE] split meta -> {out_dir/'split_meta.json'}")

    # -------------------------
    # Mining helper (can be called each epoch)
    # -------------------------
    def remine_triplets(*, model_tag: str) -> List[Tuple[str, str, str]]:
        print(f"\n==== [MINE] model_tag={model_tag} ====")
        all_triplets: List[Tuple[str, str, str]] = []
        for dom, st in domain_state.items():
            triplets = mine_hard_negs_for_collection(
                model=model,
                device=device,
                cfg=st["cfg"],
                queries=st["queries"],
                corpus=st["corpus"],
                qrels_train=st["qrels_train"],
                query_prefix=query_prefix,
                compression_ratio=args.compression_ratio,
                cache_dir=args.mine_cache_dir,
                model_tag=model_tag,
                doc_bs=args.mine_doc_bs,
                query_bs=args.mine_query_bs,
                chunk_size=args.mine_chunk_size,
                mine_topk=args.mine_topk,
                max_pos_per_query=args.max_pos_per_query,
                neg_per_pos=args.neg_per_pos,
                seed=args.seed,
                force_recompute_docs=args.force_recompute_mine_docs,
            )
            print(f"[MINE:{dom}] triplets={len(triplets)}")
            all_triplets.extend(triplets)

        if not all_triplets:
            raise RuntimeError("No training triplets built after mining. Check split/qrels/query/corpus alignment.")

        random.shuffle(all_triplets)
        print(f"[TRAIN] mined total triplets: {len(all_triplets)}")
        return all_triplets

    # -------------------------
    # Prepare loss/optimizer
    # -------------------------
    train_loss = losses.TripletLoss(
        model=model,
        distance_metric=losses.TripletDistanceMetric.COSINE,
        triplet_margin=float(args.loss_margin),
    )
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    best_score = -1.0
    best_epoch = -1
    global_step = 0

    official_model_name = args.official_model_name or _safe_tag(out_dir.name)

    official_dir = out_dir / "official_eval"
    official_dir.mkdir(parents=True, exist_ok=True)

    # If not remine_each_epoch, mine once up-front
    train_loader = None
    steps_per_epoch = None
    if not args.remine_each_epoch:
        base_mine_tag = f"{_safe_tag(out_dir.name)}__mine_ep1"
        all_triplets = remine_triplets(model_tag=base_mine_tag)
        train_ds = TripletDataset(all_triplets, seed=args.seed)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            collate_fn=model.smart_batching_collate,
        )
        steps_per_epoch = len(train_loader)
        print(f"[TRAIN] steps/epoch={steps_per_epoch} grad_accum={args.grad_accum} effective_batch={args.batch_size * args.grad_accum}")

    # -------------------------
    # Epoch loop
    # -------------------------
    for epoch in range(1, args.epochs + 1):
        # Re-mine at start of each epoch if requested
        if args.remine_each_epoch:
            mine_tag = f"{_safe_tag(out_dir.name)}__mine_ep{epoch}"
            all_triplets = remine_triplets(model_tag=mine_tag)
            train_ds = TripletDataset(all_triplets, seed=args.seed)
            train_loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=0,
                collate_fn=model.smart_batching_collate,
            )
            steps_per_epoch = len(train_loader)
            # after mining (and building loader), drop old mining doc-emb caches for this run
            run_tag = _safe_tag(out_dir.name)
            cleanup_epoch_caches(args.mine_cache_dir, run_tag, "mine", epoch, args.keep_cache_epochs)

            print(f"[TRAIN] steps/epoch={steps_per_epoch} grad_accum={args.grad_accum} effective_batch={args.batch_size * args.grad_accum}")

        # -------------------------
        # Train epoch
        # -------------------------
        model.train()
        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()
        running = 0.0

        for step, (sentence_features, labels) in enumerate(train_loader, start=1):
            sentence_features, labels = batch_to_device(sentence_features, labels, device)

            loss_val = train_loss(sentence_features, labels)
            loss_val = loss_val / args.grad_accum
            loss_val.backward()
            running += float(loss_val.item())

            if step % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if step % 50 == 0:
                print(f"[epoch {epoch}] step {step}/{steps_per_epoch} loss(avg)={running/step:.4f}")

        print(f"\n[epoch {epoch}] done. train_time={time.time() - t0:.1f}s")

        # save epoch checkpoint
        ep_dir = out_dir / f"epoch{epoch}"
        model.save(str(ep_dir))
        sync_remote_code_if_missing(str(ep_dir), args.base_dir)
        print(f"[SAVE] epoch checkpoint -> {ep_dir}")

        # -------------------------
        # OFFICIAL eval on VALID split
        # -------------------------
        model.eval()
        eval_model_tag = f"{_safe_tag(out_dir.name)}__eval_ep{epoch}"
        sub_path = str(official_dir / f"epoch{epoch}_{args.task}_valid.jsonl")
        scored_path = str(official_dir / f"epoch{epoch}_{args.task}_valid_score.jsonl")

        build_valid_submission_jsonl(
            out_path=sub_path,
            model=model,
            device=device,
            cfgs=cfgs,
            qrels_valid_by_domain=qrels_valid_by_domain,
            compression_ratio=args.compression_ratio,
            cache_dir=args.eval_cache_dir,
            model_tag=eval_model_tag,
            doc_bs=args.eval_doc_bs,
            query_bs=args.eval_query_bs,
            chunk_size=args.eval_chunk_size,
            top_k=args.eval_topk,
            force_recompute_docs=args.force_recompute_eval_docs,
        )

        with PatchDevQrels(cfgs, qrels_valid_by_domain):
            run_official_eval(
                eval_script=args.official_eval_script,
                input_file=sub_path,
                output_file=scored_path,
                model_name=official_model_name,
                task_name=args.task,
            )

        summary = compute_weighted_metrics_from_score_jsonl(
            scored_path,
            cfgs=cfgs,
            qrels_by_domain=qrels_valid_by_domain,
            k_list=(1, 3, 5, 10),
        )
        score = get_weighted_ndcg_at_5(summary)

        print("weighted_ndcg@1/3/5/10 =", summary["weighted_ndcg"])
        print("weighted_recall@1/3/5/10 =", summary["weighted_recall"])
        print(f"[EVAL] weighted_nDCG@5={score:.6f}")

        record = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": epoch,
            "global_step": global_step,
            "official_score_file": scored_path,
            "official_metrics": summary,  # ✅ fixed
            "selected_metric": "weighted_nDCG@5",
            "selected_value": score,
        }
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_dir = out_dir / "best"
            model.save(str(best_dir))
            sync_remote_code_if_missing(str(best_dir), args.base_dir)
            print(f"[BEST] epoch={epoch} weighted_nDCG@5={best_score:.6f} saved -> {best_dir}")
        # after eval, drop old eval doc-emb caches for this run
        run_tag = _safe_tag(out_dir.name)
        cleanup_epoch_caches(args.eval_cache_dir, run_tag, "eval", epoch, args.keep_cache_epochs)


    print(f"\nDone. Best weighted nDCG@5 = {best_score:.6f} (epoch={best_epoch})")
    print(f"[LOG] run log: {out_dir/'run.log'}")
    print(f"[LOG] eval metrics: {metrics_path}")


if __name__ == "__main__":
    main()
