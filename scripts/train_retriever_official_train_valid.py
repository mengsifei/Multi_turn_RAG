#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a SentenceTransformer retriever on MTRAG official HUMAN retrieval tasks (ClapNQ/FiQA/Govt/Cloud)
with an explicit TRAIN/VALID split:
  - If qrels/train.tsv exists: use it as TRAIN and qrels/dev.tsv as VALID (official).
  - Else: split qrels/dev.tsv by query into TRAIN/VALID.

Key features:
  ✅ Uses SentenceTransformer smart_batching_collate (no InputExample collate errors)
  ✅ Avoids CPU/CUDA mismatch by moving sentence_features to device
  ✅ Logging: tee ALL stdout/stderr to <output>/run.log
  ✅ Writes structured eval metrics to <output>/eval_metrics.jsonl
  ✅ Saves best model to <output>/best and epoch checkpoints to <output>/epoch{n}
  ✅ Copies Jasper remote-code files into saved folders (modeling_*.py etc.)

Example:
python train_retriever_official_train_valid.py --task lastturn  --model Jasper-Token-Compression-600M --base_remote_code_dir Jasper-Token-Compression-600M --output jasper-ft-human-lastturn  --epochs 1 --lr 2e-6 --batch_size 8 --grad_accum 4  --max_pos_per_query 2   --compression_ratio 0.3333  --freeze_last_layer_only  --prefer_train_qrels
"""

import argparse
import json
import math
import os
import random
import time
import hashlib
import sys
import atexit
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from sentence_transformers import SentenceTransformer, InputExample, losses


# -------------------------
# Reproducibility
# -------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


# -------------------------
# Remote-code sync (Jasper)
# -------------------------
def sync_remote_code_files(base_dir: str, ft_dir: str):
    """
    Copy modeling_*.py / configuration_*.py / tokenization_*.py from base_dir to ft_dir if missing.
    This avoids: OSError: missing modeling_qwen3_jasper.py
    """
    base = Path(base_dir)
    out = Path(ft_dir)
    out.mkdir(parents=True, exist_ok=True)

    pats = ["modeling_*.py", "configuration_*.py", "tokenization_*.py", "custom_st.py"]
    copied = 0
    for pat in pats:
        for src in base.glob(pat):
            dst = out / src.name
            if not dst.exists():
                dst.write_bytes(src.read_bytes())
                copied += 1
                print(f"[SYNC] {src} -> {dst}")
    if copied == 0:
        print("[SYNC] remote-code files already present (or none found).")


# -------------------------
# Query prompt prefix (for training texts)
# -------------------------
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


# -------------------------
# Jasper compression ratio alignment
# -------------------------
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


# -------------------------
# Optional: freeze encoder except last layer
# -------------------------
def freeze_encoder_except_last_layer(st_model: SentenceTransformer, unfreeze_final_norm: bool = True):
    first = st_model._first_module()
    auto_model = getattr(first, "auto_model", None)
    if auto_model is None:
        raise RuntimeError("Cannot find underlying auto_model in SentenceTransformer first module.")

    for p in auto_model.parameters():
        p.requires_grad = False

    # Qwen/LLaMA-like
    if hasattr(auto_model, "model") and hasattr(auto_model.model, "layers"):
        layers = auto_model.model.layers
        for p in layers[-1].parameters():
            p.requires_grad = True
        if unfreeze_final_norm and hasattr(auto_model.model, "norm"):
            for p in auto_model.model.norm.parameters():
                p.requires_grad = True
        print("[FREEZE] train last transformer layer (+norm)")
        return

    # BERT-like
    if hasattr(auto_model, "encoder") and hasattr(auto_model.encoder, "layer"):
        layers = auto_model.encoder.layer
        for p in layers[-1].parameters():
            p.requires_grad = True
        print("[FREEZE] train last encoder.layer")
        return

    raise RuntimeError("Unknown transformer layout; cannot locate layers to unfreeze.")


# -------------------------
# Data loading (human retrieval tasks)
# -------------------------
def load_corpus_jsonl(path: str) -> Dict[str, str]:
    """
    corpus jsonl: {"document_id":..., "title":..., "text":...} or {"_id":...}
    returns doc_id -> (title + " " + text).strip()
    """
    corpus: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            did = obj.get("document_id", obj.get("_id"))
            if did is None:
                continue
            title = (obj.get("title") or "").strip()
            text = (obj.get("text") or "").strip()
            full = (title + " " + text).strip()
            if full:
                corpus[str(did)] = full
    return corpus


def load_queries_jsonl(path: str) -> Dict[str, str]:
    """
    query jsonl: {"task_id":..., "text":...} or {"_id":...} or {"query_id":...}
    returns qid -> text
    """
    qs: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj.get("task_id") or obj.get("_id") or obj.get("query_id")
            txt = (obj.get("text") or "").strip()
            if qid is not None and txt:
                qs[str(qid)] = txt
    return qs


def load_qrels_tsv(path: str) -> Dict[str, Set[str]]:
    """
    qrels tsv: qid \t docid \t rel   OR  qid \t docid
    keep rel>0 as positive.
    returns: qid -> set(pos_doc_ids)
    """
    rels: Dict[str, Set[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            qid, did = parts[0], parts[1]
            rel = 1
            if len(parts) >= 3:
                try:
                    rel = int(float(parts[2]))
                except Exception:
                    rel = 1
            if rel <= 0:
                continue
            rels.setdefault(str(qid), set()).add(str(did))
    return rels


def split_qrels_by_query(qrels_dev: Dict[str, Set[str]], dev_ratio: float, seed: int) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    qids = list(qrels_dev.keys())
    rnd = random.Random(seed)
    rnd.shuffle(qids)
    n_dev = max(1, int(len(qids) * dev_ratio))
    dev_qids = set(qids[:n_dev])
    train = {qid: qrels_dev[qid] for qid in qids if qid not in dev_qids}
    dev = {qid: qrels_dev[qid] for qid in qids if qid in dev_qids}
    return train, dev


# -------------------------
# Training dataset
# -------------------------
class PairQrelsDataset(Dataset):
    """
    Each item: (query, positive_doc_text) for MultipleNegativesRankingLoss.
    We prefix query with query_prefix so TRAIN matches inference prompt behavior.
    """
    def __init__(
        self,
        queries: Dict[str, str],
        corpus: Dict[str, str],
        qrels_pos: Dict[str, Set[str]],
        query_prefix: str,
        max_pos_per_query: int,
        seed: int,
    ):
        self.queries = queries
        self.corpus = corpus
        self.qrels_pos = qrels_pos
        self.query_prefix = query_prefix or ""
        self.max_pos_per_query = int(max_pos_per_query)
        self.rng = random.Random(seed)

        self.items: List[Tuple[str, str]] = []  # (qid, did)

        for qid, pos_set in qrels_pos.items():
            if qid not in queries:
                continue
            pos_list = [d for d in pos_set if d in corpus]
            if not pos_list:
                continue
            self.rng.shuffle(pos_list)
            if self.max_pos_per_query > 0:
                pos_list = pos_list[: self.max_pos_per_query]
            for did in pos_list:
                self.items.append((qid, did))

        print(f"[DATASET] pair (q,pos) examples: {len(self.items)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> InputExample:
        qid, did = self.items[idx]
        q = f"{self.query_prefix}{self.queries[qid].strip()}"
        p = self.corpus[did]
        return InputExample(texts=[q, p])


# -------------------------
# Device helpers
# -------------------------
def batch_to_device(sentence_features, labels, device: str):
    # sentence_features: List[Dict[str, Tensor]]
    for sf in sentence_features:
        for k, v in sf.items():
            if isinstance(v, torch.Tensor):
                sf[k] = v.to(device, non_blocking=True)
    if isinstance(labels, torch.Tensor):
        labels = labels.to(device, non_blocking=True)
    return sentence_features, labels


# -------------------------
# IR metrics (binary relevance)
# -------------------------
def dcg_at_k(ranked: List[str], rel: Set[str], k: int) -> float:
    s = 0.0
    for i, did in enumerate(ranked[:k]):
        if did in rel:
            s += 1.0 / math.log2(i + 2)
    return s


def ndcg_at_k(ranked: List[str], rel: Set[str], k: int) -> float:
    dcg = dcg_at_k(ranked, rel, k)
    ideal = dcg_at_k(list(rel), rel, k)
    return 0.0 if ideal == 0 else dcg / ideal


def recall_at_k(ranked: List[str], rel: Set[str], k: int) -> float:
    return 1.0 if any(d in rel for d in ranked[:k]) else 0.0


@torch.no_grad()
def retrieve_topk_chunked(
    q_emb: torch.Tensor,     # [Q, D] on GPU
    doc_emb: torch.Tensor,   # [N, D] on GPU
    doc_ids: List[str],
    top_k: int,
    chunk_size: int,
) -> List[List[str]]:
    # dtype alignment
    if q_emb.dtype != doc_emb.dtype:
        q_emb = q_emb.to(dtype=doc_emb.dtype)

    N = doc_emb.shape[0]
    neg_inf = torch.finfo(doc_emb.dtype).min
    best_scores = torch.full((q_emb.size(0), top_k), neg_inf, device=doc_emb.device, dtype=doc_emb.dtype)
    best_idx = torch.full((q_emb.size(0), top_k), -1, device=doc_emb.device, dtype=torch.long)

    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)
        chunk = doc_emb[start:end]  # [c, D]
        sims = torch.matmul(q_emb, chunk.T)  # [Q, c]
        vals, idx = torch.topk(sims, k=min(top_k, sims.size(1)), dim=1)
        idx = idx + start

        cat_vals = torch.cat([best_scores, vals], dim=1)
        cat_idx = torch.cat([best_idx, idx], dim=1)
        new_vals, new_pos = torch.topk(cat_vals, k=top_k, dim=1)
        new_idx = torch.gather(cat_idx, 1, new_pos)
        best_scores, best_idx = new_vals, new_idx

    best_idx_cpu = best_idx.detach().cpu().numpy()
    out: List[List[str]] = []
    for qi in range(best_idx_cpu.shape[0]):
        out.append([doc_ids[int(j)] for j in best_idx_cpu[qi].tolist()])
    return out


# -------------------------
# Evaluator (multi-collection, weighted avg)
# -------------------------
@dataclass
class EvalCollection:
    name: str
    corpus: Dict[str, str]
    queries: Dict[str, str]
    qrels: Dict[str, Set[str]]


class HumanMultiDomainEvaluator:
    def __init__(
        self,
        collections: List[EvalCollection],
        *,
        compression_ratio: float,
        max_len: int,
        doc_bs: int,
        query_bs: int,
        top_k_eval: int,
        chunk_size: int,
        device: str,
        metrics_path: Optional[str],
        k_list=(1, 3, 5, 10),
    ):
        self.collections = collections
        self.compression_ratio = float(compression_ratio)
        self.max_len = int(max_len)
        self.doc_bs = int(doc_bs)
        self.query_bs = int(query_bs)
        self.top_k_eval = int(top_k_eval)
        self.chunk_size = int(chunk_size)
        self.device = device
        self.metrics_path = metrics_path
        self.k_list = list(k_list)

    @torch.no_grad()
    def __call__(self, model: SentenceTransformer, epoch: int, steps: int) -> float:
        t0 = time.time()
        per = []

        for col in self.collections:
            # filter qids that have qrels
            qids = [qid for qid in col.queries.keys() if qid in col.qrels and len(col.qrels[qid]) > 0]
            if not qids:
                continue

            print(f"\n[EVAL] collection={col.name} (dev_q={len(qids)})")

            doc_ids = list(col.corpus.keys())
            doc_texts = [col.corpus[d] for d in doc_ids]

            # Encode docs on GPU (float16)
            doc_emb = model.encode(
                doc_texts,
                batch_size=self.doc_bs,
                show_progress_bar=True,
                convert_to_tensor=True,
                normalize_embeddings=True,
                prompt_name=None,
                compression_ratio=self.compression_ratio,
            )
            if doc_emb.device.type != "cuda" and self.device.startswith("cuda"):
                doc_emb = doc_emb.to(self.device)
            doc_emb = doc_emb.to(torch.float16)

            qtexts = [col.queries[qid] for qid in qids]
            q_emb = model.encode(
                qtexts,
                batch_size=self.query_bs,
                show_progress_bar=True,
                convert_to_tensor=True,
                normalize_embeddings=True,
                prompt_name="query",
                compression_ratio=self.compression_ratio,
            )
            if q_emb.device.type != "cuda" and self.device.startswith("cuda"):
                q_emb = q_emb.to(self.device)
            q_emb = q_emb.to(torch.float16)

            ranked = retrieve_topk_chunked(
                q_emb=q_emb,
                doc_emb=doc_emb,
                doc_ids=doc_ids,
                top_k=self.top_k_eval,
                chunk_size=self.chunk_size,
            )

            recall = {k: [] for k in self.k_list}
            ndcg = {k: [] for k in self.k_list}
            for qi, qid in enumerate(qids):
                rel = col.qrels[qid]
                rlist = ranked[qi]
                for k in self.k_list:
                    recall[k].append(recall_at_k(rlist, rel, k))
                    ndcg[k].append(ndcg_at_k(rlist, rel, k))

            out = {
                "name": col.name,
                "count": len(qids),
                "Recall": [float(np.mean(recall[k])) for k in self.k_list],
                "nDCG": [float(np.mean(ndcg[k])) for k in self.k_list],
            }
            print(f"[EVAL] {col.name} Recall@{self.k_list}={out['Recall']}  nDCG@{self.k_list}={out['nDCG']}  count={out['count']}")
            per.append(out)

            # free GPU memory between collections (important)
            del doc_emb, q_emb
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # weighted average by #queries
        total = sum(x["count"] for x in per)
        if total == 0:
            wR = [0.0 for _ in self.k_list]
            wN = [0.0 for _ in self.k_list]
        else:
            wR = [0.0 for _ in self.k_list]
            wN = [0.0 for _ in self.k_list]
            for x in per:
                w = x["count"] / total
                for i in range(len(self.k_list)):
                    wR[i] += w * x["Recall"][i]
                    wN[i] += w * x["nDCG"][i]

        dur = time.time() - t0
        print(f"\n[WEIGHTED] Recall@{self.k_list}={wR}")
        print(f"[WEIGHTED] nDCG@{self.k_list}={wN}")

        summary = {
            "epoch": int(epoch),
            "steps": int(steps),
            "k_list": self.k_list,
            "weighted_recall": wR,
            "weighted_ndcg": wN,
            "duration_sec": float(dur),
            "per_collection": per,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if self.metrics_path:
            Path(self.metrics_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(summary, ensure_ascii=False) + "\n")

        # Return nDCG@10 if k_list contains 10, else last
        if 10 in self.k_list:
            idx = self.k_list.index(10)
            return wN[idx]
        return wN[-1]


# -------------------------
# Main
# -------------------------
@dataclass
class CollectionCfg:
    name: str
    root: str
    corpus_file: str
    query_file_tpl: str  # contains {task}
    qrels_dir: str


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, choices=["lastturn", "questions", "rewrite"], required=True)
    ap.add_argument("--model", type=str, default="Jasper-Token-Compression-600M")
    ap.add_argument("--base_remote_code_dir", type=str, default="Jasper-Token-Compression-600M")
    ap.add_argument("--output", type=str, required=True)

    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--compression_ratio", type=float, default=0.3333)
    ap.add_argument("--fix_mistral_regex", action="store_true", default=True)

    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--max_pos_per_query", type=int, default=2)

    # qrels split policy
    ap.add_argument("--prefer_train_qrels", action="store_true",
                    help="If qrels/train.tsv exists, use it for training; else split dev.tsv by query.")
    ap.add_argument("--dev_ratio_if_no_train", type=float, default=0.1)

    # eval
    ap.add_argument("--eval_doc_bs", type=int, default=256)
    ap.add_argument("--eval_query_bs", type=int, default=256)
    ap.add_argument("--eval_topk", type=int, default=10)
    ap.add_argument("--eval_chunk_size", type=int, default=50000)
    ap.add_argument("--eval_k_list", type=str, default="1,3,5,10")

    # misc
    ap.add_argument("--freeze_last_layer_only", action="store_true")
    ap.add_argument("--no_tee", action="store_true", help="Disable tee logging to output/run.log")

    args = ap.parse_args()

    if not args.no_tee:
        setup_tee(args.output)

    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Collections (match repo structure)
    COLLECTIONS = [
        CollectionCfg("clapnq", "human/retrieval_tasks/clapnq", "clapnq.jsonl", "clapnq_{task}.jsonl", "qrels"),
        CollectionCfg("fiqa",   "human/retrieval_tasks/fiqa",   "fiqa.jsonl",   "fiqa_{task}.jsonl",   "qrels"),
        CollectionCfg("govt",   "human/retrieval_tasks/govt",   "govt.jsonl",   "govt_{task}.jsonl",   "qrels"),
        CollectionCfg("cloud",  "human/retrieval_tasks/cloud",  "cloud.jsonl",  "cloud_{task}.jsonl",  "qrels"),
    ]

    tokenizer_kwargs = {"padding_side": "left", "trust_remote_code": True}
    if args.fix_mistral_regex:
        tokenizer_kwargs["fix_mistral_regex"] = True

    model = SentenceTransformer(
        args.model,
        trust_remote_code=True,
        device=device,
        tokenizer_kwargs=tokenizer_kwargs,
        model_kwargs={"trust_remote_code": True},
    )
    model.max_seq_length = args.max_len

    set_default_compression_ratio(model, args.compression_ratio)

    query_prefix = get_query_prompt_prefix(model, "query")
    if query_prefix:
        print(f"[PROMPT] Found query prefix len={len(query_prefix)} head={repr(query_prefix[:80])}")
    else:
        print("[PROMPT] WARNING: No query prefix found. Training will use raw queries.")

    if args.freeze_last_layer_only:
        freeze_encoder_except_last_layer(model, unfreeze_final_norm=True)

    # Build training examples (pairs) across all collections
    all_train_examples: List[InputExample] = []
    eval_collections: List[EvalCollection] = []

    for cfg in COLLECTIONS:
        corpus_path = os.path.join(cfg.root, cfg.corpus_file)
        query_path = os.path.join(cfg.root, cfg.query_file_tpl.format(task=args.task))
        qrels_train_path = os.path.join(cfg.root, cfg.qrels_dir, "train.tsv")
        qrels_dev_path = os.path.join(cfg.root, cfg.qrels_dir, "dev.tsv")

        print(f"\n==== Load collection: {cfg.name} ====")
        print("Corpus:", corpus_path)
        print("Queries:", query_path)
        print("Qrels(dev):", qrels_dev_path)

        corpus = load_corpus_jsonl(corpus_path)
        queries = load_queries_jsonl(query_path)
        qrels_dev = load_qrels_tsv(qrels_dev_path)

        # Choose training qrels
        if args.prefer_train_qrels and os.path.exists(qrels_train_path):
            qrels_train = load_qrels_tsv(qrels_train_path)
            print("Qrels(train):", qrels_train_path)
        else:
            qrels_train, qrels_dev_split = split_qrels_by_query(qrels_dev, args.dev_ratio_if_no_train, args.seed)
            qrels_dev = qrels_dev_split
            print(f"[SPLIT] no train.tsv -> split dev.tsv by query, train_q={len(qrels_train)} dev_q={len(qrels_dev)}")

        # filter qrels to those that exist in query+corpus
        q_train = {qid: queries[qid] for qid in qrels_train.keys() if qid in queries}
        q_dev = {qid: queries[qid] for qid in qrels_dev.keys() if qid in queries}

        # training dataset for this collection
        ds = PairQrelsDataset(
            queries=q_train,
            corpus=corpus,
            qrels_pos=qrels_train,
            query_prefix=query_prefix,
            max_pos_per_query=args.max_pos_per_query,
            seed=args.seed,
        )
        # materialize into list (simple + stable)
        for i in range(len(ds)):
            all_train_examples.append(ds[i])

        eval_collections.append(EvalCollection(cfg.name, corpus=corpus, queries=q_dev, qrels=qrels_dev))
        print(f"[COLLECT] {cfg.name}: train_examples+={len(ds)} dev_queries={len(q_dev)}")

    if not all_train_examples:
        raise RuntimeError("No training examples built. Check query/qrels alignment.")

    random.shuffle(all_train_examples)
    print(f"\n[TRAIN] total examples: {len(all_train_examples)}")

    # Loss (MNRL for pair training)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # DataLoader
    train_loader = DataLoader(
        all_train_examples,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        collate_fn=model.smart_batching_collate,  # IMPORTANT
    )

    # Evaluator
    k_list = [int(x) for x in args.eval_k_list.split(",") if x.strip()]
    metrics_path = str(Path(args.output) / "eval_metrics.jsonl")
    evaluator = HumanMultiDomainEvaluator(
        collections=eval_collections,
        compression_ratio=args.compression_ratio,
        max_len=args.max_len,
        doc_bs=args.eval_doc_bs,
        query_bs=args.eval_query_bs,
        top_k_eval=args.eval_topk,
        chunk_size=args.eval_chunk_size,
        device=device,
        metrics_path=metrics_path,
        k_list=k_list,
    )
    print(f"[LOG] eval metrics -> {metrics_path}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_steps_per_epoch = len(train_loader)
    print(f"[TRAIN] steps/epoch={total_steps_per_epoch} grad_accum={args.grad_accum} effective_batch={args.batch_size * args.grad_accum}")

    best_score = -1.0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)

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
                print(f"[epoch {epoch}] step {step}/{total_steps_per_epoch} loss(avg)={running/step:.4f}")

        train_time = time.time() - t0
        print(f"\n[epoch {epoch}] done. train_time={train_time:.1f}s")

        # ---- Evaluate on VALID (official dev.tsv or split dev) ----
        model.eval()
        score = evaluator(model, epoch=epoch, steps=global_step)

        # save best
        if score > best_score:
            best_score = score
            best_dir = out_dir / "best"
            model.save(str(best_dir))
            sync_remote_code_files(args.base_remote_code_dir, str(best_dir))
            print(f"[BEST] epoch={epoch} weighted_nDCG@10={best_score:.6f} saved -> {best_dir}")

        # save epoch checkpoint
        ep_dir = out_dir / f"epoch{epoch}"
        model.save(str(ep_dir))
        sync_remote_code_files(args.base_remote_code_dir, str(ep_dir))
        print(f"[SAVE] epoch checkpoint -> {ep_dir}")

    print(f"\nDone. Best weighted nDCG@10 = {best_score:.6f} (see {out_dir/'best'})")
    print(f"[LOG] full stdout/stderr saved in: {out_dir/'run.log'}")
    print(f"[LOG] eval metrics jsonl saved in: {out_dir/'eval_metrics.jsonl'}")


if __name__ == "__main__":
    main()
