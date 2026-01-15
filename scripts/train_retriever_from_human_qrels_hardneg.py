# train_retriever_from_human_qrels_hardneg.py
# Usage:
# python train_retriever_from_human_qrels_hardneg.py \
#   --task lastturn \
#   --model Jasper-Token-Compression-600M \
#   --base_remote_code_dir Jasper-Token-Compression-600M \
#   --output jasper-ft-human-lastturn \
#   --epochs 1 --lr 2e-6 --batch_size 8 --grad_accum 4 \
#   --loss triplet \
#   --mine_topk 50 --neg_per_pos 2 --max_pos_per_query 2 \
#   --compression_ratio 0.3333 --freeze_last_layer_only --prefer_train_qrels

import argparse
import json
import math
import os
import random
import time
import hashlib
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
# Query prompt prefix (make training match eval that uses prompt_name="query")
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
# Move batch to device (FIX: cpu vs cuda crash)
# -------------------------
def move_batch_to_device(sentence_features, labels, device: str):
    # sentence_features: List[Dict[str, Tensor]]
    for sf in sentence_features:
        for k, v in sf.items():
            if torch.is_tensor(v):
                sf[k] = v.to(device, non_blocking=True)
    if torch.is_tensor(labels):
        labels = labels.to(device, non_blocking=True)
    return sentence_features, labels


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

    pats = ["modeling_*.py", "configuration_*.py", "tokenization_*.py"]
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
    corpus = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
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
    qs = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = obj.get("task_id") or obj.get("_id") or obj.get("query_id")
            txt = (obj.get("text") or "").strip()
            if qid is not None and txt:
                qs[str(qid)] = txt
    return qs


def load_qrels_tsv(path: str) -> Dict[str, Set[str]]:
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


def train_dev_split_by_query(
    qrels: Dict[str, Set[str]],
    dev_ratio: float,
    seed: int
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    qids = list(qrels.keys())
    rnd = random.Random(seed)
    rnd.shuffle(qids)
    n_dev = max(1, int(len(qids) * dev_ratio))
    dev_qids = set(qids[:n_dev])
    train = {qid: qrels[qid] for qid in qids if qid not in dev_qids}
    dev = {qid: qrels[qid] for qid in qids if qid in dev_qids}
    return train, dev


# -------------------------
# Embedding cache for mining (docs)
# -------------------------
def _safe_tag(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-+" else "_" for ch in s)


def _cache_key(prefix: str, collection: str, model_tag: str, max_len: int, cr: float, file_path: str) -> str:
    st = os.stat(file_path)
    sig = f"{prefix}|{collection}|{model_tag}|len={max_len}|cr={cr}|size={st.st_size}|mtime={int(st.st_mtime)}"
    h = hashlib.sha1(sig.encode("utf-8")).hexdigest()[:16]
    return f"{_safe_tag(collection)}__{_safe_tag(model_tag)}__len{max_len}__cr{cr:.4f}__{h}"


def encode_docs_st(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int,
) -> torch.Tensor:
    """
    Encode docs -> CPU float16 normalized tensor.
    """
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    return embs.detach().to("cpu", dtype=torch.float16).contiguous()


def get_or_build_doc_emb_for_mining(
    cache_dir: str,
    collection_name: str,
    model_tag: str,
    corpus_path: str,
    doc_ids: List[str],
    doc_texts: List[str],
    model: SentenceTransformer,
    max_len: int,
    compression_ratio: float,
    bs: int,
    force: bool = False,
) -> torch.Tensor:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    key = _cache_key("mine_docs", collection_name, model_tag, max_len, compression_ratio, corpus_path)
    p = Path(cache_dir) / (key + ".pt")

    if p.exists() and (not force):
        pack = torch.load(str(p), map_location="cpu")
        if pack.get("doc_ids") == doc_ids:
            print(f"[MINE CACHE HIT] {p}")
            return pack["doc_emb"]
        print("[MINE CACHE STALE] doc_ids mismatch -> recompute")

    print(f"[MINE CACHE MISS] encoding docs for mining: {collection_name}")
    doc_emb = encode_docs_st(model, doc_texts, batch_size=bs)

    torch.save({"doc_ids": doc_ids, "doc_emb": doc_emb}, str(p))
    print(f"[MINE CACHE SAVE] {p} | shape={tuple(doc_emb.shape)}")
    return doc_emb


# -------------------------
# Hard negative mining (chunked CPU->GPU, dtype-safe)
# -------------------------
@torch.no_grad()
def mine_hard_negs(
    *,
    model: SentenceTransformer,
    queries: Dict[str, str],
    qrels_pos: Dict[str, Set[str]],
    doc_ids: List[str],
    doc_emb_cpu_f16: torch.Tensor,  # CPU float16 [N, D]
    topk: int,
    per_query: int,
    query_bs: int,
    chunk_size: int,
    device: str,
) -> Dict[str, List[str]]:
    """
    For each query, retrieve topk docs and pick first `per_query` that are NOT positives.
    """
    qids = [qid for qid in queries.keys() if qid in qrels_pos and len(qrels_pos[qid]) > 0]
    qtexts = [queries[qid] for qid in qids]

    q_emb = model.encode(
        qtexts,
        batch_size=query_bs,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,
    ).to(device)

    # Match dtype (FIX: float != half)
    q_emb = q_emb.to(dtype=doc_emb_cpu_f16.dtype)

    N = doc_emb_cpu_f16.shape[0]
    K = max(topk, per_query)

    neg_inf = torch.finfo(doc_emb_cpu_f16.dtype).min
    best_scores = torch.full((q_emb.size(0), K), neg_inf, device=device, dtype=doc_emb_cpu_f16.dtype)
    best_idx = torch.full((q_emb.size(0), K), -1, device=device, dtype=torch.long)

    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)

        # stream chunk to GPU (avoid moving full doc_emb to GPU)
        chunk = doc_emb_cpu_f16[start:end].to(device, non_blocking=True)  # [c, D] half
        sims = torch.matmul(q_emb, chunk.T)  # [Q, c]
        vals, idx = torch.topk(sims, k=min(K, sims.size(1)), dim=1)
        idx = idx + start

        cat_vals = torch.cat([best_scores, vals], dim=1)
        cat_idx = torch.cat([best_idx, idx], dim=1)
        new_vals, new_pos = torch.topk(cat_vals, k=K, dim=1)
        new_idx = torch.gather(cat_idx, 1, new_pos)

        best_scores, best_idx = new_vals, new_idx

    best_idx_cpu = best_idx.detach().cpu().numpy()

    negs: Dict[str, List[str]] = {}
    for qi, qid in enumerate(qids):
        pos = qrels_pos[qid]
        picked = []
        for j in best_idx_cpu[qi].tolist():
            did = doc_ids[int(j)]
            if did in pos:
                continue
            picked.append(did)
            if len(picked) >= per_query:
                break
        negs[qid] = picked

    return negs


# -------------------------
# Training dataset
# -------------------------
class TripletQrelsDataset(Dataset):
    def __init__(
        self,
        queries: Dict[str, str],
        corpus: Dict[str, str],
        qrels_pos: Dict[str, Set[str]],
        hard_negs: Dict[str, List[str]],
        max_pos_per_query: int,
        seed: int = 42,
    ):
        self.queries = queries
        self.corpus = corpus
        self.qrels_pos = qrels_pos
        self.hard_negs = hard_negs
        self.max_pos_per_query = max_pos_per_query
        self.rng = random.Random(seed)

        items = []
        for qid, pos_set in qrels_pos.items():
            if qid not in queries:
                continue
            pos_list = [d for d in pos_set if d in corpus]
            if not pos_list:
                continue

            self.rng.shuffle(pos_list)
            pos_list = pos_list[:max_pos_per_query]

            neg_list = [d for d in hard_negs.get(qid, []) if d in corpus]
            if not neg_list:
                continue

            for pd in pos_list:
                items.append((qid, pd))
        self.items = items
        print(f"[DATASET] triplet anchor-positive pairs: {len(self.items)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> InputExample:
        qid, pos_did = self.items[idx]
        q = self.queries[qid]
        pos = self.corpus[pos_did]
        neg_candidates = self.hard_negs[qid]
        neg_did = neg_candidates[self.rng.randrange(len(neg_candidates))]
        neg = self.corpus[neg_did]
        return InputExample(texts=[q, pos, neg])


# -------------------------
# Metrics (official-like)
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
    q_emb: torch.Tensor,            # [Q, D] on GPU
    doc_emb_cpu_f16: torch.Tensor,  # [N, D] CPU float16
    doc_ids: List[str],
    top_k: int,
    chunk_size: int,
    device: str,
) -> List[List[str]]:
    # match dtype
    q_emb = q_emb.to(dtype=doc_emb_cpu_f16.dtype)

    N = doc_emb_cpu_f16.shape[0]
    neg_inf = torch.finfo(doc_emb_cpu_f16.dtype).min
    best_scores = torch.full((q_emb.size(0), top_k), neg_inf, device=device, dtype=doc_emb_cpu_f16.dtype)
    best_idx = torch.full((q_emb.size(0), top_k), -1, device=device, dtype=torch.long)

    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)
        chunk = doc_emb_cpu_f16[start:end].to(device, non_blocking=True)
        sims = torch.matmul(q_emb, chunk.T)
        vals, idx = torch.topk(sims, k=min(top_k, sims.size(1)), dim=1)
        idx = idx + start

        cat_vals = torch.cat([best_scores, vals], dim=1)
        cat_idx = torch.cat([best_idx, idx], dim=1)
        new_vals, new_pos = torch.topk(cat_vals, k=top_k, dim=1)
        new_idx = torch.gather(cat_idx, 1, new_pos)

        best_scores, best_idx = new_vals, new_idx

    best_idx_cpu = best_idx.detach().cpu().numpy()
    out = []
    for qi in range(best_idx_cpu.shape[0]):
        out.append([doc_ids[int(j)] for j in best_idx_cpu[qi].tolist()])
    return out


@torch.no_grad()
def eval_one_collection(
    model: SentenceTransformer,
    corpus: Dict[str, str],
    queries: Dict[str, str],
    qrels_dev: Dict[str, Set[str]],
    doc_bs: int,
    query_bs: int,
    top_k_eval: int,
    chunk_size: int,
    device: str,
) -> Dict[str, List[float]]:
    doc_ids = list(corpus.keys())
    doc_texts = [corpus[d] for d in doc_ids]

    doc_emb = model.encode(
        doc_texts,
        batch_size=doc_bs,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,
    ).detach().to("cpu", dtype=torch.float16).contiguous()

    qids = [qid for qid in queries.keys() if qid in qrels_dev and len(qrels_dev[qid]) > 0]
    qtexts = [queries[qid] for qid in qids]

    q_emb = model.encode(
        qtexts,
        batch_size=query_bs,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,
    ).to(device)

    ranked = retrieve_topk_chunked(
        q_emb=q_emb,
        doc_emb_cpu_f16=doc_emb,
        doc_ids=doc_ids,
        top_k=top_k_eval,
        chunk_size=chunk_size,
        device=device,
    )

    ks = [1, 3, 5, 10]
    recall = {k: [] for k in ks}
    ndcg = {k: [] for k in ks}
    for qi, qid in enumerate(qids):
        rel = qrels_dev[qid]
        rlist = ranked[qi]
        for k in ks:
            recall[k].append(recall_at_k(rlist, rel, k))
            ndcg[k].append(ndcg_at_k(rlist, rel, k))

    return {
        "Recall": [float(np.mean(recall[k])) for k in ks],
        "nDCG": [float(np.mean(ndcg[k])) for k in ks],
        "count": len(qids),
    }


def weighted_avg(scores: List[Dict[str, List[float]]], key: str) -> List[float]:
    total = sum(s["count"] for s in scores)
    if total == 0:
        return [0.0, 0.0, 0.0, 0.0]
    out = [0.0, 0.0, 0.0, 0.0]
    for s in scores:
        w = s["count"] / total
        for i in range(4):
            out[i] += w * s[key][i]
    return out


# -------------------------
# Main
# -------------------------
@dataclass
class CollectionCfg:
    name: str
    root: str
    corpus_file: str
    query_file_tpl: str
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

    ap.add_argument("--loss", type=str, choices=["triplet", "mnrl"], default="triplet")
    ap.add_argument("--freeze_last_layer_only", action="store_true")

    # mining
    ap.add_argument("--mine_topk", type=int, default=50)
    ap.add_argument("--neg_per_pos", type=int, default=2)
    ap.add_argument("--max_pos_per_query", type=int, default=2)
    ap.add_argument("--mine_doc_bs", type=int, default=256)
    ap.add_argument("--mine_query_bs", type=int, default=256)
    ap.add_argument("--mine_chunk_size", type=int, default=50000)
    ap.add_argument("--mine_cache_dir", type=str, default="cache/mine_doc_emb")

    # eval
    ap.add_argument("--eval_doc_bs", type=int, default=256)
    ap.add_argument("--eval_query_bs", type=int, default=256)
    ap.add_argument("--eval_topk", type=int, default=10)
    ap.add_argument("--eval_chunk_size", type=int, default=50000)

    # qrels
    ap.add_argument("--prefer_train_qrels", action="store_true")
    ap.add_argument("--dev_ratio_if_no_train", type=float, default=0.1)

    args = ap.parse_args()
    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

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
    model.to(device)

    set_default_compression_ratio(model, args.compression_ratio)

    # IMPORTANT: make training match official query prompting
    query_prefix = get_query_prompt_prefix(model, "query")
    if query_prefix:
        print(f"[PROMPT] Found query prefix len={len(query_prefix)} head={repr(query_prefix[:80])}")
    else:
        print("[PROMPT] WARNING: No query prefix found; using raw queries.")

    if args.freeze_last_layer_only:
        freeze_encoder_except_last_layer(model, unfreeze_final_norm=True)

    # -------------------------
    # Load all collections + mine negatives
    # -------------------------
    train_examples: List[InputExample] = []
    eval_payload = []

    model_tag = _safe_tag(Path(args.model).name if os.path.isdir(args.model) else args.model)

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
        queries_raw = load_queries_jsonl(query_path)
        qrels_dev = load_qrels_tsv(qrels_dev_path)

        if args.prefer_train_qrels and os.path.exists(qrels_train_path):
            qrels_train = load_qrels_tsv(qrels_train_path)
            print("Qrels(train):", qrels_train_path)
        else:
            qrels_train, qrels_dev_split = train_dev_split_by_query(qrels_dev, args.dev_ratio_if_no_train, args.seed)
            qrels_dev = qrels_dev_split
            print(f"[SPLIT] no train.tsv -> split dev.tsv by query, train_q={len(qrels_train)} dev_q={len(qrels_dev)}")

        # Apply query prefix here (train + dev + mining + eval all consistent)
        queries = {qid: (query_prefix + txt) for qid, txt in queries_raw.items()}

        q_train = {qid: queries[qid] for qid in queries.keys() if qid in qrels_train}
        q_dev = {qid: queries[qid] for qid in queries.keys() if qid in qrels_dev}

        doc_ids = list(corpus.keys())
        doc_texts = [corpus[d] for d in doc_ids]

        doc_emb_mine = get_or_build_doc_emb_for_mining(
            cache_dir=args.mine_cache_dir,
            collection_name=cfg.name,
            model_tag=model_tag,
            corpus_path=corpus_path,
            doc_ids=doc_ids,
            doc_texts=doc_texts,
            model=model,
            max_len=args.max_len,
            compression_ratio=args.compression_ratio,
            bs=args.mine_doc_bs,
            force=False,
        )

        hard_negs = mine_hard_negs(
            model=model,
            queries=q_train,
            qrels_pos=qrels_train,
            doc_ids=doc_ids,
            doc_emb_cpu_f16=doc_emb_mine,
            topk=args.mine_topk,
            per_query=max(args.neg_per_pos, 1),
            query_bs=args.mine_query_bs,
            chunk_size=args.mine_chunk_size,
            device=device,
        )

        ds = TripletQrelsDataset(
            queries=q_train,
            corpus=corpus,
            qrels_pos=qrels_train,
            hard_negs=hard_negs,
            max_pos_per_query=args.max_pos_per_query,
            seed=args.seed,
        )

        for i in range(len(ds)):
            train_examples.append(ds[i])

        eval_payload.append((cfg.name, corpus, q_dev, qrels_dev))
        print(f"[COLLECT] {cfg.name}: train_examples+={len(ds)} dev_queries={len(q_dev)}")

    if not train_examples:
        raise RuntimeError("No training examples built. Check qrels/query alignment.")

    random.shuffle(train_examples)
    print(f"\n[TRAIN] total examples: {len(train_examples)}")

    # -------------------------
    # Loss
    # -------------------------
    if args.loss == "triplet":
        train_loss = losses.TripletLoss(
            model=model,
            distance_metric=losses.TripletDistanceMetric.COSINE,
            triplet_margin=0.2,
        )
    else:
        # MNRL uses pairs only
        pairs = [InputExample(texts=[ex.texts[0], ex.texts[1]]) for ex in train_examples]
        train_examples = pairs
        train_loss = losses.MultipleNegativesRankingLoss(model)

    # -------------------------
    # DataLoader (FIX: InputExample collate)
    # -------------------------
    train_loader = DataLoader(
        train_examples,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        collate_fn=model.smart_batching_collate,
    )

    # -------------------------
    # Optimizer
    # -------------------------
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    total_steps_per_epoch = len(train_loader)
    print(f"[TRAIN] steps/epoch={total_steps_per_epoch} grad_accum={args.grad_accum} effective_batch={args.batch_size * args.grad_accum}")

    # -------------------------
    # Train + eval + save best
    # -------------------------
    best_ndcg10 = -1.0
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    model.train()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, (sentence_features, labels) in enumerate(train_loader, start=1):
            # FIX: cpu -> gpu
            sentence_features, labels = move_batch_to_device(sentence_features, labels, device)

            loss = train_loss(sentence_features, labels)
            loss = loss / args.grad_accum
            loss.backward()
            running += float(loss.item())

            if step % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if step % 50 == 0:
                print(f"[epoch {epoch}] step {step}/{total_steps_per_epoch} loss(avg)={running/step:.4f}")

        train_time = time.time() - t0
        print(f"\n[epoch {epoch}] done. train_time={train_time:.1f}s")

        # ---- Evaluate on human dev ----
        model.eval()
        per_scores = []
        for (cname, corpus, q_dev, qrels_dev) in eval_payload:
            if len(q_dev) == 0:
                continue
            print(f"\n[EVAL] collection={cname} (dev_q={len(q_dev)})")
            s = eval_one_collection(
                model=model,
                corpus=corpus,
                queries=q_dev,
                qrels_dev=qrels_dev,
                doc_bs=args.eval_doc_bs,
                query_bs=args.eval_query_bs,
                top_k_eval=args.eval_topk,
                chunk_size=args.eval_chunk_size,
                device=device,
            )
            print(f"[EVAL] {cname} Recall@1/3/5/10={s['Recall']}  nDCG@1/3/5/10={s['nDCG']}  count={s['count']}")
            per_scores.append(s)

        wR = weighted_avg(per_scores, "Recall")
        wN = weighted_avg(per_scores, "nDCG")
        print(f"\n[WEIGHTED] Recall@1/3/5/10={wR}")
        print(f"[WEIGHTED] nDCG@1/3/5/10={wN}")

        ndcg10 = wN[3]
        if ndcg10 > best_ndcg10:
            best_ndcg10 = ndcg10
            best_dir = out_dir / "best"
            model.save(str(best_dir))
            sync_remote_code_files(args.base_remote_code_dir, str(best_dir))
            print(f"[BEST] epoch={epoch} weighted_nDCG@10={best_ndcg10:.6f} saved -> {best_dir}")

        ep_dir = out_dir / f"epoch{epoch}"
        model.save(str(ep_dir))
        sync_remote_code_files(args.base_remote_code_dir, str(ep_dir))
        print(f"[SAVE] epoch checkpoint -> {ep_dir}")

        model.train()

    print(f"\nDone. Best weighted nDCG@10 = {best_ndcg10:.6f} (see {out_dir/'best'})")


if __name__ == "__main__":
    main()
