# train_last_turn_from_synthetic_jsonl.py
import argparse
import json
import logging
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import SentenceEvaluator


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_last_user_text(inp: List[dict]) -> str:
    # inp is a list of {"speaker": "...", "text": "..."}
    for m in reversed(inp or []):
        if (m.get("speaker") or "").lower() == "user":
            t = (m.get("text") or "").strip()
            if t:
                return t
    return ""


def read_synthetic_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_turns(rows: List[dict], topk_ctx: int = 0) -> List[dict]:
    """
    Each turn:
      {
        "conversation_id": ...,
        "task_id": ...,
        "query": last_user_text,
        "passages": [ctx_text1, ctx_text2, ...]
      }
    """
    turns = []
    for r in rows:
        conv_id = r.get("conversation_id", "")
        task_id = r.get("task_id", "")
        inp = r.get("input") or []
        ctxs = r.get("contexts") or []

        q = extract_last_user_text(inp)
        if not q:
            continue

        passages = []
        for c in ctxs:
            p = (c.get("text") or "").strip()
            if p:
                passages.append(p)

        # optional: keep only first K contexts
        if topk_ctx and topk_ctx > 0:
            passages = passages[:topk_ctx]

        # dedup passages per turn
        uniq = []
        seen = set()
        for p in passages:
            if p not in seen:
                seen.add(p)
                uniq.append(p)

        if not uniq:
            continue

        turns.append(
            {
                "conversation_id": conv_id,
                "task_id": task_id,
                "query": q,
                "passages": uniq,
            }
        )
    return turns


def split_by_conversation(turns: List[dict], dev_ratio: float, seed: int):
    conv2turns = defaultdict(list)
    for t in turns:
        conv2turns[t["conversation_id"]].append(t)

    conv_ids = list(conv2turns.keys())
    rnd = random.Random(seed)
    rnd.shuffle(conv_ids)

    n_dev = max(1, int(len(conv_ids) * dev_ratio))
    dev_convs = set(conv_ids[:n_dev])

    train_turns, dev_turns = [], []
    for cid, ts in conv2turns.items():
        (dev_turns if cid in dev_convs else train_turns).extend(ts)

    return train_turns, dev_turns


def build_corpus_id_map(all_turns: List[dict]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Build a global corpus from all unique passages.
    Returns:
      text2did: passage_text -> doc_id
      corpus: doc_id -> passage_text
    """
    text2did = {}
    corpus = {}
    did = 0
    for t in all_turns:
        for p in t["passages"]:
            if p in text2did:
                continue
            doc_id = f"d{did}"
            did += 1
            text2did[p] = doc_id
            corpus[doc_id] = p
    return text2did, corpus


def build_ir_eval_sets(dev_turns: List[dict], text2did: Dict[str, str]) -> Tuple[Dict[str, str], Dict[str, set]]:
    """
    queries: qid -> query_text
    relevant_docs: qid -> set(doc_ids)
    """
    queries = {}
    relevant_docs = {}

    for i, t in enumerate(dev_turns):
        qid = f"q{i}"
        queries[qid] = t["query"]
        rel = set()
        for p in t["passages"]:
            did = text2did.get(p)
            if did is not None:
                rel.add(did)
        relevant_docs[qid] = rel

    return queries, relevant_docs


def compute_mrr(ranked: List[str], rel: set) -> float:
    for i, d in enumerate(ranked):
        if d in rel:
            return 1.0 / (i + 1)
    return 0.0


def compute_recall_at_k(ranked: List[str], rel: set, k: int) -> float:
    return 1.0 if any(d in rel for d in ranked[:k]) else 0.0


def compute_dcg_at_k(ranked: List[str], rel: set, k: int) -> float:
    dcg = 0.0
    for i, d in enumerate(ranked[:k]):
        if d in rel:
            dcg += 1.0 / np.log2(i + 2)
    return dcg


def compute_ndcg_at_k(ranked: List[str], rel: set, k: int) -> float:
    dcg = compute_dcg_at_k(ranked, rel, k)
    # ideal DCG: all relevant at top
    ideal_ranked = list(rel)
    ideal = compute_dcg_at_k(ideal_ranked, rel, k)
    return 0.0 if ideal == 0 else dcg / ideal


class LastTurnIREvaluator(SentenceEvaluator):
    def __init__(self, queries: Dict[str, str], corpus: Dict[str, str], relevant_docs: Dict[str, set], k_list=(1, 3, 5, 10), name="lastturn-ir"):
        self.queries = queries
        self.corpus = corpus
        self.relevant_docs = relevant_docs
        self.k_list = list(k_list)
        self.name = name

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        start = time.time()

        corpus_ids = list(self.corpus.keys())
        corpus_texts = [self.corpus[cid] for cid in corpus_ids]
        corpus_emb = model.encode(
            corpus_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        corpus_emb_T = corpus_emb.T

        query_ids = list(self.queries.keys())
        query_texts = [self.queries[qid] for qid in query_ids]
        query_emb = model.encode(
            query_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        # metrics
        mrrs = []
        recalls = {k: [] for k in self.k_list}
        ndcgs = {k: [] for k in self.k_list}

        for i, qid in enumerate(query_ids):
            sims = query_emb[i] @ corpus_emb_T
            ranked_idx = sims.argsort()[::-1]
            ranked_docs = [corpus_ids[j] for j in ranked_idx]

            rel = self.relevant_docs.get(qid, set())
            mrrs.append(compute_mrr(ranked_docs, rel))
            for k in self.k_list:
                recalls[k].append(compute_recall_at_k(ranked_docs, rel, k))
                ndcgs[k].append(compute_ndcg_at_k(ranked_docs, rel, k))

        out = {
            "epoch": float(epoch),
            "step": int(steps),
            "mrr": float(np.mean(mrrs)),
            "duration": time.time() - start,
        }
        for k in self.k_list:
            out[f"recall@{k}"] = float(np.mean(recalls[k]))
            out[f"ndcg@{k}"] = float(np.mean(ndcgs[k]))

        print(
            f"[{self.name} epoch={epoch} step={steps}] "
            + " | ".join([f"{k}={out[k]:.4f}" for k in ["mrr"] + [f"recall@{x}" for x in self.k_list] + [f"ndcg@{x}" for x in self.k_list]])
        )
        return out["ndcg@10"]  # used for save_best_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("synthetic.jsonl"))
    ap.add_argument("--model", type=str, default="Jasper-Token-Compression-600M")
    ap.add_argument("--output", type=str, default="jasper-ft-lastturn")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dev-ratio", type=float, default=0.1)
    ap.add_argument("--topk-ctx", type=int, default=0, help="Keep only first K contexts per turn; 0=all.")
    ap.add_argument("--use-all-contexts", action="store_true", help="Expand each turn into N (query, ctx) pairs. Default: 1 ctx per turn.")
    args = ap.parse_args()

    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    rows = read_synthetic_jsonl(args.data)
    turns = build_turns(rows, topk_ctx=args.topk_ctx)
    print(f"Loaded turns: {len(turns)}")

    train_turns, dev_turns = split_by_conversation(turns, dev_ratio=args.dev_ratio, seed=args.seed)
    print(f"Train turns: {len(train_turns)} | Dev turns: {len(dev_turns)}")

    # Build training samples
    rnd = random.Random(args.seed)
    train_samples: List[InputExample] = []
    if args.use_all_contexts:
        for t in train_turns:
            q = t["query"]
            for p in t["passages"]:
                train_samples.append(InputExample(texts=[q, p]))
    else:
        # one positive ctx per turn (recommended for MNRL)
        for t in train_turns:
            q = t["query"]
            p = rnd.choice(t["passages"])
            train_samples.append(InputExample(texts=[q, p]))

    rnd.shuffle(train_samples)
    print(f"Train samples: {len(train_samples)}")

    # Build global corpus for evaluation (all unique passages, train+dev)
    text2did, corpus = build_corpus_id_map(turns)
    queries, relevant_docs = build_ir_eval_sets(dev_turns, text2did)
    evaluator = LastTurnIREvaluator(queries, corpus, relevant_docs, k_list=(1, 3, 5, 10))

    model = SentenceTransformer(args.model, trust_remote_code=True, device=device)
    model.max_seq_length = args.max_len

    train_loader = DataLoader(train_samples, batch_size=args.batch_size, shuffle=True, drop_last=True)

    train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = int(len(train_loader) * args.epochs * 0.1)
    evaluation_steps = max(50, len(train_loader) // 2)  # tweak if you want

    # (optional) set optimizer params via fit() arguments
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        use_amp=False,
        output_path=args.output,
        evaluator=evaluator,
        evaluation_steps=evaluation_steps,
        save_best_model=True,
    )

    print("Finished! Best model saved to:", args.output)


if __name__ == "__main__":
    main()
