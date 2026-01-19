import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import SentenceEvaluator


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def set_default_compression_ratio(st_model: SentenceTransformer, ratio: float) -> None:
    """
    Ensure training forward uses the same compression setting as eval.
    We write ratio into auto_model.config under common field names.
    """
    first = st_model._first_module()
    auto_model = getattr(first, "auto_model", None)
    if auto_model is None:
        print("[CR] WARNING: auto_model not found; cannot set compression ratio in config.")
        return

    cfg = getattr(auto_model, "config", None)
    if cfg is None:
        print("[CR] WARNING: auto_model.config not found; cannot set compression ratio in config.")
        return

    # Try common names; also set attribute even if it doesn't exist (some configs allow dynamic attrs)
    names = [
        "compression_ratio",
        "default_compression_ratio",
        "token_compression_ratio",
        "jasper_compression_ratio",
    ]
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
        # still attempt to attach a generic field
        try:
            setattr(cfg, "compression_ratio", float(ratio))
            print(f"[CR] Set config field: compression_ratio = {ratio}")
        except Exception:
            print("[CR] WARNING: failed to set any compression ratio field on config.")


def extract_last_user_text(inp: List[dict]) -> str:
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
            if line:
                rows.append(json.loads(line))
    return rows


def build_turns(rows: List[dict], topk_ctx: int = 0) -> List[dict]:
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

        if topk_ctx and topk_ctx > 0:
            passages = passages[:topk_ctx]

        # dedup
        uniq, seen = [], set()
        for p in passages:
            if p not in seen:
                seen.add(p)
                uniq.append(p)

        if not uniq:
            continue

        turns.append({"conversation_id": conv_id, "task_id": task_id, "query": q, "passages": uniq})
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


class TurnDataset(Dataset):
    """
    Dynamic positive sampling; query gets prompt prefix (string) to match retrieval prompt behavior.
    """
    def __init__(self, turns: List[dict], query_prefix: str, seed: int = 42):
        self.turns = turns
        self.query_prefix = query_prefix or ""
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.turns)

    def __getitem__(self, idx: int) -> InputExample:
        t = self.turns[idx]
        q = (t["query"] or "").strip()
        q = f"{self.query_prefix}{q}"
        passages = t["passages"]
        p = passages[self.rng.randrange(len(passages))]
        return InputExample(texts=[q, p])


def build_corpus_id_map(all_turns: List[dict]) -> Tuple[Dict[str, str], Dict[str, str]]:
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


def build_ir_eval_sets(dev_turns: List[dict], text2did: Dict[str, str], query_prefix: str) -> Tuple[Dict[str, str], Dict[str, set]]:
    queries, relevant_docs = {}, {}
    for i, t in enumerate(dev_turns):
        qid = f"q{i}"
        queries[qid] = f"{query_prefix}{t['query']}"
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
    ideal = compute_dcg_at_k(list(rel), rel, k)
    return 0.0 if ideal == 0 else dcg / ideal


class LastTurnIREvaluator(SentenceEvaluator):
    def __init__(self, queries: Dict[str, str], corpus: Dict[str, str], relevant_docs: Dict[str, set],
                 compression_ratio: float, k_list=(1, 3, 5, 10), name="lastturn-ir"):
        self.queries = queries
        self.corpus = corpus
        self.relevant_docs = relevant_docs
        self.k_list = list(k_list)
        self.name = name
        self.compression_ratio = float(compression_ratio)

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        start = time.time()

        corpus_ids = list(self.corpus.keys())
        corpus_texts = [self.corpus[cid] for cid in corpus_ids]
        corpus_emb = model.encode(
            corpus_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            compression_ratio=self.compression_ratio,
        )
        corpus_emb_T = corpus_emb.T

        query_ids = list(self.queries.keys())
        query_texts = [self.queries[qid] for qid in query_ids]
        query_emb = model.encode(
            query_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            compression_ratio=self.compression_ratio,
        )

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

        out = {"epoch": float(epoch), "step": int(steps), "mrr": float(np.mean(mrrs)), "duration": time.time() - start}
        for k in self.k_list:
            out[f"recall@{k}"] = float(np.mean(recalls[k]))
            out[f"ndcg@{k}"] = float(np.mean(ndcgs[k]))

        print(f"[{self.name} epoch={epoch} step={steps}] " +
              " | ".join([f"{k}={out[k]:.4f}" for k in ["mrr"] + [f"recall@{x}" for x in self.k_list] + [f"ndcg@{x}" for x in self.k_list]]))
        return out["ndcg@10"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model", type=str, default="Jasper-Token-Compression-600M")
    ap.add_argument("--output", type=str, default="jasper-ft-lastturn")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dev-ratio", type=float, default=0.1)
    ap.add_argument("--topk-ctx", type=int, default=3)
    ap.add_argument("--freeze_last_layer_only", action="store_true")
    ap.add_argument("--compression-ratio", type=float, default=0.3333)
    ap.add_argument("--fix_mistral_regex", action="store_true", default=True)
    ap.add_argument("--save_last", action="store_true", help="Also save final weights to <output>-last")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

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

    # IMPORTANT: align default compression ratio for TRAIN forward
    set_default_compression_ratio(model, args.compression_ratio)

    query_prefix = get_query_prompt_prefix(model, "query")
    if query_prefix:
        print(f"[PROMPT] Found query prompt prefix (len={len(query_prefix)}). head={repr(query_prefix[:60])}")
    else:
        print("[PROMPT] WARNING: No query prompt prefix found. Training will use raw last-turn queries.")

    if args.freeze_last_layer_only:
        freeze_encoder_except_last_layer(model, unfreeze_final_norm=True)

    rows = read_synthetic_jsonl(args.data)
    turns = build_turns(rows, topk_ctx=args.topk_ctx)
    print(f"Loaded turns: {len(turns)}")

    train_turns, dev_turns = split_by_conversation(turns, dev_ratio=args.dev_ratio, seed=args.seed)
    print(f"Train turns: {len(train_turns)} | Dev turns: {len(dev_turns)}")

    train_dataset = TurnDataset(train_turns, query_prefix=query_prefix, seed=args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)

    text2did, corpus = build_corpus_id_map(turns)
    queries, relevant_docs = build_ir_eval_sets(dev_turns, text2did, query_prefix=query_prefix)
    evaluator = LastTurnIREvaluator(queries, corpus, relevant_docs, compression_ratio=args.compression_ratio)

    train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = int(len(train_loader) * args.epochs * 0.1)
    evaluation_steps = max(50, len(train_loader) // 2)

    fit_kwargs = dict(
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

    # gradient accumulation may not exist in older ST versions; try safely
    try:
        model.fit(**fit_kwargs, gradient_accumulation_steps=args.gradient_accumulation_steps)
    except TypeError:
        if args.gradient_accumulation_steps != 1:
            print("[WARN] sentence-transformers.fit() doesn't support gradient_accumulation_steps in this version. Ignoring.")
        model.fit(**fit_kwargs)

    print("Finished! Best model saved to:", args.output)

    if args.save_last:
        out_last = args.output + "-last"
        model.save(out_last)
        print("Also saved LAST model to:", out_last)


if __name__ == "__main__":
    main()
