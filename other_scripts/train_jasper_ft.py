import os
import json
import random
import time
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import SentenceEvaluator

SEED = 42

K_LIST = [1, 3, 5, 10]

QUERY_PROMPT_NAME = "query"
DOC_PROMPT_NAME = None  
DATA_PATH = Path("jasper_train_pairs.jsonl")
BASE_MODEL_DIR = Path("Jasper-Token-Compression-600M")      
OUTPUT_DIR = Path("jasper-token-compression-600M-rag-ft")   

LOG_TRAIN = Path("train_loss.jsonl")
LOG_EVAL = Path("eval_metrics.jsonl")


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def dataloader_worker_init_fn(worker_id: int):
    worker_seed = SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


# ===============================
# JSONL logger
# ===============================
class JSONLLogger:
    def __init__(self, path: Path):
        self.path = path
        self.step = 0

    def log(self, obj: dict):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


loss_logger = JSONLLogger(LOG_TRAIN)
eval_logger = JSONLLogger(LOG_EVAL)



def compute_mrr(ranked, rel_set):
    for i, d in enumerate(ranked):
        if d in rel_set:
            return 1.0 / (i + 1)
    return 0.0

def compute_recall_at_k(ranked, rel_set, k: int):
    return float(any(d in rel_set for d in ranked[:k]))

def compute_ndcg_at_k(ranked, rel_set, k: int):
    dcg = 0.0
    for i, d in enumerate(ranked[:k]):
        if d in rel_set:
            dcg += 1.0 / np.log2(i + 2)
    ideal_dcg = 1.0 if len(rel_set) > 0 else 0.0
    return 0.0 if ideal_dcg == 0 else (dcg / ideal_dcg)


class JasperIREvaluator(SentenceEvaluator):
    def __init__(self, queries, corpus, rel, k_list):
        self.queries = queries
        self.corpus = corpus
        self.rel = rel
        self.k_list = list(sorted(k_list))

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        t0 = time.time()

        doc_ids = list(self.corpus.keys())
        doc_texts = [self.corpus[d] for d in doc_ids]
        doc_emb = model.encode(
            doc_texts,
            prompt_name=DOC_PROMPT_NAME,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        q_ids = list(self.queries.keys())
        q_texts = [self.queries[q] for q in q_ids]
        q_emb = model.encode(
            q_texts,
            prompt_name=QUERY_PROMPT_NAME,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        docT = doc_emb.T

        mrrs = []
        recalls = {k: [] for k in self.k_list}
        ndcgs = {k: [] for k in self.k_list}

        for i, qid in enumerate(q_ids):
            sims = q_emb[i] @ docT
            order = sims.argsort()[::-1]
            ranked_docs = [doc_ids[j] for j in order]

            rel_set = self.rel[qid]

            mrrs.append(compute_mrr(ranked_docs, rel_set))
            for k in self.k_list:
                recalls[k].append(compute_recall_at_k(ranked_docs, rel_set, k))
                ndcgs[k].append(compute_ndcg_at_k(ranked_docs, rel_set, k))

        metrics = {
            "type": "eval",
            "epoch": float(epoch),
            "step": int(steps) if steps is not None else -1,
            "mrr": float(np.mean(mrrs)),
            "duration": float(time.time() - t0),
            "time": float(time.time()),
        }

        for k in self.k_list:
            metrics[f"recall_{k}"] = float(np.mean(recalls[k]))
            metrics[f"ndcg_{k}"] = float(np.mean(ndcgs[k]))

        metrics["best_metric"] = "ndcg_5"
        metrics["best_score"] = float(metrics["ndcg_5"])

        eval_logger.log(metrics)

        parts = [f"MRR={metrics['mrr']:.4f}"]
        for k in self.k_list:
            parts.append(f"R@{k}={metrics[f'recall_{k}']:.4f}")
        for k in self.k_list:
            parts.append(f"nDCG@{k}={metrics[f'ndcg_{k}']:.4f}")
        parts.append(f"BEST=ndcg@5({metrics['best_score']:.4f})")
        print(f"[Eval epoch={epoch}] " + " | ".join(parts))

        return float(metrics["ndcg_5"])

    
def load_pairs(path: Path):
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            q = (obj.get("query") or "").strip()
            p = (obj.get("passage") or "").strip()
            if q and p:
                samples.append(InputExample(texts=[q, p]))
    return samples


def patch_output_dir(out_dir: Path, base_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    whitelist = {
        "modeling_qwen3_jasper.py",
        "custom_st.py",
        "configuration.json",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "README.md",
        "modules.json",
        "config_sentence_transformers.json",
        "sentence_bert_config.json",
    }

    for p in base_dir.iterdir():
        if p.is_file() and p.suffix == ".py":
            whitelist.add(p.name)

    for name in sorted(whitelist):
        if name == "model.safetensors":
            continue

        src = base_dir / name
        dst = out_dir / name

        if not src.exists():
            continue
        if not src.is_file():
            continue  

        if not dst.exists():
            dst.write_bytes(src.read_bytes())

    print(f"[PATCH] ensured remote-code/tokenizer files in: {out_dir}")


def should_resume(out_dir: Path) -> bool:
    return (out_dir / "modules.json").exists() and (out_dir / "config_sentence_transformers.json").exists()

def main():
    seed_everything(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if OUTPUT_DIR.exists():
        patch_output_dir(OUTPUT_DIR, BASE_MODEL_DIR)

    model_path = str(OUTPUT_DIR) if should_resume(OUTPUT_DIR) else str(BASE_MODEL_DIR)
    print("Loading model from:", model_path)

    model = SentenceTransformer(
        model_path,
        trust_remote_code=True,
        device=device,
        local_files_only=True,
    )
    model.max_seq_length = 512

    samples = load_pairs(DATA_PATH)
    if len(samples) < 10:
        raise ValueError("Too few samples to split train/dev.")

    rng = random.Random(SEED)
    rng.shuffle(samples)

    dev_size = max(1, int(0.1 * len(samples)))
    dev = samples[:dev_size]
    train = samples[dev_size:]

    g = torch.Generator()
    g.manual_seed(SEED)

    train_dataloader = DataLoader(
        train,
        batch_size=10,
        shuffle=True,
        drop_last=True,
        num_workers=0, 
        generator=g,
        worker_init_fn=dataloader_worker_init_fn,
    )

    train_loss = losses.MultipleNegativesRankingLoss(model)

    old_forward = train_loss.forward
    LOG_EVERY = 50 

    def wrapped_forward(*args, **kwargs):
        loss = old_forward(*args, **kwargs)
        if loss_logger.step % LOG_EVERY == 0:
            loss_logger.log({
                "type": "train",
                "step": int(loss_logger.step),
                "loss": float(loss.item()),
                "time": float(time.time()),
            })
        loss_logger.step += 1
        return loss

    train_loss.forward = wrapped_forward


    queries = {f"q{i}": ex.texts[0] for i, ex in enumerate(dev)}
    corpus = {f"d{i}": ex.texts[1] for i, ex in enumerate(dev)}
    rel = {f"q{i}": {f"d{i}"} for i in range(len(dev))}
    evaluator = JasperIREvaluator(queries, corpus, rel, K_LIST)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=10,
        warmup_steps=max(1, int(0.1 * len(train_dataloader))),
        use_amp=False,

        output_path=str(OUTPUT_DIR),

        evaluator=evaluator,
        evaluation_steps=0,     # ★ 每个 epoch eval 一次
        save_best_model=True,   # ★ 只保留 best

        checkpoint_path=None,
        checkpoint_save_steps=None,
    )


    patch_output_dir(OUTPUT_DIR, BASE_MODEL_DIR)

    print("Done. Best model saved to:", OUTPUT_DIR)
    print("Loss log:", LOG_TRAIN)
    print("Eval log:", LOG_EVAL)


if __name__ == "__main__":
    main()
