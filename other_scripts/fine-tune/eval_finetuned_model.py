import os, json, time, hashlib
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def _safe_tag(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-+" else "_" for ch in s)

def _cache_key(model_tag: str, collection_name: str, max_len: int, compression_ratio: float, corpus_path: str) -> str:
    st = os.stat(corpus_path)
    sig = f"{model_tag}|{collection_name}|len={max_len}|cr={compression_ratio}|size={st.st_size}|mtime={int(st.st_mtime)}"
    h = hashlib.sha1(sig.encode("utf-8")).hexdigest()[:16]
    return f"{_safe_tag(collection_name)}__{_safe_tag(model_tag)}__len{max_len}__cr{compression_ratio:.4f}__{h}"

def load_corpus(corpus_path):
    corpus = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            doc_id = item["document_id"] if "document_id" in item else item["_id"]
            title = item.get("title", "")
            text = item.get("text", "")
            corpus[doc_id] = {
                "title": title,
                "text": text,
            }
    return corpus

def load_queries(query_path):
    queries = {}
    with open(query_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            qid = item.get("_id") or item.get("query_id") or item["task_id"]
            text = item["text"]
            queries[qid] = text
    return queries


def get_or_build_doc_embeddings(
    *,
    model,
    doc_texts,
    doc_ids,
    cache_dir: str,
    model_tag: str,
    collection_name: str,
    max_len: int,
    compression_ratio: float,
    corpus_path: str,
    batch_size: int = 256,
    force_recompute: bool = False,
    device: str = "cuda",
):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    key = _cache_key(model_tag, collection_name, max_len, compression_ratio, corpus_path)
    cache_path = os.path.join(cache_dir, key + ".pt")

    if (not force_recompute) and os.path.exists(cache_path):
        pack = torch.load(cache_path, map_location="cpu")
        if pack["doc_ids"] == doc_ids:
            print(f"[CACHE HIT] Loading doc embeddings: {cache_path}")
            doc_emb = pack["doc_emb"]  # CPU
            return pack["doc_ids"], doc_emb
        else:
            print("[CACHE STALE] Doc ids mismatch -> recompute")

    print(f"[CACHE MISS] Encoding docs for {collection_name} ...")
    # 注意：doc 用 prompt_name=None
    # Jasper 的 encode_cuda 支持 compression_ratio
    doc_emb = model.encode_cuda(
        doc_texts,
        batch_size=batch_size,
        prompt_name=None,
        compression_ratio=compression_ratio,
    )  # returns CPU tensor (你encode_cuda里 append(cpu))

    # 存 float16 降磁盘/内存；相似度时再搬上 GPU
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

def run_retrieval_for_collection(
    name, cfg, model, top_k=50,
    cache_dir="cache/doc_emb",
    model_tag="jasper-ft",
    doc_bs=256,
    query_bs=256,
    compression_ratio=0.3333,
    force_recompute_docs=False,
):
    import os, torch
    from tqdm import tqdm

    device = "cuda" if torch.cuda.is_available() else "cpu"

    root = cfg["root"]
    corpus_path = os.path.join(root, cfg["corpus_file"])
    query_path = os.path.join(root, cfg["query_file"])

    print(f"\n========== Collection: {name} ==========")
    print("Corpus:", corpus_path)
    print("Queries:", query_path)

    corpus = load_corpus(corpus_path)
    queries = load_queries(query_path)

    # -------- docs --------
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
        collection_name=cfg["collection_name"],
        max_len=getattr(model, "max_seq_length", 512),
        compression_ratio=compression_ratio,
        corpus_path=corpus_path,
        batch_size=doc_bs,
        force_recompute=force_recompute_docs,
        device=device,
    )

    # doc_emb 放 GPU 做相似度（float16 更快/省显存）
    doc_emb = doc_emb_cpu.to(device, non_blocking=True)

    # -------- queries --------
    q_ids = list(queries.keys())
    q_texts = [queries[q] for q in q_ids]

    # query 用 prompt_name="query"
    q_emb = model.encode_cuda(
        q_texts,
        batch_size=query_bs,
        prompt_name="query",
        compression_ratio=compression_ratio,
    ).to(device, non_blocking=True)

    # 相似度（已经 normalize_embeddings=True 了）
    sims = torch.matmul(q_emb, doc_emb.T)

    results = []
    for qi, qid in enumerate(tqdm(q_ids, desc="Building results")):
        vals, idx = torch.topk(sims[qi], top_k)
        ctxs = [
            {"document_id": doc_ids_cached[j], "score": float(v)}
            for v, j in zip(vals.tolist(), idx.tolist())
        ]
        results.append({
            "task_id": qid,
            "contexts": ctxs,
            "Collection": cfg["collection_name"],
        })

    return results


def load_jasper_st(
    model_name="Jasper-Token-Compression-600M",
    device=None,
    default_compression_ratio=0.3333,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Using device:", device)

    model = SentenceTransformer(
        model_name,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
            "trust_remote_code": True,
            "device_map": "cuda",
        },
        tokenizer_kwargs={
            "padding_side": "left",
            "trust_remote_code": True,
        },
        trust_remote_code=True,
        local_files_only=True,
        device=device,
    )

    model.max_seq_length = 512

    def encode(texts, batch_size=32, prompt_name=None, compression_ratio=None):
        if compression_ratio is None:
            compression_ratio = default_compression_ratio

        all_emb = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
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

    model.encode_cuda = encode
    return model



def build_submission(submission_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ✅ 用你 fine-tuned 的 Jasper 目录
    model = load_jasper_st(
        model_name="jasper-ft-lastturn",
        device=device,
        default_compression_ratio=0.3333,
    )

    # 缓存目录（每个 collection 一份 doc_emb）
    cache_dir = "cache/doc_emb_jasper_ft"
    model_tag = "jasper-ft-lastturn"
    compression_ratio = 0.3333

    all_results = []
    for name, cfg in COLLECTIONS.items():
        res = run_retrieval_for_collection(
            name, cfg, model, top_k=50,
            cache_dir=cache_dir,
            model_tag=model_tag,
            doc_bs=256,          # 如果 OOM 就调小，比如 64/128
            query_bs=256,
            compression_ratio=compression_ratio,
            force_recompute_docs=False,  # 第一次没有缓存会自动算；想强制重算改 True
        )
        all_results.extend(res)

    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    with open(submission_path, "w", encoding="utf-8") as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Saved submission to:", submission_path)
    return submission_path

for task_name in ['lastturn']:
    model_name = "jasper_1222"

    COLLECTIONS = {
        "clapnq": {
            "collection_name": "mt-rag-clapnq-elser-512-100-20240503",
            "root": "human/retrieval_tasks/clapnq",
            "corpus_file": "clapnq.jsonl",
            "query_file": f"clapnq_{task_name}.jsonl",
            "qrels_file": "qrels/dev.tsv",
        },
        "fiqa": {
            "collection_name": "mt-rag-fiqa-beir-elser-512-100-20240501",
            "root": "human/retrieval_tasks/fiqa",
            "corpus_file": "fiqa.jsonl",
            "query_file": f"fiqa_{task_name}.jsonl",
            "qrels_file": "qrels/dev.tsv",
        },
        "govt": {
            "collection_name": "mt-rag-govt-elser-512-100-20240611",
            "root": "human/retrieval_tasks/govt",
            "corpus_file": "govt.jsonl",
            "query_file": f"govt_{task_name}.jsonl",
            "qrels_file": "qrels/dev.tsv",
        },
        "cloud": {
            "collection_name": "mt-rag-ibmcloud-elser-512-100-20240502",
            "root": "human/retrieval_tasks/cloud",
            "corpus_file": "cloud.jsonl",
            "query_file": f"cloud_{task_name}.jsonl",
            "qrels_file": "qrels/dev.tsv",
        },
    }


    SUB_PATH = f"outputs/{model_name}_{task_name}.jsonl"
    SCORED_PATH = f"outputs/{model_name}_{task_name}_score.jsonl"

    print("Start!")
    print("Current task", task_name)
    sub_path = build_submission(SUB_PATH)
    run_official_eval(SUB_PATH, SCORED_PATH)