#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DocSem (doc-level semantic chunk) retrieval for MTRAG

- Load chunk corpus (*.jsonl.gz) with fields: {id,parent_id,start,end,title,text}
- Encode chunks (title + text) with Jasper SentenceTransformer (encode_cuda wrapper)
- Encode queries
- Retrieve top-k chunks via streaming batched topk (no full sims matrix)
- Aggregate chunk hits to official passage ids using passage span index
- Write submission jsonl and run official eval

This version supports custom corpus filename suffix:
  --docsem_suffix docsem512_100_outlier
so it loads:
  <docsem_corpus_dir>/<domain>_<docsem_suffix>.jsonl.gz
"""

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
from collections import Counter
from bisect import bisect_right

import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# -----------------------------
# Utils: Jasper remote-code sync
# -----------------------------
def sync_remote_code_if_missing(model_dir: str, base_dir: str | None):
    ft = Path(model_dir)
    need = ft / "modeling_qwen3_jasper.py"
    if need.exists():
        return
    if base_dir is None:
        raise OSError(f"{model_dir} missing {need.name}. Provide --base_dir to copy remote-code files.")
    base = Path(base_dir)
    if not base.exists():
        raise OSError(f"base_dir not found: {base_dir}")
    print(f"[SYNC] {need.name} missing in {model_dir}. Copying remote-code files from {base_dir} ...")
    for pat in ["modeling_*.py", "configuration_*.py", "tokenization_*.py", "custom_st.py"]:
        for src in base.glob(pat):
            dst = ft / src.name
            shutil.copy2(src, dst)
            print(f"[SYNC] {src} -> {dst}")


# -----------------------------
# Cache helpers
# -----------------------------
def _safe_tag(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-+" else "_" for ch in str(s))

def _cache_key(
    model_tag: str,
    collection_tag: str,
    max_len: int,
    compression_ratio: float,
    corpus_path: str,
    *,
    docsem_suffix: str,
    max_chunks_per_parent: int,
    skip_parent_ids_path: str,
) -> str:
    st = os.stat(corpus_path)
    sig = (
        f"{model_tag}|{collection_tag}|suffix={docsem_suffix}|len={max_len}|cr={compression_ratio}"
        f"|size={st.st_size}|mtime={int(st.st_mtime)}"
        f"|mcpp={max_chunks_per_parent}|skip={skip_parent_ids_path}"
    )
    h = hashlib.sha1(sig.encode("utf-8")).hexdigest()[:16]
    return f"{_safe_tag(collection_tag)}__{_safe_tag(model_tag)}__{_safe_tag(docsem_suffix)}__len{max_len}__cr{compression_ratio:.4f}__{h}"

def get_or_build_doc_embeddings(
    *,
    model,
    doc_texts: List[str],
    doc_ids: List[str],
    cache_dir: str,
    model_tag: str,
    collection_tag: str,
    max_len: int,
    compression_ratio: float,
    corpus_path: str,
    batch_size: int = 256,
    force_recompute: bool = False,
    docsem_suffix: str = "",
    max_chunks_per_parent: int = 0,
    skip_parent_ids_path: str = "",
) -> Tuple[List[str], torch.Tensor]:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    key = _cache_key(
        model_tag, collection_tag, max_len, compression_ratio, corpus_path,
        docsem_suffix=docsem_suffix,
        max_chunks_per_parent=max_chunks_per_parent,
        skip_parent_ids_path=skip_parent_ids_path,
    )
    cache_path = os.path.join(cache_dir, key + ".pt")

    if (not force_recompute) and os.path.exists(cache_path):
        try:
            pack = torch.load(cache_path, map_location="cpu")
            if pack.get("doc_ids") == doc_ids and isinstance(pack.get("doc_emb"), torch.Tensor):
                print(f"[CACHE HIT] {cache_path}")
                return pack["doc_ids"], pack["doc_emb"]
            print("[CACHE STALE] doc_ids mismatch or missing doc_emb -> recompute")
        except Exception as e:
            print(f"[CACHE BROKEN] failed to load {cache_path}: {e} -> recompute")

    print(f"[CACHE MISS] Encoding docs for {collection_tag} ...")
    doc_emb = model.encode_cuda(
        doc_texts,
        batch_size=batch_size,
        prompt_name=None,
        compression_ratio=compression_ratio,
    )
    doc_emb = doc_emb.to(torch.float16).contiguous()
    torch.save({"doc_ids": doc_ids, "doc_emb": doc_emb, "meta": {"saved_time": time.time()}}, cache_path)
    print(f"[CACHE SAVE] {cache_path} | shape={tuple(doc_emb.shape)}")
    return doc_ids, doc_emb


# -----------------------------
# Load corpora / queries
# -----------------------------
def load_docsem_corpus_gz(path: str):
    ids, texts, parents, starts, ends, titles = [], [], [], [], [], []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            ids.append(str(o["id"]))
            parents.append(str(o["parent_id"]))
            starts.append(int(o["start"]))
            ends.append(int(o["end"]))
            titles.append((o.get("title") or ""))
            texts.append((o.get("text") or ""))
    return ids, texts, parents, starts, ends, titles


def load_passage_span_index(path: str):
    by_base: Dict[str, List[Tuple[int, int, str]]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            base = str(o["base_id"])
            by_base.setdefault(base, []).append((int(o["start"]), int(o["end"]), str(o["passage_id"])))

    out = {}
    for base, spans in by_base.items():
        spans.sort(key=lambda x: x[0])
        starts = [s for (s, e, pid) in spans]
        ends = [e for (s, e, pid) in spans]
        pids = [pid for (s, e, pid) in spans]
        out[base] = (starts, ends, pids)
    return out


def load_queries(query_path: str) -> Dict[str, str]:
    queries: Dict[str, str] = {}
    with open(query_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            qid = item.get("_id") or item.get("query_id") or item.get("task_id")
            if qid is None:
                continue
            queries[str(qid)] = str(item["text"])
    return queries


def load_skip_parent_ids(path: str) -> set:
    if not path:
        return set()
    s = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            s.add(t)
    return s


def filter_by_parent_noise(
    chunk_ids, chunk_texts, parents, starts, ends, titles,
    *, max_chunks_per_parent: int, skip_parent_ids: set
):
    if max_chunks_per_parent <= 0 and not skip_parent_ids:
        stats = {
            "orig_chunks": len(chunk_ids),
            "orig_parents": len(set(parents)),
            "banned_parents_total": 0,
            "kept_chunks": len(chunk_ids),
            "kept_parents": len(set(parents)),
        }
        return chunk_ids, chunk_texts, parents, starts, ends, titles, stats

    cnt = Counter(parents)
    too_many = {p for p, c in cnt.items() if c > max_chunks_per_parent} if max_chunks_per_parent > 0 else set()
    banned = too_many | set(skip_parent_ids)
    kept_idx = [i for i, p in enumerate(parents) if p not in banned]

    stats = {
        "orig_chunks": len(chunk_ids),
        "orig_parents": len(cnt),
        "banned_parents_too_many": len(too_many),
        "banned_parents_explicit": len(skip_parent_ids),
        "banned_parents_total": len(banned),
        "kept_chunks": len(kept_idx),
        "kept_parents": len({parents[i] for i in kept_idx}) if kept_idx else 0,
    }

    def take(xs):
        return [xs[i] for i in kept_idx]

    return take(chunk_ids), take(chunk_texts), take(parents), take(starts), take(ends), take(titles), stats


# -----------------------------
# Jasper loader (with encode_cuda)
# -----------------------------
def load_jasper_st(model_name: str, device: str, default_compression_ratio: float = 0.3333):
    print("Using device:", device)

    model = SentenceTransformer(
        model_name,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
            "trust_remote_code": True,
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

    def encode_cuda(texts, batch_size=64, prompt_name=None, compression_ratio=None):
        if compression_ratio is None:
            compression_ratio = default_compression_ratio
        emb = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=True if len(texts) > batch_size else False,
            normalize_embeddings=True,
            prompt_name=prompt_name,
            compression_ratio=compression_ratio,
        )
        return emb.detach().cpu()

    model.encode_cuda = encode_cuda
    return model


# -----------------------------
# Exact chunked top-k (no big sims matrix)
# -----------------------------
@torch.no_grad()
def batched_topk_stream(q_emb_cpu: torch.Tensor, doc_emb_cpu: torch.Tensor, k: int, chunk_size: int, device: str):
    Q = q_emb_cpu.size(0)
    N = doc_emb_cpu.size(0)
    if N <= 0:
        raise ValueError("Empty doc embeddings.")
    k = min(int(k), int(N))
    if k <= 0:
        raise ValueError(f"Invalid k={k} with N={N}")

    q = q_emb_cpu.to(device, non_blocking=True).float()
    top_vals = torch.full((Q, k), -1e9, device=device, dtype=torch.float32)
    top_idx = torch.full((Q, k), -1, device=device, dtype=torch.int64)

    for start in range(0, N, chunk_size):
        chunk = doc_emb_cpu[start:start + chunk_size].to(device, non_blocking=True).float()
        sims = q @ chunk.T
        kk = min(k, sims.size(1))
        vals, idx = torch.topk(sims, kk, dim=1)
        idx = idx + start

        merged_vals = torch.cat([top_vals, vals], dim=1)
        merged_idx = torch.cat([top_idx, idx], dim=1)
        new_vals, pos = torch.topk(merged_vals, k, dim=1)
        new_idx = torch.gather(merged_idx, 1, pos)
        top_vals, top_idx = new_vals, new_idx

    return top_vals.cpu(), top_idx.cpu()


def overlap(a0, a1, b0, b1) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def aggregate_chunks_to_passages_best_assign(chunk_hits, chunk_meta, span_index):
    out = {}
    for cid, s in chunk_hits:
        meta = chunk_meta.get(cid)
        if meta is None:
            continue
        base_id, cs, ce = meta
        pack = span_index.get(base_id)
        if pack is None:
            continue
        starts, ends, pids = pack

        i = bisect_right(starts, cs) - 1
        if i < 0:
            i = 0

        best_pid, best_ov = None, 0
        while i < len(starts) and starts[i] < ce:
            ps, pe, pid = starts[i], ends[i], pids[i]
            ov = max(0, min(ce, pe) - max(cs, ps))
            if ov > best_ov:
                best_ov = ov
                best_pid = pid
            i += 1

        if best_pid is None:
            continue

        prev = out.get(best_pid)
        if prev is None or float(s) > prev:
            out[best_pid] = float(s)
    return out



def aggregate_chunks_to_passages(
    chunk_hits: List[Tuple[str, float]],
    chunk_meta: Dict[str, Tuple[str, int, int]],
    span_index: Dict[str, Tuple[List[int], List[int], List[str]]],
    agg: str = "max",
):
    out: Dict[str, float] = {}
    for cid, s in chunk_hits:
        meta = chunk_meta.get(cid)
        if meta is None:
            continue
        base_id, cs, ce = meta
        spans = span_index.get(base_id)
        if spans is None:
            continue
        starts, ends, pids = spans

        i = bisect_right(starts, cs) - 1
        if i < 0:
            i = 0

        while i < len(starts) and starts[i] < ce:
            ps, pe, pid = starts[i], ends[i], pids[i]
            ov = overlap(cs, ce, ps, pe)
            if ov > 0:
                if agg == "max":
                    val = float(s)
                else:
                    denom = max(1, (ce - cs))
                    val = float(s) * (ov / denom)
                prev = out.get(pid)
                if prev is None or val > prev:
                    out[pid] = val
            i += 1
    return out


# -----------------------------
# Main
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, default="lastturn", choices=["lastturn", "questions", "rewrite"])

    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--base_dir", type=str, default=None)
    ap.add_argument("--model_name", type=str, default="jasper_docsem512_100")

    ap.add_argument("--domains", type=str, default="fiqa")

    ap.add_argument("--docsem_corpus_dir", type=str, required=True)
    ap.add_argument("--docsem_suffix", type=str, default="docsem512_100",
                    help="corpus filename suffix: <domain>_<suffix>.jsonl.gz")
    ap.add_argument("--passage_span_dir", type=str, default="corpora/passage_level")

    ap.add_argument("--cache_dir", type=str, default="cache/doc_emb_jasper_docsem512_100")
    ap.add_argument("--model_tag", type=str, default=None)
    ap.add_argument("--compression_ratio", type=float, default=0.3333)

    ap.add_argument("--top_k_passages", type=int, default=50)
    ap.add_argument("--top_k_chunks", type=int, default=1000)
    ap.add_argument("--agg", type=str, default="max", choices=["max", "max_overlap", "best_assign"])

    ap.add_argument("--doc_bs", type=int, default=256)
    ap.add_argument("--query_bs", type=int, default=256)
    ap.add_argument("--chunk_size", type=int, default=50000)
    ap.add_argument("--force_recompute_docs", action="store_true")

    # optional safety filter (default off to keep your outlier corpus as-is)
    ap.add_argument("--max_chunks_per_parent", type=int, default=0)
    ap.add_argument("--skip_parent_ids", type=str, default="")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_eval", action="store_true")

    return ap.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sync_remote_code_if_missing(args.model_dir, args.base_dir)

    model = load_jasper_st(args.model_dir, device=device, default_compression_ratio=args.compression_ratio)
    model_tag = args.model_tag or Path(args.model_dir).name

    COLLECTIONS = {
        "clapnq": {"collection_name": "mt-rag-clapnq-elser-512-100-20240503", "root": "human/retrieval_tasks/clapnq"},
        "fiqa":   {"collection_name": "mt-rag-fiqa-beir-elser-512-100-20240501", "root": "human/retrieval_tasks/fiqa"},
        "govt":   {"collection_name": "mt-rag-govt-elser-512-100-20240611", "root": "human/retrieval_tasks/govt"},
        "cloud":  {"collection_name": "mt-rag-ibmcloud-elser-512-100-20240502", "root": "human/retrieval_tasks/cloud"},
    }

    wanted = [x.strip() for x in args.domains.split(",") if x.strip()]
    for d in wanted:
        if d not in COLLECTIONS:
            raise ValueError(f"Unknown domain '{d}'. Must be one of {list(COLLECTIONS.keys())}")

    os.makedirs("outputs", exist_ok=True)
    sub_path = f"outputs/{args.model_name}_{args.task}.jsonl"
    scored_path = f"outputs/{args.model_name}_{args.task}_score.jsonl"

    skip_parent_ids = load_skip_parent_ids(args.skip_parent_ids)
    all_results = []

    print("Start!")
    print("Task:", args.task)
    print("Domains:", wanted)
    print("docsem dir:", args.docsem_corpus_dir)
    print("docsem suffix:", args.docsem_suffix)
    print("span dir:", args.passage_span_dir)

    for name in wanted:
        cfg = COLLECTIONS[name]
        print(f"\n========== Collection: {name} ==========")

        # docsem_path = os.path.join(args.docsem_corpus_dir, f"{name}_{args.docsem_suffix}.jsonl.gz")
        docsem_path = os.path.join(args.docsem_corpus_dir, "fiqa_budgetpad512_100.jsonl.gz")

        span_path = os.path.join(args.passage_span_dir, f"{name}_passage_spans.jsonl.gz")
        query_path = os.path.join(cfg["root"], f"{name}_{args.task}.jsonl")

        if not os.path.exists(docsem_path):
            raise FileNotFoundError(f"Missing docsem corpus: {docsem_path}")
        if not os.path.exists(span_path):
            raise FileNotFoundError(f"Missing passage span index: {span_path}")
        if not os.path.exists(query_path):
            raise FileNotFoundError(f"Missing query file: {query_path}")

        print("DocSem corpus:", docsem_path)
        print("Span index:", span_path)
        print("Queries:", query_path)

        chunk_ids, chunk_texts, parents, starts, ends, titles = load_docsem_corpus_gz(docsem_path)
        span_index = load_passage_span_index(span_path)
        queries = load_queries(query_path)

        # optional parent filter (default off)
        chunk_ids, chunk_texts, parents, starts, ends, titles, stats = filter_by_parent_noise(
            chunk_ids, chunk_texts, parents, starts, ends, titles,
            max_chunks_per_parent=args.max_chunks_per_parent,
            skip_parent_ids=skip_parent_ids,
        )
        print("[PARENT_FILTER]", stats)
        if not chunk_ids:
            raise RuntimeError(f"After filtering, no chunks left for domain {name}.")

        chunk_meta = {cid: (parents[i], starts[i], ends[i]) for i, cid in enumerate(chunk_ids)}

        doc_texts = []
        for i in range(len(chunk_ids)):
            t = titles[i].strip()
            x = chunk_texts[i].strip()
            doc_texts.append((t + "\n" + x).strip() if (t and x) else (t or x))

        collection_tag = cfg["collection_name"] + f"__{args.docsem_suffix}"

        _, doc_emb_cpu = get_or_build_doc_embeddings(
            model=model,
            doc_texts=doc_texts,
            doc_ids=chunk_ids,
            cache_dir=args.cache_dir,
            model_tag=model_tag,
            collection_tag=collection_tag,
            max_len=getattr(model, "max_seq_length", 512),
            compression_ratio=args.compression_ratio,
            corpus_path=docsem_path,
            batch_size=args.doc_bs,
            force_recompute=args.force_recompute_docs,
            docsem_suffix=args.docsem_suffix,
            max_chunks_per_parent=args.max_chunks_per_parent,
            skip_parent_ids_path=args.skip_parent_ids,
        )

        q_ids = list(queries.keys())
        q_texts = [queries[q] for q in q_ids]
        q_emb_cpu = model.encode_cuda(q_texts, batch_size=args.query_bs, prompt_name="query", compression_ratio=args.compression_ratio)

        top_vals, top_idx = batched_topk_stream(
            q_emb_cpu=q_emb_cpu,
            doc_emb_cpu=doc_emb_cpu,
            k=args.top_k_chunks,
            chunk_size=args.chunk_size,
            device=device,
        )

        for qi, qid in enumerate(tqdm(q_ids, desc=f"Aggregate {name}")):
            idx_row = top_idx[qi].tolist()
            val_row = top_vals[qi].tolist()
            chunk_hits = [(chunk_ids[j], float(v)) for j, v in zip(idx_row, val_row)]

            if args.agg == "best_assign":
                passage_scores = aggregate_chunks_to_passages_best_assign(
                    chunk_hits=chunk_hits,
                    chunk_meta=chunk_meta,
                    span_index=span_index,
                )
            else:
                passage_scores = aggregate_chunks_to_passages(
                    chunk_hits=chunk_hits,
                    chunk_meta=chunk_meta,
                    span_index=span_index,
                    agg=args.agg,
                )


            ranked = sorted(passage_scores.items(), key=lambda x: x[1], reverse=True)[:args.top_k_passages]
            ctxs = [{"document_id": pid, "score": float(sc)} for pid, sc in ranked]

            all_results.append({"task_id": str(qid), "contexts": ctxs, "Collection": cfg["collection_name"]})

    with open(sub_path, "w", encoding="utf-8") as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Saved submission to:", sub_path)

    if args.skip_eval:
        print("[SKIP_EVAL] Done.")
        return

    cmd = [
        "python3", "scripts/evaluation/run_retrieval_eval.py",
        "--input_file", sub_path,
        "--output_file", scored_path,
        "--model_name", args.model_name,
        "--task_name", args.task,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Done, scored file:", scored_path)


if __name__ == "__main__":
    main()
