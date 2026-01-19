#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ast import arg
import os
import json
import time
import hashlib
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import AutoTokenizer
import sys, shlex, datetime

import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

import gzip


def _norm_collection(x: str) -> str:
    s = (x or "").strip().lower()
    if s == "ibmcloud":
        return "cloud"
    return s



def _linearize_dialog(msgs, mode="last_user", last_n=None):
    """
    mode:
      - "last_user": 只用最后一个 user utterance（最接近你旧的 lastturn 格式）
      - "full": 把整段对话串起来（多轮信息更全）
    """
    if mode == "last_user":
        for m in reversed(msgs):
            if m.get("speaker") == "user":
                return "|user|: " + (m.get("text","") or "")
        return ""

    # full
    lines = []
    for m in msgs[-last_n:] if (last_n and last_n > 0) else msgs:
        sp = m.get("speaker")
        role = "user" if sp == "user" else "assistant"  # agent -> assistant
        lines.append(f"|{role}|: {m.get('text','') or ''}")
    return "\n".join(lines).strip()


def load_test_taska_queries(test_input_jsonl: str, query_mode="last_user", last_n=None):
    by_domain = defaultdict(dict)
    with open(test_input_jsonl, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            # 需要 domain（用于按 fiqa/govt/... 分流）
            if "Collection" not in item:
                raise ValueError(f"Line {line_no}: missing Collection in {test_input_jsonl}")
            # domain = item["Collection"]
            domain = _norm_collection(item["Collection"])


            # 兼容 task_id / _id
            task_id = item.get("task_id") or item.get("_id")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(f"Line {line_no}: missing task_id/_id in {test_input_jsonl}")

            # Case1: 原始 test（有 input）
            if "input" in item:
                msgs = item.get("input", [])
                q = _linearize_dialog(msgs, mode=query_mode, last_n=last_n)

            # Case2: 你预处理后的 lastturn 文件（有 text）
            elif "text" in item:
                q = item.get("text", "") or ""
                if query_mode == "last_user":
                    # 如果 text 里有多行，只取最后一个 |user|:
                    last = ""
                    for ln in reversed([x for x in q.splitlines() if x.strip()]):
                        if ln.startswith("|user|:"):
                            last = ln
                            break
                    q = last or q.strip()

            else:
                q = ""

            by_domain[domain][task_id] = q
    return by_domain


import re
from collections import defaultdict

# _NUM_RE = re.compile(r"\d+")
# clapnq 类：822086267_22716-22948-0-232  或 822086267_18870-20795-1227-1925
_RE_CLAP4 = re.compile(r"^(?P<doc>.+)_(?P<s>\d+)-(?P<e>\d+)-(?P<a>\d+)-(?P<b>\d+)$")

# fiqa / govt / cloud 类：10171-0-2129 / 45cbe...-2-2092 / ibmcld_00422-0-387
_RE_2NUM = re.compile(r"^(?P<doc>.+)-(?P<s>\d+)-(?P<b>\d+)$")

def _parse_docid_start(passage_id: str):
    m = _RE_CLAP4.match(passage_id)
    if m:
        return m.group("doc"), int(m.group("s"))

    m = _RE_2NUM.match(passage_id)
    if m:
        # doc 是 “最后两个 -数字-数字 之前的全部”
        # cloud: doc=ibmcld_00422, s=0
        # fiqa:  doc=10171,       s=0
        # govt:  doc=<hash>,      s=2
        return m.group("doc"), int(m.group("s"))

    return None

# def _parse_docid_start(passage_id: str):
#     # 适配你这种：822086267_22716-22948-0-232
#     if "_" not in passage_id:
#         return None
#     docid, rest = passage_id.split("_", 1)
#     nums = _NUM_RE.findall(rest)
#     if not nums:
#         return None
#     return docid, int(nums[0])


def build_packs_from_passage_corpus(
    corpus: Dict[str, Dict[str, str]],
    tok,
    *,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
    sep: str = "\n\n",
) -> Tuple[List[str], List[str], Dict[str, List[str]], Dict[str, str]]:
    """
    Returns:
      pack_ids:   list[str]
      pack_texts: list[str]
      pack2child: dict[pack_id] -> [passage_id...]
      passage_text: dict[passage_id] -> full_text (title+text)
    """

    # 先把每个 passage 的 full text 建出来（后面 expand / rerank 会用）
    passage_text: Dict[str, str] = {}
    items = []  # (docid, start, passage_id, tlen)
    for pid, o in corpus.items():
        full = ((o.get("title", "") + " " + o.get("text", "")).strip())
        passage_text[pid] = full
        key = _parse_docid_start(pid)
        if key is None:
            docid, start = (pid, 0)   # 解析失败就每条自己一桶，不合并
        else:
            docid, start = key

        # if key is None:
        #     # parse 不出来的放到一个虚拟 doc 里，按 pid 排序（至少稳定）
        #     docid, start = ("__UNKNOWN__", 0)
        # else:
        #     docid, start = key

        ids = tok(full, add_special_tokens=False)["input_ids"]
        tlen = len(ids)
        items.append((docid, start, pid, tlen))

    # 按 doc 分组 + doc 内按 start 排序
    buckets = defaultdict(list)
    for docid, start, pid, tlen in items:
        buckets[docid].append((start, pid, tlen))
    for docid in buckets:
        buckets[docid].sort(key=lambda x: x[0])

    pack_ids, pack_texts = [], []
    pack2child: Dict[str, List[str]] = {}

    def _make_pack_id(docid: str, k: int) -> str:
        return f"{docid}__pack{k:05d}"

    for docid, arr in buckets.items():
        i = 0
        pack_k = 0
        n = len(arr)
        while i < n:
            cur_toks = 0
            child = []
            parts = []
            j = i

            while j < n:
                _, pid, tlen = arr[j]
                add = tlen + (1 if parts else 0)  # 轻微分隔开销
                if child and (cur_toks + add) > max_tokens:
                    break
                child.append(pid)
                parts.append(passage_text[pid])
                cur_toks += add
                j += 1

            pid_pack = _make_pack_id(docid, pack_k)
            pack_k += 1

            text_pack = sep.join(parts)
            pack_ids.append(pid_pack)
            pack_texts.append(text_pack)
            pack2child[pid_pack] = child

            if j >= n:
                break

            # overlap：保留尾部 passages 直到 overlap_tokens
            if overlap_tokens <= 0:
                i = j
            else:
                keep = 0
                k = j - 1
                while k >= i and keep < overlap_tokens:
                    keep += arr[k][2]  # tlen
                    k -= 1
                i = max(k + 1, i)
                if i >= j:
                    i = j

    return pack_ids, pack_texts, pack2child, passage_text


def expand_pack_hits_to_passages(
    pack_ctxs: List[Dict[str, float]],         # [{"document_id": pack_id, "score": s}, ...]
    pack2child: Dict[str, List[str]],
    *,
    agg: str = "max",                          # "max" or "sum"
    topm: int = 1200,
) -> List[Tuple[str, float]]:
    scores = defaultdict(float)
    for c in pack_ctxs:
        pid = c["document_id"]
        ps = float(c["score"])
        for child in pack2child.get(pid, []):
            if agg == "sum":
                scores[child] += ps
            else:
                if ps > scores[child]:
                    scores[child] = ps

    cand = sorted(scores.items(), key=lambda x: -x[1])[:topm]
    return cand  # [(passage_id, coarse_score), ...]


# new end

def load_blacklist(path: str | None) -> set[str]:
    if not path:
        return set()
    bl = set()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s:
                bl.add(s)
    print(f"[BLACKLIST] loaded {len(bl)} ids from {path}")
    return bl


def open_maybe_gz(path: str, mode: str = "rt"):
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8", errors="ignore")
    return open(path, mode, encoding="utf-8", errors="ignore")



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
# Data loading
# -----------------------------
# def load_corpus(corpus_path: str) -> Dict[str, Dict[str, str]]:
#     corpus = {}
#     with open(corpus_path, "r", encoding="utf-8") as f:
#         for line in f:
#             item = json.loads(line)
#             doc_id = item["document_id"] if "document_id" in item else item["_id"]
#             corpus[doc_id] = {"title": item.get("title", ""), "text": item.get("text", "")}
#     return corpus

# def load_corpus(corpus_path: str, blacklist: set[str] | None = None) -> Dict[str, Dict[str, str]]:
#     corpus = {}
#     blacklist = blacklist or set()

#     # 如果你的 corpus 可能是 .gz，建议用 open_maybe_gz；否则 open() 也行
#     # with open(corpus_path, "r", encoding="utf-8") as f:
#     with open_maybe_gz(corpus_path, "rt") as f:   # <- 如果你已经加了 open_maybe_gz
#         for line in f:
#             item = json.loads(line)
#             # doc_id = item["document_id"] if "document_id" in item else item["_id"]
#             doc_id = str(item.get("document_id") or item.get("_id") or item.get("id") or "")
#             if not doc_id:
#                 continue

#             # blacklist filter
#             if doc_id in blacklist:
#                 continue

#             corpus[doc_id] = {"title": item.get("title", ""), "text": item.get("text", "")}

#     return corpus


def load_corpus(corpus_path: str, blacklist: set[str] | None = None, text_key: str = "text") -> Dict[str, Dict[str, str]]:
    corpus = {}
    blacklist = blacklist or set()

    with open_maybe_gz(corpus_path, "rt") as f:
        for line in f:
            item = json.loads(line)
            doc_id = str(item.get("document_id") or item.get("_id") or item.get("id") or "")
            if not doc_id:
                continue
            if doc_id in blacklist:
                continue

            # title 保留；text 允许从 ctx_text 等字段取
            title = item.get("title", "") or ""
            text = item.get(text_key, None)
            if text is None:
                # fallback：如果指定字段不存在，就退回 text
                text = item.get("text", "") or ""
            corpus[doc_id] = {"title": title, "text": text}

    return corpus


def load_queries(query_path: str) -> Dict[str, str]:
    queries = {}
    with open(query_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            qid = item.get("_id") or item.get("query_id") or item["task_id"]
            queries[qid] = item["text"]
    return queries


# -----------------------------
# Jasper loader (with encode_cuda)
# -----------------------------
def load_jasper_st(
    model_name: str,
    device: str,
    default_compression_ratio: float = 0.3333,
    max_len: int = 512,
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
            "fix_mistral_regex": True,
            # "use_fast": False
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
    q_f = q.float()  # score in fp32

    Q = q_f.size(0)
    top_vals = torch.full((Q, k), -1e9, device=device, dtype=torch.float32)
    top_idx = torch.full((Q, k), -1, device=device, dtype=torch.int64)

    N = doc_emb_cpu.size(0)
    for start in range(0, N, chunk_size):
        chunk = doc_emb_cpu[start:start + chunk_size].to(device, non_blocking=True)
        sims = q_f @ chunk.float().T                       # [Q, C] fp32
        vals, idx = torch.topk(sims, k, dim=1)            # [Q, k]
        idx = idx + start                                  # global indices

        merged_vals = torch.cat([top_vals, vals], dim=1)   # [Q, 2k]
        merged_idx  = torch.cat([top_idx,  idx],  dim=1)   # [Q, 2k]
        new_vals, pos = torch.topk(merged_vals, k, dim=1)  # [Q, k]
        new_idx = torch.gather(merged_idx, 1, pos)         # [Q, k]
        top_vals, top_idx = new_vals, new_idx

    return top_vals.cpu(), top_idx.cpu()


# -----------------------------
# Retrieval runner (with cache + chunk-topk)
# -----------------------------
def run_retrieval_for_collection(
    name: str,
    cfg: dict,
    model,
    top_k: int,
    cache_dir: str,
    model_tag: str,
    compression_ratio: float,
    doc_bs: int,
    query_bs: int,
    force_recompute_docs: bool,
    chunk_size: int,
    corpus_override_dir: str | None = None,
    blacklist: set[str] | None = None,
    corpus_override_suffix=".en_only.jsonl.gz",
    split2: bool = False,
    tok=None,
    max_len: int = 512,
    pack_adjacent: bool = False,
    pack_max_tokens: int = 512,
    pack_overlap_tokens: int = 64,
    pack_topk: int = 200,
    expand_topm: int = 1200,
    pack_agg: str = "max",
    corpus_text_key: str = "text",
    queries_override: Dict[str, str] | None = None,
    collection_out: str | None = None,

):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    root = cfg["root"]
    # corpus_path = os.path.join(root, cfg["corpus_file"])
    # query_path = os.path.join(root, cfg["query_file"])
    if corpus_override_dir:
        corpus_path = os.path.join(corpus_override_dir, f"{name}{corpus_override_suffix}") #os.path.join(corpus_override_dir, f"{name}.en_only.jsonl.gz")
    else:
        corpus_path = os.path.join(root, cfg["corpus_file"])

    query_path = os.path.join(root, cfg["query_file"])

    print(f"\n========== Collection: {name} ==========")
    print("Corpus:", corpus_path)
    print("Queries:", query_path)

    # corpus = load_corpus(corpus_path, blacklist=blacklist)
    corpus = load_corpus(corpus_path, blacklist=blacklist, text_key=corpus_text_key)

    # queries = load_queries(query_path)
    if queries_override is not None:
        queries = queries_override
    else:
        queries = load_queries(query_path)


    # doc_ids = list(corpus.keys())
    # doc_texts = [
    #     (corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip()
    #     for d in doc_ids
    # ]
    def _seg2_texts(parent_id: str, text: str, tok, max_len: int):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        L = len(ids)
        if L <= max_len:
            return [(f"{parent_id}__seg0", text)]
        # head + tail（如果 max_len >= ceil(L/2)，两段能覆盖全文）
        head = ids[:max_len]
        tail = ids[-max_len:]
        t0 = tok.decode(head, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        t1 = tok.decode(tail, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return [(f"{parent_id}__seg0", t0), (f"{parent_id}__seg1", t1)]

    parent_doc_ids = list(corpus.keys())

    seg_ids = []
    seg_texts = []
    if split2 and tok is None:
        raise ValueError("split2=True but tok is None. Pass tokenizer via tok=...")

        # --- Build retrieval units ---
    if pack_adjacent:
        # pack 作为粗排检索单元
        pack_ids, pack_texts, pack2child, passage_text = build_packs_from_passage_corpus(
            corpus, tok,
            max_tokens=pack_max_tokens,
            overlap_tokens=pack_overlap_tokens,
        )
        retr_ids = pack_ids
        retr_texts = pack_texts

        # cache tag 要区分一下，否则会复用旧 cache
        collection_for_cache = cfg["collection_name"] + f"__pack{pack_max_tokens}_ov{pack_overlap_tokens}"
    else:
        # 原逻辑：passage 直接检索（可 split2）
        parent_doc_ids = list(corpus.keys())
        seg_ids = []
        seg_texts = []
        if split2 and tok is None:
            raise ValueError("split2=True but tok is None. Pass tokenizer via tok=...")

        for did in parent_doc_ids:
            full = ((corpus[did].get("title","") + " " + corpus[did].get("text","")).strip())
            if split2:
                segs = _seg2_texts(did, full, tok, max_len)
                for sid, st in segs:
                    seg_ids.append(sid)
                    seg_texts.append(st)
            else:
                seg_ids.append(did)
                seg_texts.append(full)

        retr_ids = seg_ids
        retr_texts = seg_texts
        collection_for_cache = cfg["collection_name"]

    doc_ids_cached, doc_emb_cpu = get_or_build_doc_embeddings(
        model=model,
        doc_texts=retr_texts,
        doc_ids=retr_ids,
        cache_dir=cache_dir,
        model_tag=model_tag,
        collection_name=collection_for_cache,  # <- 这里用新名字
        max_len=getattr(model, "max_seq_length", 512),
        compression_ratio=compression_ratio,
        corpus_path=corpus_path,
        batch_size=doc_bs,
        force_recompute=force_recompute_docs,
    )


    # for did in parent_doc_ids:
    #     full = ((corpus[did].get("title","") + " " + corpus[did].get("text","")).strip())
    #     if split2:
    #         segs = _seg2_texts(did, full, tok, max_len)
    #         for sid, st in segs:
    #             seg_ids.append(sid)
    #             seg_texts.append(st)
    #     else:
    #         seg_ids.append(did)       # 不拆
    #         seg_texts.append(full)

    # doc_ids_cached, doc_emb_cpu = get_or_build_doc_embeddings(
    #     model=model,
    #     doc_texts=seg_texts,
    #     doc_ids=seg_ids,
    #     # doc_texts=doc_texts,
    #     # doc_ids=doc_ids,
    #     cache_dir=cache_dir,
    #     model_tag=model_tag,
    #     collection_name=cfg["collection_name"],
    #     max_len=getattr(model, "max_seq_length", 512),
    #     compression_ratio=compression_ratio,
    #     corpus_path=corpus_path,
    #     batch_size=doc_bs,
    #     force_recompute=force_recompute_docs,
    # )

    q_ids = list(queries.keys())
    q_texts = [queries[q] for q in q_ids]

    # queries: prompt_name="query"
    q_emb_cpu = model.encode_cuda(
        q_texts,
        batch_size=query_bs,
        prompt_name="query",
        compression_ratio=compression_ratio,
    )  # CPU

    # k_seg = top_k * 2 if split2 else top_k
    if pack_adjacent:
        k_retr = pack_topk
    else:
        k_retr = top_k * 2 if split2 else top_k

    top_vals, top_idx = batched_topk_stream(
        q_emb_cpu=q_emb_cpu,
        doc_emb_cpu=doc_emb_cpu,
        k=k_retr,
        chunk_size=chunk_size,
        device=device,
    )

    results = []
    for qi, qid in enumerate(tqdm(q_ids, desc="Building results")):
        idx_row = top_idx[qi].tolist()
        val_row = top_vals[qi].tolist()

        if pack_adjacent:
            # 1) pack contexts
            pack_ctxs = [{"document_id": doc_ids_cached[j], "score": float(v)}
                         for j, v in zip(idx_row, val_row)]
            # 2) expand -> passage candidates
            cand = expand_pack_hits_to_passages(
                pack_ctxs, pack2child,
                agg=pack_agg,
                topm=expand_topm,
            )
            # 3) 先不接 reranker：直接用 coarse 传递分数输出 top_k
            ctxs = [{"document_id": pid, "score": float(sc)} for pid, sc in cand[:top_k]]

        else:
            if not split2:
                ctxs = [{"document_id": doc_ids_cached[j], "score": float(v)}
                        for j, v in zip(idx_row, val_row)]
            else:
                best = {}
                for j, v in zip(idx_row, val_row):
                    segid = doc_ids_cached[j]
                    parent = segid.rsplit("__seg", 1)[0]
                    if (parent not in best) or (v > best[parent]):
                        best[parent] = v
                top_parents = sorted(best.items(), key=lambda x: -x[1])[:top_k]
                ctxs = [{"document_id": pid, "score": float(sc)} for pid, sc in top_parents]

        # results.append({"task_id": qid, "contexts": ctxs, "Collection": cfg["collection_name"]})
        results.append({
            "task_id": qid,
            "contexts": ctxs,
            "Collection": (collection_out or cfg["collection_name"]),
        })

    return results



    # results = []
    # for qi, qid in enumerate(tqdm(q_ids, desc="Building results")):
    #     idx_row = top_idx[qi].tolist()
    #     val_row = top_vals[qi].tolist()
    #     ctxs = [{"document_id": doc_ids_cached[j], "score": float(v)} for j, v in zip(idx_row, val_row)]
    #     results.append({"task_id": qid, "contexts": ctxs, "Collection": cfg["collection_name"]})
    # return results


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
    ap.add_argument("--task", type=str, default="lastturn", choices=["lastturn", "questions", "rewrite", "rewrite_gpt", "rewrite_gpt_ir", "rewrite_gpt_keywords"])
    ap.add_argument("--model_dir", type=str, default="jasper-ft-lastturn", help="FT model folder")
    ap.add_argument("--base_dir", type=str, default=None, help="Base Jasper folder to sync remote-code *.py if missing")
    ap.add_argument("--model_name", type=str, default="jasper_1222", help="Name used in output filenames / official eval")
    ap.add_argument("--cache_dir", type=str, default="cache/doc_emb_jasper_ft")
    ap.add_argument("--model_tag", type=str, default=None, help="Cache tag; default uses model_dir name")
    ap.add_argument("--compression_ratio", type=float, default=0.3333)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--doc_bs", type=int, default=64)
    ap.add_argument("--query_bs", type=int, default=256)
    ap.add_argument("--chunk_size", type=int, default=50000, help="Doc chunk size for topk streaming (tune for GPU mem)")
    ap.add_argument("--force_recompute_docs", action="store_true")
    ap.add_argument("--corpus_override_dir", type=str, default=None,
                help="If set, load corpus from this dir as <domain>.en_only.jsonl.gz (queries stay official)")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--blacklist_path", type=str, default=None,
                help="txt file, one doc_id per line. Filter these docs from corpus")
    ap.add_argument(
        "--corpus_override_suffix",
        type=str,
        default=".en_only.jsonl.gz",
        help="Override corpus filename suffix, e.g. .cleaned.jsonl or .en_only.jsonl.gz"
    )
    ap.add_argument("--split2", action="store_true", help="Split long docs into 2 segments and aggregate back to parent doc_id")
    ap.add_argument("--pack_adjacent", action="store_true",
                help="Pack adjacent passages (~512 tokens) for coarse retrieval, then expand back to passage ids.")
    ap.add_argument("--pack_max_tokens", type=int, default=512)
    ap.add_argument("--pack_overlap_tokens", type=int, default=64)
    ap.add_argument("--pack_topk", type=int, default=200, help="TopK packs to retrieve per query")
    ap.add_argument("--expand_topm", type=int, default=1200, help="After expanding packs, keep topM passage candidates")
    ap.add_argument("--pack_agg", type=str, default="max", choices=["max", "sum"])
    ap.add_argument(
        "--tasks_root",
        type=str,
        default="human/retrieval_tasks",
        help="Root directory containing <domain>/<domain>.jsonl, <domain>_<task>.jsonl, and qrels/dev.tsv"
    )
    ap.add_argument(
        "--corpus_text_key",
        type=str,
        default="text",
        help="Which field to use as document text in corpus JSONL. e.g. text or ctx_text"
    )

    ap.add_argument("--test_input_jsonl", type=str, default=None,
                    help="If set, run TaskA submission on this test jsonl (conversation-format).")
    ap.add_argument("--query_mode", type=str, default="last_user", choices=["last_user", "full"])
    ap.add_argument("--query_last_n", type=int, default=None,
                    help="Only used when query_mode=full. Keep last N messages.")
    ap.add_argument("--skip_official_eval", action="store_true",
                    help="Do not run scripts/evaluation/run_retrieval_eval.py (useful for test submission).")



    args = ap.parse_args()

    cmd = " ".join(shlex.quote(x) for x in sys.argv)
    print(f"[CMD] {cmd}")
    print(f"[TIME] {datetime.datetime.now().isoformat(timespec='seconds')}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Ensure remote-code exists in FT dir
    sync_remote_code_if_missing(args.model_dir, args.base_dir)

    # Load Jasper FT
    model = load_jasper_st(
        model_name=args.model_dir,
        device=device,
        default_compression_ratio=args.compression_ratio,
    )
    model.max_seq_length = args.max_len
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)



    model_tag = args.model_tag or Path(args.model_dir).name

    # COLLECTIONS = {
    #     "clapnq": {
    #         "collection_name": "mt-rag-clapnq-elser-512-100-20240503",
    #         "root": "human/retrieval_tasks/clapnq",
    #         "corpus_file": "clapnq.jsonl",
    #         "query_file": f"clapnq_{args.task}.jsonl",
    #     },
    #     "fiqa": {
    #         "collection_name": "mt-rag-fiqa-beir-elser-512-100-20240501",
    #         "root": "human/retrieval_tasks/fiqa",
    #         "corpus_file": "fiqa.jsonl",
    #         "query_file": f"fiqa_{args.task}.jsonl",
    #     },
    #     "govt": {
    #         "collection_name": "mt-rag-govt-elser-512-100-20240611",
    #         "root": "human/retrieval_tasks/govt",
    #         "corpus_file": "govt.jsonl",
    #         "query_file": f"govt_{args.task}.jsonl",
    #     },
    #     "cloud": {
    #         "collection_name": "mt-rag-ibmcloud-elser-512-100-20240502",
    #         "root": "human/retrieval_tasks/cloud",
    #         "corpus_file": "cloud.jsonl",
    #         "query_file": f"cloud_{args.task}.jsonl",
    #     },
    # }

    COLLECTIONS = {
        "clapnq": {
            "collection_name": "mt-rag-clapnq-elser-512-100-20240503",
            "root": os.path.join(args.tasks_root, "clapnq"),
            "corpus_file": "clapnq.jsonl",
            "query_file": f"clapnq_{args.task}.jsonl",
        },
        "fiqa": {
            "collection_name": "mt-rag-fiqa-beir-elser-512-100-20240501",
            "root": os.path.join(args.tasks_root, "fiqa"),
            "corpus_file": "fiqa.jsonl",
            "query_file": f"fiqa_{args.task}.jsonl",
        },
        "govt": {
            "collection_name": "mt-rag-govt-elser-512-100-20240611",
            "root": os.path.join(args.tasks_root, "govt"),
            "corpus_file": "govt.jsonl",
            "query_file": f"govt_{args.task}.jsonl",
        },
        "cloud": {
            "collection_name": "mt-rag-ibmcloud-elser-512-100-20240502",
            "root": os.path.join(args.tasks_root, "cloud"),
            "corpus_file": "cloud.jsonl",
            "query_file": f"cloud_{args.task}.jsonl",
        },
    }

    os.makedirs("outputs", exist_ok=True)
    sub_path = f"outputs/{args.model_name}_{args.task}.jsonl"
    scored_path = f"outputs/{args.model_name}_{args.task}_score.jsonl"

    blacklist = load_blacklist(args.blacklist_path)


    all_results = []
    print("Start!")
    print("Current task:", args.task)

    test_queries_by_domain = None
    if args.test_input_jsonl:
        test_queries_by_domain = load_test_taska_queries(
            args.test_input_jsonl,
            query_mode=args.query_mode,
            last_n=args.query_last_n,
        )


    for name, cfg in COLLECTIONS.items():
        if test_queries_by_domain is not None:
            if name not in test_queries_by_domain:
                continue
            queries_override = test_queries_by_domain[name]
            collection_out = name   # 输出用短名 fiqa/govt/...
        else:
            queries_override = None
            collection_out = None
        res = run_retrieval_for_collection(
            name=name,
            cfg=cfg,
            model=model,
            top_k=args.top_k,
            cache_dir=args.cache_dir,
            model_tag=model_tag,
            compression_ratio=args.compression_ratio,
            doc_bs=args.doc_bs,
            query_bs=args.query_bs,
            force_recompute_docs=args.force_recompute_docs,
            chunk_size=args.chunk_size,
            corpus_override_dir=args.corpus_override_dir,
            blacklist=blacklist,
            corpus_override_suffix=args.corpus_override_suffix,
            split2=args.split2,
            tok=tok,
            max_len=args.max_len,
            pack_adjacent=args.pack_adjacent,
            pack_max_tokens=args.pack_max_tokens,
            pack_overlap_tokens=args.pack_overlap_tokens,
            pack_topk=args.pack_topk,
            expand_topm=args.expand_topm,
            pack_agg=args.pack_agg,
            corpus_text_key=args.corpus_text_key,
            queries_override=queries_override,
            collection_out=collection_out,
        )
        all_results.extend(res)

    with open(sub_path, "w", encoding="utf-8") as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Saved submission to:", sub_path)
    if not args.skip_official_eval:
        run_official_eval(sub_path, scored_path, model_name=args.model_name, task_name=args.task)

    # run_official_eval(sub_path, scored_path, model_name=args.model_name, task_name=args.task)


if __name__ == "__main__":
    main()
