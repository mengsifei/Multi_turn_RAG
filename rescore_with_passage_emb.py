#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, argparse, hashlib, time, gzip
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

DOMAINS = ["clapnq", "fiqa", "govt", "cloud"]

def guess_domain(collection_name: str) -> str:
    s = (collection_name or "").lower()
    for d in DOMAINS:
        if d in s:
            return d
    raise ValueError(f"Cannot infer domain from Collection={collection_name}")

def _norm_qid(qid: str) -> str:
    return qid.split("<::>", 1)[0] if "<::>" in qid else qid

def load_jsonl_text_map(path: Path,
                        id_keys=("id","_id","query_id","task_id","doc_id","corpus_id","document_id"),
                        text_keys=("text","query","contents","content","document","passage","body")) -> Dict[str,str]:
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            o = json.loads(line)
            _id = None
            for k in id_keys:
                v = o.get(k)
                if isinstance(v, str) and v:
                    _id = v; break
            if _id is None: continue

            txt = None
            for k in text_keys:
                v = o.get(k)
                if isinstance(v, str) and v.strip():
                    txt = v.strip(); break
            if txt is None:
                title, text = o.get("title"), o.get("text")
                if isinstance(title, str) and isinstance(text, str):
                    txt = (title.strip() + "\n" + text.strip()).strip()
            if txt:
                out[_id] = txt
    return out

def open_any(path: str):
    return gzip.open(path,"rt",encoding="utf-8",errors="ignore") if path.endswith(".gz") else open(path,"r",encoding="utf-8",errors="ignore")

def cache_sig(model_tag: str, dom: str, corpus_path: str, ids_sorted: List[str], max_len: int, cr: float) -> str:
    st = os.stat(corpus_path)
    h = hashlib.sha1()
    h.update(f"{model_tag}|{dom}|len={max_len}|cr={cr}|size={st.st_size}|mtime={int(st.st_mtime)}|n={len(ids_sorted)}".encode())
    # hash ids (stream)
    for x in ids_sorted:
        h.update(b"\0")
        h.update(x.encode("utf-8"))
    return h.hexdigest()[:16]

def load_or_build_doc_emb(
    model: SentenceTransformer,
    doc_texts: List[str],
    doc_ids: List[str],
    *,
    cache_dir: str,
    cache_name: str,
    doc_bs: int,
    compression_ratio: float,
):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_path = os.path.join(cache_dir, cache_name + ".pt")
    if os.path.exists(cache_path):
        pack = torch.load(cache_path, map_location="cpu")
        if pack.get("doc_ids") == doc_ids:
            return pack["doc_emb"]  # CPU fp16
    # build
    all_emb = []
    for i in tqdm(range(0, len(doc_texts), doc_bs), desc=f"Encode docs ({cache_name})"):
        batch = doc_texts[i:i+doc_bs]
        emb = model.encode(
            batch,
            batch_size=doc_bs,
            convert_to_tensor=True,
            show_progress_bar=False,
            normalize_embeddings=True,
            prompt_name=None,
            compression_ratio=compression_ratio,
        )
        all_emb.append(emb.cpu())
    doc_emb = torch.cat(all_emb, dim=0).to(torch.float16).contiguous()
    torch.save({"doc_ids": doc_ids, "doc_emb": doc_emb, "saved_time": time.time()}, cache_path)
    return doc_emb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["lastturn","questions","rewrite"])
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)

    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--model_tag", default=None)

    ap.add_argument("--retrieval_tasks_root", default="cleaned_dataset",
                    help="queries root containing <dom>/<dom>_<task>.jsonl")
    ap.add_argument("--corpus_override_dir", default="human/retrieval_tasks_derived")
    ap.add_argument("--corpus_override_suffix", default=".cleaned.jsonl")

    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--compression_ratio", type=float, default=0.3333)
    ap.add_argument("--doc_bs", type=int, default=128)
    ap.add_argument("--query_bs", type=int, default=256)

    ap.add_argument("--top_k_out", type=int, default=1000)
    ap.add_argument("--local_files_only", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_tag = args.model_tag or Path(args.model_dir).name

    # load model
    model = SentenceTransformer(
        args.model_dir,
        model_kwargs={"torch_dtype": torch.bfloat16, "attn_implementation": "sdpa", "trust_remote_code": True,
                      "device_map": "cuda" if device=="cuda" else None},
        tokenizer_kwargs={"padding_side":"left","trust_remote_code":True, "fix_mistral_regex":True},
        trust_remote_code=True,
        local_files_only=bool(args.local_files_only),
        device=device,
    )
    model.max_seq_length = args.max_len

    # load queries per domain
    qmap = {}
    for d in DOMAINS:
        qf = Path(args.retrieval_tasks_root) / d / f"{d}_{args.task}.jsonl"
        qmap[d] = load_jsonl_text_map(qf, id_keys=("task_id","_id","id","query_id"), text_keys=("text","query"))

    # first pass: read run, collect per-domain candidate ids and per-query candidate list
    run_items = []  # keep parsed lines (to rewrite)
    dom_cand_ids = defaultdict(set)
    for line in open(args.in_jsonl, "r", encoding="utf-8"):
        o = json.loads(line)
        dom = guess_domain(o.get("Collection",""))
        qid = o["task_id"]
        ctxs = o.get("contexts", [])
        cids = [c["document_id"] for c in ctxs]
        for cid in cids:
            dom_cand_ids[dom].add(cid)
        run_items.append(o)

    # load corpus texts (only for candidate ids)
    dom_doc_text = {}
    for dom in DOMAINS:
        need = dom_cand_ids.get(dom, set())
        if not need:
            dom_doc_text[dom] = {}
            continue
        corpus_path = os.path.join(args.corpus_override_dir, dom + args.corpus_override_suffix)
        m = {}
        with open_any(corpus_path) as f:
            for line in f:
                o = json.loads(line)
                _id = str(o.get("document_id") or o.get("_id") or o.get("id") or "")
                if _id in need:
                    title = o.get("title") or ""
                    text = o.get("text") or o.get("contents") or ""
                    full = (str(title).strip() + " " + str(text).strip()).strip()
                    m[_id] = full
        dom_doc_text[dom] = m

    # build/load doc embeddings per domain
    dom_doc_ids = {}
    dom_doc_emb = {}
    dom_id2idx = {}
    for dom in DOMAINS:
        need = sorted(dom_cand_ids.get(dom, set()))
        if not need:
            continue
        corpus_path = os.path.join(args.corpus_override_dir, dom + args.corpus_override_suffix)
        sig = cache_sig(model_tag, dom, corpus_path, need, args.max_len, args.compression_ratio)
        cache_name = f"{dom}__candpassage__{model_tag}__{sig}"
        texts = [dom_doc_text[dom].get(i, "") for i in need]
        # (极少数缺失文本时保底空串，但你前面 sanity 已经验证 missing=0)
        emb = load_or_build_doc_emb(
            model, texts, need,
            cache_dir=args.cache_dir,
            cache_name=cache_name,
            doc_bs=args.doc_bs,
            compression_ratio=args.compression_ratio,
        )
        dom_doc_ids[dom] = need
        dom_doc_emb[dom] = emb  # CPU fp16
        dom_id2idx[dom] = {pid:i for i,pid in enumerate(need)}

    # query embeddings per domain
    dom_q_emb = {}
    for dom in DOMAINS:
        # collect qids in this domain
        qids = [o["task_id"] for o in run_items if guess_domain(o.get("Collection","")) == dom]
        if not qids:
            continue
        qtexts = []
        for qid in qids:
            qt = qmap[dom].get(qid) or qmap[dom].get(_norm_qid(qid))
            if not qt:
                raise KeyError(f"Query text not found: dom={dom} qid={qid}")
            qtexts.append(qt)
        all_emb = []
        for i in tqdm(range(0, len(qtexts), args.query_bs), desc=f"Encode queries ({dom})"):
            batch = qtexts[i:i+args.query_bs]
            emb = model.encode(
                batch,
                batch_size=args.query_bs,
                convert_to_tensor=True,
                show_progress_bar=False,
                normalize_embeddings=True,
                prompt_name="query",
                compression_ratio=args.compression_ratio,
            )
            all_emb.append(emb.cpu())
        dom_q_emb[dom] = (qids, torch.cat(all_emb, dim=0).to(torch.float16).contiguous())

    # rescore and write
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fout:
        # build quick qid->row lookup per domain
        dom_qid2row = {}
        for dom, (qids, qemb) in dom_q_emb.items():
            dom_qid2row[dom] = {qid:i for i,qid in enumerate(qids)}

        for o in tqdm(run_items, desc="Rescoring"):
            dom = guess_domain(o.get("Collection",""))
            ctxs = o.get("contexts", [])
            if not ctxs:
                fout.write(json.dumps(o, ensure_ascii=False) + "\n")
                continue

            qid = o["task_id"]
            row = dom_qid2row[dom][qid]
            qv = dom_q_emb[dom][1][row].to("cuda" if torch.cuda.is_available() else "cpu").float()  # [D]

            # gather doc vectors
            id2idx = dom_id2idx[dom]
            idx = [id2idx[c["document_id"]] for c in ctxs]
            dv = dom_doc_emb[dom][torch.tensor(idx, dtype=torch.long)].to(qv.device).float()  # [M, D]
            scores = dv @ qv  # [M]
            k = min(args.top_k_out, scores.numel())
            vals, pos = torch.topk(scores, k=k, largest=True)

            new_ctxs = []
            for v, p in zip(vals.tolist(), pos.tolist()):
                c = ctxs[p]
                new_ctxs.append({
                    "document_id": c["document_id"],
                    "score": float(v),
                    "coarse_score": float(c.get("score", 0.0)),
                })

            o2 = {"task_id": o["task_id"], "contexts": new_ctxs, "Collection": o.get("Collection")}
            fout.write(json.dumps(o2, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
