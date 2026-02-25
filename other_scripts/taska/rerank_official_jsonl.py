#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rerank_official_jsonl.py

- Supports two reranker modes:
  1) yesno_llm: Qwen-style yes/no via CausalLM last-token probability
  2) cross_encoder:
      - If reranker name contains "zerank": uses sentence_transformers.CrossEncoder (trust_remote_code=True)
      - Otherwise: uses transformers AutoModelForSequenceClassification

- Adds:
  * --fallback_to_human : if doc missing in cleaned corpus, try human/retrieval_tasks corpus
  * --human_tasks_root  : where to load the fallback corpus from (default human/retrieval_tasks)
  * robust handling when a doc is missing even after fallback:
      - keep it with a very low score so output still has keep_topk items
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys, shlex, datetime

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from sentence_transformers import CrossEncoder


# ---------------------------
# Qwen yes/no reranker template
# ---------------------------
PREFIX = (
    '<|im_start|>system\n'
    'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
    'Note that the answer can only be "yes" or "no".<|im_end|>\n'
    '<|im_start|>user\n'
)
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


# -----------------------------
# Helpers
# -----------------------------
def guess_domain(collection_name: str) -> str:
    s = (collection_name or "").lower()
    for d in ["clapnq", "fiqa", "govt", "cloud"]:
        if d in s:
            return d
    raise ValueError(f"Cannot infer domain from Collection={collection_name}")


def load_jsonl_text_map(
    path: Path,
    *,
    id_keys=("id", "_id", "query_id", "task_id", "doc_id", "corpus_id", "document_id"),
    text_keys=("ctx_text", "text", "query", "contents", "content", "document", "passage", "body"),
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            _id = None
            for k in id_keys:
                v = obj.get(k)
                if isinstance(v, str) and v:
                    _id = v
                    break
            if _id is None:
                continue

            txt = None
            for k in text_keys:
                v = obj.get(k)
                if isinstance(v, str) and v.strip():
                    txt = v.strip()
                    break

            # BEIR corpus common: {"_id": "...", "title": "...", "text": "..."}
            if txt is None:
                title = obj.get("title")
                text = obj.get("text")
                if isinstance(title, str) and isinstance(text, str):
                    txt = (title.strip() + "\n" + text.strip()).strip()

            if isinstance(txt, str) and txt:
                out[_id] = txt
    return out


def _norm_qid(qid: str) -> str:
    return qid.split("<::>", 1)[0] if "<::>" in qid else qid


def get_query_text(qmap: Dict[str, str], task_id: str) -> Optional[str]:
    if task_id in qmap:
        return qmap[task_id]
    base = _norm_qid(task_id)
    return qmap.get(base)


def format_header(instruction: str, query: str) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: "


def build_inputs_doc_trunc(
    tokenizer,
    headers: List[str],
    docs: List[str],
    max_length: int,
    prefix_tokens: List[int],
    suffix_tokens: List[int],
):
    avail = max_length - len(prefix_tokens) - len(suffix_tokens)
    if avail <= 0:
        raise ValueError(
            f"max_length={max_length} too small for prefix+suffix "
            f"({len(prefix_tokens)}+{len(suffix_tokens)})"
        )

    input_ids = []
    for h, d in zip(headers, docs):
        h_ids = tokenizer.encode(h, add_special_tokens=False, truncation=False)

        if len(h_ids) > avail:
            h_ids = h_ids[-avail:]
            remain = 0
        else:
            remain = avail - len(h_ids)

        if remain > 0:
            d_ids = tokenizer.encode(d, add_special_tokens=False, truncation=True, max_length=remain)
        else:
            d_ids = []

        ids = prefix_tokens + h_ids + d_ids + suffix_tokens
        if len(ids) > max_length:
            ids = ids[-max_length:]
        input_ids.append(ids)

    batch = tokenizer.pad(
        {"input_ids": input_ids},
        padding=True,
        return_tensors="pt",
    )
    return batch


# -----------------------------
# Cross-encoder scoring (transformers)
# -----------------------------
@torch.no_grad()
def rerank_scores_ce_tf(
    model,
    tokenizer,
    query: str,
    docs: List[str],
    *,
    max_length=1024,
    batch_size=32,
    sigmoid=False,
) -> List[float]:
    device = next(model.parameters()).device
    out: List[float] = []

    for s in range(0, len(docs), batch_size):
        d_b = docs[s : s + batch_size]
        features = tokenizer(
            [query] * len(d_b),
            d_b,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        features = {k: v.to(device) for k, v in features.items()}

        logits = model(**features).logits  # [B], [B,1], or [B,2]

        if logits.ndim == 1:
            scores = logits
            if sigmoid:
                scores = torch.sigmoid(scores)

        elif logits.ndim == 2 and logits.size(-1) == 1:
            scores = logits.squeeze(-1)
            if sigmoid:
                scores = torch.sigmoid(scores)

        elif logits.ndim == 2 and logits.size(-1) == 2:
            if sigmoid:
                scores = torch.softmax(logits, dim=-1)[:, 1]
            else:
                scores = logits[:, 1]
        else:
            scores = logits.squeeze()
            if scores.ndim != 1:
                raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")
            if sigmoid:
                scores = torch.sigmoid(scores)

        out.extend(scores.detach().float().cpu().tolist())

    return out


# -----------------------------
# Qwen yes/no scoring (CausalLM)
# -----------------------------
def _token_id_safe(tokenizer, s: str) -> Optional[int]:
    tid = tokenizer.convert_tokens_to_ids(s)
    if tid is None:
        return None
    return int(tid)


def _collect_variant_ids(tokenizer, variants) -> List[int]:
    ids = []
    for v in variants:
        tid = _token_id_safe(tokenizer, v)
        if tid is not None:
            ids.append(tid)
    seen = set()
    out = []
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    if not out:
        raise ValueError(f"None of these tokens exist in vocab: {variants}")
    return out


@torch.no_grad()
def rerank_scores_yesno_llm(model, tokenizer, headers, docs, *, max_length=2048, batch_size=8) -> List[float]:
    yes_ids = _collect_variant_ids(tokenizer, [" yes", "yes"])
    no_ids = _collect_variant_ids(tokenizer, [" no", "no"])

    prefix_tokens = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(SUFFIX, add_special_tokens=False)

    out: List[float] = []
    for s in range(0, len(docs), batch_size):
        h_b = headers[s : s + batch_size]
        d_b = docs[s : s + batch_size]

        inputs = build_inputs_doc_trunc(
            tokenizer, h_b, d_b, max_length, prefix_tokens, suffix_tokens
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        logits_last = model(**inputs, use_cache=False).logits[:, -1, :]  # [B, V]

        yes_logits = logits_last.index_select(1, torch.tensor(yes_ids, device=logits_last.device))
        no_logits = logits_last.index_select(1, torch.tensor(no_ids, device=logits_last.device))

        yes_v = torch.logsumexp(yes_logits, dim=1)
        no_v = torch.logsumexp(no_logits, dim=1)

        two = torch.stack([no_v, yes_v], dim=1)
        p_yes = F.softmax(two, dim=1)[:, 1]
        out.extend(p_yes.detach().cpu().tolist())

    return out


# -----------------------------
# Model resolver
# -----------------------------
def resolve_local_or_id(model_arg: str, *, local_only: bool) -> str:
    p = Path(model_arg)
    if p.exists():
        return str(p.resolve())
    if local_only and (model_arg.startswith("./") or model_arg.startswith("/") or model_arg.startswith("../")):
        raise FileNotFoundError(f"--reranker points to a local path but it does not exist: {model_arg}")
    return model_arg


def count_lines(path: Path) -> int:
    n = 0
    with path.open("rb") as f:
        for _ in f:
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--task", required=True, choices=["lastturn", "questions", "rewrite", "rewrite_gpt", "concat_lastturn_rewrite_gpt"])
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)

    ap.add_argument("--reranker", required=True)
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION)

    ap.add_argument("--retrieval_tasks_root", default="human/retrieval_tasks")

    ap.add_argument("--corpus_clapnq", default=None)
    ap.add_argument("--corpus_fiqa", default=None)
    ap.add_argument("--corpus_govt", default=None)
    ap.add_argument("--corpus_cloud", default=None)

    ap.add_argument("--cand_topn", type=int, default=100)
    ap.add_argument("--keep_topk", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=1.0)

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=8192)

    ap.add_argument("--local_files_only", action="store_true")

    ap.add_argument("--reranker_mode", choices=["yesno_llm", "cross_encoder"], default="yesno_llm")
    ap.add_argument("--ce_max_length", type=int, default=1024)
    ap.add_argument("--ce_sigmoid", action="store_true")

    # Fallback to human corpus when doc missing in cleaned corpus
    ap.add_argument(
        "--fallback_to_human",
        action="store_true",
        help="If doc missing in retrieval_tasks_root corpus, fallback to human_tasks_root corpus.",
    )
    ap.add_argument(
        "--human_tasks_root",
        default="human/retrieval_tasks",
        help="Root path for fallback human corpus (default: human/retrieval_tasks).",
    )

    # Progress logging
    ap.add_argument("--progress_every", type=int, default=10, help="Print progress every N queries.")

    args = ap.parse_args()

    cmd = " ".join(shlex.quote(x) for x in sys.argv)
    print(f"[CMD] {cmd}")
    print(f"[TIME] {datetime.datetime.now().isoformat(timespec='seconds')}")

    root = Path(args.retrieval_tasks_root)
    human_root = Path(args.human_tasks_root)

    def query_file(dom: str) -> Path:
        return root / dom / f"{dom}_{args.task}.jsonl"

    def corpus_file(dom: str) -> Path:
        override = getattr(args, f"corpus_{dom}", None)
        if override:
            return Path(override)
        return root / dom / f"{dom}.jsonl"

    def human_corpus_file(dom: str) -> Path:
        return human_root / dom / f"{dom}.jsonl"

    domains = ["clapnq", "fiqa", "govt", "cloud"]

    # Load query/corpus maps once
    queries: Dict[str, Dict[str, str]] = {}
    corpus_clean: Dict[str, Dict[str, str]] = {}
    corpus_human: Dict[str, Dict[str, str]] = {}

    for dom in domains:
        qf = query_file(dom)
        cf = corpus_file(dom)

        print(f"[QUERY] {dom}: {qf}")
        if not qf.exists():
            raise FileNotFoundError(f"Missing query file for {dom}: {qf}")
        if not cf.exists():
            raise FileNotFoundError(f"Missing corpus file for {dom}: {cf}")

        queries[dom] = load_jsonl_text_map(qf)
        corpus_clean[dom] = load_jsonl_text_map(cf)

        if args.fallback_to_human:
            hf = human_corpus_file(dom)
            if not hf.exists():
                raise FileNotFoundError(f"--fallback_to_human enabled but missing human corpus for {dom}: {hf}")
            corpus_human[dom] = load_jsonl_text_map(hf)

    reranker_id = resolve_local_or_id(args.reranker, local_only=bool(args.local_files_only))
    reranker_name = str(reranker_id).lower()
    is_zerank = ("zerank" in reranker_name)

    # Load reranker
    model = None
    tokenizer = None
    ce = None  # sentence-transformers CrossEncoder (zerank)

    if args.reranker_mode == "yesno_llm":
        tokenizer = AutoTokenizer.from_pretrained(
            reranker_id,
            padding_side="left",
            local_files_only=bool(args.local_files_only),
            trust_remote_code=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            reranker_id,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            local_files_only=bool(args.local_files_only),
            trust_remote_code=True,
        ).eval()

    else:
        if is_zerank:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            ce = CrossEncoder(reranker_id, trust_remote_code=True, device=device)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                reranker_id,
                padding_side="right",
                local_files_only=bool(args.local_files_only),
                trust_remote_code=True,
            )

            if tokenizer.pad_token_id is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                elif tokenizer.sep_token is not None:
                    tokenizer.pad_token = tokenizer.sep_token
                else:
                    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

            dtype_arg = torch.float16 if torch.cuda.is_available() else None

            model = AutoModelForSequenceClassification.from_pretrained(
                reranker_id,
                device_map="auto",
                torch_dtype=dtype_arg,
                local_files_only=bool(args.local_files_only),
                trust_remote_code=True,
            ).eval()

            if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
                model.resize_token_embeddings(len(tokenizer))

            if getattr(model.config, "pad_token_id", None) is None:
                model.config.pad_token_id = tokenizer.pad_token_id

    in_path = Path(args.in_jsonl)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = count_lines(in_path)
    print(f"[INFO] total_queries={total}", flush=True)

    missing_total = 0
    missing_after_fallback_total = 0

    LOW_SCORE = -1e9  # for truly-missing docs, keep but push to the bottom

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin, 1):
            if i == 1 or (args.progress_every > 0 and i % args.progress_every == 0):
                pct = (i / total) if total > 0 else 0.0
                print(
                    f"[PROGRESS] {i}/{total} ({pct:.1%}) missing_total={missing_total} missing_after_fallback={missing_after_fallback_total}",
                    flush=True,
                )

            item = json.loads(line)
            dom = guess_domain(item.get("Collection", ""))
            task_id = item["task_id"]

            qtext = get_query_text(queries[dom], task_id)
            if not qtext:
                raise KeyError(
                    f"Query text not found for task_id={task_id} (domain={dom}). "
                    f"Loaded queries from: {query_file(dom)}"
                )

            ctxs = item.get("contexts", [])
            if not ctxs:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                continue

            ctxs_sorted = sorted(ctxs, key=lambda c: float(c.get("score", 0.0)), reverse=True)
            cand_all = ctxs_sorted[: args.cand_topn]

            # Build docs list for those we can actually fetch text for
            docs_to_score: List[str] = []
            cand_to_score: List[dict] = []
            cand_missing: List[dict] = []

            for c in cand_all:
                did = c.get("document_id")
                if not did:
                    # weird candidate; keep but push down
                    c["orig_score"] = float(c.get("score", 0.0))
                    c["rerank_score"] = None
                    c["score"] = LOW_SCORE
                    cand_missing.append(c)
                    missing_after_fallback_total += 1
                    continue

                dtext = corpus_clean[dom].get(did)
                if dtext is None and args.fallback_to_human:
                    missing_total += 1
                    dtext = corpus_human.get(dom, {}).get(did)

                if dtext is None:
                    # still missing after fallback -> keep but make it last
                    missing_after_fallback_total += 1
                    c["orig_score"] = float(c.get("score", 0.0))
                    c["rerank_score"] = None
                    c["score"] = LOW_SCORE
                    cand_missing.append(c)
                    continue

                docs_to_score.append(dtext)
                cand_to_score.append(c)

            # Score only those we can fetch text for
            rr: List[float] = []
            if docs_to_score:
                if args.reranker_mode == "yesno_llm":
                    headers = [format_header(args.instruction, qtext)] * len(docs_to_score)
                    rr = rerank_scores_yesno_llm(
                        model, tokenizer, headers, docs_to_score,
                        max_length=args.max_length,
                        batch_size=args.batch_size,
                    )
                else:
                    if ce is not None:
                        rr = ce.predict(list(zip([qtext] * len(docs_to_score), docs_to_score)), batch_size=args.batch_size)
                        rr = [float(x) for x in rr]
                    else:
                        rr = rerank_scores_ce_tf(
                            model, tokenizer, qtext, docs_to_score,
                            max_length=args.ce_max_length,
                            batch_size=args.batch_size,
                            sigmoid=bool(args.ce_sigmoid),
                        )

                if len(rr) != len(cand_to_score):
                    raise RuntimeError(f"Internal error: rr len {len(rr)} != cand_to_score len {len(cand_to_score)}")

                alpha = float(args.alpha)
                for c, s in zip(cand_to_score, rr):
                    orig = float(c.get("score", 0.0))
                    c["orig_score"] = orig
                    c["rerank_score"] = float(s)
                    c["score"] = alpha * float(s) + (1.0 - alpha) * orig

            # Merge back: scored + missing (missing already has LOW_SCORE)
            merged = cand_to_score + cand_missing
            merged = sorted(merged, key=lambda c: float(c.get("score", 0.0)), reverse=True)

            # Keep topk
            item["contexts"] = merged[: args.keep_topk]
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("[DONE] wrote:", out_path, flush=True)
    if missing_total or missing_after_fallback_total:
        print(
            f"[STATS] missing_in_clean={missing_total} still_missing_after_fallback={missing_after_fallback_total}",
            flush=True,
        )


if __name__ == "__main__":
    main()
