#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys, shlex, datetime

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
    # text_keys=("text", "query", "contents", "content", "document", "passage", "body"),
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
    # exact first
    if task_id in qmap:
        return qmap[task_id]
    # fallback base id (strip <::>turn)
    base = _norm_qid(task_id)
    return qmap.get(base)


# def format_pair(instruction: str, query: str, doc: str) -> str:
#     return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


# def build_inputs(tokenizer, pairs: List[str], max_length: int,
#                  prefix_tokens: List[int], suffix_tokens: List[int]):
#     inputs = tokenizer(
#         pairs,
#         padding=False,
#         truncation="longest_first",
#         return_attention_mask=False,
#         max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
#     )
#     for i, ele in enumerate(inputs["input_ids"]):
#         inputs["input_ids"][i] = prefix_tokens + ele + suffix_tokens
#     inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
#     return inputs


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
        raise ValueError(f"max_length={max_length} too small for prefix+suffix "
                         f"({len(prefix_tokens)}+{len(suffix_tokens)})")

    input_ids = []
    for h, d in zip(headers, docs):
        # 1) header：不使用自动截断，避免把 "<Document>:" 截掉
        h_ids = tokenizer.encode(h, add_special_tokens=False, truncation=False)

        # 如果 header 超了 avail：保留 header 尾部（更可能保住 Query + <Document>:）
        if len(h_ids) > avail:
            h_ids = h_ids[-avail:]
            remain = 0
        else:
            remain = avail - len(h_ids)

        # 2) doc：只在剩余预算里截断
        if remain > 0:
            d_ids = tokenizer.encode(d, add_special_tokens=False, truncation=True, max_length=remain)
        else:
            d_ids = []

        ids = prefix_tokens + h_ids + d_ids + suffix_tokens
        # debug sanity：确保不超过 max_length
        if len(ids) > max_length:
            ids = ids[-max_length:]  # 理论上不会触发，保险兜底（保尾部）
        input_ids.append(ids)

    batch = tokenizer.pad(
        {"input_ids": input_ids},
        padding=True,
        return_tensors="pt",
        # return_attention_mask=True,  # 默认就是 True；想显式也可打开
    )
    return batch




# @torch.no_grad()
# def rerank_scores(model, tokenizer, headers, docs, *, max_length=2048, batch_size=8):
#     token_no = tokenizer.convert_tokens_to_ids("no")
#     token_yes = tokenizer.convert_tokens_to_ids("yes")
#     prefix_tokens = tokenizer.encode(PREFIX, add_special_tokens=False)
#     suffix_tokens = tokenizer.encode(SUFFIX, add_special_tokens=False)

#     out = []
#     for s in range(0, len(docs), batch_size):
#         h_b = headers[s:s+batch_size]
#         d_b = docs[s:s+batch_size]
#         inputs = build_inputs_doc_trunc(tokenizer, h_b, d_b, max_length, prefix_tokens, suffix_tokens)
#         inputs = {k: v.to(model.device) for k, v in inputs.items()}

#         # 关键：禁用 cache，显著降显存风险
#         logits_last = model(**inputs, use_cache=False).logits[:, -1, :]

#         yes_v = logits_last[:, token_yes]
#         no_v  = logits_last[:, token_no]
#         two = torch.stack([no_v, yes_v], dim=1)
#         p_yes = torch.nn.functional.log_softmax(two, dim=1).exp()[:, 1]
#         out.extend(p_yes.detach().cpu().tolist())
#     return out

import torch
import torch.nn.functional as F

def _token_id_safe(tokenizer, s: str) -> int | None:
    tid = tokenizer.convert_tokens_to_ids(s)
    # 有些 tokenizer 遇到不存在 token 会返回 unk 或 None；这里简单过滤掉无效 id
    if tid is None:
        return None
    # 经验：有的实现会返回 0/unk_id，你也可以按 tokenizer.unk_token_id 过滤
    return int(tid)

def _collect_variant_ids(tokenizer, variants):
    ids = []
    for v in variants:
        tid = _token_id_safe(tokenizer, v)
        if tid is not None:
            ids.append(tid)
    # 去重，保持顺序
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
def rerank_scores(model, tokenizer, headers, docs, *, max_length=2048, batch_size=8):
    # 关键：Qwen 常用的是 " yes"/" no"（带空格），但也保留不带空格的备选
    yes_ids = _collect_variant_ids(tokenizer, [" yes", "yes"])
    no_ids  = _collect_variant_ids(tokenizer, [" no",  "no"])

    prefix_tokens = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(SUFFIX, add_special_tokens=False)

    out = []
    for s in range(0, len(docs), batch_size):
        h_b = headers[s:s+batch_size]
        d_b = docs[s:s+batch_size]

        inputs = build_inputs_doc_trunc(
            tokenizer, h_b, d_b,
            max_length, prefix_tokens, suffix_tokens
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # 禁用 KV cache 降显存风险
        logits_last = model(**inputs, use_cache=False).logits[:, -1, :]  # [B, V]

        # 把 " yes"/"yes" 合成一个“yes logit”，" no"/"no" 合成一个“no logit”
        yes_logits = logits_last.index_select(1, torch.tensor(yes_ids, device=logits_last.device))
        no_logits  = logits_last.index_select(1, torch.tensor(no_ids,  device=logits_last.device))

        # 多个 token 变体用 logsumexp 合并（比 max 更平滑）
        yes_v = torch.logsumexp(yes_logits, dim=1)  # [B]
        no_v  = torch.logsumexp(no_logits,  dim=1)  # [B]

        two = torch.stack([no_v, yes_v], dim=1)     # [B, 2]
        p_yes = F.softmax(two, dim=1)[:, 1]
        out.extend(p_yes.detach().cpu().tolist())

    return out



def resolve_local_or_id(model_arg: str, *, local_only: bool) -> str:
    p = Path(model_arg)
    if p.exists():
        return str(p.resolve())
    # if local-only requested, fail fast with a clear message
    if local_only and (model_arg.startswith("./") or model_arg.startswith("/") or model_arg.startswith("../")):
        raise FileNotFoundError(f"--reranker points to a local path but it does not exist: {model_arg}")
    return model_arg


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--task", required=True, choices=["lastturn", "questions", "rewrite", "rewrite_gpt"],
                    help="Which query file to use: <dom>_<task>.jsonl")

    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)

    ap.add_argument("--reranker", required=True)
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION)

    ap.add_argument("--retrieval_tasks_root", default="human/retrieval_tasks",
                    help="Root containing per-domain folders (fiqa/, clapnq/, ...).")

    # Optional overrides for corpus paths (if you don't use <root>/<dom>/<dom>.jsonl)
    ap.add_argument("--corpus_clapnq", default=None)
    ap.add_argument("--corpus_fiqa", default=None)
    ap.add_argument("--corpus_govt", default=None)
    ap.add_argument("--corpus_cloud", default=None)

    ap.add_argument("--cand_topn", type=int, default=100)
    ap.add_argument("--keep_topk", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=1.0)

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=8192)

    ap.add_argument("--local_files_only", action="store_true", help="Force offline load for reranker model/tokenizer.")

    args = ap.parse_args()

    cmd = " ".join(shlex.quote(x) for x in sys.argv)
    print(f"[CMD] {cmd}")
    print(f"[TIME] {datetime.datetime.now().isoformat(timespec='seconds')}")

    root = Path(args.retrieval_tasks_root)

    # Build per-domain query-text paths: <root>/<dom>/<dom>_<task>.jsonl
    def query_file(dom: str) -> Path:
        return root / dom / f"{dom}_{args.task}.jsonl"

    # Build per-domain corpus paths: <root>/<dom>/<dom>.jsonl (or override)
    def corpus_file(dom: str) -> Path:
        override = getattr(args, f"corpus_{dom}", None)
        if override:
            return Path(override)
        return root / dom / f"{dom}.jsonl"

    # Load queries + corpus maps
    domains = ["clapnq", "fiqa", "govt", "cloud"]

    queries: Dict[str, Dict[str, str]] = {}
    corpus: Dict[str, Dict[str, str]] = {}

    for dom in domains:
        qf = query_file(dom)
        cf = corpus_file(dom)
        print(f"[QUERY] {dom}: {qf}")
        if not qf.exists():
            raise FileNotFoundError(f"Missing query file for {dom}: {qf}")
        if not cf.exists():
            raise FileNotFoundError(f"Missing corpus file for {dom}: {cf}")

        queries[dom] = load_jsonl_text_map(qf)
        corpus[dom] = load_jsonl_text_map(cf)

    # Load reranker
    reranker_id = resolve_local_or_id(args.reranker, local_only=bool(args.local_files_only))
    tokenizer = AutoTokenizer.from_pretrained(
        reranker_id,
        padding_side="left",
        local_files_only=bool(args.local_files_only),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Transformers warns torch_dtype deprecated in some versions; keep compatible:
    model = AutoModelForCausalLM.from_pretrained(
        reranker_id,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        local_files_only=bool(args.local_files_only),
    ).eval()
    tokenizer.pad_token = tokenizer.eos_token

    in_path = Path(args.in_jsonl)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
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

            # topN by original score
            ctxs_sorted = sorted(ctxs, key=lambda c: float(c.get("score", 0.0)), reverse=True)
            cand = ctxs_sorted[: args.cand_topn]

            docs: List[str] = []
            for c in cand:
                did = c["document_id"]
                dtext = corpus[dom].get(did)
                if not dtext:
                    raise KeyError(
                        f"Doc text not found for document_id={did} (domain={dom}). "
                        f"Loaded corpus from: {corpus_file(dom)}"
                    )
                docs.append(dtext)
            
            
            # pairs = [format_pair(args.instruction, qtext, d) for d in docs]
            # rr = rerank_scores(model, tokenizer, pairs, max_length=args.max_length, batch_size=args.batch_size)

            headers = [format_header(args.instruction, qtext)] * len(docs)
            rr = rerank_scores(model, tokenizer, headers, docs, max_length=args.max_length, batch_size=args.batch_size)


            alpha = float(args.alpha)
            for c, s in zip(cand, rr):
                orig = float(c.get("score", 0.0))
                c["orig_score"] = orig
                c["rerank_score"] = float(s)
                c["score"] = alpha * float(s) + (1.0 - alpha) * orig

            cand = sorted(cand, key=lambda c: float(c["score"]), reverse=True)[: args.keep_topk]
            item["contexts"] = cand
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
