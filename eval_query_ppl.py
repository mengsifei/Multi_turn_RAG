#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


@torch.no_grad()
def batch_ppl(
    model,
    tok,
    texts: List[str],
    device: torch.device,
    max_length: int,
) -> Tuple[List[float], float, int]:
    """
    Returns:
      ppl_per_sample: list of per-sample perplexity (NaN if too short)
      batch_nll_sum: sum of negative log-likelihood over all tokens in batch
      batch_tok_cnt: number of tokens counted (excluding padding, excluding first token)
    """
    enc = tok(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    out = model(input_ids=input_ids, attention_mask=attn)
    logits = out.logits  # [B, T, V]

    # shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attn[:, 1:].contiguous()

    B, Tm1 = shift_labels.shape
    V = shift_logits.size(-1)

    # token-level CE loss (no reduction)
    loss_flat = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_labels.view(-1),
        reduction="none",
    )
    loss_tok = loss_flat.view(B, Tm1)

    # mask padding
    nll_sum_per = (loss_tok * shift_mask).sum(dim=1)            # [B]
    tok_cnt_per = shift_mask.sum(dim=1)                         # [B]

    # per-sample ppl
    ppl_per = []
    for i in range(B):
        tc = int(tok_cnt_per[i].item())
        if tc <= 0:
            ppl_per.append(float("nan"))
        else:
            ppl_per.append(float(torch.exp(nll_sum_per[i] / tok_cnt_per[i]).item()))

    batch_nll_sum = float(nll_sum_per.sum().item())
    batch_tok_cnt = int(tok_cnt_per.sum().item())
    return ppl_per, batch_nll_sum, batch_tok_cnt


def safe_percentile(xs: List[float], p: float) -> float:
    ys = sorted([x for x in xs if not (math.isnan(x) or math.isinf(x))])
    if not ys:
        return float("nan")
    k = (len(ys) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ys[int(k)]
    return ys[f] * (c - k) + ys[c] * (k - f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True, help="Input jsonl")
    ap.add_argument("--model", required=True, help="Local HF model path, e.g. ./Qwen3-0.6B")
    ap.add_argument("--text_key", default="text", help="Field name for query text (default: text)")
    ap.add_argument("--id_key", default="_id", help="Field name for id (default: _id)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=256, help="Max token length for each query (truncate)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--out_jsonl", default="", help="Optional: write jsonl with ppl field added")
    ap.add_argument("--out_field", default="ppl", help="Field name for ppl in output jsonl")
    ap.add_argument("--topk_print", type=int, default=20, help="Print top-K max perplexity queries")
    ap.add_argument("--topk_out_jsonl", default="", help="Optional: write top-K max ppl queries to jsonl")
    ap.add_argument("--preview_chars", type=int, default=200, help="Preview chars when printing")
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()

    rows = load_jsonl(Path(args.in_jsonl))

    ppl_all: List[float] = []
    total_nll = 0.0
    total_tok = 0

    out_rows = []
    bs = args.batch_size

    # store (ppl, id, text, global_idx)
    items: List[Tuple[float, str, str, int]] = []

    for i in tqdm(range(0, len(rows), bs), desc="PPL", dynamic_ncols=True):
        batch = rows[i:i + bs]
        texts = []
        for r in batch:
            t = r.get(args.text_key, "")
            if not isinstance(t, str):
                t = str(t)
            texts.append(t)

        ppl_per, nll_sum, tok_cnt = batch_ppl(
            model=model,
            tok=tok,
            texts=texts,
            device=model.device if device.type == "cuda" else device,
            max_length=args.max_length,
        )

        ppl_all.extend(ppl_per)
        total_nll += nll_sum
        total_tok += tok_cnt

        # collect id/text mapping for max ppl search
        for j, (r, ppl, t) in enumerate(zip(batch, ppl_per, texts)):
            _id = r.get(args.id_key, f"idx_{i+j}")
            items.append((float(ppl), str(_id), t, i + j))

        if args.out_jsonl:
            for r, ppl in zip(batch, ppl_per):
                rr = dict(r)
                rr[args.out_field] = float(ppl)
                out_rows.append(rr)

    # corpus-level ppl (more stable than mean of per-sample ppl)
    corpus_ppl = math.exp(total_nll / max(total_tok, 1))

    finite = [x for x in ppl_all if not (math.isnan(x) or math.isinf(x))]
    mean_ppl = sum(finite) / max(len(finite), 1)

    print("========================================")
    print(f"n_queries            = {len(rows)}")
    print(f"valid_ppl_queries    = {len(finite)}")
    print(f"total_tokens(counted)= {total_tok}  (excludes padding, excludes first token)")
    print("----------------------------------------")
    print(f"corpus_ppl           = {corpus_ppl:.4f}")
    print(f"mean_sample_ppl      = {mean_ppl:.4f}")
    print(f"median_sample_ppl    = {safe_percentile(finite, 0.50):.4f}")
    print(f"p90_sample_ppl       = {safe_percentile(finite, 0.90):.4f}")
    print(f"p99_sample_ppl       = {safe_percentile(finite, 0.99):.4f}")
    print("========================================")

    # Find and print top-K max perplexity queries
    finite_items = [x for x in items if not (math.isnan(x[0]) or math.isinf(x[0]))]
    finite_items.sort(key=lambda x: x[0], reverse=True)
    k = max(0, min(args.topk_print, len(finite_items)))

    print(f"==== Top-{k} max perplexity queries ====")
    for rank, (ppl, _id, text, idx) in enumerate(finite_items[:k], 1):
        preview = text.replace("\n", " ")[: args.preview_chars]
        print(f"[{rank:02d}] ppl={ppl:.4f}  idx={idx}  id={_id}  text={preview}")

    if finite_items:
        ppl_max, id_max, text_max, idx_max = finite_items[0]
        print("==== Max perplexity query (full text) ====")
        print(f"ppl={ppl_max:.4f}  idx={idx_max}  id={id_max}")
        print(text_max)

    if args.topk_out_jsonl and finite_items:
        topk_rows = []
        for rank, (ppl, _id, text, idx) in enumerate(finite_items[:k], 1):
            topk_rows.append({
                "rank": rank,
                "ppl": float(ppl),
                "idx": int(idx),
                args.id_key: _id,
                args.text_key: text,
            })
        out_path = Path(args.topk_out_jsonl)
        write_jsonl(out_path, topk_rows)
        print(f"[OK] wrote top-{k} max ppl queries to: {out_path}")

    if args.out_jsonl:
        out_path = Path(args.out_jsonl)
        write_jsonl(out_path, out_rows)
        print(f"[OK] wrote per-query ppl to: {out_path}")


if __name__ == "__main__":
    main()
