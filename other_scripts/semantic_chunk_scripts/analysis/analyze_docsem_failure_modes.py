#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, gzip, random
from pathlib import Path
from bisect import bisect_right
from collections import Counter
import numpy as np
from transformers import AutoTokenizer


def open_text(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, "r", encoding="utf-8")


def load_span_index(span_path: str):
    """
    Accepts either:
      - corpora/passage_level/<domain>_passage_spans.jsonl.gz
      - corpora/span_index/<domain>_passage_span_index.jsonl.gz
    Records assumed to contain: base_id, passage_id, start, end
    """
    by_base = {}
    with open_text(span_path) as f:
        for line in f:
            o = json.loads(line)
            base = o["base_id"]
            by_base.setdefault(base, []).append((int(o["start"]), int(o["end"]), o["passage_id"]))
    out = {}
    for base, spans in by_base.items():
        spans.sort(key=lambda x: x[0])
        starts = [s for s, e, pid in spans]
        ends = [e for s, e, pid in spans]
        pids = [pid for s, e, pid in spans]
        out[base] = (starts, ends, pids)
    return out


def overlap(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


def count_overlapped_passages(span_index, base_id: str, cs: int, ce: int):
    if base_id not in span_index:
        return 0, 0.0, None
    starts, ends, pids = span_index[base_id]
    i = bisect_right(starts, cs) - 1
    if i < 0:
        i = 0
    cnt = 0
    best_ratio = 0.0
    best_pid = None
    denom = max(1, ce - cs)
    while i < len(starts) and starts[i] < ce:
        ps, pe, pid = starts[i], ends[i], pids[i]
        ov = overlap(cs, ce, ps, pe)
        if ov > 0:
            cnt += 1
            r = ov / denom
            if r > best_ratio:
                best_ratio = r
                best_pid = pid
        i += 1
    return cnt, best_ratio, best_pid


def reservoir_sample(stream_iter, k: int, seed: int = 0):
    rnd = random.Random(seed)
    sample = []
    n = 0
    for item in stream_iter:
        n += 1
        if len(sample) < k:
            sample.append(item)
        else:
            j = rnd.randrange(n)
            if j < k:
                sample[j] = item
    return sample, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--chunk_gz", required=True)
    ap.add_argument("--span_gz", required=True)
    ap.add_argument("--hf_tokenizer", required=True)
    ap.add_argument("--sample_k", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.hf_tokenizer, trust_remote_code=True)
    span_index = load_span_index(args.span_gz)

    # ---- stream chunks
    def chunk_stream():
        with open_text(args.chunk_gz) as f:
            for line in f:
                o = json.loads(line)
                yield (o["id"], o["parent_id"], int(o["start"]), int(o["end"]), o.get("text", ""))

    # sample
    sample, total = reservoir_sample(chunk_stream(), args.sample_k, args.seed)

    # stats containers
    lens = []
    cross_cnt = []
    best_ratios = []
    parent_counts = Counter()

    # For parent_counts we need full pass; do a light pass counting only parents (fast)
    with open_text(args.chunk_gz) as f:
        for line in f:
            o = json.loads(line)
            parent_counts[o["parent_id"]] += 1

    for cid, pid, s, e, text in sample:
        # token length
        L = len(tok.encode(text, add_special_tokens=False))
        lens.append(L)

        # how many passages overlapped
        c, br, _ = count_overlapped_passages(span_index, pid, s, e)
        cross_cnt.append(c)
        best_ratios.append(br)

    lens = np.array(lens, dtype=np.int64)
    cross_cnt = np.array(cross_cnt, dtype=np.int64)
    best_ratios = np.array(best_ratios, dtype=np.float32)

    def pct(x):
        return float(x) * 100.0

    print("\n" + "=" * 96)
    print(f"[DOMAIN] {args.domain}")
    print(f"total_chunks = {total}")
    print(f"sample_k     = {len(sample)}")
    print("=" * 96)

    # 1) token length distribution
    print("\n[1] chunk token-length")
    for q in [0, 10, 25, 50, 75, 90, 95, 99, 100]:
        v = np.percentile(lens, q)
        print(f"  p{q:02d} = {v:.0f}")
    print(f"  mean = {lens.mean():.1f}")
    print(f"  frac <128  = {pct((lens < 128).mean()):.2f}%")
    print(f"  frac <256  = {pct((lens < 256).mean()):.2f}%")
    print(f"  frac <384  = {pct((lens < 384).mean()):.2f}%")
    print(f"  frac >=480 = {pct((lens >= 480).mean()):.2f}%  (理想情况下应很高，接近 fixed 512/100)")

    # 2) boundary leakage: chunk overlaps multiple passages?
    print("\n[2] chunk overlaps how many passages")
    print(f"  mean_overlap_passages = {cross_cnt.mean():.3f}")
    print(f"  frac overlap>=2       = {pct((cross_cnt >= 2).mean()):.2f}%")
    print(f"  frac overlap>=3       = {pct((cross_cnt >= 3).mean()):.2f}%")
    print(f"  p95 overlap_passages  = {np.percentile(cross_cnt, 95):.0f}")
    print(f"  max overlap_passages  = {cross_cnt.max()}")

    print("\n[3] best overlap ratio within a chunk (max passage overlap / chunk_len)")
    print(f"  mean best_ratio = {best_ratios.mean():.4f}")
    print(f"  frac best_ratio <0.6 = {pct((best_ratios < 0.6).mean()):.2f}%   (大量<0.6 说明 chunk 经常跨 span)")
    print(f"  frac best_ratio <0.8 = {pct((best_ratios < 0.8).mean()):.2f}%")

    # 4) corpus fragmentation
    vals = np.array(list(parent_counts.values()), dtype=np.int64)
    print("\n[4] chunks per parent_id (fragmentation)")
    print(f"  unique_parents = {len(parent_counts)}")
    for q in [50, 75, 90, 95, 99]:
        print(f"  p{q:02d} chunks/parent = {np.percentile(vals, q):.0f}")
    top10 = parent_counts.most_common(10)
    print("  top parents:")
    for pid, n in top10:
        print(f"    {n:6d}  {pid}")

    print("\nDONE.\n")


if __name__ == "__main__":
    main()
