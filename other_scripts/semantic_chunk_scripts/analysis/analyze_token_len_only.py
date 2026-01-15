#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, gzip
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

def open_text(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, "r", encoding="utf-8")

def derive_out_base(in_path: Path) -> str:
    name = in_path.name
    for suf in [".jsonl.gz", ".jsonl", ".gz", ".json"]:
        if name.endswith(suf):
            name = name[: -len(suf)]
            break
    return name

def hist_to_json(values: np.ndarray, bins, name="token_length_histogram"):
    edges = np.array(bins, dtype=np.int64)
    hist, edges = np.histogram(values, bins=edges)
    n = int(values.shape[0])

    labels = []
    for i in range(len(edges) - 1):
        lo = int(edges[i])
        hi = int(edges[i + 1])
        if hi >= 10**9:
            labels.append(f"[{lo}, +inf)")
        else:
            labels.append(f"[{lo}, {hi})")

    counts = hist.astype(int).tolist()
    pct = [(c / max(1, n)) * 100.0 for c in counts]
    return {
        "name": name,
        "n": n,
        "bin_edges": edges.astype(int).tolist(),
        "bin_labels": labels,
        "counts": counts,
        "pct": pct,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True, help="fiqa.jsonl or fiqa.jsonl.gz")
    ap.add_argument("--hf_tokenizer", required=True)
    ap.add_argument("--text_key", default="text", help="field name that holds text (default: text)")
    ap.add_argument("--max_docs", type=int, default=0, help="0=all, else only first N docs")
    ap.add_argument("--report_dir", default="reports/token_stats")
    ap.add_argument("--save_png", action="store_true")
    args = ap.parse_args()

    inp = Path(args.in_jsonl)
    out_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = derive_out_base(inp)

    tok = AutoTokenizer.from_pretrained(args.hf_tokenizer, trust_remote_code=True)

    lens = []
    n_total = 0
    n_empty = 0

    with open_text(str(inp)) as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            txt = o.get(args.text_key, "")
            txt = "" if txt is None else str(txt)
            n_total += 1
            if txt.strip() == "":
                n_empty += 1
            L = len(tok.encode(txt, add_special_tokens=False))
            lens.append(L)

            if args.max_docs and n_total >= args.max_docs:
                break

    lens = np.array(lens, dtype=np.int64)

    stats = {
        "n": int(lens.shape[0]),
        "empty_or_whitespace": int(n_empty),
        "min": int(lens.min()) if lens.size else None,
        "max": int(lens.max()) if lens.size else None,
        "mean": float(lens.mean()) if lens.size else None,
        "p10": float(np.percentile(lens, 10)) if lens.size else None,
        "p25": float(np.percentile(lens, 25)) if lens.size else None,
        "p50": float(np.percentile(lens, 50)) if lens.size else None,
        "p75": float(np.percentile(lens, 75)) if lens.size else None,
        "p90": float(np.percentile(lens, 90)) if lens.size else None,
        "p95": float(np.percentile(lens, 95)) if lens.size else None,
        "p99": float(np.percentile(lens, 99)) if lens.size else None,
        "frac_lt_128": float((lens < 128).mean() * 100.0) if lens.size else None,
        "frac_lt_256": float((lens < 256).mean() * 100.0) if lens.size else None,
        "frac_lt_384": float((lens < 384).mean() * 100.0) if lens.size else None,
        "frac_ge_480": float((lens >= 480).mean() * 100.0) if lens.size else None,
    }

    # 你之前用的 bins（最后一个是 +inf）
    bins = [0, 2, 16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 480, 512, 10**9]
    hist = hist_to_json(lens, bins, name="token_length_histogram")

    report = {
        "meta": {
            "in_jsonl": str(inp),
            "hf_tokenizer": args.hf_tokenizer,
            "text_key": args.text_key,
            "max_docs": args.max_docs,
        },
        "stats": stats,
        "histogram": hist,
    }

    out_json = out_dir / f"{out_base}.token_stats.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote: {out_json}")

    if args.save_png:
        import matplotlib.pyplot as plt
        out_png = out_dir / f"{out_base}.token_hist.png"
        plt.figure()
        plt.hist(lens, bins=200)
        plt.title(f"Token length histogram: {out_base}")
        plt.xlabel("token length")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"[OK] wrote: {out_png}")

if __name__ == "__main__":
    main()
