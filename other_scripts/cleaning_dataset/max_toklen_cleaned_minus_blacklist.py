#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, gzip
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer, AutoConfig

def open_maybe_gz(path: str, mode="rt"):
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8", errors="ignore")
    return open(path, mode, encoding="utf-8", errors="ignore")

def load_blacklist(path: str | None) -> set[str]:
    if not path:
        return set()
    s = set()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            t = line.strip()
            if t:
                s.add(t)
    return s

def get_doc_id(o: dict) -> str:
    return str(o.get("document_id") or o.get("_id") or o.get("id") or "")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="tokenizer source (same as eval model_dir)")
    ap.add_argument("--clean_dir", required=True, help="dir containing <domain>.cleaned.jsonl")
    ap.add_argument("--suffix", default=".cleaned.jsonl")
    ap.add_argument("--domains", default="clapnq,cloud,fiqa,govt")
    ap.add_argument("--blacklist", default=None)
    ap.add_argument("--out", default="reports/max_toklen_cleaned_minus_blacklist.json")
    ap.add_argument("--topk_show", type=int, default=5, help="store top-k longest samples per domain")
    args = ap.parse_args()

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    bl = load_blacklist(args.blacklist)
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    # print model context hints (best-effort)
    try:
        cfg = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
        print("[config] loaded:", type(cfg).__name__)
        for k in ["max_position_embeddings", "seq_length", "n_positions", "max_seq_len"]:
            if hasattr(cfg, k):
                print(f"[config] {k} =", getattr(cfg, k))
    except Exception as e:
        print("[config] warning:", e)

    print("[tokenizer] model_max_length =", getattr(tok, "model_max_length", None))

    report = {"model_dir": args.model_dir, "blacklist_size": len(bl), "domains": {}}
    overall_lens = []

    for d in domains:
        fp = Path(args.clean_dir) / f"{d}{args.suffix}"
        if not fp.exists():
            print("[skip] missing:", fp)
            continue

        lens = []
        top = []  # (len, id, head)
        kept = 0
        skipped_bl = 0

        with open_maybe_gz(str(fp), "rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                did = get_doc_id(o)
                if not did:
                    continue
                if did in bl:
                    skipped_bl += 1
                    continue

                title = o.get("title", "") or ""
                text = o.get("text", "") or ""
                s = (title + " " + text).strip()
                L = len(tok(s, add_special_tokens=False)["input_ids"])
                lens.append(L)
                overall_lens.append(L)
                kept += 1

                if args.topk_show > 0:
                    head = s[:200].replace("\n", " ")
                    top.append((L, did, head))
                    top.sort(key=lambda x: -x[0])
                    if len(top) > args.topk_show:
                        top.pop()

        if not lens:
            report["domains"][d] = {"file": str(fp), "kept": kept, "skipped_blacklist": skipped_bl}
            continue

        arr = np.array(lens, dtype=np.int32)
        info = {
            "file": str(fp),
            "kept": kept,
            "skipped_blacklist": skipped_bl,
            "p50": int(np.percentile(arr, 50)),
            "p75": int(np.percentile(arr, 75)),
            "p90": int(np.percentile(arr, 90)),
            "p95": int(np.percentile(arr, 95)),
            "p99": int(np.percentile(arr, 99)),
            "max": int(arr.max()),
            "topk": [{"toklen": int(L), "id": did, "head": head} for (L, did, head) in top],
        }
        report["domains"][d] = info
        print(f"[{d}] kept={kept} skipped_bl={skipped_bl} p99={info['p99']} max={info['max']}")

    if overall_lens:
        a = np.array(overall_lens, dtype=np.int32)
        report["overall"] = {
            "p50": int(np.percentile(a, 50)),
            "p75": int(np.percentile(a, 75)),
            "p90": int(np.percentile(a, 90)),
            "p95": int(np.percentile(a, 95)),
            "p99": int(np.percentile(a, 99)),
            "max": int(a.max()),
            "count": int(a.size),
        }
        print("[overall]", report["overall"])

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[ok] wrote", outp)

if __name__ == "__main__":
    main()
