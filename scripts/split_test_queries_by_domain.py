#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path
from collections import defaultdict

# mt-rag 的“目录域名”是 clapnq/fiqa/govt/cloud
# 但你的 test Collection 里可能写成 ibmcloud（以及可能出现 mt-rag-ibmcloud...）
DOMAIN_ALIASES = {
    "clapnq": "clapnq",
    "fiqa": "fiqa",
    "govt": "govt",
    "cloud": "cloud",
    "ibmcloud": "cloud",
    "mt-rag-ibmcloud-elser-512-100-20240502": "cloud",
    "mt-rag-clapnq-elser-512-100-20240503": "clapnq",
    "mt-rag-fiqa-beir-elser-512-100-20240501": "fiqa",
    "mt-rag-govt-elser-512-100-20240611": "govt",
}

VALID_DOM_DIRS = ["clapnq", "fiqa", "govt", "cloud"]

def normalize_collection(x: str) -> str:
    s = (x or "").strip().lower()
    if s in DOMAIN_ALIASES:
        return DOMAIN_ALIASES[s]
    # 兜底：只要包含关键词也认
    for k, v in [("clapnq","clapnq"), ("fiqa","fiqa"), ("govt","govt"), ("ibmcloud","cloud"), ("cloud","cloud")]:
        if k in s:
            return v
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_root", required=True, help="temp root to write per-domain query jsonl")
    ap.add_argument("--task", default="rewrite_gpt", choices=["lastturn","questions","rewrite","rewrite_gpt"])
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    buckets = defaultdict(list)

    with open(args.in_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            dom = normalize_collection(o.get("Collection", ""))
            if dom not in VALID_DOM_DIRS:
                raise ValueError(f"Unknown Collection/domain={o.get('Collection')} normalized={dom} in line: {o}")

            buckets[dom].append({
                "_id": o["task_id"],
                "text": o.get("text", "")
            })

    for dom in VALID_DOM_DIRS:
        dom_dir = out_root / dom
        dom_dir.mkdir(parents=True, exist_ok=True)
        out_path = dom_dir / f"{dom}_{args.task}.jsonl"
        with open(out_path, "w", encoding="utf-8") as w:
            for row in buckets.get(dom, []):
                w.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[WROTE] {out_path} lines={len(buckets.get(dom, []))}")

if __name__ == "__main__":
    main()
