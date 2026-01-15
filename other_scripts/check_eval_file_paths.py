#!/usr/bin/env python3
import os, json
from pathlib import Path

chunk_dir = Path("corpora/chunk_level_docsem512_100")
domains = ["clapnq","fiqa","govt","cloud"]

def p(domain):
    return {
        "chunk": chunk_dir / f"{domain}_docsem512_100.jsonl.gz",
        "map":   chunk_dir / f"{domain}_docsem512_100_map.jsonl.gz",
        # 这个 span-index 你如果有生成，也放这里检查（文件名按你的实现来）
        "span":  Path("corpora/span_index") / f"{domain}_passage_span_index.jsonl.gz",
    }

for d in domains:
    print("\n==", d, "==")
    for k,v in p(d).items():
        print(f"{k:6s} {str(v):80s} exists={v.exists()}")
