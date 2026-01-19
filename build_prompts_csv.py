#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import csv
import argparse
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm

DOMAINS = ["clapnq", "cloud", "fiqa", "govt"]

def infer_domain(collection_name: str) -> str:
    s = (collection_name or "").lower()
    for d in DOMAINS:
        if d in s:
            return d
    raise ValueError(f"Cannot infer domain from Collection={collection_name}")

def load_doc_map(corpus_jsonl: Path) -> Dict[str, str]:
    mp: Dict[str, str] = {}
    with open(corpus_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            if "id" not in j:
                raise KeyError(f"Expected field `id` in corpus, got keys={list(j.keys())}")
            mp[j["id"]] = (j.get("text", "") or "")
    return mp

def load_query_map(rewrite_jsonl: Path) -> Dict[str, str]:
    mp: Dict[str, str] = {}
    with open(rewrite_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            if "_id" not in j or "text" not in j:
                raise KeyError(f"Expected _id and text in rewrite file, got keys={list(j.keys())}")
            tid = j["_id"]
            q = j.get("text", "") or ""
            if q.startswith("|user|:"):
                q = q[len("|user|:"):].strip()
            mp[tid] = q
    return mp

def build_prompt(question: str, contexts: List[Dict[str, Any]], max_doc_chars: int) -> str:
    docs = []
    for i, c in enumerate(contexts, 1):
        docs.append(f"[Document {i}]\n{(c['text'] or '')[:max_doc_chars]}")
    docs_str = "\n\n".join(docs)

    return (
        "Answer the question using ONLY the provided documents.\n"
        "If the answer cannot be determined from the documents, say so.\n\n"
        f"Question:\n{question}\n\n"
        f"Documents:\n{docs_str}\n\n"
        "Answer:"
    )

def main(args):
    cleaned_root = Path(args.cleaned_root)

    # preload 4 domains
    doc_maps = {}
    query_maps = {}
    for d in DOMAINS:
        doc_maps[d] = load_doc_map(cleaned_root / d / f"{d}.jsonl")
        query_maps[d] = load_query_map(cleaned_root / d / f"{d}_{args.task_name}.jsonl")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = n_written = 0
    n_skip_no_query = n_skip_no_ctx = 0

    with open(args.taska_file, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8", newline="") as fout:

        writer = csv.DictWriter(
            fout,
            fieldnames=["task_id", "conversation_id", "Collection", "domain", "prompt"]
        )
        writer.writeheader()

        for line in tqdm(fin, desc="Building prompts.csv"):
            n_total += 1
            j = json.loads(line)

            task_id = j.get("task_id")
            collection = j.get("Collection")
            if not task_id or not collection:
                continue

            conv_id = task_id.split("::")[0]
            domain = infer_domain(collection)

            question = query_maps[domain].get(task_id)
            if not question:
                n_skip_no_query += 1
                continue

            ctxs = []
            for c in (j.get("contexts") or [])[: args.topk]:
                did = c.get("document_id")
                if not did:
                    continue
                txt = doc_maps[domain].get(did)
                if not txt:
                    continue
                ctxs.append({"document_id": did, "score": c.get("score", None), "text": txt})

            if not ctxs:
                n_skip_no_ctx += 1
                continue

            prompt = build_prompt(question, ctxs, args.max_doc_chars)

            writer.writerow({
                "task_id": task_id,
                "conversation_id": conv_id,
                "Collection": collection,
                "domain": domain,
                "prompt": prompt
            })
            n_written += 1

    print("========== PROMPTS CSV SUMMARY ==========")
    print(f"Input lines           : {n_total}")
    print(f"Rows written          : {n_written}")
    print(f"Skipped (no query)    : {n_skip_no_query}")
    print(f"Skipped (no contexts) : {n_skip_no_ctx}")
    print(f"Output file           : {out_path}")
    print("========================================")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--taska_file", required=True, help="TaskA retrieval output jsonl (has contexts)")
    ap.add_argument("--cleaned_root", default="cleaned_dataset", help="cleaned_dataset root")
    ap.add_argument("--task_name", type=str, default="rewrite_gpt", help="e.g. rewrite_gpt")
    ap.add_argument("--out_csv", required=True, help="output prompts.csv")
    ap.add_argument("--topk", type=int, default=5, help="use top-k contexts")
    ap.add_argument("--max_doc_chars", type=int, default=1200, help="truncate each doc text")
    args = ap.parse_args()
    main(args)
