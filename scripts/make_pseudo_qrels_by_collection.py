#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, csv
from pathlib import Path
from collections import defaultdict

def read_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", type=Path, required=True,
                    help="Merged official jsonl with Collection + contexts")
    ap.add_argument("--out_dir", type=Path, required=True,
                    help="Output dir for pseudo qrels TSV files")
    ap.add_argument("--rel_k", type=int, default=1,
                    help="Mark top-k docs as relevant (default: 1)")
    ap.add_argument("--rel_grade", type=int, default=1,
                    help="Relevance grade (default: 1)")
    args = ap.parse_args()

    grouped = defaultdict(list)  # collection -> list[(qid, docid, grade)]

    for item in read_jsonl(args.input_jsonl):
        qid = item["task_id"]
        coll = item["Collection"]
        ctxs = item.get("contexts", []) or []
        for c in ctxs[: max(args.rel_k, 0)]:
            grouped[coll].append((qid, c["document_id"], args.rel_grade))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for coll, triples in grouped.items():
        out_tsv = args.out_dir / f"{coll}.pseudo_qrels.tsv"
        with out_tsv.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            w.writerow(["query_id", "corpus_id", "score"])
            for qid, docid, grade in triples:
                w.writerow([qid, docid, grade])
        print(f"[DONE] wrote {out_tsv} rows={len(triples)}")

if __name__ == "__main__":
    main()
