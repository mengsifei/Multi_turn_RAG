#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from collections import defaultdict

def load_qrels(tsv_path: Path):
    # format: query-id \t corpus-id \t score
    qrels = defaultdict(set)
    with tsv_path.open("r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2: 
                continue
            qid, did = parts[0], parts[1]
            qrels[qid].add(did)
    return qrels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_jsonl", required=True)
    ap.add_argument("--qrels_tsv", required=True)
    ap.add_argument("--k", type=int, default=1000)
    args = ap.parse_args()

    qrels = load_qrels(Path(args.qrels_tsv))

    hit = 0
    tot = 0
    with Path(args.run_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            qid = o["task_id"]
            # 有些任务 qid 带 <::>turn，需要 strip
            qid_base = qid.split("<::>", 1)[0]
            rel = qrels.get(qid_base) or qrels.get(qid)
            if not rel:
                continue
            ctx = o.get("contexts", [])[:args.k]
            cand = {c["document_id"] for c in ctx}
            tot += 1
            if len(rel & cand) > 0:
                hit += 1

    print(f"Hit@{args.k} (at least one positive in topK): {hit}/{tot} = {hit/(tot+1e-9):.4f}")

if __name__ == "__main__":
    main()
