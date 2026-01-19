#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_test_jsonl", required=True, help="original mixed test with Collection domain")
    ap.add_argument("--run_jsonl", required=True, help="retrieval output jsonl (task_id + contexts + Collection?)")
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    # load run into map: task_id -> contexts
    run_map = {}
    with open(args.run_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            run_map[o["task_id"]] = o.get("contexts", [])

    missing = 0
    with open(args.orig_test_jsonl, "r", encoding="utf-8") as fin, open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            o = json.loads(line)
            tid = o["task_id"]
            dom = o["Collection"]
            ctxs = run_map.get(tid)
            if ctxs is None:
                missing += 1
                ctxs = []
            fout.write(json.dumps({
                "task_id": tid,
                "contexts": ctxs,
                "Collection": dom,
            }, ensure_ascii=False) + "\n")

    print(f"[DONE] wrote={args.out_jsonl} missing={missing}")

if __name__ == "__main__":
    main()
