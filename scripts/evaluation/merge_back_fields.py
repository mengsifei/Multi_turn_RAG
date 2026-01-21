#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="eval_input_official.jsonl (has input/contexts/predictions/targets)")
    ap.add_argument("--pred", required=True, help="outputs/taskc/gen_eval_official.jsonl (has metrics, maybe missing input)")
    ap.add_argument("--out", required=True, help="merged output jsonl")
    args = ap.parse_args()

    base_map = {}
    for j in read_jsonl(Path(args.base)):
        tid = str(j.get("task_id", ""))
        if tid:
            base_map[tid] = j

    out_rows = []
    kept = 0
    for j in read_jsonl(Path(args.pred)):
        tid = str(j.get("task_id", ""))
        b = base_map.get(tid)
        if b:
            # 用 pred 覆盖 base（保留 base 的 input/contexts/predictions/targets 等）
            merged = dict(b)
            merged.update(j)
            out_rows.append(merged)
            kept += 1
        else:
            out_rows.append(j)

    write_jsonl(Path(args.out), out_rows)
    print(f"[MERGE_BACK] base={len(base_map)} pred={len(out_rows)} kept_with_base={kept} -> {args.out}")

if __name__ == "__main__":
    main()
