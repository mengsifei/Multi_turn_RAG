#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, gzip
from pathlib import Path
from typing import Optional, Tuple

def open_text(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, "r", encoding="utf-8")

def parse_span_from_passage_id(pid: str) -> Optional[Tuple[str, int, int]]:
    """
    Works for ids like:
      - "10171-0-2129"                -> base="10171", start=0, end=2129
      - "837799097_6931-7548-0-617"   -> base="837799097_6931-7548", start=0, end=617
    Rule: split by '-', take last two fields as ints, remaining prefix is base_id.
    """
    parts = pid.split("-")
    if len(parts) < 3:
        return None
    a, b = parts[-2], parts[-1]
    if not (a.isdigit() and b.isdigit()):
        return None
    start = int(a); end = int(b)
    base = "-".join(parts[:-2])
    if base == "" or end <= start:
        return None
    return base, start, end

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_passages", required=True, help="passage corpus jsonl (e.g. human/retrieval_tasks/fiqa/fiqa.jsonl)")
    ap.add_argument("--out_gz", required=True, help="output spans jsonl.gz (e.g. corpora/passage_level/fiqa_passage_spans.jsonl.gz)")
    args = ap.parse_args()

    out_path = Path(args.out_gz)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    ok = 0
    bad = 0

    with open_text(args.in_passages) as fin, gzip.open(out_path, "wt", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            pid = str(o.get("id") or o.get("_id") or "")
            if not pid:
                bad += 1
                continue

            parsed = parse_span_from_passage_id(pid)
            if parsed is None:
                bad += 1
                continue

            base, s, e = parsed
            rec = {"base_id": base, "passage_id": pid, "start": s, "end": e}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ok += 1
            total += 1

    print(f"[OK] wrote: {out_path}")
    print(f"  ok={ok}  bad={bad}")

if __name__ == "__main__":
    main()
