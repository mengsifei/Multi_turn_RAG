#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path

def load_blacklist(path: Path) -> set[str]:
    s = set()
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            t = line.strip()
            if t:
                s.add(t)
    return s

def get_doc_id(item: dict) -> str:
    # 兼容你项目里不同字段
    if "document_id" in item:
        return str(item["document_id"])
    if "_id" in item:
        return str(item["_id"])
    if "id" in item:
        return str(item["id"])
    raise KeyError("No document id field found (document_id/_id/id)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--blacklist", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--out_removed", default="", help="optional: save removed docs for inspection")
    args = ap.parse_args()

    in_path = Path(args.in_jsonl)
    bl_path = Path(args.blacklist)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bl = load_blacklist(bl_path)
    print(f"[blacklist] unique ids = {len(bl)} from {bl_path}")

    total = kept = removed = 0
    removed_fp = None
    if args.out_removed:
        removed_path = Path(args.out_removed)
        removed_path.parent.mkdir(parents=True, exist_ok=True)
        removed_fp = removed_path.open("w", encoding="utf-8")

    with in_path.open("r", encoding="utf-8", errors="ignore") as fin, \
         out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            item = json.loads(line)
            doc_id = get_doc_id(item)

            if doc_id in bl:
                removed += 1
                if removed_fp:
                    removed_fp.write(json.dumps(item, ensure_ascii=False) + "\n")
                continue

            kept += 1
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    if removed_fp:
        removed_fp.close()

    print(f"[done] total={total} kept={kept} removed={removed} removed_ratio={removed/max(1,total)*100:.3f}%")
    print(f"[out]  {out_path}")
    if args.out_removed:
        print(f"[out_removed] {args.out_removed}")

if __name__ == "__main__":
    main()
