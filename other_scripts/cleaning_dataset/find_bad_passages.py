#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Find "bad" passages by token length / whitespace.
Outputs:
  1) JSONL with details of bad passages
  2) TXT with bad passage ids (one per line)

Supports input:
  - .jsonl
  - .jsonl.gz
  - .jsonl.zip (member defaults to "<stem>.jsonl" if not provided)
"""

import argparse, json, gzip, zipfile
from pathlib import Path
from typing import Iterable, Dict, Any, Optional

from transformers import AutoTokenizer


def iter_jsonl_any(path: Path, *, zip_member: Optional[str] = None) -> Iterable[Dict[str, Any]]:
    p = str(path)
    if p.endswith(".jsonl.gz") or p.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    if p.endswith(".zip"):
        member = zip_member
        if member is None:
            # e.g. fiqa.jsonl.zip -> stem = "fiqa.jsonl" -> member = "fiqa.jsonl"
            member = Path(p).stem
            if not member.endswith(".jsonl"):
                member = member + ".jsonl"
        with zipfile.ZipFile(path, "r") as zf, zf.open(member, "r") as fin:
            for raw in fin:
                line = raw.decode("utf-8", errors="replace").strip()
                if line:
                    yield json.loads(line)
        return

    # plain jsonl
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def get_id(obj: Dict[str, Any]) -> str:
    for k in ("_id", "id", "passage_id", "doc_id", "docid"):
        if k in obj and obj[k] is not None:
            s = str(obj[k])
            if s != "":
                return s
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True, help="input corpus jsonl/jsonl.gz/jsonl.zip")
    ap.add_argument("--zip_member", default=None, help="zip member name (if in_path is .zip)")
    ap.add_argument("--hf_tokenizer", required=True)
    ap.add_argument("--text_key", default="text")

    ap.add_argument("--lt_tokens", type=int, default=2, help="flag if tok_len < this")
    ap.add_argument("--gt_tokens", type=int, default=850, help="flag if tok_len > this")
    ap.add_argument("--include_empty", action="store_true", help="also flag text.strip()=='' (recommended)")

    ap.add_argument("--out_jsonl", required=True, help="output bad passages jsonl (plain, not gz)")
    ap.add_argument("--out_txt", required=True, help="output bad passage ids txt")
    ap.add_argument("--snip_chars", type=int, default=300, help="snippet chars in jsonl")
    ap.add_argument("--max_docs", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_jsonl = Path(args.out_jsonl)
    out_txt = Path(args.out_txt)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.hf_tokenizer, trust_remote_code=True)

    n = 0
    bad_n = 0
    counts = {"empty_or_whitespace": 0, f"lt_{args.lt_tokens}": 0, f"gt_{args.gt_tokens}": 0}
    bad_ids = []

    with open(out_jsonl, "w", encoding="utf-8") as fj:
        for obj in iter_jsonl_any(in_path, zip_member=args.zip_member):
            n += 1
            if args.max_docs and n > args.max_docs:
                break

            pid = get_id(obj)
            text = obj.get(args.text_key, "")
            if text is None:
                text = ""
            text = str(text)

            reasons = []
            empty = (text.strip() == "")
            if args.include_empty and empty:
                reasons.append("empty_or_whitespace")

            # 对空白也可以不算 tok_len（更快），但这里还是算一下，方便你排查
            tok_len = len(tok.encode(text, add_special_tokens=False))

            if tok_len < args.lt_tokens:
                reasons.append(f"lt_{args.lt_tokens}")
            if tok_len > args.gt_tokens:
                reasons.append(f"gt_{args.gt_tokens}")

            # 只输出“确实要flag”的
            if not reasons:
                continue

            bad_n += 1
            bad_ids.append(pid)

            for r in set(reasons):
                if r in counts:
                    counts[r] += 1

            rec = {
                "id": pid,
                "tok_len": tok_len,
                "reasons": reasons,
                "text_is_whitespace": bool(empty),
                "snippet": text[: args.snip_chars],
            }
            fj.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ids txt
    with open(out_txt, "w", encoding="utf-8") as ft:
        for pid in bad_ids:
            ft.write(pid + "\n")

    print("DONE")
    print(f"  scanned_docs : {n}")
    print(f"  bad_docs     : {bad_n}")
    print(f"  counts       : {counts}")
    print(f"  out_jsonl    : {out_jsonl}")
    print(f"  out_txt      : {out_txt}")


if __name__ == "__main__":
    main()
