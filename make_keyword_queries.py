#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--prefix", default="|user|: ")
    args = ap.parse_args()

    with open(args.in_jsonl, "r", encoding="utf-8") as f, open(args.out_jsonl, "w", encoding="utf-8") as w:
        for line in f:
            if not line.strip():
                continue
            j = json.loads(line)
            kw = (j.get("keyword_query") or "").strip()
            if not kw:
                # fallback：用 clean_query 或 text
                kw = (j.get("clean_query") or j.get("text") or "").strip()
                kw = kw.replace("|user|:", "").strip()
            j["text"] = args.prefix + kw
            w.write(json.dumps(j, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
