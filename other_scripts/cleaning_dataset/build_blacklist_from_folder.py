#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="folder containing many *.txt, each line is an id")
    ap.add_argument("--out", required=True, help="output blacklist.txt")
    ap.add_argument("--glob", default="*.txt", help="glob pattern, default *.txt")
    ap.add_argument("--sort", action="store_true", help="sort ids before writing")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    ids = set()
    files = sorted(in_dir.glob(args.glob))
    if not files:
        raise SystemExit(f"[err] no files matched {args.glob} in {in_dir}")

    for fp in files:
        with fp.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if s:
                    ids.add(s)

    out_list = sorted(ids) if args.sort else list(ids)
    outp.write_text("\n".join(out_list) + ("\n" if out_list else ""), encoding="utf-8")
    print(f"[ok] files={len(files)} unique_ids={len(ids)} -> {outp}")

if __name__ == "__main__":
    main()
