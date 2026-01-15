#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build passage span index for docsem aggregation.

Output (gz jsonl):
  {"passage_id": ..., "parent_id": ..., "start": int, "end": int}

We only need to parse the official passage_id to get (parent_id, start, end).
Supported patterns:
  - clapnq:  837799097_6931-7548-0-617          -> parent=837799097_6931-7548 start=0 end=617
  - fiqa:    10171-0-2129                      -> parent=10171             start=0 end=2129
  - govt:    45cbe52725a3cd97-2-2092           -> parent=45cbe52725a3cd97   start=2 end=2092
  - cloud:   ibmcld_00422-0-387                -> parent=ibmcld_00422       start=0 end=387

Input can be:
  - a .jsonl file
  - a .jsonl.zip with --jsonl_name
"""

import argparse
import gzip
import json
import zipfile
from pathlib import Path
from typing import Optional, Tuple, Iterator


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True, help="official corpus: .jsonl or .jsonl.zip")
    ap.add_argument("--jsonl_name", default=None, help="member name inside zip (if in_path is .zip)")
    ap.add_argument("--out_path", required=True, help="output .jsonl.gz span index")
    ap.add_argument("--max_warn", type=int, default=5)
    return ap.parse_args()


def iter_jsonl(in_path: Path, jsonl_name: Optional[str]) -> Iterator[dict]:
    if in_path.suffix == ".zip":
        if not jsonl_name:
            raise ValueError("--jsonl_name is required when --in_path is a .zip")
        with zipfile.ZipFile(in_path, "r") as zf, zf.open(jsonl_name, "r") as f:
            for raw in f:
                yield json.loads(raw.decode("utf-8"))
    else:
        with in_path.open("r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


def parse_passage_id(pid: str) -> Optional[Tuple[str, int, int]]:
    """
    Generic rule:
      take last 2 '-' separated fields if both are digits => (parent, start, end)
      parent is the remaining prefix joined by '-'
    """
    s = pid.strip()
    if not s:
        return None
    parts = s.split("-")
    if len(parts) < 3:
        return None
    a, b = parts[-2], parts[-1]
    if not (a.isdigit() and b.isdigit()):
        return None
    start = int(a)
    end = int(b)
    parent = "-".join(parts[:-2])
    if parent == "":
        return None
    if end <= start:
        return None
    return parent, start, end


def main():
    args = parse_args()
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    n_ok = 0
    n_bad = 0
    warned = 0

    with gzip.open(out_path, "wt", encoding="utf-8", compresslevel=1) as fout:
        for obj in iter_jsonl(in_path, args.jsonl_name):
            n += 1
            pid = obj.get("id") or obj.get("_id")
            if pid is None:
                n_bad += 1
                if warned < args.max_warn:
                    warned += 1
                    print("[WARN] missing id/_id keys:", list(obj.keys())[:20])
                continue
            pid = str(pid)

            parsed = parse_passage_id(pid)
            if parsed is None:
                n_bad += 1
                if warned < args.max_warn:
                    warned += 1
                    print("[WARN] cannot parse passage_id:", pid)
                continue

            parent, start, end = parsed
            rec = {
                "passage_id": pid,
                "parent_id": parent,
                "start": start,
                "end": end,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += 1

    print(f"[DONE] wrote: {out_path}")
    print(f" total_lines={n} ok={n_ok} bad={n_bad}")


if __name__ == "__main__":
    main()

    
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# Build span index from passage_level corpus IDs:
#   <docid>_<pstart>-<pend>-<cstart>-<cend>

# Output jsonl.gz records:
#   {"base_id": "<docid>_<pstart>-<pend>", "passage_id": "<full_id>", "start": cstart, "end": cend}
# """

# import argparse, json, re, zipfile, gzip
# from pathlib import Path
# from tqdm import tqdm


# _ID_RE = re.compile(r"^(.+?)_(\d+)-(\d+)-(\d+)-(\d+)$")


# def parse_args():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--in_zip", required=True)
#     ap.add_argument("--jsonl_name", required=True)
#     ap.add_argument("--out_index", required=True)
#     ap.add_argument("--gzip_level", type=int, default=1)
#     return ap.parse_args()


# def main():
#     args = parse_args()
#     outp = Path(args.out_index)
#     outp.parent.mkdir(parents=True, exist_ok=True)

#     with zipfile.ZipFile(args.in_zip, "r") as zf, \
#          zf.open(args.jsonl_name, "r") as fin, \
#          gzip.open(outp, "wt", encoding="utf-8", compresslevel=args.gzip_level) as fout:

#         for raw in tqdm(fin, desc="build-span-index"):
#             obj = json.loads(raw.decode("utf-8"))
#             pid = obj.get("id") or obj.get("_id")
#             if not pid:
#                 continue

#             m = _ID_RE.match(pid)
#             if not m:
#                 # some corpora might have non-matching ids; skip
#                 continue

#             docid, pstart, pend, cstart, cend = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
#             base_id = f"{docid}_{pstart}-{pend}"
#             rec = {"base_id": base_id, "passage_id": pid, "start": cstart, "end": cend}
#             fout.write(json.dumps(rec, ensure_ascii=False) + "\n")


# if __name__ == "__main__":
#     main()
