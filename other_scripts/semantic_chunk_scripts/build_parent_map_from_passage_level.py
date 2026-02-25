#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_parent_map_from_passage_level.py

Step 1 (recommended): build a pseudo-document grouping ("parent") from the official
`passage_level` corpus (or your cleaned passage_level JSONL).

Why: You can later compute doc-level keywords/summary once per parent, then attach
to each passage without changing passage boundaries (aligns with official ingestion).

Input JSONL format (your example):
{"_id": "...", "id": "...", "url": "", "text": "...", "title": "", "_clean_stats": {...}}

Output:
1) parent_map.jsonl  (one line per parent)
   {
     "domain": "fiqa",
     "parent_id": "10171",
     "n_passages": 3,
     "passages": [
        {"id":"10171-0-2129","_id":"10171-0-2129","start":0,"end":2129,"len":2129,"title":"","url":""},
        ...
     ]
   }

2) parent_stats.json  (summary stats + parsing fallback counters)

This script tries to robustly infer:
- parent_id
- (start, end) offsets
from "id" strings like:
- FIQA:   10171-0-2129
- GOVT:   similar patterns often exist
- CLAPNQ: 837799097_6931-7548-0-617  (parent likely 837799097_6931-7548)

If offsets cannot be parsed, passages are still grouped by parent_id, but sorting
falls back to input order and start/end=-1.
"""

import argparse
import json
import gzip
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, Any, Tuple, Optional, List


def open_text(path: str):
    p = Path(path)
    if p.suffix == ".gz":
        return gzip.open(p, "rt", encoding="utf-8")
    return open(p, "rt", encoding="utf-8")


def write_text(path: str):
    p = Path(path)
    if p.suffix == ".gz":
        return gzip.open(p, "wt", encoding="utf-8")
    return open(p, "wt", encoding="utf-8")


def try_parse_start_end_from_id(id_str: str) -> Tuple[Optional[int], Optional[int], str]:
    """
    Return (start, end, method_tag).
    Methods:
      - tail_dash_pair: last 2 dash-separated tokens are integers: ...-<start>-<end>
      - any_dash_pair:  some adjacent dash-separated tokens are integers (rare fallback)
      - none
    """
    if not id_str:
        return None, None, "none"

    parts = id_str.split("-")

    # Most common for passage_level: ...-start-end
    if len(parts) >= 3:
        a, b = parts[-2], parts[-1]
        if a.isdigit() and b.isdigit():
            return int(a), int(b), "tail_dash_pair"

    # Fallback: find any adjacent numeric pair
    for i in range(len(parts) - 1):
        if parts[i].isdigit() and parts[i + 1].isdigit():
            return int(parts[i]), int(parts[i + 1]), "any_dash_pair"

    return None, None, "none"


def infer_parent_id(domain: str, id_str: str) -> Tuple[str, str]:
    """
    Return (parent_id, method_tag).

    Heuristics:
      - If contains '_' and has a 'start-end' right after underscore (CLAPNQ-like):
          parent = "<prefix_before_underscore>_<start-end>"
          Example: 837799097_6931-7548-0-617 -> 837799097_6931-7548
      - Else if tail is "-start-end" (FIQA-like):
          parent = everything before "-start-end"
          Example: 10171-0-2129 -> 10171
                   822086267_7384-8758-5-3624 -> 822086267_7384-8758-5  (maybe too long)
        But we also try to strip to a more "doc-ish" parent:
          if first token is digits and others are offsets, parent may be first token.
      - Else:
          parent = id_str up to last '-' token (coarser) or full id_str.
    """
    s = id_str or ""
    if not s:
        return "", "empty"

    # CLAPNQ-style: <doc>_<docStart-docEnd>-<chunkStart>-<chunkEnd> (often)
    if "_" in s:
        left, right = s.split("_", 1)
        # right begins with something like "6931-7548-0-617"
        right_parts = right.split("-")
        if len(right_parts) >= 2 and right_parts[0].isdigit() and right_parts[1].isdigit():
            parent = f"{left}_{right_parts[0]}-{right_parts[1]}"
            return parent, "underscore_doc_span"

    # If it ends with -start-end, parent is prefix before those
    start, end, se_method = try_parse_start_end_from_id(s)
    if se_method in ("tail_dash_pair",):
        parts = s.split("-")
        prefix_parts = parts[:-2]
        prefix = "-".join(prefix_parts)

        # Extra FIQA-like tightening: if prefix itself is a pure int token or starts with it.
        # Example FIQA: "10171-0-2129" -> prefix "10171"
        # If prefix contains only digits and maybe other numeric offsets, prefer first token.
        if domain.lower() in ("fiqa", "govt", "cloud", "all"):
            # If first token is digits, and remaining prefix tokens are digits too, parent=first token
            if prefix_parts and prefix_parts[0].isdigit():
                if all(p.isdigit() for p in prefix_parts):
                    return prefix_parts[0], "tail_dash_pair_first_token"
        return prefix, "tail_dash_pair_prefix"

    # Generic fallback: drop the last "-X" segment if it exists
    if "-" in s:
        return s.rsplit("-", 1)[0], "rsplit_last_dash"

    return s, "as_is"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, help="e.g. fiqa/clapnq/govt/cloud (for stats + heuristics)")
    ap.add_argument("--in_jsonl", required=True, help="passage_level jsonl (or cleaned passage-level jsonl). .gz ok")
    ap.add_argument("--out_parent_map", required=True, help="output parent_map.jsonl(.gz ok)")
    ap.add_argument("--out_stats", required=True, help="output stats json")
    ap.add_argument("--max_passages_per_parent", type=int, default=0,
                    help="0 = keep all; >0 = truncate passages list per parent (after sorting). Useful for huge parents.")
    ap.add_argument("--keep_text", action="store_true",
                    help="If set, include 'text' in each passage entry (big files!). Default: off.")
    ap.add_argument("--sort_by_offset", action="store_true",
                    help="If set, sort passages per parent by (start,end) when available. Otherwise keep input order.")
    args = ap.parse_args()

    domain = args.domain.lower()
    parse_se_counter = Counter()
    parent_method_counter = Counter()

    # parent_id -> list[passage_record]
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    total = 0
    missing_id = 0

    with open_text(args.in_jsonl) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except Exception:
                # skip malformed
                continue

            pid = obj.get("id") or obj.get("_id") or ""
            if not pid:
                missing_id += 1
                continue

            parent_id, pm = infer_parent_id(domain, pid)
            parent_method_counter[pm] += 1

            start, end, sm = try_parse_start_end_from_id(pid)
            parse_se_counter[sm] += 1

            text = obj.get("text", "") or ""
            rec = {
                "id": pid,
                "_id": obj.get("_id", pid),
                "start": int(start) if start is not None else -1,
                "end": int(end) if end is not None else -1,
                "len": len(text),
                "title": obj.get("title", "") or "",
                "url": obj.get("url", "") or "",
                "_line": line_no,  # stable fallback ordering
            }
            if args.keep_text:
                rec["text"] = text

            groups[parent_id].append(rec)

    # Write parent_map
    n_parents = 0
    sizes = []
    with write_text(args.out_parent_map) as out:
        for parent_id, passages in groups.items():
            n_parents += 1

            if args.sort_by_offset:
                # sort by start/end when available; otherwise by input line order
                def sort_key(r):
                    s = r.get("start", -1)
                    e = r.get("end", -1)
                    if s >= 0 and e >= 0:
                        return (0, s, e, r["_line"])
                    return (1, r["_line"])
                passages = sorted(passages, key=sort_key)

            if args.max_passages_per_parent and args.max_passages_per_parent > 0:
                passages = passages[: args.max_passages_per_parent]

            for r in passages:
                r.pop("_line", None)

            sizes.append(len(passages))
            out_obj = {
                "domain": domain,
                "parent_id": parent_id,
                "n_passages": len(passages),
                "passages": passages,
            }
            out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    # Stats
    sizes_sorted = sorted(sizes)
    def pct(p):
        if not sizes_sorted:
            return 0
        idx = int(round((p / 100.0) * (len(sizes_sorted) - 1)))
        return sizes_sorted[idx]

    stats = {
        "domain": domain,
        "in_jsonl": args.in_jsonl,
        "total_lines_parsed": total,
        "missing_id": missing_id,
        "n_parents": n_parents,
        "parent_size_min": sizes_sorted[0] if sizes_sorted else 0,
        "parent_size_p50": pct(50),
        "parent_size_p90": pct(90),
        "parent_size_p99": pct(99),
        "parent_size_max": sizes_sorted[-1] if sizes_sorted else 0,
        "start_end_parse_methods": dict(parse_se_counter),
        "parent_infer_methods": dict(parent_method_counter),
        "notes": {
            "keep_text": bool(args.keep_text),
            "sort_by_offset": bool(args.sort_by_offset),
            "max_passages_per_parent": args.max_passages_per_parent,
            "recommendation": "Inspect a few parents manually; if parents look mixed-topic, adjust infer_parent_id() for that domain.",
        }
    }

    Path(args.out_stats).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote parent_map: {args.out_parent_map}")
    print(f"[OK] wrote stats     : {args.out_stats}")
    print(f"[INFO] parents={n_parents} passages={total} missing_id={missing_id}")
    print(f"[INFO] start/end parse methods: {dict(parse_se_counter)}")
    print(f"[INFO] parent infer methods   : {dict(parent_method_counter)}")


if __name__ == "__main__":
    main()
