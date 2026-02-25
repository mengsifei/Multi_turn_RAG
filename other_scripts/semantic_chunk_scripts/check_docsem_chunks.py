#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validate doc-level semantic chunks:
- chunk_id format
- parent_id exists in document_level
- 0 <= start < end <= len(parent_text)
- chunk_text == parent_text[start:end]
- token_len(chunk_text) <= max_tokens (+ small tolerance)
- optional: chunk_map consistency

Designed for corpora/document_level/*.jsonl.zip + chunk_level_docsem*.jsonl.gz
"""

import argparse
import gzip
import json
import random
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer


# -------------------------
# helpers
# -------------------------
def get_base_id(obj: dict) -> Optional[str]:
    """
    Robust id getter across domains.
    Accepts:
      - string ids (strip, non-empty)
      - numeric ids (including 0) -> str
    Returns None if missing/empty/None/un-stringifiable.
    """
    for key in ("_id", "id", "document_id", "doc_id", "docid"):
        if key not in obj:
            continue
        v: Any = obj.get(key)
        if v is None:
            continue
        if isinstance(v, str):
            v2 = v.strip()
            if v2 == "":
                continue
            return v2
        try:
            return str(v)
        except Exception:
            continue
    return None


def iter_gz_jsonl(path: Path) -> Iterable[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def reservoir_sample(stream: Iterable[dict], k: int, seed: int = 0) -> Tuple[List[dict], int]:
    """
    Reservoir sampling over a stream. Returns (samples, total_count_seen).
    """
    rng = random.Random(seed)
    sample: List[dict] = []
    n = 0
    for item in stream:
        n += 1
        if len(sample) < k:
            sample.append(item)
        else:
            j = rng.randrange(1, n + 1)
            if j <= k:
                sample[j - 1] = item
    return sample, n


def read_doc_texts_from_zip(
    doc_zip: Path,
    member_name: str,
    need_ids: Set[str],
    *,
    stop_when_all_found: bool = True,
) -> Dict[str, Tuple[str, str]]:
    """
    Returns dict: parent_id -> (title, text) for ids in need_ids
    """
    found: Dict[str, Tuple[str, str]] = {}
    if not need_ids:
        return found

    with zipfile.ZipFile(doc_zip, "r") as zf, zf.open(member_name, "r") as fin:
        for raw in fin:
            obj = json.loads(raw.decode("utf-8"))
            pid = get_base_id(obj)
            if pid is None or pid not in need_ids:
                continue

            text = obj.get("text")
            title = obj.get("title") or ""
            if text is None:
                text = ""
            text = str(text)

            found[pid] = (str(title), text)

            if stop_when_all_found and len(found) >= len(need_ids):
                break
    return found


@dataclass
class CheckResult:
    total_chunks: int = 0
    unique_parents: int = 0
    sample_k: int = 0

    err_missing_parent: int = 0
    err_bad_id_format: int = 0
    err_bad_offsets: int = 0
    err_slice_mismatch: int = 0
    err_token_too_long: int = 0

    err_map_missing: int = 0
    err_map_mismatch: int = 0

    max_chunks_parent: str = ""
    max_chunks_count: int = 0

    examples: Dict[str, List[dict]] = None

    def __post_init__(self):
        if self.examples is None:
            self.examples = defaultdict(list)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", type=str, default="clapnq,fiqa,govt,cloud",
                    help="comma-separated domains")
    ap.add_argument("--doc_dir", type=str, default="corpora/document_level")
    ap.add_argument("--chunk_dir", type=str, default="corpora/chunk_level_docsem512_100")

    ap.add_argument("--chunk_suffix", type=str, default="docsem512_100",
                    help="chunk file name suffix: <domain>_<suffix>.jsonl.gz")
    ap.add_argument("--map_suffix", type=str, default="docsem512_100_map",
                    help="map file name suffix: <domain>_<suffix>.jsonl.gz")

    ap.add_argument("--hf_tokenizer", type=str, required=True)

    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--token_tolerance", type=int, default=5,
                    help="allow a small tolerance above max_tokens to avoid false positives")
    ap.add_argument("--sample_k", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--check_map", action="store_true",
                    help="also validate chunk_map file for sampled chunk_ids")

    ap.add_argument("--report_dir", type=str, default="reports/docsem_check")
    args = ap.parse_args()

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    doc_dir = Path(args.doc_dir)
    chunk_dir = Path(args.chunk_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.hf_tokenizer, trust_remote_code=True)

    for d in domains:
        doc_zip = doc_dir / f"{d}.jsonl.zip"
        member = f"{d}.jsonl"
        chunk_path = chunk_dir / f"{d}_{args.chunk_suffix}.jsonl.gz"
        map_path = chunk_dir / f"{d}_{args.map_suffix}.jsonl.gz"

        print("\n" + "=" * 90)
        print(f"[DOMAIN] {d}")
        print(f" doc_zip   = {doc_zip}")
        print(f" chunk_gz  = {chunk_path}")
        print(f" map_gz    = {map_path}  (check_map={args.check_map})")
        print("=" * 90)

        res = CheckResult()

        # Pass 1: stream chunks -> count + reservoir sample + parent counter
        parent_counter = Counter()
        chunk_stream = iter_gz_jsonl(chunk_path)
        sample, total = reservoir_sample(chunk_stream, k=args.sample_k, seed=args.seed)

        # Need a second pass to count parents (reservoir consumed the iterator)
        for obj in tqdm(iter_gz_jsonl(chunk_path), desc=f"[{d}] counting", leave=False):
            pid = obj.get("parent_id")
            if pid is None:
                pid = obj.get("parent")  # fallback if someone used a different key
            if pid is not None:
                parent_counter[str(pid)] += 1

        res.total_chunks = total
        res.unique_parents = len(parent_counter)
        if parent_counter:
            max_pid, max_cnt = parent_counter.most_common(1)[0]
            res.max_chunks_parent = max_pid
            res.max_chunks_count = int(max_cnt)

        res.sample_k = len(sample)

        # Build needed parent_ids for sample validation
        need_parents: Set[str] = set()
        need_chunk_ids: Set[str] = set()
        for ch in sample:
            pid = ch.get("parent_id") or ch.get("parent")
            cid = ch.get("id") or ch.get("chunk_id")
            if pid is not None:
                need_parents.add(str(pid))
            if cid is not None:
                need_chunk_ids.add(str(cid))

        # Load parent texts for sampled parent_ids only
        parent_texts = read_doc_texts_from_zip(doc_zip, member, need_parents)
        missing = need_parents - set(parent_texts.keys())
        if missing:
            print(f"[WARN] sampled parents missing in doc_zip: {len(missing)} (show up to 5) -> {list(sorted(missing))[:5]}")

        # Optional: load chunk_map entries for sampled chunk_ids only
        map_entries: Dict[str, dict] = {}
        if args.check_map:
            if not map_path.exists():
                print(f"[WARN] map file not found: {map_path} (skip map check)")
            else:
                for mo in tqdm(iter_gz_jsonl(map_path), desc=f"[{d}] map-scan", leave=False):
                    mid = mo.get("chunk_id") or mo.get("id")
                    if mid is None:
                        continue
                    mid = str(mid)
                    if mid in need_chunk_ids:
                        map_entries[mid] = mo
                        if len(map_entries) >= len(need_chunk_ids):
                            break

        # Validate sampled chunks
        for ch in tqdm(sample, desc=f"[{d}] validate-sample"):
            cid = ch.get("id") or ch.get("chunk_id")
            pid = ch.get("parent_id") or ch.get("parent")
            start = ch.get("start")
            end = ch.get("end")
            text = ch.get("text", "")

            if cid is None or pid is None:
                res.err_bad_id_format += 1
                if len(res.examples["bad_record"]) < 5:
                    res.examples["bad_record"].append({"chunk": ch})
                continue

            cid = str(cid)
            pid = str(pid)

            if pid not in parent_texts:
                res.err_missing_parent += 1
                if len(res.examples["missing_parent"]) < 5:
                    res.examples["missing_parent"].append({"chunk_id": cid, "parent_id": pid})
                continue

            _, ptxt = parent_texts[pid]
            if ptxt is None:
                ptxt = ""
            ptxt = str(ptxt)

            # id format
            if not cid.startswith(pid + "__c"):
                res.err_bad_id_format += 1
                if len(res.examples["bad_id_format"]) < 5:
                    res.examples["bad_id_format"].append({"chunk_id": cid, "parent_id": pid})
                # still continue to check other errors

            # offset validity
            ok_offsets = True
            if not isinstance(start, int) or not isinstance(end, int):
                ok_offsets = False
            else:
                if start < 0 or end <= start or end > len(ptxt):
                    ok_offsets = False

            if not ok_offsets:
                res.err_bad_offsets += 1
                if len(res.examples["bad_offsets"]) < 5:
                    res.examples["bad_offsets"].append({
                        "chunk_id": cid, "parent_id": pid, "start": start, "end": end,
                        "parent_len": len(ptxt)
                    })
                continue  # can't slice-check if offsets invalid

            # slice match
            slice_txt = ptxt[start:end]
            if slice_txt != text:
                res.err_slice_mismatch += 1
                if len(res.examples["slice_mismatch"]) < 5:
                    res.examples["slice_mismatch"].append({
                        "chunk_id": cid, "parent_id": pid, "start": start, "end": end,
                        "expected_prefix": slice_txt[:120],
                        "got_prefix": str(text)[:120],
                    })

            # token length check (best-effort)
            tok_len = len(tokenizer.encode(str(text), add_special_tokens=False))
            if tok_len > args.max_tokens + args.token_tolerance:
                res.err_token_too_long += 1
                if len(res.examples["token_too_long"]) < 5:
                    res.examples["token_too_long"].append({
                        "chunk_id": cid, "parent_id": pid, "tok_len": tok_len,
                        "max_tokens": args.max_tokens, "tolerance": args.token_tolerance
                    })

            # map check
            if args.check_map and map_path.exists():
                m = map_entries.get(cid)
                if m is None:
                    res.err_map_missing += 1
                else:
                    mpid = str(m.get("parent_id") or m.get("parent") or "")
                    ms = m.get("start")
                    me = m.get("end")
                    if mpid != pid or ms != start or me != end:
                        res.err_map_mismatch += 1
                        if len(res.examples["map_mismatch"]) < 5:
                            res.examples["map_mismatch"].append({
                                "chunk_id": cid,
                                "chunk": {"parent_id": pid, "start": start, "end": end},
                                "map": {"parent_id": mpid, "start": ms, "end": me},
                            })

        # Print summary
        print("\n[SUMMARY]")
        print(f" total_chunks      : {res.total_chunks}")
        print(f" unique_parents    : {res.unique_parents}")
        if res.unique_parents:
            print(f" avg_chunks/parent : {res.total_chunks / max(1, res.unique_parents):.3f}")
        print(f" max_chunks_parent : {res.max_chunks_parent}  (count={res.max_chunks_count})")
        print(f" sample_checked    : {res.sample_k}")
        print(" errors:")
        print(f"   missing_parent  : {res.err_missing_parent}")
        print(f"   bad_id_format   : {res.err_bad_id_format}")
        print(f"   bad_offsets     : {res.err_bad_offsets}")
        print(f"   slice_mismatch  : {res.err_slice_mismatch}")
        print(f"   token_too_long  : {res.err_token_too_long}")
        if args.check_map:
            print(f"   map_missing     : {res.err_map_missing}")
            print(f"   map_mismatch    : {res.err_map_mismatch}")

        # Write report
        out_report = report_dir / f"{d}_check.json"
        payload = {
            "domain": d,
            "paths": {
                "doc_zip": str(doc_zip),
                "chunk_path": str(chunk_path),
                "map_path": str(map_path),
            },
            "params": {
                "max_tokens": args.max_tokens,
                "token_tolerance": args.token_tolerance,
                "sample_k": args.sample_k,
                "seed": args.seed,
                "check_map": bool(args.check_map),
            },
            "summary": {
                "total_chunks": res.total_chunks,
                "unique_parents": res.unique_parents,
                "max_chunks_parent": res.max_chunks_parent,
                "max_chunks_count": res.max_chunks_count,
                "sample_checked": res.sample_k,
                "errors": {
                    "missing_parent": res.err_missing_parent,
                    "bad_id_format": res.err_bad_id_format,
                    "bad_offsets": res.err_bad_offsets,
                    "slice_mismatch": res.err_slice_mismatch,
                    "token_too_long": res.err_token_too_long,
                    "map_missing": res.err_map_missing,
                    "map_mismatch": res.err_map_mismatch,
                },
            },
            "examples": {k: v for k, v in res.examples.items()},
        }
        out_report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[REPORT] wrote: {out_report}")


if __name__ == "__main__":
    main()
