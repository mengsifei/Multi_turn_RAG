#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare corpora between:
A) tasks_root/<dom>/<dom>.jsonl(.gz)  (e.g., cleaned_dataset)
B) corpus_override_dir/<dom><suffix> (e.g., human/retrieval_tasks_derived/<dom>.cleaned.jsonl(.gz))
   after removing blacklist ids

We verify:
1) tasks_root corpus contains NO blacklisted ids
2) ID sets match: tasks_root == (override - blacklist)
3) Text hashes match per ID (md5) for all IDs (sampled mismatches shown if any)
"""

import argparse, json, os, sys, gzip, hashlib, datetime, shlex
from pathlib import Path
from typing import Dict, Tuple, Iterable, Optional

DOMAINS_DEFAULT = ["clapnq", "fiqa", "cloud", "govt"]

def open_any(path: Path):
    p = str(path)
    if p.endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8")
    return open(p, "r", encoding="utf-8")

def md5_text(s: str) -> str:
    return hashlib.md5((s or "").encode("utf-8")).hexdigest()

def load_blacklist(path: Optional[Path]) -> set:
    if not path:
        return set()
    s = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # 每行一个 doc_id（你的 blacklist.valid.txt 就是这样）
            s.add(line)
    return s

def guess_id_and_text(obj: dict) -> Tuple[Optional[str], Optional[str]]:
    """
    Try to extract (document_id, text) from common corpus schemas.
    Supports:
      {"document_id": "...", "text": "..."}
      {"id": "...", "text": "..."}
      {"_id": "...", "title": "...", "text": "..."}  (BEIR style)
    """
    _id = obj.get("document_id") or obj.get("id") or obj.get("_id")
    if isinstance(_id, (int, float)):
        _id = str(_id)
    if not isinstance(_id, str) or not _id:
        return None, None

    txt = obj.get("text")
    if isinstance(txt, str) and txt.strip():
        return _id, txt.strip()

    # BEIR-style: title+text
    title = obj.get("title")
    body = obj.get("body") or obj.get("contents") or obj.get("content")
    if isinstance(title, str) and isinstance(body, str):
        t = (title.strip() + "\n" + body.strip()).strip()
        return _id, t if t else None

    # fallback: some corpora use "contents"
    if isinstance(body, str) and body.strip():
        return _id, body.strip()

    return _id, None

def load_corpus_hashmap(path: Path) -> Dict[str, str]:
    """
    Return map: doc_id -> md5(text)
    """
    out: Dict[str, str] = {}
    with open_any(path) as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise RuntimeError(f"Bad JSON in {path} line {ln}: {e}")
            did, txt = guess_id_and_text(obj)
            if did is None:
                continue
            if txt is None:
                txt = ""  # keep empty text deterministic
            out[did] = md5_text(txt)
    return out

def find_corpus_file_tasks_root(tasks_root: Path, dom: str) -> Path:
    """
    Expect: tasks_root/<dom>/<dom>.jsonl or .jsonl.gz
    """
    p1 = tasks_root / dom / f"{dom}.jsonl"
    p2 = tasks_root / dom / f"{dom}.jsonl.gz"
    if p1.exists(): return p1
    if p2.exists(): return p2
    raise FileNotFoundError(f"Cannot find corpus under tasks_root for {dom}: tried {p1} and {p2}")

def find_corpus_file_override(override_dir: Path, dom: str, suffix: str) -> Path:
    """
    Expect: override_dir/<dom><suffix> or gz
    e.g., human/retrieval_tasks_derived/clapnq.cleaned.jsonl
    """
    p1 = override_dir / f"{dom}{suffix}"
    p2 = override_dir / f"{dom}{suffix}.gz"
    if p1.exists(): return p1
    if p2.exists(): return p2
    raise FileNotFoundError(f"Cannot find corpus under override_dir for {dom}: tried {p1} and {p2}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks_root", required=True, type=Path)
    ap.add_argument("--corpus_override_dir", required=True, type=Path)
    ap.add_argument("--corpus_override_suffix", required=True, type=str)
    ap.add_argument("--blacklist_path", default=None, type=Path)
    ap.add_argument("--domains", nargs="*", default=DOMAINS_DEFAULT)
    ap.add_argument("--show_mismatches", type=int, default=20, help="How many mismatch examples to print per domain")
    args = ap.parse_args()

    cmd = " ".join(shlex.quote(x) for x in sys.argv)
    print(f"[CMD] {cmd}")
    print(f"[TIME] {datetime.datetime.now().isoformat(timespec='seconds')}")

    tasks_root = args.tasks_root
    override_dir = args.corpus_override_dir
    suffix = args.corpus_override_suffix
    blacklist = load_blacklist(args.blacklist_path)
    print(f"[INFO] blacklist size = {len(blacklist)}")

    all_ok = True

    for dom in args.domains:
        print(f"\n========== {dom} ==========")
        a_path = find_corpus_file_tasks_root(tasks_root, dom)
        b_path = find_corpus_file_override(override_dir, dom, suffix)
        print(f"[A] tasks_root corpus    : {a_path}")
        print(f"[B] override corpus      : {b_path}")

        A = load_corpus_hashmap(a_path)
        B = load_corpus_hashmap(b_path)

        A_ids = set(A.keys())
        B_ids = set(B.keys())

        # 1) ensure tasks_root doesn't contain blacklisted ids
        in_black_A = len(A_ids & blacklist)
        if in_black_A != 0:
            all_ok = False
            some = list((A_ids & blacklist))[:args.show_mismatches]
            print(f"[FAIL] tasks_root contains {in_black_A} blacklisted ids. Examples: {some}")
        else:
            print(f"[OK] tasks_root contains 0 blacklisted ids")

        # 2) compare ID sets: A == (B - blacklist)
        Bf_ids = B_ids - blacklist
        miss_in_A = sorted(list(Bf_ids - A_ids))[:args.show_mismatches]
        extra_in_A = sorted(list(A_ids - Bf_ids))[:args.show_mismatches]

        if miss_in_A or extra_in_A:
            all_ok = False
            print(f"[FAIL] ID set mismatch between A and (B-blacklist)")
            print(f"  |A|={len(A_ids)}  |B|={len(B_ids)}  |B-blacklist|={len(Bf_ids)}")
            if miss_in_A:
                print(f"  Missing in A (present in B-blacklist): {len(Bf_ids - A_ids)} examples: {miss_in_A}")
            if extra_in_A:
                print(f"  Extra in A (not in B-blacklist): {len(A_ids - Bf_ids)} examples: {extra_in_A}")
        else:
            print(f"[OK] ID sets match: A == (B-blacklist)  (|A|={len(A_ids)})")

        # 3) compare text hashes for all shared IDs
        shared = A_ids & Bf_ids
        mism = []
        for did in shared:
            # B might contain did but blacklisted removed already; did is in Bf
            if A[did] != B[did]:
                mism.append(did)
                if len(mism) >= args.show_mismatches:
                    break
        if mism:
            all_ok = False
            print(f"[FAIL] Text hash mismatch for {len(mism)} (showing up to {args.show_mismatches})")
            print(f"  Examples: {mism}")
        else:
            print(f"[OK] Text hashes match for all shared IDs (n={len(shared)})")

    print("\n========== SUMMARY ==========")
    if all_ok:
        print("[PASS] corpora are equivalent: tasks_root == (override - blacklist) for all domains.")
        print("=> 只看语料层面，你的两个 root 是一致的。")
    else:
        print("[FAIL] corpora are NOT equivalent. See logs above for details.")
        sys.exit(2)

if __name__ == "__main__":
    main()
