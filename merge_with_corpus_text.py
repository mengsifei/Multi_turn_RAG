#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple, Optional
from collections import defaultdict

# short -> official long
OFFICIAL_COLLECTION_MAP = {
    "clapnq": "mt-rag-clapnq-elser-512-100-20240503",
    "govt": "mt-rag-govt-elser-512-100-20240611",
    "fiqa": "mt-rag-fiqa-beir-elser-512-100-20240501",
    "cloud": "mt-rag-ibmcloud-elser-512-100-20240502",
    "ibmcloud": "mt-rag-ibmcloud-elser-512-100-20240502",
}

# official long -> domain folder name (your corpus path key)
OFFICIAL_COLLECTION_TO_DOMAIN = {
    "mt-rag-clapnq-elser-512-100-20240503": "clapnq",
    "mt-rag-govt-elser-512-100-20240611": "govt",
    "mt-rag-fiqa-beir-elser-512-100-20240501": "fiqa",
    "mt-rag-ibmcloud-elser-512-100-20240502": "cloud",
}

def read_jsonl(p: Path) -> List[Dict[str, Any]]:
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"JSON parse error in {p} line {i}: {e}") from e
    return rows

def iter_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise ValueError(f"JSON parse error in {p} line {i}: {e}") from e

def write_jsonl(p: Path, rows: List[Dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def normalize_collection(coll: Optional[str]) -> str:
    if not coll:
        raise ValueError("Missing Collection in original input jsonl row.")
    c = coll.strip()
    # already official long name
    if c.startswith("mt-rag-") and ("elser-512-100" in c):
        return c
    key = c.lower()
    if key in OFFICIAL_COLLECTION_MAP:
        return OFFICIAL_COLLECTION_MAP[key]
    for short, official in OFFICIAL_COLLECTION_MAP.items():
        if short in key:
            return official
    raise ValueError(f"Cannot map Collection='{coll}' to an official long collection name.")

def to_float(x: Any, what: str) -> float:
    try:
        return float(x)
    except Exception:
        raise ValueError(f"Non-numeric {what}: {x}")

def get_domain_from_collection(official_collection: str) -> str:
    if official_collection not in OFFICIAL_COLLECTION_TO_DOMAIN:
        raise ValueError(f"Unknown official Collection: {official_collection}")
    return OFFICIAL_COLLECTION_TO_DOMAIN[official_collection]

def load_needed_texts_from_corpus(
    corpus_jsonl: Path,
    needed_ids: Set[str],
    id_key: str,
    text_key: str
) -> Dict[str, str]:
    """
    Stream scan corpus and only keep doc_id in needed_ids.
    """
    found: Dict[str, str] = {}
    if not needed_ids:
        return found

    for doc in iter_jsonl(corpus_jsonl):
        did = doc.get(id_key)
        if did in needed_ids:
            txt = doc.get(text_key)
            if isinstance(txt, str):
                found[did] = txt
            else:
                # keep empty string if text missing/non-string
                found[did] = ""
            if len(found) == len(needed_ids):
                break
    return found

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_jsonl", type=Path, required=True,
                    help="eval_data/rag_taskAC.jsonl (contains conversation_id/task_id/Collection/input)")
    ap.add_argument("--run_jsonl", type=Path, required=True,
                    help="run/submission jsonl (contains task_id + contexts with document_id + score)")
    ap.add_argument("--out_jsonl", type=Path, required=True,
                    help="output merged jsonl (sample-like minimal schema)")
    ap.add_argument("--corpus_root", type=Path, default=Path("cleaned_dataset"),
                    help="root directory that contains domain corpus jsonl, e.g. cleaned_dataset/<domain>/<domain>.jsonl")
    ap.add_argument("--score_key", type=str, default="score",
                    help="score field name in run contexts (default: score)")
    ap.add_argument("--topk", type=int, default=None,
                    help="truncate contexts to top-k")
    ap.add_argument("--corpus_id_key", type=str, default="id",
                    help="document id key in corpus jsonl (default: id)")
    ap.add_argument("--corpus_text_key", type=str, default="text",
                    help="text key in corpus jsonl (default: text)")
    ap.add_argument("--missing_text", type=str, default="",
                    help="fallback text if document_id not found in corpus (default: empty string)")
    args = ap.parse_args()

    orig_rows = read_jsonl(args.orig_jsonl)
    run_rows = read_jsonl(args.run_jsonl)

    # task_id -> contexts(list)
    run_map: Dict[str, List[Dict[str, Any]]] = {}
    for r in run_rows:
        tid = r.get("task_id")
        if not tid:
            raise KeyError(f"run row missing task_id: keys={list(r.keys())}")
        ctxs = r.get("contexts", []) or []
        if not isinstance(ctxs, list):
            raise TypeError(f"run contexts must be list for task_id={tid}, got {type(ctxs)}")
        run_map[tid] = ctxs

    # 1) collect needed doc_ids per domain
    needed_by_domain: Dict[str, Set[str]] = defaultdict(set)

    normalized_collection_by_task: Dict[str, str] = {}
    for o in orig_rows:
        tid = o.get("task_id")
        if not tid:
            raise KeyError(f"orig row missing task_id: keys={list(o.keys())}")

        official_coll = normalize_collection(o.get("Collection"))
        normalized_collection_by_task[tid] = official_coll
        domain = get_domain_from_collection(official_coll)

        ctxs = run_map.get(tid, []) or []
        if args.topk is not None:
            ctxs = ctxs[:args.topk]

        for c in ctxs:
            did = c.get("document_id")
            if not did:
                raise KeyError(f"Missing document_id in run contexts for task_id={tid}")
            needed_by_domain[domain].add(did)

    # 2) load texts from each domain corpus (streaming, only needed ids)
    text_by_domain: Dict[str, Dict[str, str]] = {}
    for domain, needed_ids in needed_by_domain.items():
        corpus_path = args.corpus_root / domain / f"{domain}.jsonl"
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found: {corpus_path}")
        mp = load_needed_texts_from_corpus(
            corpus_jsonl=corpus_path,
            needed_ids=needed_ids,
            id_key=args.corpus_id_key,
            text_key=args.corpus_text_key,
        )
        text_by_domain[domain] = mp
        print(f"[LOAD] domain={domain} needed={len(needed_ids)} found={len(mp)} corpus={corpus_path}")

    # 3) write merged output with minimal fields + corpus text
    merged: List[Dict[str, Any]] = []
    missing_in_run = 0
    missing_text_total = 0

    for o in orig_rows:
        tid = o["task_id"]
        official_coll = normalized_collection_by_task[tid]
        domain = get_domain_from_collection(official_coll)

        ctxs = run_map.get(tid)
        if ctxs is None:
            missing_in_run += 1
            ctxs = []
        if args.topk is not None:
            ctxs = ctxs[:args.topk]

        out_ctxs = []
        domain_text_map = text_by_domain.get(domain, {})
        for c in ctxs:
            did = c["document_id"]
            score = to_float(c.get(args.score_key), f"{args.score_key} (task_id={tid}, doc_id={did})")
            txt = domain_text_map.get(did)
            if txt is None:
                missing_text_total += 1
                txt = args.missing_text
            out_ctxs.append({"document_id": did, "text": txt, "score": score})

        merged.append({
            "conversation_id": o.get("conversation_id"),
            "task_id": tid,
            "Collection": official_coll,
            "input": o.get("input", []),
            "contexts": out_ctxs,
        })

    write_jsonl(args.out_jsonl, merged)
    print(f"[DONE] wrote: {args.out_jsonl}")
    print(f"[STATS] orig_rows={len(orig_rows)} run_rows={len(run_rows)} missing_in_run={missing_in_run} missing_text={missing_text_total}")

if __name__ == "__main__":
    main()
