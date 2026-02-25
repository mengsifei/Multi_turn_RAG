#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict

DOMAINS_DEFAULT = ["clapnq", "cloud", "fiqa", "govt"]

def load_ids_txt(path: Path) -> list[str]:
    ids = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s:
                ids.append(s)
    return ids

def iter_corpus_docids(corpus_path: Path):
    with corpus_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            doc_id = o.get("document_id") if "document_id" in o else o.get("_id") or o.get("id")
            if doc_id is not None:
                yield str(doc_id)

def load_qrels_docids(qrels_path: Path, rel_min: float):
    """
    Expected format (no header): qid \\t docid \\t relevance
    Some datasets may have more cols; we read first 3.
    Returns:
      qrels_any: set(docid) appeared in qrels regardless of relevance
      qrels_rel: set(docid) with relevance >= rel_min
    """
    qrels_any = set()
    qrels_rel = set()

    with qrels_path.open("r", encoding="utf-8", errors="ignore") as f:
        r = csv.reader(f, delimiter="\t")
        for row in r:
            if not row or len(row) < 2:
                continue
            # tolerate header-ish lines
            if row[0].lower() in ("query-id", "query_id", "qid") and len(row) >= 2:
                continue

            docid = row[1].strip()
            if not docid:
                continue
            qrels_any.add(docid)

            rel = 0.0
            if len(row) >= 3:
                try:
                    rel = float(row[2])
                except:
                    rel = 0.0
            if rel >= rel_min:
                qrels_rel.add(docid)

    return qrels_any, qrels_rel

def write_set(path: str, s: set[str]):
    outp = Path(path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("\n".join(sorted(s)) + ("\n" if s else ""), encoding="utf-8")
    print(f"[ok] wrote {len(s)} -> {outp}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blacklist", required=True, help="blacklist.txt, one id per line")
    ap.add_argument("--corpus_root", default="human/retrieval_tasks",
                    help="root containing <domain>/<domain>.jsonl and <domain>/qrels/dev.tsv")
    ap.add_argument("--domains", default=",".join(DOMAINS_DEFAULT),
                    help="comma-separated domains, default clapnq,cloud,fiqa,govt")
    ap.add_argument("--rel_min", type=float, default=1.0,
                    help="relevance threshold for 'relevant' qrels overlap (default 1)")
    ap.add_argument("--max_print", type=int, default=20, help="print at most N sample ids for each warning")

    # optional outputs
    ap.add_argument("--out_valid", default="", help="ids that exist in ANY domain corpus")
    ap.add_argument("--out_missing", default="", help="ids that do NOT exist in ANY domain corpus")
    ap.add_argument("--out_in_qrels_any", default="", help="ids that appear in ANY qrels (any relevance)")
    ap.add_argument("--out_in_qrels_relevant", default="", help="ids that appear in ANY qrels with rel>=rel_min")

    args = ap.parse_args()

    bl_path = Path(args.blacklist)
    root = Path(args.corpus_root)
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]

    bl_ids = load_ids_txt(bl_path)
    bl_set = set(bl_ids)
    print(f"[in] blacklist lines={len(bl_ids)} unique={len(bl_set)} file={bl_path}")

    # load corpora + qrels per domain
    corpus_ids = {}
    qrels_any_ids = {}
    qrels_rel_ids = {}

    for d in domains:
        corpus_path = root / d / f"{d}.jsonl"
        qrels_path = root / d / "qrels" / "dev.tsv"

        if corpus_path.exists():
            s = set(iter_corpus_docids(corpus_path))
            corpus_ids[d] = s
            print(f"[load] corpus {d}: docs={len(s)} from {corpus_path}")
        else:
            print(f"[skip] corpus missing for {d}: {corpus_path}")

        if qrels_path.exists():
            any_set, rel_set = load_qrels_docids(qrels_path, rel_min=args.rel_min)
            qrels_any_ids[d] = any_set
            qrels_rel_ids[d] = rel_set
            print(f"[load] qrels  {d}: any={len(any_set)} relevant(rel>={args.rel_min})={len(rel_set)} from {qrels_path}")
        else:
            print(f"[skip] qrels missing for {d}: {qrels_path}")

    # compute membership
    hit_any_corpus = set()
    hit_by_domain_corpus = defaultdict(set)

    in_qrels_any = set()
    in_qrels_relevant = set()
    in_qrels_any_by_domain = defaultdict(set)
    in_qrels_rel_by_domain = defaultdict(set)

    for bid in bl_set:
        # corpus membership
        for d, s in corpus_ids.items():
            if bid in s:
                hit_any_corpus.add(bid)
                hit_by_domain_corpus[d].add(bid)

        # qrels membership (any)
        for d, s in qrels_any_ids.items():
            if bid in s:
                in_qrels_any.add(bid)
                in_qrels_any_by_domain[d].add(bid)

        # qrels membership (relevant)
        for d, s in qrels_rel_ids.items():
            if bid in s:
                in_qrels_relevant.add(bid)
                in_qrels_rel_by_domain[d].add(bid)

    missing = bl_set - hit_any_corpus

    print("\n========== Summary (per domain) ==========")
    for d in domains:
        if d not in corpus_ids and d not in qrels_any_ids:
            continue
        hits_c = len(hit_by_domain_corpus.get(d, set()))
        hits_q_any = len(in_qrels_any_by_domain.get(d, set()))
        hits_q_rel = len(in_qrels_rel_by_domain.get(d, set()))
        print(f"[{d}] corpus_hit={hits_c} | in_qrels_any={hits_q_any} | in_qrels_relevant={hits_q_rel}")

    print("\n========== Summary (overall) ==========")
    print(f"[corpus] valid ids (exist in any corpus): {len(hit_any_corpus)} / {len(bl_set)} ({len(hit_any_corpus)/max(1,len(bl_set))*100:.2f}%)")
    print(f"[corpus] missing ids (exist in none):      {len(missing)} / {len(bl_set)} ({len(missing)/max(1,len(bl_set))*100:.2f}%)")
    print(f"[qrels]  in_qrels_any (any relevance):     {len(in_qrels_any)} / {len(bl_set)} ({len(in_qrels_any)/max(1,len(bl_set))*100:.2f}%)")
    print(f"[qrels]  in_qrels_relevant (rel>={args.rel_min}): {len(in_qrels_relevant)} / {len(bl_set)} ({len(in_qrels_relevant)/max(1,len(bl_set))*100:.2f}%)")

    if in_qrels_any:
        sample = sorted(in_qrels_any)[:args.max_print]
        print(f"\n[WARN] blacklist ids found in qrels (any relevance), sample {len(sample)}:")
        for x in sample:
            print(x)

    if in_qrels_relevant:
        sample = sorted(in_qrels_relevant)[:args.max_print]
        print(f"\n[CRITICAL] blacklist ids found in qrels with rel>={args.rel_min}, sample {len(sample)}:")
        for x in sample:
            print(x)

    # optional outputs
    if args.out_valid:
        write_set(args.out_valid, hit_any_corpus)
    if args.out_missing:
        write_set(args.out_missing, missing)
    if args.out_in_qrels_any:
        write_set(args.out_in_qrels_any, in_qrels_any)
    if args.out_in_qrels_relevant:
        write_set(args.out_in_qrels_relevant, in_qrels_relevant)

if __name__ == "__main__":
    main()
