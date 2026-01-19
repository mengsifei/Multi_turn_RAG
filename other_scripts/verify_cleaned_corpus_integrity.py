#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, gzip, re
from pathlib import Path

def open_maybe_gz(path: str, mode="rt"):
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8", errors="ignore")
    return open(path, mode, encoding="utf-8", errors="ignore")

def load_blacklist(path: str | None) -> set[str]:
    if not path:
        return set()
    s = set()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            t = line.strip()
            if t:
                s.add(t)
    return s

def get_doc_id(o: dict) -> str:
    return str(o.get("document_id") or o.get("_id") or o.get("id") or "")

def load_corpus_ids(corpus_path: str) -> set[str]:
    ids = set()
    with open_maybe_gz(corpus_path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            did = get_doc_id(o)
            if did:
                ids.add(did)
    return ids

def load_qrels_docids(qrels_path: str, rel_min: float = 1.0) -> tuple[set[str], set[str]]:
    """
    Supports:
      - TREC qrels style: qid 0 docid rel  (whitespace separated)
      - BEIR-like: qid docid rel
    Returns:
      any_docids, relmin_docids
    """
    any_ids = set()
    rel_ids = set()
    with open_maybe_gz(qrels_path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"\s+", line)
            # skip header rows like: "query-id corpus-id score"
            if parts[0].lower().startswith("query") and any("corpus" in p.lower() for p in parts[:3]):
                continue
            if any(p.lower() in ("query-id", "corpus-id", "score") for p in parts[:3]):
                continue

            if len(parts) < 3:
                continue

            # qid 0 docid rel  OR  qid docid rel
            if len(parts) >= 4 and parts[1] == "0":
                docid = parts[2]
                rel = parts[3]
            else:
                docid = parts[1]
                rel = parts[-1]

            any_ids.add(str(docid))
            try:
                if float(rel) >= rel_min:
                    rel_ids.add(str(docid))
            except Exception:
                pass
    return any_ids, rel_ids

def fmt_path(pattern: str, domain: str) -> str:
    return pattern.format(domain=domain)

def write_list(path: Path, items: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as w:
        for x in items:
            w.write(x + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="clapnq,cloud,fiqa,govt")
    ap.add_argument("--corpus_pattern", required=True,
                    help='e.g. "human/retrieval_tasks_derived/{domain}.cleaned.jsonl" '
                         'or "human/retrieval_tasks_cleaned/{domain}/{domain}.jsonl"')
    ap.add_argument("--qrels_pattern", default="human/retrieval_tasks/{domain}/qrels/dev.tsv",
                    help='default: "human/retrieval_tasks/{domain}/qrels/dev.tsv"')
    ap.add_argument("--blacklist", default=None, help="txt file, one id per line")
    ap.add_argument("--rel_min", type=float, default=1.0)
    ap.add_argument("--out_dir", default="reports/verify_cleaned")
    ap.add_argument("--sample", type=int, default=20)
    args = ap.parse_args()

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    bl = load_blacklist(args.blacklist)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    overall_missing_any = 0
    overall_missing_rel = 0
    overall_blacklist_hits = 0

    print("\n========== Verify cleaned corpora ==========")
    print("[domains]", domains)
    print("[blacklist] size =", len(bl))
    print("[corpus_pattern]", args.corpus_pattern)
    print("[qrels_pattern ]", args.qrels_pattern)

    for d in domains:
        corpus_path = fmt_path(args.corpus_pattern, d)
        qrels_path = fmt_path(args.qrels_pattern, d)

        if not Path(corpus_path).exists():
            print(f"[skip] {d}: missing corpus: {corpus_path}")
            continue
        if not Path(qrels_path).exists():
            print(f"[skip] {d}: missing qrels:  {qrels_path}")
            continue

        corpus_ids = load_corpus_ids(corpus_path)
        q_any, q_rel = load_qrels_docids(qrels_path, rel_min=args.rel_min)

        

        missing_any = sorted(q_any - corpus_ids)
        missing_rel = sorted(q_rel - corpus_ids)

        bl_hits = sorted((bl & corpus_ids) if bl else [])

        overall_missing_any += len(missing_any)
        overall_missing_rel += len(missing_rel)
        overall_blacklist_hits += len(bl_hits)

        print(f"\n[{d}] corpus_unique={len(corpus_ids)} | qrels_any={len(q_any)} | qrels_rel>={args.rel_min}={len(q_rel)}")
        print(f"[{d}] missing qrels_any={len(missing_any)} | missing qrels_rel={len(missing_rel)} | blacklist_hits_in_corpus={len(bl_hits)}")

        # write reports
        write_list(out_dir / f"{d}.missing_qrels_any.txt", missing_any)
        write_list(out_dir / f"{d}.missing_qrels_rel.txt", missing_rel)
        write_list(out_dir / f"{d}.blacklist_hits_in_corpus.txt", bl_hits)

        if missing_rel:
            print(f"[WARN] {d} missing relevant in corpus (sample {min(args.sample,len(missing_rel))}): {missing_rel[:args.sample]}")
        if bl_hits:
            print(f"[WARN] {d} blacklist ids still present in corpus (sample {min(args.sample,len(bl_hits))}): {bl_hits[:args.sample]}")

    print("\n========== Overall ==========")
    print("[overall] missing_qrels_any =", overall_missing_any)
    print("[overall] missing_qrels_rel =", overall_missing_rel)
    print("[overall] blacklist_hits_in_corpus =", overall_blacklist_hits)
    print("[ok] wrote reports to:", out_dir)

if __name__ == "__main__":
    main()
