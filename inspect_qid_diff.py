#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspect per-qid differences between two runs:
- prints query text
- prints relevant doc ids from qrels
- prints topN from run_a and run_b with relevance flags and snippets

Example:
python inspect_qid_diff.py \
  --domain fiqa \
  --task lastturn \
  --retrieval_tasks_root cleaned_dataset \
  --qrels_tsv human/retrieval_tasks/fiqa/qrels/dev.tsv \
  --run_a outputs/cleaned_hybrid_rrf_lastturn.rerank.jsonl \
  --run_b outputs/hybrid_rrf_pool1000_lastturn.rerank_cand1000.jsonl \
  --qid "1c74814752c3f9d4ed0b99c16e1ed192<::>1" \
  --topn 20 --snippet 220
"""

import argparse, csv, json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

def load_qrels(path: str) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        _ = next(r, None)
        for row in r:
            if len(row) < 3: 
                continue
            qid, did, s = row[0], row[1], row[2]
            try:
                si = int(float(s))
            except Exception:
                si = 0
            qrels.setdefault(qid, {})[did] = si
    return qrels

def load_run(path: str) -> Dict[str, List[Tuple[str, float]]]:
    out: Dict[str, List[Tuple[str, float]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): 
                continue
            o = json.loads(line)
            qid = o.get("task_id")
            if not qid: 
                continue
            ctxs = o.get("contexts") or []
            arr = []
            for c in ctxs:
                did = c.get("document_id")
                if not did:
                    continue
                try:
                    sc = float(c.get("score", 0.0))
                except Exception:
                    sc = 0.0
                arr.append((did, sc))
            out[qid] = arr
    return out

def _norm_qid(qid: str) -> str:
    return qid.split("<::>", 1)[0] if "<::>" in qid else qid

def load_query_text_map(path: Path) -> Dict[str, str]:
    m: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): 
                continue
            o = json.loads(line)
            qid = o.get("task_id") or o.get("query_id") or o.get("id") or o.get("_id")
            q  = o.get("query") or o.get("text") or o.get("question")
            if isinstance(qid, str) and isinstance(q, str) and q.strip():
                m[qid] = q.strip()
    return m

def get_query_text(qmap: Dict[str, str], qid: str) -> Optional[str]:
    if qid in qmap:
        return qmap[qid]
    base = _norm_qid(qid)
    return qmap.get(base)

def collect_needed_docids(runA, runB, rels: Set[str], qid: str, topn: int) -> Set[str]:
    need = set(rels)
    need.update([d for d,_ in runA.get(qid, [])[:topn]])
    need.update([d for d,_ in runB.get(qid, [])[:topn]])
    return need

def load_corpus_snippets(path: Path, need_ids: Set[str], snippet: int) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not need_ids:
        return out
    id_keys = ("id","_id","doc_id","corpus_id","document_id")
    text_keys = ("text","contents","content","document","passage","body")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if len(out) == len(need_ids):
                break
            if not line.strip():
                continue
            o = json.loads(line)
            _id = None
            for k in id_keys:
                v = o.get(k)
                if isinstance(v, str) and v:
                    _id = v
                    break
            if not _id or _id not in need_ids:
                continue

            txt = None
            for k in text_keys:
                v = o.get(k)
                if isinstance(v, str) and v.strip():
                    txt = v.strip()
                    break
            if txt is None:
                title = o.get("title")
                text  = o.get("text")
                if isinstance(title, str) and isinstance(text, str):
                    txt = (title.strip() + "\n" + text.strip()).strip()
            if not txt:
                txt = ""
            out[_id] = (txt[:snippet].replace("\n"," ") + ("..." if len(txt) > snippet else ""))
    return out

def show_run(name: str, arr: List[Tuple[str,float]], rels: Set[str], snippets: Dict[str,str], topn: int):
    print(f"\n== {name} top{topn} ==")
    print(f"{'rk':>3}  {'rel':>3}  {'score':>9}  {'doc_id':<35}  snippet")
    for i, (did, sc) in enumerate(arr[:topn], start=1):
        r = 1 if did in rels else 0
        sn = snippets.get(did, "")
        print(f"{i:3d}  {r:3d}  {sc:9.4f}  {did:<35}  {sn}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["fiqa","clapnq","govt","cloud"])
    ap.add_argument("--task", required=True, choices=["lastturn","questions","rewrite"])
    ap.add_argument("--retrieval_tasks_root", default="cleaned_dataset")
    ap.add_argument("--qrels_tsv", required=True)
    ap.add_argument("--run_a", required=True)
    ap.add_argument("--run_b", required=True)
    ap.add_argument("--qid", required=True)
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--snippet", type=int, default=220)
    args = ap.parse_args()

    root = Path(args.retrieval_tasks_root) / args.domain
    qfile = root / f"{args.domain}_{args.task}.jsonl"
    cfile = root / f"{args.domain}.jsonl"

    qrels = load_qrels(args.qrels_tsv)
    runA = load_run(args.run_a)
    runB = load_run(args.run_b)
    qmap = load_query_text_map(qfile)

    rels = {d for d,s in qrels.get(args.qid, {}).items() if int(s) > 0}
    qtxt = get_query_text(qmap, args.qid)

    print("QID:", args.qid)
    print("Query:", (qtxt or "<NOT FOUND>"))
    print("Relevant doc ids (qrels):", len(rels))
    for d in list(sorted(rels))[:15]:
        print("  -", d)
    if len(rels) > 15:
        print("  ...")

    need = collect_needed_docids(runA, runB, rels, args.qid, args.topn)
    snippets = load_corpus_snippets(cfile, need, args.snippet)

    show_run("RUN_A", runA.get(args.qid, []), rels, snippets, args.topn)
    show_run("RUN_B", runB.get(args.qid, []), rels, snippets, args.topn)

if __name__ == "__main__":
    main()
