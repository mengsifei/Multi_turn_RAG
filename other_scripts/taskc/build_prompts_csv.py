#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import csv
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm import tqdm

DOMAINS_DEFAULT = ["clapnq", "cloud", "fiqa", "govt"]


# ----------------------- io utils -----------------------

def infer_domain(collection_name: str, domains: List[str]) -> str:
    s = (collection_name or "").lower()
    for d in domains:
        if d in s:
            return d
    raise ValueError(f"Cannot infer domain from Collection={collection_name!r}, domains={domains}")


def load_doc_map(corpus_jsonl: Path) -> Dict[str, str]:
    mp: Dict[str, str] = {}
    if not corpus_jsonl.exists():
        return mp  # allow missing corpus if contexts already have text
    with open(corpus_jsonl, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            if "id" not in j:
                raise KeyError(f"[{corpus_jsonl}] line {ln}: expected field `id`, got keys={list(j.keys())}")
            mp[str(j["id"])] = (j.get("text", "") or "")
    return mp


def _extract_query_id(j: Dict[str, Any]) -> Optional[str]:
    # rewrite 文件里常见几种 key
    for k in ("task_id", "_id", "id"):
        if k in j and j[k]:
            return str(j[k])
    return None


def _extract_query_text(j: Dict[str, Any]) -> str:
    # 常见几种字段名，默认 text
    for k in ("text", "query", "question"):
        if k in j and j[k]:
            q = str(j[k]).strip()
            if q.startswith("|user|:"):
                q = q[len("|user|:"):].strip()
            return q
    return ""


def load_query_map(rewrite_jsonl: Path) -> Dict[str, str]:
    """
    支持 rewrite 文件每行字段是:
      - task_id 或 _id 或 id
      - text (或 query/question)
    """
    mp: Dict[str, str] = {}
    if not rewrite_jsonl.exists():
        return mp
    with open(rewrite_jsonl, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            tid = _extract_query_id(j)
            if not tid:
                continue
            q = _extract_query_text(j)
            if not q:
                continue
            mp[tid] = q
    return mp


def load_query_map_global(rewrite_jsonl: Path, domains: List[str]) -> Dict[str, Dict[str, str]]:
    """
    如果你给的是一个混合 domains 的 rewrite jsonl（比如 eval_data/rag_taskAC_rewrite_gpt.jsonl）
    就用 Collection 来分流到 domain->map
    """
    maps: Dict[str, Dict[str, str]] = {d: {} for d in domains}
    if not rewrite_jsonl.exists():
        return maps
    with open(rewrite_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            tid = _extract_query_id(j)
            if not tid:
                continue
            q = _extract_query_text(j)
            if not q:
                continue
            col = j.get("Collection", "")
            try:
                d = infer_domain(col, domains)
            except Exception:
                continue
            maps[d][tid] = q
    return maps


def get_last_user_question(record: Dict[str, Any]) -> Optional[str]:
    """
    从 taska_file 的 record["input"] 里取最后一条 user 问句
    """
    turns = record.get("input") or []
    for t in reversed(turns):
        if (t.get("speaker") or "").lower() == "user" and t.get("text"):
            return str(t["text"]).strip()
    return None


# ----------------------- prompt builders -----------------------

def docs_to_context(
    docs: List[Dict[str, Any]],
    max_doc_chars: int = 1200,
    use_doc_id_in_header: bool = False,
    include_score: bool = False,
) -> str:
    parts: List[str] = ["Documents:"]
    for i, d in enumerate(docs, 1):
        did = d.get("document_id", d.get("id", None))
        txt = (d.get("text", d.get("content", "")) or "")
        txt = txt[:max_doc_chars]

        header = f"[Document {i}]"
        meta = []
        if use_doc_id_in_header and did is not None:
            meta.append(f"id={did}")
        if include_score and d.get("score") is not None:
            meta.append(f"score={d.get('score')}")
        if meta:
            header += " " + " ".join(meta)

        parts.append(header)
        parts.append(txt)
        parts.append("")

    return "\n".join(parts).rstrip()


def generate_qa_prompt(question: str, context: str) -> str:
    return f"""<Question>:
{question}
</Question>

<Task>
Answer the question using ONLY the provided documents. You should think step by step, carefully reading throught the context provided, provide evidence (quotations) from the text. Your answer should be in format:
```

<Arguments>\n\n
Your thinking and quotations\n\n </Arguments>\n\n
===\n\n <Answer>: your final answer

```
If the answer cannot be determined from the documents, say "I don't know." 
</Task>

<Contexts>
{context}
</Contexts>
"""


def generate_qa_prompt_from_docs(
    question: str,
    docs: List[Dict[str, Any]],
    max_doc_chars: int = 1200,
    use_doc_id_in_header: bool = False,
    include_score: bool = False,
) -> str:
    context = docs_to_context(
        docs=docs,
        max_doc_chars=max_doc_chars,
        use_doc_id_in_header=use_doc_id_in_header,
        include_score=include_score,
    )
    return generate_qa_prompt(question, context)


def load_rewrite_map_global(rewrite_jsonl: Path) -> Dict[str, str]:
    mp: Dict[str, str] = {}
    with open(rewrite_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            j = json.loads(line)

            # 关键：按 task_id（兼容老字段 _id）
            tid = j.get("task_id") or j.get("_id") or j.get("id")
            if not tid:
                continue
            tid = str(tid)

            q = (j.get("text", "") or "").strip()
            if q.startswith("|user|:"):
                q = q[len("|user|:"):].strip()

            if q:
                mp[tid] = q
    return mp


# ----------------------- main -----------------------

def main(args: argparse.Namespace) -> None:
    domains = [d.strip() for d in args.domains.split(",") if d.strip()] or DOMAINS_DEFAULT
    cleaned_root = Path(args.cleaned_root)

    # preload corpora (optional)
    doc_maps: Dict[str, Dict[str, str]] = {}
    for d in domains:
        doc_maps[d] = load_doc_map(cleaned_root / d / f"{d}.jsonl")

    # preload rewrites
    rewrite_path = Path("/home/smen/mt-rag/mt-rag-benchmark/eval_data/rag_taskAC_rewrite_gpt.jsonl")
    rewrite_map = load_rewrite_map_global(rewrite_path)
    print(f"[INFO] loaded rewrites: {len(rewrite_map)} from {rewrite_path}")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = n_written = 0
    n_skip_bad = 0
    n_skip_no_task_or_collection = 0
    n_skip_no_query = 0
    n_skip_no_ctx = 0
    n_fallback_last_user = 0

    with open(args.taska_file, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8", newline="") as fout:

        fieldnames = ["task_id", "conversation_id", "Collection", "domain", "prompt"]
        if args.add_ctx_count:
            fieldnames.append("n_contexts")
        if args.add_question_source:
            fieldnames.append("question_source")

        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for line in tqdm(fin, desc="Building prompts.csv"):
            n_total += 1
            line = line.strip()
            if not line:
                n_skip_bad += 1
                continue
            try:
                j = json.loads(line)
            except Exception:
                n_skip_bad += 1
                continue

            task_id = j.get("task_id")
            collection = j.get("Collection")
            if not task_id or not collection:
                n_skip_no_task_or_collection += 1
                continue

            try:
                domain = infer_domain(collection, domains)
            except Exception:
                n_skip_no_task_or_collection += 1
                continue

            conv_id = str(task_id).split("::")[0]

            # question: prefer rewrite; fallback to last user in input
            # question = query_maps.get(domain, {}).get(str(task_id))
            question = rewrite_map.get(task_id)
            question_source = "rewrite"
            if not question and args.fallback_to_last_user:
                q2 = get_last_user_question(j)
                if q2:
                    question = q2
                    question_source = "last_user"
                    n_fallback_last_user += 1

            if not question:
                n_skip_no_query += 1
                continue

            # contexts: prefer c["text"]; else lookup by doc_id if corpus available
            ctxs: List[Dict[str, Any]] = []
            for c in (j.get("contexts") or [])[: args.topk]:
                did = c.get("document_id")
                txt = c.get("text")  # many pipelines already include text
                if not txt and did:
                    txt = doc_maps.get(domain, {}).get(str(did))
                if not txt:
                    continue
                ctxs.append({
                    "document_id": str(did) if did is not None else None,
                    "score": c.get("score", None),
                    "text": txt,
                })

            if not ctxs:
                n_skip_no_ctx += 1
                continue

            prompt = generate_qa_prompt_from_docs(
                question=question,
                docs=ctxs,
                max_doc_chars=args.max_doc_chars,
                use_doc_id_in_header=args.doc_id_in_header,
                include_score=args.include_score,
            )

            row = {
                "task_id": task_id,
                "conversation_id": conv_id,
                "Collection": collection,
                "domain": domain,
                "prompt": prompt,
            }
            if args.add_ctx_count:
                row["n_contexts"] = len(ctxs)
            if args.add_question_source:
                row["question_source"] = question_source

            writer.writerow(row)
            n_written += 1

    print("========== PROMPTS CSV SUMMARY ==========")
    print(f"Input lines                 : {n_total}")
    print(f"Rows written                : {n_written}")
    print(f"Skipped (bad/empty jsonl)    : {n_skip_bad}")
    print(f"Skipped (missing id/col/dom) : {n_skip_no_task_or_collection}")
    print(f"Skipped (no query)           : {n_skip_no_query}")
    print(f"Skipped (no contexts)        : {n_skip_no_ctx}")
    if args.fallback_to_last_user:
        print(f"Fallback to last_user used   : {n_fallback_last_user}")
    print(f"Output file                  : {out_path}")
    print("========================================")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--taska_file", required=True, help="TaskA retrieval output jsonl (has contexts)")
    ap.add_argument("--cleaned_root", default="cleaned_dataset", help="cleaned_dataset root (optional)")
    ap.add_argument("--task_name", type=str, default="rewrite_gpt", help="suffix: <domain>_<task_name>.jsonl")
    ap.add_argument("--rewrite_jsonl", type=str, default=None,
                    help="Optional: a single rewrite jsonl containing multiple domains (with Collection). "
                         "If provided, overrides per-domain <cleaned_root>/<domain>/<domain>_<task_name>.jsonl")
    ap.add_argument("--out_csv", required=True, help="output prompts.csv")
    ap.add_argument("--topk", type=int, default=5, help="use top-k contexts")
    ap.add_argument("--max_doc_chars", type=int, default=1200, help="truncate each doc text")
    ap.add_argument("--domains", type=str, default="clapnq,cloud,fiqa,govt", help="comma-separated domain list")
    ap.add_argument("--include_score", action="store_true", help="include score in document header metadata")
    ap.add_argument("--doc_id_in_header", action="store_true", help="include document_id in document header metadata")
    ap.add_argument("--fallback_to_last_user", action="store_true", default=True,
                    help="if rewrite query missing, fallback to last user question in `input` (default on)")
    ap.add_argument("--add_ctx_count", action="store_true", help="add n_contexts column to CSV")
    ap.add_argument("--add_question_source", action="store_true", help="add question_source column (rewrite/last_user)")
    args = ap.parse_args()
    main(args)
