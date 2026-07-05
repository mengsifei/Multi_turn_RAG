#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a CSV by replacing RAG.jsonl contexts with contexts from best_res_777.jsonl,
resolving context text from original retrieval task chunk files, and adding query fields.

Main inputs:
    human/generation_tasks/RAG.jsonl
    best_res_777.jsonl
    human/retrieval_tasks/{domain}/{domain}.jsonl
    cleaned_dataset/{domain}/{domain}_rewrite_gpt.jsonl
    cleaned_dataset/{domain}/{domain}_lastturn.jsonl

Output CSV fields:
    task_id
    conversation_id
    rewrite_query
    lastturn_query
    questions
    contexts
    answerability
    collection
    targets

Special rule for UNANSWERABLE and CONVERSATIONAL tasks:
    - questions is built from RAG.jsonl input by concatenating every user turn:
        |user|: ...
        |user|: ...
    - lastturn_query is replaced by the last user turn from those questions.
    - contexts is [] if the task_id is not in best_res_777.jsonl.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_DOMAINS = ["clapnq", "govt", "cloud", "fiqa"]
TEXT_KEYS = ("text", "contents", "content", "body")
ID_KEYS = ("document_id", "id", "_id")


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    """Yield (line_no, json_obj) from a JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path} 第 {line_no} 行不是合法 JSON: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"{path} 第 {line_no} 行不是 JSON object")
            yield line_no, obj


def json_dumps(obj: Any) -> str:
    """Dump JSON for CSV cells, preserving non-ASCII characters."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def normalize_answerability(rag_obj: Dict[str, Any]) -> Any:
    """
    RAG sample uses key 'Answerability': ['UNANSWERABLE'].
    Keep the original value if present; fallback to common variants.
    """
    for key in ("Answerability", "answerability", "Answerable"):
        if key in rag_obj:
            return rag_obj[key]
    return None


def answerability_values(answerability: Any) -> List[str]:
    """Normalize answerability into a lowercase string list."""
    if answerability is None:
        return []
    if isinstance(answerability, list):
        return [str(x).strip().lower() for x in answerability]
    if isinstance(answerability, tuple):
        return [str(x).strip().lower() for x in answerability]
    return [str(answerability).strip().lower()]


def needs_questions_from_rag_input(answerability: Any) -> bool:
    """
    Return True for labels that should use full conversational user-turn history
    from RAG.jsonl input.

    Currently applies to:
      - UNANSWERABLE
      - CONVERSATIONAL
    """
    vals = answerability_values(answerability)
    return any(
        v == "unanswerable"
        or "unanswerable" in v
        or v == "conversational"
        or "conversational" in v
        for v in vals
    )


def format_user_turn(text: Any) -> str:
    """Format one user turn in the same style as cleaned_dataset query files."""
    text = "" if text is None else str(text).strip()
    return f"|user|: {text}"


def build_questions_from_rag_input(rag_obj: Dict[str, Any]) -> Tuple[str, str]:
    """
    For UNANSWERABLE / CONVERSATIONAL examples, concatenate all user turns from RAG input.

    Returns:
        questions_text: all user turns joined with newline
        lastturn_text: the last formatted user turn
    """
    user_turns: List[str] = []

    for item in rag_obj.get("input", []) or []:
        if not isinstance(item, dict):
            continue
        speaker = str(item.get("speaker", "")).strip().lower()
        if speaker != "user":
            continue
        user_turns.append(format_user_turn(item.get("text", "")))

    questions_text = "\n".join(user_turns)
    lastturn_text = user_turns[-1] if user_turns else ""
    return questions_text, lastturn_text


def get_collection(obj: Dict[str, Any]) -> str:
    return str(obj.get("Collection", obj.get("collection", "")) or "")


def detect_domain_from_collection(collection: str) -> Optional[str]:
    """
    Examples:
      mt-rag-clapnq-elser-512-100-20240503 -> clapnq
      mt-rag-fiqa-...                         -> fiqa
    """
    c = (collection or "").lower()
    for domain in DEFAULT_DOMAINS:
        if domain in c:
            return domain

    # Fallback: try mt-rag-{domain}-...
    prefix = "mt-rag-"
    if c.startswith(prefix):
        rest = c[len(prefix):]
        if "-" in rest:
            return rest.split("-", 1)[0]
        return rest or None

    return None


def first_existing_value(obj: Dict[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for key in keys:
        value = obj.get(key)
        if value is not None:
            return value
    return None


def get_doc_text(obj: Dict[str, Any], include_title: bool = False) -> str:
    value = first_existing_value(obj, TEXT_KEYS)
    text = "" if value is None else str(value)

    if include_title:
        title = obj.get("title")
        if title:
            text = f"{title}\n{text}"

    return text


def get_all_doc_ids(obj: Dict[str, Any]) -> List[str]:
    ids: List[str] = []
    for key in ID_KEYS:
        value = obj.get(key)
        if value is None:
            continue
        value = str(value)
        if value and value not in ids:
            ids.append(value)
    return ids


def suffix_after_first_underscore(doc_id: str) -> Optional[str]:
    if "_" not in doc_id:
        return None
    suffix = doc_id.split("_", 1)[1]
    return suffix or None


class RetrievalIndex:
    """
    Stores document text from original retrieval_tasks files.

    It supports:
    1. exact lookup by document_id/id/_id
    2. fallback lookup by suffix after first underscore:
       e.g. 850931827_11086-12414-0-1328 -> 11086-12414-0-1328
    3. reverse suffix lookup:
       if query is 11086-12414-0-1328 but stored id is
       850931827_11086-12414-0-1328
    """

    def __init__(self) -> None:
        self.by_domain: Dict[str, Dict[str, str]] = defaultdict(dict)
        self.suffix_to_ids: Dict[str, Dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        self.conflicts: List[Dict[str, Any]] = []
        self.loaded_counts: Dict[str, int] = {}
        self.loaded_files: Dict[str, str] = {}

    def add_doc(
        self,
        domain: str,
        doc_ids: Sequence[str],
        text: str,
        source_path: Path,
        line_no: int,
    ) -> None:
        if not doc_ids:
            return

        for doc_id in doc_ids:
            existing = self.by_domain[domain].get(doc_id)
            if existing is not None and existing != text:
                self.conflicts.append(
                    {
                        "domain": domain,
                        "document_id": doc_id,
                        "source_file": str(source_path),
                        "line_no": line_no,
                        "old_text_preview": existing[:200],
                        "new_text_preview": text[:200],
                    }
                )
                # Keep first occurrence to avoid unstable output.
                continue

            self.by_domain[domain][doc_id] = text

            suffix = suffix_after_first_underscore(doc_id)
            if suffix:
                self.suffix_to_ids[domain][suffix].add(doc_id)

            # Also let the full doc_id point to itself in suffix map.
            self.suffix_to_ids[domain][doc_id].add(doc_id)

    def load_domain_file(self, domain: str, path: Path, include_title: bool = False) -> None:
        count = 0
        for line_no, obj in iter_jsonl(path):
            doc_ids = get_all_doc_ids(obj)
            text = get_doc_text(obj, include_title=include_title)
            self.add_doc(domain, doc_ids, text, path, line_no)
            count += 1

        self.loaded_counts[domain] = count
        self.loaded_files[domain] = str(path)

    def lookup(
        self,
        document_id: str,
        domain: Optional[str],
    ) -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
        """
        Return:
            text, matched_domain, match_method, candidate_ids

        match_method:
            exact
            query_suffix_exact
            reverse_suffix
            global_exact
            global_query_suffix_exact
            global_reverse_suffix
            ambiguous_suffix
            not_found
        """
        document_id = str(document_id)

        domain_order: List[str] = []
        if domain:
            domain_order.append(domain)
        domain_order.extend([d for d in self.by_domain.keys() if d not in domain_order])

        # 1. Exact lookup in preferred domain first.
        if domain and document_id in self.by_domain.get(domain, {}):
            return self.by_domain[domain][document_id], domain, "exact", [document_id]

        # 2. If query has prefix_, try suffix as exact id in preferred domain.
        suffix = suffix_after_first_underscore(document_id)
        if domain and suffix and suffix in self.by_domain.get(domain, {}):
            return self.by_domain[domain][suffix], domain, "query_suffix_exact", [suffix]

        # 3. Reverse suffix lookup in preferred domain.
        if domain:
            candidates = sorted(self.suffix_to_ids.get(domain, {}).get(document_id, set()))
            if len(candidates) == 1:
                candidate = candidates[0]
                return self.by_domain[domain][candidate], domain, "reverse_suffix", candidates
            if len(candidates) > 1:
                return None, domain, "ambiguous_suffix", candidates

        # 4. Global exact lookup.
        exact_matches: List[Tuple[str, str]] = []
        for d in domain_order:
            if document_id in self.by_domain.get(d, {}):
                exact_matches.append((d, document_id))
        if len(exact_matches) == 1:
            d, doc_id = exact_matches[0]
            return self.by_domain[d][doc_id], d, "global_exact", [doc_id]
        if len(exact_matches) > 1:
            return None, None, "ambiguous_suffix", [f"{d}:{doc_id}" for d, doc_id in exact_matches]

        # 5. Global query suffix exact lookup.
        if suffix:
            suffix_exact_matches: List[Tuple[str, str]] = []
            for d in domain_order:
                if suffix in self.by_domain.get(d, {}):
                    suffix_exact_matches.append((d, suffix))
            if len(suffix_exact_matches) == 1:
                d, doc_id = suffix_exact_matches[0]
                return self.by_domain[d][doc_id], d, "global_query_suffix_exact", [doc_id]
            if len(suffix_exact_matches) > 1:
                return None, None, "ambiguous_suffix", [f"{d}:{doc_id}" for d, doc_id in suffix_exact_matches]

        # 6. Global reverse suffix lookup.
        reverse_matches: List[Tuple[str, str]] = []
        for d in domain_order:
            for candidate in sorted(self.suffix_to_ids.get(d, {}).get(document_id, set())):
                reverse_matches.append((d, candidate))

        if len(reverse_matches) == 1:
            d, doc_id = reverse_matches[0]
            return self.by_domain[d][doc_id], d, "global_reverse_suffix", [doc_id]
        if len(reverse_matches) > 1:
            return None, None, "ambiguous_suffix", [f"{d}:{doc_id}" for d, doc_id in reverse_matches]

        return None, domain, "not_found", []


def load_retrieval_index(
    retrieval_root: Path,
    domains: Sequence[str],
    include_title: bool = False,
) -> RetrievalIndex:
    index = RetrievalIndex()

    for domain in domains:
        path = retrieval_root / domain / f"{domain}.jsonl"
        if not path.exists():
            print(f"[WARN] retrieval file 不存在，跳过: {path}", file=sys.stderr)
            continue

        index.load_domain_file(domain=domain, path=path, include_title=include_title)
        print(f"[OK] loaded retrieval {domain}: {index.loaded_counts[domain]} chunks from {path}")

    if not index.by_domain:
        raise FileNotFoundError(
            f"没有成功加载任何 retrieval jsonl。请检查 --retrieval_root: {retrieval_root}"
        )

    return index


def load_rag_task_domains(rag_path: Path) -> Dict[str, str]:
    task_to_domain: Dict[str, str] = {}
    for _, obj in iter_jsonl(rag_path):
        task_id = obj.get("task_id")
        if not task_id:
            continue
        domain = detect_domain_from_collection(get_collection(obj))
        if domain:
            task_to_domain[str(task_id)] = domain
    return task_to_domain


def load_query_file(path: Path) -> Dict[str, str]:
    """
    Load a cleaned_dataset query jsonl file:
        {"_id": "...", "text": "..."}
    """
    mapping: Dict[str, str] = {}

    for line_no, obj in iter_jsonl(path):
        qid = obj.get("_id", obj.get("id", obj.get("task_id")))
        if not qid:
            print(f"[WARN] {path} 第 {line_no} 行缺少 _id/id/task_id，跳过", file=sys.stderr)
            continue

        text = obj.get("text", "")
        qid = str(qid)
        text = "" if text is None else str(text)

        if qid in mapping and mapping[qid] != text:
            print(
                f"[WARN] {path} 中 _id 重复且 text 不同，保留第一次: {qid}",
                file=sys.stderr,
            )
            continue

        mapping[qid] = text

    return mapping


def load_cleaned_query_maps(
    cleaned_root: Path,
    domains: Sequence[str],
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Any]]:
    """
    Load rewrite_gpt and lastturn jsonl files for all domains.

    Returns:
        rewrite_by_task_id
        lastturn_by_task_id
        summary
    """
    rewrite_by_task_id: Dict[str, str] = {}
    lastturn_by_task_id: Dict[str, str] = {}

    summary: Dict[str, Any] = {
        "loaded_files": {},
        "loaded_counts": {},
        "missing_files": [],
        "duplicate_ids": [],
    }

    def merge_mapping(kind: str, domain: str, path: Path, loaded: Dict[str, str]) -> None:
        target = rewrite_by_task_id if kind == "rewrite" else lastturn_by_task_id

        for qid, text in loaded.items():
            if qid in target and target[qid] != text:
                summary["duplicate_ids"].append(
                    {
                        "kind": kind,
                        "domain": domain,
                        "_id": qid,
                        "old_text": target[qid],
                        "new_text": text,
                    }
                )
                continue
            target[qid] = text

        summary["loaded_files"][f"{domain}_{kind}"] = str(path)
        summary["loaded_counts"][f"{domain}_{kind}"] = len(loaded)

    for domain in domains:
        rewrite_path = cleaned_root / domain / f"{domain}_rewrite_gpt.jsonl"
        lastturn_path = cleaned_root / domain / f"{domain}_lastturn.jsonl"

        if rewrite_path.exists():
            rewrite_loaded = load_query_file(rewrite_path)
            merge_mapping("rewrite", domain, rewrite_path, rewrite_loaded)
            print(f"[OK] loaded rewrite queries {domain}: {len(rewrite_loaded)} from {rewrite_path}")
        else:
            summary["missing_files"].append(str(rewrite_path))
            print(f"[WARN] rewrite query file 不存在，跳过: {rewrite_path}", file=sys.stderr)

        if lastturn_path.exists():
            lastturn_loaded = load_query_file(lastturn_path)
            merge_mapping("lastturn", domain, lastturn_path, lastturn_loaded)
            print(f"[OK] loaded lastturn queries {domain}: {len(lastturn_loaded)} from {lastturn_path}")
        else:
            summary["missing_files"].append(str(lastturn_path))
            print(f"[WARN] lastturn query file 不存在，跳过: {lastturn_path}", file=sys.stderr)

    summary["total_rewrite_queries"] = len(rewrite_by_task_id)
    summary["total_lastturn_queries"] = len(lastturn_by_task_id)

    return rewrite_by_task_id, lastturn_by_task_id, summary


def load_best_contexts(
    best_path: Path,
    retrieval_index: RetrievalIndex,
    rag_task_domains: Dict[str, str],
    include_scores: bool = False,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]], int, Dict[str, int]]:
    """
    Load best_res contexts by task_id, replacing each context with:
        {"document_id": ..., "text": text_from_original_retrieval_docs}

    Returns:
        best_by_task_id
        missing_doc_ids details
        total number of document_id references checked
        match_method_counts
    """
    best_by_task_id: Dict[str, List[Dict[str, Any]]] = {}
    missing_doc_ids: List[Dict[str, Any]] = []
    total_doc_refs = 0
    match_method_counts: Dict[str, int] = defaultdict(int)

    for line_no, obj in iter_jsonl(best_path):
        task_id = obj.get("task_id")
        if not task_id:
            raise ValueError(f"{best_path} 第 {line_no} 行缺少 task_id")
        task_id = str(task_id)

        best_domain = detect_domain_from_collection(get_collection(obj))
        domain = best_domain or rag_task_domains.get(task_id)

        replaced_contexts: List[Dict[str, Any]] = []

        for ctx_idx, ctx in enumerate(obj.get("contexts", []) or []):
            if not isinstance(ctx, dict):
                continue

            doc_id = ctx.get("document_id")
            if not doc_id:
                continue

            doc_id = str(doc_id)
            total_doc_refs += 1

            text, matched_domain, match_method, candidates = retrieval_index.lookup(
                document_id=doc_id,
                domain=domain,
            )
            match_method_counts[match_method or "unknown"] += 1

            out_ctx: Dict[str, Any] = {
                "document_id": doc_id,
                "text": "" if text is None else text,
            }

            if include_scores:
                for score_key in ("score", "orig_score", "rerank_score"):
                    if score_key in ctx:
                        out_ctx[score_key] = ctx[score_key]

            if text is None:
                missing_doc_ids.append(
                    {
                        "task_id": task_id,
                        "best_res_line_no": line_no,
                        "context_index": ctx_idx,
                        "collection": get_collection(obj),
                        "expected_domain": domain,
                        "document_id": doc_id,
                        "match_method": match_method,
                        "candidate_ids": candidates[:20],
                        "num_candidates": len(candidates),
                    }
                )

            replaced_contexts.append(out_ctx)

        if task_id in best_by_task_id:
            raise ValueError(f"{best_path} 中 task_id 重复: {task_id}")

        best_by_task_id[task_id] = replaced_contexts

    return best_by_task_id, missing_doc_ids, total_doc_refs, dict(match_method_counts)


def write_csv(
    rag_path: Path,
    out_csv: Path,
    best_by_task_id: Dict[str, List[Dict[str, Any]]],
    rewrite_by_task_id: Dict[str, str],
    lastturn_by_task_id: Dict[str, str],
    questions_format: str = "text",
) -> Dict[str, int]:
    """
    Iterate through RAG.jsonl and write output CSV.

    If a RAG task_id is not in best_by_task_id, contexts = [].

    For UNANSWERABLE / CONVERSATIONAL rows:
        questions = concatenated user turns from RAG input
        lastturn_query = last user turn from RAG input
    """
    stats = {
        "rag_rows": 0,
        "rows_with_best_contexts": 0,
        "rows_without_best_contexts_unanswerable": 0,
        "rag_input_questions_rows": 0,
        "rag_input_questions_rows_with_questions": 0,
        "rewrite_query_hits": 0,
        "lastturn_query_hits_from_cleaned": 0,
        "lastturn_query_hits_from_rag_input": 0,
    }

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task_id",
                "conversation_id",
                "rewrite_query",
                "lastturn_query",
                "questions",
                "contexts",
                "answerability",
                "collection",
                "targets",
            ],
        )
        writer.writeheader()

        for _, obj in iter_jsonl(rag_path):
            stats["rag_rows"] += 1

            task_id = str(obj.get("task_id", ""))
            answerability = normalize_answerability(obj)
            row_needs_questions_from_rag_input = needs_questions_from_rag_input(answerability)

            if task_id in best_by_task_id:
                new_contexts = best_by_task_id[task_id]
                stats["rows_with_best_contexts"] += 1
            else:
                new_contexts = []
                stats["rows_without_best_contexts_unanswerable"] += 1

            rewrite_query = rewrite_by_task_id.get(task_id, "")
            if rewrite_query:
                stats["rewrite_query_hits"] += 1

            lastturn_query = lastturn_by_task_id.get(task_id, "")
            if lastturn_query:
                stats["lastturn_query_hits_from_cleaned"] += 1

            questions = ""

            if row_needs_questions_from_rag_input:
                stats["rag_input_questions_rows"] += 1
                questions_text, rag_lastturn = build_questions_from_rag_input(obj)

                if questions_text:
                    stats["rag_input_questions_rows_with_questions"] += 1

                if questions_format == "json":
                    questions = json_dumps({"_id": task_id, "text": questions_text}) if questions_text else ""
                else:
                    questions = questions_text

                # For UNANSWERABLE / CONVERSATIONAL examples, lastturn_query should come from the last user turn in RAG input.
                if rag_lastturn:
                    lastturn_query = rag_lastturn
                    stats["lastturn_query_hits_from_rag_input"] += 1

            writer.writerow(
                {
                    "task_id": task_id,
                    "conversation_id": obj.get("conversation_id", ""),
                    "rewrite_query": rewrite_query,
                    "lastturn_query": lastturn_query,
                    "questions": questions,
                    "contexts": json_dumps(new_contexts),
                    "answerability": json_dumps(answerability),
                    "collection": get_collection(obj),
                    "targets": json_dumps(obj.get("targets", [])),
                }
            )

    return stats


def write_json_report(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace RAG.jsonl contexts with best_res contexts, resolve context text from "
            "human/retrieval_tasks/{domain}/{domain}.jsonl, and add rewrite/lastturn/questions fields. UNANSWERABLE and CONVERSATIONAL rows use RAG input for questions/lastturn."
        )
    )
    parser.add_argument("--rag", required=True, type=Path, help="Path to RAG.jsonl")
    parser.add_argument("--best", required=True, type=Path, help="Path to best_res_777.jsonl")
    parser.add_argument(
        "--retrieval_root",
        type=Path,
        default=Path("human/retrieval_tasks"),
        help="Root path containing {domain}/{domain}.jsonl. Default: human/retrieval_tasks",
    )
    parser.add_argument(
        "--cleaned_root",
        type=Path,
        default=Path("cleaned_dataset"),
        help="Root path containing {domain}/{domain}_rewrite_gpt.jsonl and {domain}_lastturn.jsonl. Default: cleaned_dataset",
    )
    parser.add_argument(
        "--domains",
        nargs="*",
        default=DEFAULT_DOMAINS,
        help="Domains to load. Default: clapnq govt cloud fiqa",
    )
    parser.add_argument("--out_csv", required=True, type=Path, help="Output CSV path")
    parser.add_argument(
        "--missing_report",
        type=Path,
        default=Path("missing_document_ids.json"),
        help="Where to write missing document_id report if any. Default: missing_document_ids.json",
    )
    parser.add_argument(
        "--conflict_report",
        type=Path,
        default=Path("duplicate_doc_id_text_conflicts.json"),
        help="Where to write duplicate document_id text conflict report if any.",
    )
    parser.add_argument(
        "--match_report",
        type=Path,
        default=Path("document_id_match_summary.json"),
        help="Where to write match method summary. Default: document_id_match_summary.json",
    )
    parser.add_argument(
        "--query_report",
        type=Path,
        default=Path("query_coverage_summary.json"),
        help="Where to write query coverage summary. Default: query_coverage_summary.json",
    )
    parser.add_argument(
        "--questions_format",
        choices=["text", "json"],
        default="text",
        help=(
            "Format for CSV questions column on unanswerable rows. "
            "'text' writes only '|user|: ...\\n|user|: ...'. "
            "'json' writes {'_id': task_id, 'text': questions_text}. "
            "Default: text."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="If set, stop immediately when best_res has document_id not found in original retrieval docs.",
    )
    parser.add_argument(
        "--include_scores",
        action="store_true",
        help="If set, keep score/orig_score/rerank_score inside output contexts.",
    )
    parser.add_argument(
        "--include_title",
        action="store_true",
        help="If set, prepend title to text when original chunk has a title field.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    retrieval_index = load_retrieval_index(
        retrieval_root=args.retrieval_root,
        domains=args.domains,
        include_title=args.include_title,
    )

    if retrieval_index.conflicts:
        write_json_report(args.conflict_report, retrieval_index.conflicts)
        print(
            f"[WARN] 原始 retrieval docs 中发现 {len(retrieval_index.conflicts)} 个重复 ID 但 text 不同，"
            f"已写入: {args.conflict_report}",
            file=sys.stderr,
        )

    rag_task_domains = load_rag_task_domains(args.rag)

    rewrite_by_task_id, lastturn_by_task_id, query_summary = load_cleaned_query_maps(
        cleaned_root=args.cleaned_root,
        domains=args.domains,
    )
    write_json_report(args.query_report, query_summary)

    best_by_task_id, missing_doc_ids, total_doc_refs, match_method_counts = load_best_contexts(
        best_path=args.best,
        retrieval_index=retrieval_index,
        rag_task_domains=rag_task_domains,
        include_scores=args.include_scores,
    )

    match_summary = {
        "loaded_files": retrieval_index.loaded_files,
        "loaded_counts": retrieval_index.loaded_counts,
        "best_res_task_count": len(best_by_task_id),
        "best_res_context_document_id_refs": total_doc_refs,
        "match_method_counts": match_method_counts,
        "missing_count": len(missing_doc_ids),
    }
    write_json_report(args.match_report, match_summary)

    print(f"[OK] best_res task_id 数量: {len(best_by_task_id)}")
    print(f"[OK] best_res context document_id 引用总数: {total_doc_refs}")
    print(f"[OK] query coverage 统计已写入: {args.query_report}")
    print(f"[OK] document_id 匹配统计已写入: {args.match_report}")
    print(f"[OK] match_method_counts: {json_dumps(match_method_counts)}")

    if missing_doc_ids:
        write_json_report(args.missing_report, missing_doc_ids)
        msg = (
            f"[ERROR] best_res 中有 {len(missing_doc_ids)} 个 document_id "
            f"无法在原始 retrieval docs 里唯一找到。报告: {args.missing_report}"
        )
        if args.strict:
            print(msg, file=sys.stderr)
            return 1
        print(msg, file=sys.stderr)
        print("[WARN] 未使用 --strict，因此继续生成 CSV；缺失 text 会写为空字符串。", file=sys.stderr)
    else:
        print("[OK] 校验通过：best_res 里的所有 context document_id 都能在原始 retrieval docs 中找到。")

    stats = write_csv(
        rag_path=args.rag,
        out_csv=args.out_csv,
        best_by_task_id=best_by_task_id,
        rewrite_by_task_id=rewrite_by_task_id,
        lastturn_by_task_id=lastturn_by_task_id,
        questions_format=args.questions_format,
    )

    print(f"[OK] CSV 已生成: {args.out_csv}")
    print(
        "[OK] 统计: "
        f"RAG总行数={stats['rag_rows']}, "
        f"使用best_res替换context的行数={stats['rows_with_best_contexts']}, "
        f"contexts=[]的行数={stats['rows_without_best_contexts_unanswerable']}, "
        f"从RAG input生成questions的行数={stats['rag_input_questions_rows']}, "
        f"questions非空行数={stats['rag_input_questions_rows_with_questions']}, "
        f"rewrite命中={stats['rewrite_query_hits']}, "
        f"cleaned lastturn命中={stats['lastturn_query_hits_from_cleaned']}, "
        f"lastturn来自RAG input行数={stats['lastturn_query_hits_from_rag_input']}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
