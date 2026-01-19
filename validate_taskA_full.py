#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import Counter, defaultdict

OFFICIAL_COLLECTIONS = {
    "mt-rag-clapnq-elser-512-100-20240503",
    "mt-rag-govt-elser-512-100-20240611",
    "mt-rag-fiqa-beir-elser-512-100-20240501",
    "mt-rag-ibmcloud-elser-512-100-20240502",
}

def is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and len(x.strip()) > 0

def as_float(x: Any) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        return float(x)  # may raise ValueError
    raise ValueError("not a number")

def check_record(obj: Dict[str, Any], require_nonempty_contexts: bool) -> List[str]:
    errs: List[str] = []

    # required top-level keys
    for k in ["conversation_id", "task_id", "Collection", "input", "contexts"]:
        if k not in obj:
            errs.append(f"missing_top_key:{k}")

    if "task_id" in obj and not is_nonempty_str(obj["task_id"]):
        errs.append("bad_task_id")

    if "conversation_id" in obj and not is_nonempty_str(obj["conversation_id"]):
        errs.append("bad_conversation_id")

    if "Collection" in obj:
        coll = obj["Collection"]
        if not is_nonempty_str(coll):
            errs.append("bad_collection_type")
        elif coll not in OFFICIAL_COLLECTIONS:
            errs.append(f"collection_not_official:{coll}")

    # input checks
    if "input" in obj:
        inp = obj["input"]
        if not isinstance(inp, list):
            errs.append("input_not_list")
        else:
            if len(inp) == 0:
                errs.append("input_empty")
            for i, turn in enumerate(inp[:50]):  # limit checks for speed
                if not isinstance(turn, dict):
                    errs.append(f"input_item_not_dict:i={i}")
                    continue
                if "speaker" not in turn:
                    errs.append(f"input_missing_speaker:i={i}")
                if "text" not in turn:
                    errs.append(f"input_missing_text:i={i}")
                else:
                    if not is_nonempty_str(turn["text"]):
                        errs.append(f"input_bad_text:i={i}")

    # contexts checks
    if "contexts" in obj:
        ctxs = obj["contexts"]
        if not isinstance(ctxs, list):
            errs.append("contexts_not_list")
        else:
            if require_nonempty_contexts and len(ctxs) == 0:
                errs.append("contexts_empty_but_required")
            for j, c in enumerate(ctxs[:500]):  # cap to avoid pathological huge entries
                if not isinstance(c, dict):
                    errs.append(f"context_item_not_dict:j={j}")
                    continue
                for ck in ["document_id", "score", "text"]:
                    if ck not in c:
                        errs.append(f"context_missing_{ck}:j={j}")
                if "document_id" in c and not is_nonempty_str(c["document_id"]):
                    errs.append(f"context_bad_document_id:j={j}")
                if "text" in c and not isinstance(c["text"], str):
                    errs.append(f"context_text_not_str:j={j}")
                if "score" in c:
                    try:
                        _ = as_float(c["score"])
                    except Exception:
                        errs.append(f"context_score_not_numeric:j={j}")

    return errs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", type=Path, required=True,
                    help="Path to Sifei_taskA_full.jsonl")
    ap.add_argument("--max_errors_print", type=int, default=20,
                    help="How many bad rows to print (task_id + first error)")
    ap.add_argument("--require_nonempty_contexts", action="store_true",
                    help="If set, contexts must be non-empty")
    ap.add_argument("--out_report", type=Path, default=None,
                    help="Optional: write a JSON report with error summary + samples")
    args = ap.parse_args()

    total = 0
    bad = 0
    err_counter = Counter()
    samples = []  # list of dicts: {line, task_id, errors_preview}

    with args.input_jsonl.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except Exception as e:
                bad += 1
                err_counter["json_parse_error"] += 1
                if len(samples) < args.max_errors_print:
                    samples.append({"line": line_no, "task_id": None, "errors_preview": [f"json_parse_error:{e}"]})
                continue

            errs = check_record(obj, args.require_nonempty_contexts)
            if errs:
                bad += 1
                for e in errs:
                    err_counter[e] += 1
                if len(samples) < args.max_errors_print:
                    samples.append({
                        "line": line_no,
                        "task_id": obj.get("task_id"),
                        "errors_preview": errs[:6],
                    })

    ok = total - bad

    print(f"[DONE] file={args.input_jsonl}")
    print(f"[STATS] total_rows={total} ok_rows={ok} bad_rows={bad}")

    if bad == 0:
        print("[PASS] All rows match the required schema.")
    else:
        print("[FAIL] Found schema violations. Top error types:")
        for k, v in err_counter.most_common(20):
            print(f"  {k}: {v}")
        print("\n[EXAMPLES] bad rows (preview):")
        for s in samples:
            print(f"  line={s['line']} task_id={s['task_id']} errors={s['errors_preview']}")

    if args.out_report:
        report = {
            "file": str(args.input_jsonl),
            "total_rows": total,
            "ok_rows": ok,
            "bad_rows": bad,
            "require_nonempty_contexts": args.require_nonempty_contexts,
            "errors_top": err_counter.most_common(200),
            "samples": samples,
        }
        args.out_report.parent.mkdir(parents=True, exist_ok=True)
        args.out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[DONE] wrote report: {args.out_report}")

if __name__ == "__main__":
    main()
