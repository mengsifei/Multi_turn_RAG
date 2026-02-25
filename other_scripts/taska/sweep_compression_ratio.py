#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ast
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

def _cr_name(cr: float) -> str:
    s = f"{cr:.4f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def _safe_tag(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-+" else "_" for ch in s)


def _file_sig(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"exists": False, "path": str(p)}
    st = p.stat()
    return {
        "exists": True,
        "path": str(p),
        "size": st.st_size,
        "mtime": int(st.st_mtime),
    }


def _sweep_cache_key(base_script: str, ratio: float, known_args: Dict[str, Any], extra: List[str]) -> str:
    # key should change if base script changes, or args change
    base_sig = _file_sig(base_script)
    payload = {
        "base_script": str(Path(base_script).resolve()),
        "base_sig": base_sig,
        "ratio": float(ratio),
        "known_args": known_args,
        "extra": extra,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return h


def _parse_aggregate_scores(stdout: str) -> List[Dict[str, Any]]:
    """
    Parse lines like:
      Retriever Evaluation Aggregate Scores: {'nDCG': [...], 'Recall': [...], 'collection': '...', 'count': 180}
    """
    out: List[Dict[str, Any]] = []
    lines = stdout.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if "Retriever Evaluation Aggregate Scores:" not in line:
            i += 1
            continue

        buf = line[line.find("Retriever Evaluation Aggregate Scores:"):]
        buf = buf[buf.find("{"):] if "{" in buf else ""

        brace = buf.count("{") - buf.count("}")
        j = i + 1
        while brace > 0 and j < len(lines):
            buf += "\n" + lines[j]
            brace += lines[j].count("{") - lines[j].count("}")
            j += 1

        try:
            d = ast.literal_eval(buf.strip())
            if isinstance(d, dict) and ("nDCG" in d or "Recall" in d):
                out.append(d)
        except Exception:
            pass

        i = j
    return out


def _weighted_avg(rows: List[Dict[str, Any]], key: str) -> Optional[List[float]]:
    if not rows:
        return None
    total = 0.0
    acc: Optional[List[float]] = None
    for r in rows:
        cnt = float(r.get("count", 0) or 0)
        arr = r.get(key)
        if not isinstance(arr, list) or cnt <= 0:
            continue
        if acc is None:
            acc = [0.0] * len(arr)
        if len(arr) != len(acc):
            continue
        for i, v in enumerate(arr):
            acc[i] += float(v) * cnt
        total += cnt
    if acc is None or total <= 0:
        return None
    return [x / total for x in acc]


def main():
    ap = argparse.ArgumentParser(
        description="Sweep compression_ratio for your eval script, with sweep-level cache/resume."
    )

    # ---- compatibility: allow base_script as positional (like you used) ----
    ap.add_argument("base_script", nargs="?", help="Path to your eval script (e.g., eval_jasper_ft_cached.py)")
    ap.add_argument("--base_script", dest="base_script_opt", default=None,
                    help="Same as positional base_script, but explicit.")

    ap.add_argument("--python", type=str, default="python3")

    ap.add_argument("--ratios", type=str, default="0.33,0.5,0.7,0.8,1.0")

    # if not set, we will derive a prefix
    ap.add_argument("--sweep_name", type=str, default=None,
                    help="Prefix for base script --model_name. If not set, derived from model_dir name.")

    # sweep cache
    ap.add_argument("--out_dir", type=str, default="outputs")
    ap.add_argument("--sweep_cache_dir", type=str, default=None,
                    help="Where sweep cache JSON files are stored. Default: <out_dir>/sweep_cache")
    ap.add_argument("--reuse", action="store_true", help="Reuse cached sweep result if available (default: true)")
    ap.add_argument("--no_reuse", action="store_true", help="Disable reuse (force run base script).")

    # Known args (we will pass through to base script)
    ap.add_argument("--task", type=str, default="lastturn", choices=["lastturn", "questions", "rewrite"])
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--base_dir", type=str, default=None)
    ap.add_argument("--cache_dir", type=str, default="cache/doc_emb_jasper_ft")
    ap.add_argument("--model_tag", type=str, default=None)

    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--doc_bs", type=int, default=256)
    ap.add_argument("--query_bs", type=int, default=256)
    ap.add_argument("--chunk_size", type=int, default=50000)
    ap.add_argument("--max_len", type=int, default=512)

    ap.add_argument("--corpus_override_dir", type=str, default=None)
    ap.add_argument("--corpus_override_suffix", type=str, default=".en_only.jsonl.gz")
    ap.add_argument("--blacklist_path", type=str, default=None)
    ap.add_argument("--split2", action="store_true")
    ap.add_argument("--force_recompute_docs", action="store_true")

    args, extra = ap.parse_known_args()

    base_script = args.base_script_opt or args.base_script
    if not base_script:
        raise SystemExit("Need base_script: e.g. python3 sweep_compression_ratio.py eval_jasper_ft_cached.py ...")
    base_script = str(Path(base_script).resolve())
    if not Path(base_script).exists():
        raise SystemExit(f"base_script not found: {base_script}")

    ratios = []
    for s in args.ratios.split(","):
        s = s.strip()
        if s:
            ratios.append(float(s))
    if not ratios:
        raise SystemExit("No valid ratios parsed from --ratios")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_dir = out_dir / "sweep_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    sweep_cache_dir = Path(args.sweep_cache_dir) if args.sweep_cache_dir else (out_dir / "sweep_cache")
    sweep_cache_dir.mkdir(parents=True, exist_ok=True)

    # reuse default: on
    reuse = True
    if args.no_reuse:
        reuse = False
    elif args.reuse:
        reuse = True

    # derive prefix if not provided
    prefix = args.sweep_name
    if not prefix:
        prefix = _safe_tag(Path(args.model_dir).name)

    summary_rows: List[Dict[str, Any]] = []
    details: Dict[str, Any] = {}

    # pack some known args to be part of cache key
    known_args_for_key = {
        "python": args.python,
        "task": args.task,
        "model_dir": args.model_dir,
        "base_dir": args.base_dir,
        "cache_dir": args.cache_dir,
        "model_tag": args.model_tag,
        "top_k": args.top_k,
        "doc_bs": args.doc_bs,
        "query_bs": args.query_bs,
        "chunk_size": args.chunk_size,
        "max_len": args.max_len,
        "corpus_override_dir": args.corpus_override_dir,
        "corpus_override_suffix": args.corpus_override_suffix,
        "blacklist_path": args.blacklist_path,
        "split2": args.split2,
        "force_recompute_docs": args.force_recompute_docs,
        "prefix": prefix,
    }

    for cr in ratios:
        cr_tag = _cr_name(cr)
        model_name = f"{prefix}_cr{cr_tag}"
        log_path = log_dir / f"{model_name}_{args.task}.log"

        # expected base outputs (your base script uses outputs/<model_name>_<task>*.jsonl)
        sub_path = Path("outputs") / f"{model_name}_{args.task}.jsonl"
        scored_path = Path("outputs") / f"{model_name}_{args.task}_score.jsonl"

        cache_key = _sweep_cache_key(base_script, cr, known_args_for_key, extra)
        cache_path = sweep_cache_dir / f"{prefix}_{args.task}_cr{cr_tag}_{cache_key}.json"

        if reuse and cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            # If you want stricter check, also ensure scored file exists.
            if cached.get("ok") is True and (not cached.get("require_score_file", True) or scored_path.exists()):
                print(f"\n[CACHE REUSE] cr={cr} -> {cache_path}")
                summary_rows.append(cached["summary_row"])
                details[str(cr)] = cached
                continue

        cmd = [
            args.python,
            base_script,
            "--task", args.task,
            "--model_dir", args.model_dir,
            "--model_name", model_name,
            "--cache_dir", args.cache_dir,
            "--compression_ratio", str(cr),
            "--top_k", str(args.top_k),
            "--doc_bs", str(args.doc_bs),
            "--query_bs", str(args.query_bs),
            "--chunk_size", str(args.chunk_size),
            "--max_len", str(args.max_len),
            "--corpus_override_suffix", args.corpus_override_suffix,
        ]

        if args.base_dir is not None:
            cmd += ["--base_dir", args.base_dir]
        if args.model_tag is not None:
            cmd += ["--model_tag", args.model_tag]
        if args.corpus_override_dir is not None:
            cmd += ["--corpus_override_dir", args.corpus_override_dir]
        if args.blacklist_path is not None:
            cmd += ["--blacklist_path", args.blacklist_path]
        if args.split2:
            cmd += ["--split2"]
        if args.force_recompute_docs:
            cmd += ["--force_recompute_docs"]

        # forward the rest (pack_adjacent / pack_* / anything else)
        if extra:
            cmd += extra

        print("\n" + "=" * 90)
        print(f"[RUN] compression_ratio={cr}  model_name={model_name}")
        print("[CMD]", " ".join(cmd))
        print("=" * 90)

        # proc = subprocess.run(cmd, text=True, capture_output=True)
        if sys.version_info >= (3, 7):
            proc = subprocess.run(cmd, text=True, capture_output=True)
        else:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,  # 3.6 用这个代替 text=True
            )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""

        # write log
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("### CMD\n")
            f.write(" ".join(cmd) + "\n\n")
            f.write("### STDOUT\n")
            f.write(stdout + "\n\n")
            f.write("### STDERR\n")
            f.write(stderr + "\n")
        print(f"[LOG] {log_path}")

        cached_obj: Dict[str, Any] = {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "ratio": cr,
            "model_name": model_name,
            "task": args.task,
            "cmd": cmd,
            "base_script": base_script,
            "log_path": str(log_path),
            "sub_path": str(sub_path),
            "scored_path": str(scored_path),
            "sub_sig": _file_sig(str(sub_path)),
            "scored_sig": _file_sig(str(scored_path)),
            "aggregate_by_collection": None,
            "weighted_ndcg": None,
            "weighted_recall": None,
            "summary_row": None,
            "require_score_file": True,
        }

        if proc.returncode != 0:
            row = {
                "compression_ratio": cr,
                "model_name": model_name,
                "ok": False,
                "error": f"returncode={proc.returncode}",
                "log_path": str(log_path),
            }
            cached_obj["summary_row"] = row
            summary_rows.append(row)
            details[str(cr)] = cached_obj
            cache_path.write_text(json.dumps(cached_obj, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[CACHE SAVE] {cache_path}")
            continue

        agg_rows = _parse_aggregate_scores(stdout)
        ndcg_w = _weighted_avg(agg_rows, "nDCG")
        recall_w = _weighted_avg(agg_rows, "Recall")

        row = {
            "compression_ratio": cr,
            "model_name": model_name,
            "ok": True,
            "log_path": str(log_path),
        }

        # assume 4 ks (often [1,3,5,10]) – if official changes, you'll still get the lists in JSON cache
        if ndcg_w is not None and len(ndcg_w) >= 4:
            row["ndcg@k1"], row["ndcg@k3"], row["ndcg@k5"], row["ndcg@k10"] = ndcg_w[:4]
        if recall_w is not None and len(recall_w) >= 4:
            row["recall@k1"], row["recall@k3"], row["recall@k5"], row["recall@k10"] = recall_w[:4]

        cached_obj["aggregate_by_collection"] = agg_rows
        cached_obj["weighted_ndcg"] = ndcg_w
        cached_obj["weighted_recall"] = recall_w
        cached_obj["summary_row"] = row
        cached_obj["sub_sig"] = _file_sig(str(sub_path))
        cached_obj["scored_sig"] = _file_sig(str(scored_path))

        summary_rows.append(row)
        details[str(cr)] = cached_obj
        cache_path.write_text(json.dumps(cached_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[CACHE SAVE] {cache_path}")

        print("[PARSED] weighted_nDCG:", ndcg_w)
        print("[PARSED] weighted_Recall:", recall_w)

    # summary CSV
    csv_path = out_dir / f"sweep_cr_{prefix}_{args.task}.csv"
    keys = [
        "compression_ratio", "model_name", "ok",
        "ndcg@k1", "ndcg@k3", "ndcg@k5", "ndcg@k10",
        "recall@k1", "recall@k3", "recall@k5", "recall@k10",
        "error", "log_path",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    print(f"\n[SUMMARY CSV] {csv_path}")

    # details JSON (lightweight pointer; heavy details already in per-run cache json)
    json_path = out_dir / f"sweep_cr_{prefix}_{args.task}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)
    print(f"[DETAIL JSON] {json_path}")


if __name__ == "__main__":
    main()
