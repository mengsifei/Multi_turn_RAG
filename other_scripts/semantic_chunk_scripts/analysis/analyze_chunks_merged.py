#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chunk analysis + validation (merged):
- Sampling + token length stats + representative examples + outliers
- Span overlap (boundary leakage) using passage_spans/span_index
- Fragmentation (chunks per parent_id)
- Validation vs document_level zip:
    * parent exists
    * 0 <= start < end <= len(parent_text)
    * chunk_text == parent_text[start:end]
    * token_len <= max_tokens + tolerance
- Optional map check: chunk_id -> (parent_id,start,end) consistency
- Optional compare two chunk files (semantic vs fixed) in one report

Example:
python analyze_chunks_merged.py \
  --domain fiqa \
  --chunk_gz corpora/chunk_level_docsem512_100_st_45/fiqa_docsem512_100_st_45.jsonl.gz \
  --span_gz corpora/passage_level/fiqa_passage_spans.jsonl.gz \
  --hf_tokenizer Jasper-Token-Compression-600M \
  --doc_dir corpora/document_level \
  --report_dir reports/chunk_reports \
  --sample_k 20000 --seed 0 --max_tokens 512 --token_tolerance 5

Compare:
python analyze_chunks_merged.py \
  --domain fiqa \
  --chunk_gz corpora/chunk_level_docsem512_100_st_45/fiqa_docsem512_100_st_45.jsonl.gz \
  --compare_chunk_gz corpora/chunk_level_fixed512_100/fiqa_fixed512_100.jsonl.gz \
  --span_gz corpora/passage_level/fiqa_passage_spans.jsonl.gz \
  --hf_tokenizer Jasper-Token-Compression-600M \
  --doc_dir corpora/document_level \
  --report_dir reports/chunk_reports \
  --sample_k 20000 --seed 0
"""

import argparse
import gzip
import json
import random
import zipfile
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

import matplotlib.pyplot as plt

def save_hist_png(values, bins, out_png: Path, title: str, xlabel: str):
    plt.figure()
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

# -------------------------
# IO helpers
# -------------------------
def open_text(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, "r", encoding="utf-8")


def iter_jsonl(path: Path) -> Iterable[dict]:
    with open_text(str(path)) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default


def get_str(x, default=""):
    if x is None:
        return default
    return str(x)


def derive_out_name(chunk_gz: Path) -> str:
    name = chunk_gz.name
    # strip common suffixes
    for suf in [".jsonl.gz", ".json.gz", ".jsonl", ".gz"]:
        if name.endswith(suf):
            name = name[: -len(suf)]
            break
    return name


# -------------------------
# Span overlap / leakage
# -------------------------
def load_span_index(span_path: str):
    """
    Load passage spans and build:
      - span_by_base: base_id -> (starts[], ends[], passage_ids[]) for fast overlap queries
      - span_by_passage: passage_id -> (base_id, start, end) for inferring parent/start/end

    Accepts files like:
      corpora/passage_level/<domain>_passage_spans.jsonl.gz
      corpora/span_index/<domain>_passage_span_index.jsonl.gz

    Each record must have: base_id, passage_id, start, end
    """
    by_base: Dict[str, List[Tuple[int, int, str]]] = {}
    by_passage: Dict[str, Tuple[str, int, int]] = {}

    with open_text(span_path) as f:
        for line in f:
            o = json.loads(line)
            base = str(o["base_id"])
            pid = str(o["passage_id"])
            s = int(o["start"])
            e = int(o["end"])

            by_base.setdefault(base, []).append((s, e, pid))
            by_passage[pid] = (base, s, e)

    span_by_base: Dict[str, Tuple[List[int], List[int], List[str]]] = {}
    for base, spans in by_base.items():
        spans.sort(key=lambda x: x[0])
        starts = [s for s, e, pid in spans]
        ends = [e for s, e, pid in spans]
        pids = [pid for s, e, pid in spans]
        span_by_base[base] = (starts, ends, pids)

    return span_by_base, by_passage




def overlap(a0, a1, b0, b1) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def count_overlapped_passages(span_index, base_id: str, cs: int, ce: int) -> Tuple[int, float, Optional[str]]:
    """
    Returns:
      cnt        : how many passage spans overlap [cs,ce)
      best_ratio : max(overlap_len / chunk_len) among overlapped passages
      best_pid   : passage_id achieving best_ratio
    """
    if not span_index or base_id not in span_index:
        return 0, 0.0, None

    starts, ends, pids = span_index[base_id]
    i = bisect_right(starts, cs) - 1
    if i < 0:
        i = 0

    cnt = 0
    best_ratio = 0.0
    best_pid = None
    denom = max(1, ce - cs)

    while i < len(starts) and starts[i] < ce:
        ps, pe, pid = starts[i], ends[i], pids[i]
        ov = overlap(cs, ce, ps, pe)
        if ov > 0:
            cnt += 1
            r = ov / denom
            if r > best_ratio:
                best_ratio = r
                best_pid = pid
        i += 1

    return cnt, float(best_ratio), best_pid


# -------------------------
# Zip doc reader (sampled parents only)
# -------------------------
def get_base_id(obj: dict) -> Optional[str]:
    for key in ("_id", "id", "document_id", "doc_id", "docid"):
        if key not in obj:
            continue
        v: Any = obj.get(key)
        if v is None:
            continue
        if isinstance(v, str):
            v2 = v.strip()
            if v2:
                return v2
            continue
        try:
            return str(v)
        except Exception:
            continue
    return None


def read_doc_texts_from_zip(
    doc_zip: Path,
    member_name: str,
    need_ids: Set[str],
    *,
    stop_when_all_found: bool = True,
) -> Dict[str, Tuple[str, str]]:
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
            found[str(pid)] = (str(title), str(text))

            if stop_when_all_found and len(found) >= len(need_ids):
                break
    return found


# -------------------------
# Sampling
# -------------------------
def reservoir_sample_dicts(stream: Iterable[dict], k: int, seed: int = 0) -> Tuple[List[dict], int]:
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


# -------------------------
# Report structs
# -------------------------
@dataclass
class ValidationErrors:
    missing_parent: int = 0
    bad_id_format: int = 0
    bad_offsets: int = 0
    slice_mismatch: int = 0
    token_too_long: int = 0
    map_missing: int = 0
    map_mismatch: int = 0

    examples: Dict[str, List[dict]] = None

    def __post_init__(self):
        if self.examples is None:
            self.examples = {}


def add_example(examples: Dict[str, List[dict]], key: str, obj: dict, limit: int = 5):
    if key not in examples:
        examples[key] = []
    if len(examples[key]) < limit:
        examples[key].append(obj)


def hist_to_json(values: np.ndarray, bins: List[int], *, name: str):
    """
    bins: 递增边界，比如 [0,16,32,...,512,10**9]
    返回 JSON-friendly dict: edges + counts + pct + labels
    """
    edges = np.array(bins, dtype=np.int64)
    hist, edges = np.histogram(values, bins=edges)

    n = int(len(values))
    counts = hist.astype(int).tolist()
    pct = [(c / max(1, n)) * 100.0 for c in counts]

    labels = []
    for i in range(len(edges) - 1):
        lo = int(edges[i])
        hi = int(edges[i + 1])
        if hi >= 10**9:
            labels.append(f"[{lo}, +inf)")
        else:
            labels.append(f"[{lo}, {hi})")

    return {
        "name": name,
        "n": n,
        "bin_edges": edges.astype(int).tolist(),   # len = nbins+1
        "bin_labels": labels,                      # len = nbins
        "counts": counts,                          # len = nbins
        "pct": pct,                                # len = nbins
    }


# -------------------------
# Core analysis per chunk file
# -------------------------
def analyze_one(
    *,
    domain: Optional[str],
    chunk_gz: Path,
    span_gz: Optional[Path],
    tokenizer,
    doc_zip: Optional[Path],
    doc_member: Optional[str],
    map_gz: Optional[Path],
    check_map: bool,
    sample_k: int,
    seed: int,
    max_tokens: int,
    token_tolerance: int,
    id_prefix: str,
    repr_snip_chars: int,
    top_outlier_k: int,
    infer_from_spans: bool,
) -> Dict[str, Any]:

    # ---- span helpers
    span_by_base = None
    span_by_passage = None
    if span_gz:
        span_by_base, span_by_passage = load_span_index(str(span_gz))

    # Pass 1: sample + count total + parent_counter
    parent_counter = Counter()
    sample: List[dict] = []
    total = 0

    rng = random.Random(seed)

    with open_text(str(chunk_gz)) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            total += 1

            # --- infer parent_id/start/end for passage-style records (id/text only)
            cid = str(o.get("id") or o.get("chunk_id") or "")
            pid = o.get("parent_id") or o.get("parent")
            s = o.get("start")
            e = o.get("end")

            if infer_from_spans and (pid is None or s is None or e is None) and span_by_passage:
                t = span_by_passage.get(cid)
                if t is not None:
                    pid2, s2, e2 = t
                    o["parent_id"] = str(pid2)
                    o["start"] = int(s2)
                    o["end"] = int(e2)

            # --- now count parents using inferred fields (if any)
            pid2 = o.get("parent_id") or o.get("parent")
            if pid2 is not None:
                parent_counter[str(pid2)] += 1

            # --- reservoir sample
            if len(sample) < sample_k:
                sample.append(o)
            else:
                j = rng.randrange(1, total + 1)
                if j <= sample_k:
                    sample[j - 1] = o

    # Token lengths (sample)
    def chunk_token_len(obj: dict) -> int:
        return len(tokenizer.encode(get_str(obj.get("text", "")), add_special_tokens=False))

    sample_tok_lens = np.array([chunk_token_len(o) for o in sample], dtype=np.int64) if sample else np.array([], dtype=np.int64)

    # Span overlap stats (sample)
    cross_cnt = []
    best_ratios = []
    if span_by_base:
        for o in sample:
            pid = get_str(o.get("parent_id") or o.get("parent") or "")
            s = safe_int(o.get("start"), None)
            e = safe_int(o.get("end"), None)
            if not pid or s is None or e is None:
                cross_cnt.append(0)
                best_ratios.append(0.0)
                continue
            c, br, _ = count_overlapped_passages(span_by_base, pid, s, e)
            cross_cnt.append(c)
            best_ratios.append(br)

    cross_cnt = np.array(cross_cnt, dtype=np.int64) if cross_cnt else np.array([], dtype=np.int64)
    best_ratios = np.array(best_ratios, dtype=np.float32) if best_ratios else np.array([], dtype=np.float32)

    # Fragmentation stats (full)
    frag_vals = np.array(list(parent_counter.values()), dtype=np.int64) if parent_counter else np.array([], dtype=np.int64)

    # Representative chunks from sample by percentile of token length
    representatives = []
    if len(sample_tok_lens) > 0:
        pct_points = [10, 25, 50, 75, 90, 95, 99]
        for p in pct_points:
            target = np.percentile(sample_tok_lens, p)
            idx = int(np.argmin(np.abs(sample_tok_lens - target)))
            o = sample[idx]
            cid = get_str(o.get("id") or o.get("chunk_id"))
            pid = get_str(o.get("parent_id") or o.get("parent"))
            txt = get_str(o.get("text", ""))
            representatives.append({
                "label": f"p{p:02d}",
                "tok_len": int(sample_tok_lens[idx]),
                "chunk_id": cid,
                "parent_id": pid,
                "start": safe_int(o.get("start"), None),
                "end": safe_int(o.get("end"), None),
                "snippet": txt[:repr_snip_chars],
            })

    # Outliers: top by token length (sample only; simple & reliable)
    outlier_items = []
    for o in sample:
        cid = get_str(o.get("id") or o.get("chunk_id") or "")
        if not cid:
            continue
        outlier_items.append((chunk_token_len(o), cid, o))
    outlier_items.sort(key=lambda x: x[0], reverse=True)
    outliers_top = []
    for L, cid, o in outlier_items[:top_outlier_k]:
        txt = get_str(o.get("text", ""))
        outliers_top.append({
            "tok_len": int(L),
            "chunk_id": cid,
            "parent_id": get_str(o.get("parent_id") or o.get("parent")),
            "start": safe_int(o.get("start"), None),
            "end": safe_int(o.get("end"), None),
            "snippet": txt[:repr_snip_chars],
        })

    # Validation (sample only)
    verr = ValidationErrors()
    parent_texts = {}

    if doc_zip and doc_member:
        need_parents = set()
        need_chunk_ids = set()
        for o in sample:
            pid = o.get("parent_id") or o.get("parent")
            cid = o.get("id") or o.get("chunk_id")
            if pid is not None:
                need_parents.add(str(pid))
            if cid is not None:
                need_chunk_ids.add(str(cid))

        parent_texts = read_doc_texts_from_zip(doc_zip, doc_member, need_parents)

        # map scan (sample chunk ids only)
        map_entries: Dict[str, dict] = {}
        if check_map and map_gz and map_gz.exists():
            for mo in iter_jsonl(map_gz):
                mid = mo.get("chunk_id") or mo.get("id")
                if mid is None:
                    continue
                mid = str(mid)
                if mid in need_chunk_ids:
                    map_entries[mid] = mo
                    if len(map_entries) >= len(need_chunk_ids):
                        break

        for o in sample:
            cid = o.get("id") or o.get("chunk_id")
            pid = o.get("parent_id") or o.get("parent")
            start = o.get("start")
            end = o.get("end")
            text = get_str(o.get("text", ""))

            if cid is None or pid is None:
                verr.bad_id_format += 1
                add_example(verr.examples, "bad_record", {"chunk": o})
                continue

            cid = str(cid)
            pid = str(pid)

            if pid not in parent_texts:
                verr.missing_parent += 1
                add_example(verr.examples, "missing_parent", {"chunk_id": cid, "parent_id": pid})
                continue

            _, ptxt = parent_texts[pid]
            ptxt = get_str(ptxt, "")

            # id format: only enforce if id_prefix is provided AND this looks like semantic-chunk ids
            # (For official passages like "10171-0-2129", run with --id_prefix "" to disable.)
            if id_prefix:
                expect = pid + id_prefix
                if not cid.startswith(expect):
                    verr.bad_id_format += 1
                    add_example(verr.examples, "bad_id_format", {"chunk_id": cid, "parent_id": pid, "expect_prefix": expect})

            # offset validity
            s = safe_int(start, None)
            e = safe_int(end, None)
            if s is None or e is None or s < 0 or e <= s or e > len(ptxt):
                verr.bad_offsets += 1
                add_example(verr.examples, "bad_offsets", {
                    "chunk_id": cid, "parent_id": pid, "start": start, "end": end, "parent_len": len(ptxt)
                })
                continue

            # slice match
            slice_txt = ptxt[s:e]
            if slice_txt != text:
                verr.slice_mismatch += 1
                add_example(verr.examples, "slice_mismatch", {
                    "chunk_id": cid, "parent_id": pid, "start": s, "end": e,
                    "expected_prefix": slice_txt[:120],
                    "got_prefix": text[:120],
                })

            # token len
            tok_len = len(tokenizer.encode(text, add_special_tokens=False))
            if tok_len > max_tokens + token_tolerance:
                verr.token_too_long += 1
                add_example(verr.examples, "token_too_long", {
                    "chunk_id": cid, "parent_id": pid,
                    "tok_len": tok_len, "max_tokens": max_tokens, "tolerance": token_tolerance
                })

            # map check
            if check_map and map_gz and map_gz.exists():
                m = map_entries.get(cid)
                if m is None:
                    verr.map_missing += 1
                else:
                    mpid = get_str(m.get("parent_id") or m.get("parent") or "")
                    ms = m.get("start")
                    me = m.get("end")
                    if mpid != pid or ms != s or me != e:
                        verr.map_mismatch += 1
                        add_example(verr.examples, "map_mismatch", {
                            "chunk_id": cid,
                            "chunk": {"parent_id": pid, "start": s, "end": e},
                            "map": {"parent_id": mpid, "start": ms, "end": me},
                        })

    # build stats summaries
    def pct(frac: float) -> float:
        return float(frac) * 100.0

    token_stats = {}
    if len(sample_tok_lens) > 0:
        token_stats = {
            "sample_n": int(len(sample_tok_lens)),
            "min": int(sample_tok_lens.min()),
            "max": int(sample_tok_lens.max()),
            "mean": float(sample_tok_lens.mean()),
            "p10": float(np.percentile(sample_tok_lens, 10)),
            "p25": float(np.percentile(sample_tok_lens, 25)),
            "p50": float(np.percentile(sample_tok_lens, 50)),
            "p75": float(np.percentile(sample_tok_lens, 75)),
            "p90": float(np.percentile(sample_tok_lens, 90)),
            "p95": float(np.percentile(sample_tok_lens, 95)),
            "p99": float(np.percentile(sample_tok_lens, 99)),
            "frac_lt_128": pct((sample_tok_lens < 128).mean()),
            "frac_lt_256": pct((sample_tok_lens < 256).mean()),
            "frac_lt_384": pct((sample_tok_lens < 384).mean()),
            "frac_ge_480": pct((sample_tok_lens >= 480).mean()),
            "frac_gt_max_tokens": pct((sample_tok_lens > (max_tokens + token_tolerance)).mean()),
        }

    leakage_stats = {}
    if len(cross_cnt) > 0:
        leakage_stats = {
            "mean_overlap_passages": float(cross_cnt.mean()),
            "p95_overlap_passages": float(np.percentile(cross_cnt, 95)),
            "max_overlap_passages": int(cross_cnt.max()),
            "frac_overlap_ge_2": pct((cross_cnt >= 2).mean()),
            "frac_overlap_ge_3": pct((cross_cnt >= 3).mean()),
            "mean_best_ratio": float(best_ratios.mean()) if len(best_ratios) else 0.0,
            "frac_best_ratio_lt_06": pct((best_ratios < 0.6).mean()) if len(best_ratios) else 0.0,
            "frac_best_ratio_lt_08": pct((best_ratios < 0.8).mean()) if len(best_ratios) else 0.0,
        }

    frag_stats = {}
    if len(frag_vals) > 0:
        top_parents = parent_counter.most_common(10)
        frag_stats = {
            "unique_parents": int(len(parent_counter)),
            "avg_chunks_per_parent": float(total / max(1, len(parent_counter))),
            "p50": float(np.percentile(frag_vals, 50)),
            "p75": float(np.percentile(frag_vals, 75)),
            "p90": float(np.percentile(frag_vals, 90)),
            "p95": float(np.percentile(frag_vals, 95)),
            "p99": float(np.percentile(frag_vals, 99)),
            "max_parent": top_parents[0][0] if top_parents else "",
            "max_parent_chunks": int(top_parents[0][1]) if top_parents else 0,
            "top10_parents": [{"parent_id": pid, "chunks": int(n)} for pid, n in top_parents],
        }

    validation = {
        "enabled": bool(doc_zip and doc_member),
        "missing_parent": verr.missing_parent,
        "bad_id_format": verr.bad_id_format,
        "bad_offsets": verr.bad_offsets,
        "slice_mismatch": verr.slice_mismatch,
        "token_too_long": verr.token_too_long,
        "map_missing": verr.map_missing,
        "map_mismatch": verr.map_mismatch,
        "examples": verr.examples,
    }

    tok_bins = [0, 2, 16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 480, 512, 10**9]
    tok_len_hist = hist_to_json(sample_tok_lens, tok_bins, name="token_length_histogram")

    return {
        "domain": domain,
        "chunk_gz": str(chunk_gz),
        "total_chunks": int(total),
        "sample_k": int(len(sample)),
        "token_stats_sample": token_stats,
        "leakage_stats_sample": leakage_stats,
        "fragmentation_full": frag_stats,
        "validation_sample": validation,
        "representative_chunks": representatives,
        "outliers_top": outliers_top,
        "notes": {
            "token_stats_based_on": "sample",
            "leakage_stats_based_on": "sample (needs span_gz)",
            "fragmentation_based_on": "full pass (counts only)",
            "validation_based_on": "sample (needs doc_zip)",
        },
        "token_len_histogram": tok_len_hist,
    }



def compare_summaries(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce a compact diff-like comparison for the most diagnostic metrics.
    """
    def pick(d, *keys, default=None):
        cur = d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    A = a.get("token_stats_sample", {})
    B = b.get("token_stats_sample", {})
    LA = a.get("leakage_stats_sample", {})
    LB = b.get("leakage_stats_sample", {})
    FA = a.get("fragmentation_full", {})
    FB = b.get("fragmentation_full", {})

    return {
        "token_fill": {
            "mean": [pick(A, "mean"), pick(B, "mean")],
            "frac_ge_480": [pick(A, "frac_ge_480"), pick(B, "frac_ge_480")],
            "frac_lt_256": [pick(A, "frac_lt_256"), pick(B, "frac_lt_256")],
            "frac_gt_max_tokens": [pick(A, "frac_gt_max_tokens"), pick(B, "frac_gt_max_tokens")],
        },
        "boundary_leakage": {
            "frac_overlap_ge_2": [pick(LA, "frac_overlap_ge_2"), pick(LB, "frac_overlap_ge_2")],
            "mean_best_ratio": [pick(LA, "mean_best_ratio"), pick(LB, "mean_best_ratio")],
            "frac_best_ratio_lt_08": [pick(LA, "frac_best_ratio_lt_08"), pick(LB, "frac_best_ratio_lt_08")],
        },
        "fragmentation": {
            "avg_chunks_per_parent": [pick(FA, "avg_chunks_per_parent"), pick(FB, "avg_chunks_per_parent")],
            "p95": [pick(FA, "p95"), pick(FB, "p95")],
            "p99": [pick(FA, "p99"), pick(FB, "p99")],
            "max_parent_chunks": [pick(FA, "max_parent_chunks"), pick(FB, "max_parent_chunks")],
        },
        "validation_errors_sample": {
            "slice_mismatch": [pick(a, "validation_sample", "slice_mismatch"), pick(b, "validation_sample", "slice_mismatch")],
            "bad_offsets": [pick(a, "validation_sample", "bad_offsets"), pick(b, "validation_sample", "bad_offsets")],
            "token_too_long": [pick(a, "validation_sample", "token_too_long"), pick(b, "validation_sample", "token_too_long")],
        },
    }





def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", type=str, default=None)

    ap.add_argument("--chunk_gz", type=str, required=True)
    ap.add_argument("--compare_chunk_gz", type=str, default=None, help="optional second chunk file to compare")

    ap.add_argument("--span_gz", type=str, default=None, help="optional passage spans/index for leakage analysis")

    ap.add_argument("--hf_tokenizer", type=str, required=True)

    ap.add_argument("--doc_dir", type=str, default="corpora/document_level")
    ap.add_argument("--doc_zip", type=str, default=None, help="override doc zip path (else use --doc_dir/--domain)")
    ap.add_argument("--doc_member", type=str, default=None, help="override zip member (else '<domain>.jsonl')")

    ap.add_argument("--map_gz", type=str, default=None, help="optional chunk map gz path")
    ap.add_argument("--check_map", action="store_true")

    ap.add_argument("--sample_k", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--token_tolerance", type=int, default=5)

    ap.add_argument("--id_prefix", type=str, default="__c", help="expected chunk_id prefix after parent_id, e.g. '__c'")

    ap.add_argument("--repr_snip_chars", type=int, default=500)
    ap.add_argument("--top_outlier_k", type=int, default=20)

    ap.add_argument("--report_dir", type=str, default="reports/chunk_reports")

    ap.add_argument("--infer_from_spans", action="store_true",
                help="if chunk has no parent_id/start/end, infer them from span_gz using passage id")

    args = ap.parse_args()

    chunk_gz = Path(args.chunk_gz)
    compare_gz = Path(args.compare_chunk_gz) if args.compare_chunk_gz else None
    span_gz = Path(args.span_gz) if args.span_gz else None
    map_gz = Path(args.map_gz) if args.map_gz else None

    # docs
    doc_zip = Path(args.doc_zip) if args.doc_zip else None
    doc_member = args.doc_member
    if doc_zip is None and args.domain:
        doc_zip = Path(args.doc_dir) / f"{args.domain}.jsonl.zip"
    if doc_member is None and args.domain:
        doc_member = f"{args.domain}.jsonl"

    if doc_zip and not doc_zip.exists():
        print(f"[WARN] doc_zip not found: {doc_zip} (validation will be skipped)")
        doc_zip = None
        doc_member = None

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.hf_tokenizer, trust_remote_code=True)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    out_base = derive_out_name(chunk_gz)
    if compare_gz:
        out_base = f"{derive_out_name(chunk_gz)}__VS__{derive_out_name(compare_gz)}"

    payload: Dict[str, Any] = {
        "meta": {
            "domain": args.domain,
            "hf_tokenizer": args.hf_tokenizer,
            "params": {
                "sample_k": args.sample_k,
                "seed": args.seed,
                "max_tokens": args.max_tokens,
                "token_tolerance": args.token_tolerance,
                "id_prefix": args.id_prefix,
                "repr_snip_chars": args.repr_snip_chars,
                "top_outlier_k": args.top_outlier_k,
                "check_map": bool(args.check_map),
            },
            "paths": {
                "chunk_gz": str(chunk_gz),
                "compare_chunk_gz": str(compare_gz) if compare_gz else None,
                "span_gz": str(span_gz) if span_gz else None,
                "doc_zip": str(doc_zip) if doc_zip else None,
                "doc_member": doc_member,
                "map_gz": str(map_gz) if map_gz else None,
            },
        }
    }

    print("\n" + "=" * 96)
    print(f"[RUN] chunk_gz = {chunk_gz}")
    if compare_gz:
        print(f"[RUN] compare = {compare_gz}")
    print(f"[RUN] span_gz  = {span_gz}")
    print(f"[RUN] doc_zip  = {doc_zip} member={doc_member}")
    print(f"[RUN] report_dir = {report_dir} out_base={out_base}")
    print("=" * 96)

    a = analyze_one(
        domain=args.domain,
        chunk_gz=chunk_gz,
        span_gz=span_gz,
        tokenizer=tokenizer,
        doc_zip=doc_zip,
        doc_member=doc_member,
        map_gz=map_gz,
        check_map=args.check_map,
        sample_k=args.sample_k,
        seed=args.seed,
        max_tokens=args.max_tokens,
        token_tolerance=args.token_tolerance,
        id_prefix=args.id_prefix,
        repr_snip_chars=args.repr_snip_chars,
        top_outlier_k=args.top_outlier_k,
        infer_from_spans=args.infer_from_spans,
    )

    
    payload["analysis"] = {"A": a}

    if compare_gz:
        b = analyze_one(
            domain=args.domain,
            chunk_gz=compare_gz,
            span_gz=span_gz,
            tokenizer=tokenizer,
            doc_zip=doc_zip,
            doc_member=doc_member,
            map_gz=None,              # usually different map; keep off by default
            check_map=False,
            sample_k=args.sample_k,
            seed=args.seed,
            max_tokens=args.max_tokens,
            token_tolerance=args.token_tolerance,
            id_prefix=args.id_prefix,  # may differ; override if needed
            repr_snip_chars=args.repr_snip_chars,
            top_outlier_k=args.top_outlier_k,
            infer_from_spans=args.infer_from_spans,
        )
        payload["analysis"]["B"] = b
        payload["compare"] = compare_summaries(a, b)

    # Save JSON report
    out_json = report_dir / f"{out_base}.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Also save a short TXT summary for quick glance
    def fmt_line(k, v):
        return f"{k:28s}: {v}"

    lines = []
    lines.append(f"chunk report: {out_base}")
    lines.append("-" * 80)
    for tag in ["A", "B"] if compare_gz else ["A"]:
        x = payload["analysis"][tag]
        lines.append(f"[{tag}] {x['chunk_gz']}")
        lines.append(fmt_line("total_chunks", x.get("total_chunks")))
        lines.append(fmt_line("sample_k", x.get("sample_k")))
        ts = x.get("token_stats_sample", {})
        if ts:
            lines.append(fmt_line("tok_mean", f"{ts.get('mean'):.2f}"))
            lines.append(fmt_line("tok_min/max", f"{ts.get('min')} / {ts.get('max')}"))
            lines.append(fmt_line("frac_ge_480", f"{ts.get('frac_ge_480'):.2f}%"))
            lines.append(fmt_line("frac_lt_256", f"{ts.get('frac_lt_256'):.2f}%"))
            lines.append(fmt_line("frac_gt_max_tokens", f"{ts.get('frac_gt_max_tokens'):.2f}%"))
        ls = x.get("leakage_stats_sample", {})
        if ls:
            lines.append(fmt_line("overlap>=2", f"{ls.get('frac_overlap_ge_2'):.2f}%"))
            lines.append(fmt_line("best_ratio_mean", f"{ls.get('mean_best_ratio'):.4f}"))
            lines.append(fmt_line("best_ratio<0.8", f"{ls.get('frac_best_ratio_lt_08'):.2f}%"))
        fs = x.get("fragmentation_full", {})
        if fs:
            lines.append(fmt_line("avg_chunks/parent", f"{fs.get('avg_chunks_per_parent'):.3f}"))
            lines.append(fmt_line("chunks/parent p95", f"{fs.get('p95'):.0f}"))
            lines.append(fmt_line("chunks/parent p99", f"{fs.get('p99'):.0f}"))
            lines.append(fmt_line("max_parent_chunks", fs.get("max_parent_chunks")))
        vs = x.get("validation_sample", {})
        if vs and vs.get("enabled"):
            lines.append(fmt_line("slice_mismatch", vs.get("slice_mismatch")))
            lines.append(fmt_line("bad_offsets", vs.get("bad_offsets")))
            lines.append(fmt_line("token_too_long", vs.get("token_too_long")))
        lines.append("")

    if compare_gz:
        lines.append("[COMPARE] (A vs B)")
        lines.append(json.dumps(payload["compare"], ensure_ascii=False, indent=2))

    out_txt = report_dir / f"{out_base}.txt"
    out_txt.write_text("\n".join(lines), encoding="utf-8")


    print(f"\n[OK] wrote report:")
    print(f"  JSON: {out_json}")
    print(f"  TXT : {out_txt}\n")


if __name__ == "__main__":
    main()
