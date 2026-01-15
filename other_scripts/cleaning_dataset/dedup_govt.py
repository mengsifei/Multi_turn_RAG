#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python dedup_govt.py   --in_jsonl human/retrieval_tasks/cloud/cloud.jsonl  --out_dir reports/cloud_dedup   --qrels human/retrieval_tasks/cloud/qrels/dev.tsv

import argparse, gzip, json, re, hashlib
from pathlib import Path
from collections import defaultdict

def open_text(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, "r", encoding="utf-8")

# 你可以把 govt 常见模板继续往里加
_BOILERPLATE_PATTERNS = [
    r"Saving your location allows us to provide you with more relevant information",
    r"\bSet Location\b",
    r"Office of the Patient Advocate - State of California",
    r"^\s*[×x]\s*$",
]
_BOILERPLATE_RE = re.compile("|".join(f"(?:{p})" for p in _BOILERPLATE_PATTERNS), re.I)

_WS_RE = re.compile(r"\s+")

def normalize_text(text: str) -> str:
    """
    用于 dedup 的归一化：
    - 去掉明显 boilerplate 行
    - 压缩空白
    - 去掉首尾空白
    """
    if text is None:
        return ""
    # 按行过滤模板
    lines = []
    for ln in str(text).splitlines():
        ln2 = ln.strip()
        if not ln2:
            continue
        if _BOILERPLATE_RE.search(ln2):
            continue
        lines.append(ln2)
    s = "\n".join(lines)
    s = _WS_RE.sub(" ", s).strip()
    return s

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def load_qrels_ids(qrels_path: str) -> set:
    """
    兼容常见 qrels：
    - TSV: query_id  doc_id  relevance  (或带 iteration 字段)
    - JSONL: {"qid":..., "doc_id":...}
    只要能抽到 doc_id 就行。
    """
    if not qrels_path:
        return set()
    p = Path(qrels_path)
    if not p.exists():
        raise FileNotFoundError(f"qrels not found: {qrels_path}")
    keep = set()
    if p.suffix == ".jsonl":
        with open_text(str(p)) as f:
            for line in f:
                o = json.loads(line)
                for k in ("doc_id", "did", "passage_id", "id"):
                    if k in o:
                        keep.add(str(o[k]))
                        break
    else:
        # tsv/space separated
        with open_text(str(p)) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = re.split(r"\s+", line)
                # 常见格式：qid docid rel 或 qid 0 docid rel
                if len(parts) >= 3:
                    if len(parts) >= 4 and parts[1].isdigit() is False and parts[1] == "0":
                        docid = parts[2]
                    elif len(parts) >= 4 and parts[1] == "0":
                        docid = parts[2]
                    else:
                        docid = parts[1]
                    keep.add(str(docid))
    return keep

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--text_key", default="text")
    ap.add_argument("--id_key", default="id")
    ap.add_argument("--qrels", default="", help="optional: qrels path for protection")
    ap.add_argument("--max_members_dump", type=int, default=200, help="dump up to N member ids per group in dups.jsonl")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_dups = out_dir / "dups.jsonl.gz"
    out_black = out_dir / f"../{out_dir.stem}_blacklist.txt"
    out_stats = out_dir / "stats.json"

    protect_ids = load_qrels_ids(args.qrels) if args.qrels else set()

    # pass1: hash -> count, rep_id
    counts = defaultdict(int)
    rep_id = {}
    rep_norm = {}
    id2h_path = out_dir / "id2hash.tsv.gz"

    total = 0
    empty_norm = 0

    with open_text(args.in_jsonl) as f, gzip.open(id2h_path, "wt", encoding="utf-8") as w:
        for line in f:
            o = json.loads(line)
            pid = str(o.get(args.id_key) or o.get("_id") or "")
            text = o.get(args.text_key, "")
            norm = normalize_text(text)
            if not norm:
                empty_norm += 1
            h = sha1(norm)
            counts[h] += 1
            if h not in rep_id:
                rep_id[h] = pid
                rep_norm[h] = norm[:300]
            w.write(f"{pid}\t{h}\n")
            total += 1

    # pass2: collect duplicates + build blacklist (skip non-protected duplicates)
    dup_groups = 0
    blacklist = []

    # 先把所有 duplicate hash 标出来
    dup_hashes = {h for h, c in counts.items() if c >= 2}

    # hash -> members（只保留少量用于输出；真正 blacklist 还是逐条生成）
    members_dump = defaultdict(list)

    with gzip.open(id2h_path, "rt", encoding="utf-8") as f:
        for line in f:
            pid, h = line.rstrip("\n").split("\t")
            if h not in dup_hashes:
                continue
            if len(members_dump[h]) < args.max_members_dump:
                members_dump[h].append(pid)

    # 生成 blacklist：每个 duplicate group 默认保留 rep_id，
    # 但如果有 qrels 保护 id，则这些 protected 必须保留。
    with gzip.open(out_dups, "wt", encoding="utf-8") as wdup, open(out_black, "w", encoding="utf-8") as wbl:
        with gzip.open(id2h_path, "rt", encoding="utf-8") as f:
            # 需要知道一个 group 里哪些 id 被保护，所以先做 hash->has_protected & protected_ids
            protected_in_group = defaultdict(set)
            for line in f:
                pid, h = line.rstrip("\n").split("\t")
                if h in dup_hashes and pid in protect_ids:
                    protected_in_group[h].add(pid)

        # 输出 duplicate groups summary
        for h in sorted(dup_hashes, key=lambda x: -counts[x]):
            dup_groups += 1
            wdup.write(json.dumps({
                "hash": h,
                "count": counts[h],
                "rep_id": rep_id[h],
                "protected_ids": sorted(protected_in_group.get(h, set())),
                "members_sample": members_dump.get(h, []),
                "rep_norm_snip": rep_norm.get(h, ""),
            }, ensure_ascii=False) + "\n")

        # 再扫一遍 id2hash，写 blacklist
        with gzip.open(id2h_path, "rt", encoding="utf-8") as f2:
            for line in f2:
                pid, h = line.rstrip("\n").split("\t")
                if h not in dup_hashes:
                    continue

                # 保护：在 qrels 里出现过的不要 blacklist
                if pid in protect_ids:
                    continue

                # 每个 group 默认保留 rep_id；但如果 rep_id 不是 protected，也保留它作为代表
                if pid == rep_id[h]:
                    continue

                # 其余 duplicate 全部建议跳过
                blacklist.append(pid)
                wbl.write(pid + "\n")

    stats = {
        "in_jsonl": args.in_jsonl,
        "total": total,
        "empty_norm_after_clean": empty_norm,
        "unique_hashes": len(counts),
        "dup_hashes": len(dup_hashes),
        "dup_groups": dup_groups,
        "blacklist_n": len(blacklist),
        "qrels_protect_n": len(protect_ids),
        "notes": "blacklist are duplicate passages after normalize_text; qrels-protected ids are kept",
    }
    out_stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK]", json.dumps(stats, ensure_ascii=False))

if __name__ == "__main__":
    main()
