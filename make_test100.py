#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
from pathlib import Path
from collections import defaultdict, Counter

DOMAINS = ["clapnq", "fiqa", "govt", "cloud"]

def infer_domain(collection: str) -> str:
    s = (collection or "").lower()
    for d in DOMAINS:
        if d in s:
            return d
    return "unknown"

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def write_jsonl(path: str, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    # ====== 配置区 ======
    taska_path = "train_best_taska.jsonl"
    rag_path   = "human/generation_tasks/RAG.jsonl"
    out_path   = "test_100.jsonl"

    N = 100
    SEED = 42

    # True: 每个 domain 尽量抽 25 条凑 100
    # False: 直接按 merge 后的顺序取前 100
    STRATIFIED_BY_DOMAIN = True
    PER_DOMAIN = 25
    # ====================

    random.seed(SEED)

    # 1) 从官方 RAG.jsonl 建 task_id -> contexts(full) 的索引
    ctx_by_tid = {}
    meta_by_tid = {}  # 可选：也把 input/targets/Answerability/Question Type 等带过来（如果你想覆盖）
    total_rag = 0
    for rec in read_jsonl(rag_path):
        total_rag += 1
        tid = str(rec.get("task_id", ""))
        if not tid:
            continue
        if "contexts" in rec and isinstance(rec["contexts"], list):
            ctx_by_tid[tid] = rec["contexts"]
        # 你如果还想从官方带回其他字段，可从这里取：
        meta_by_tid[tid] = rec

    print(f"[RAG] loaded: {total_rag} lines, ctx_index={len(ctx_by_tid)}")

    # 2) 读你的 taska，按 task_id merge contexts；找不到就略过
    merged = []
    skipped = 0
    for rec in read_jsonl(taska_path):
        tid = str(rec.get("task_id", ""))
        if not tid or tid not in ctx_by_tid:
            skipped += 1
            continue

        # 用官方 contexts 覆盖（带 text）
        rec["contexts"] = ctx_by_tid[tid]

        # （可选）如果你希望把官方的 input/targets/Answerability 也覆盖回来，取消注释：
        # off = meta_by_tid[tid]
        # for k in ["input", "targets", "Answerability", "Question Type", "No. References", "Multi-Turn", "Collection", "conversation_id", "turn", "task_type", "dataset"]:
        #     if k in off and k not in rec:
        #         rec[k] = off[k]
        # 或者你想强制覆盖 rec[k] = off[k] 也行

        merged.append(rec)

    print(f"[MERGE] kept={len(merged)} skipped(no task_id in RAG)={skipped}")

    if not merged:
        raise SystemExit("No records merged. Check task_id consistency between your file and RAG.jsonl.")

    # 3) 统计 domain 分布
    for r in merged:
        r["_domain"] = infer_domain(r.get("Collection", ""))

    dist = Counter([r["_domain"] for r in merged])
    print("[MERGE] domain dist:", dict(dist))

    # 4) 选 100 个
    if not STRATIFIED_BY_DOMAIN:
        test = merged[:N]
    else:
        buckets = defaultdict(list)
        for r in merged:
            buckets[r["_domain"]].append(r)

        # 每个 bucket shuffle
        for d in buckets:
            random.shuffle(buckets[d])

        test = []

        # 先每域取 PER_DOMAIN
        for d in DOMAINS:
            take = min(PER_DOMAIN, len(buckets[d]))
            test.extend(buckets[d][:take])
            buckets[d] = buckets[d][take:]

        # 不够再从剩余里补齐（也 shuffle 一下）
        if len(test) < N:
            rest = []
            for d, lst in buckets.items():
                rest.extend(lst)
            random.shuffle(rest)
            test.extend(rest[: (N - len(test))])

        test = test[:N]

    # 5) 清理临时字段并写出
    for r in test:
        r.pop("_domain", None)

    print(f"[TEST] selected={len(test)} -> {out_path}")
    write_jsonl(out_path, test)

if __name__ == "__main__":
    main()
