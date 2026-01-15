#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple


def load_qrels_tsv(path: str) -> Dict[str, Set[str]]:
    rels: Dict[str, Set[str]] = defaultdict(set)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            qid, did = str(parts[0]), str(parts[1])
            rel = 1
            if len(parts) >= 3:
                try:
                    rel = int(float(parts[2]))
                except Exception:
                    rel = 1
            if rel > 0:
                rels[qid].add(did)
    return dict(rels)


def save_qrels_tsv(path: Path, qrels: Dict[str, Set[str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for qid in sorted(qrels.keys()):
            for did in sorted(qrels[qid]):
                f.write(f"{qid}\t{did}\t1\n")


def split_by_query(qrels_dev: Dict[str, Set[str]], valid_ratio: float, seed: int) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    qids = list(qrels_dev.keys())
    rnd = random.Random(seed)
    rnd.shuffle(qids)
    n_valid = max(1, int(len(qids) * valid_ratio))
    valid_qids = set(qids[:n_valid])
    train = {qid: qrels_dev[qid] for qid in qids if qid not in valid_qids}
    valid = {qid: qrels_dev[qid] for qid in qids if qid in valid_qids}
    return train, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, choices=["lastturn", "questions", "rewrite"], required=True)

    ap.add_argument("--out_dir", type=str, required=True, help="Where to save splits, e.g. splits/human_s42_r01")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--valid_ratio_if_no_train", type=float, default=0.1)

    ap.add_argument("--prefer_train_qrels", action="store_true",
                    help="If qrels/train.tsv exists, use it as train and dev.tsv as valid.")
    args = ap.parse_args()

    collections = {
        "clapnq": "human/retrieval_tasks/clapnq",
        "fiqa":   "human/retrieval_tasks/fiqa",
        "govt":   "human/retrieval_tasks/govt",
        "cloud":  "human/retrieval_tasks/cloud",
    }

    out_root = Path(args.out_dir) / args.task
    out_root.mkdir(parents=True, exist_ok=True)

    meta = {
        "task": args.task,
        "seed": args.seed,
        "valid_ratio_if_no_train": args.valid_ratio_if_no_train,
        "prefer_train_qrels": bool(args.prefer_train_qrels),
        "domains": {},
    }

    for dom, root in collections.items():
        qrels_dir = Path(root) / "qrels"
        train_path = qrels_dir / "train.tsv"
        dev_path = qrels_dir / "dev.tsv"

        if not dev_path.exists():
            raise FileNotFoundError(f"Missing {dev_path}")

        qrels_dev = load_qrels_tsv(str(dev_path))

        if args.prefer_train_qrels and train_path.exists():
            qrels_train = load_qrels_tsv(str(train_path))
            qrels_valid = qrels_dev  # valid = full dev (official)
            policy = "official_train+dev"
        else:
            qrels_train, qrels_valid = split_by_query(qrels_dev, args.valid_ratio_if_no_train, args.seed)
            policy = "split_dev_by_query"

        dom_out = out_root / dom
        save_qrels_tsv(dom_out / "train.tsv", qrels_train)
        save_qrels_tsv(dom_out / "valid.tsv", qrels_valid)

        (dom_out / "train_qids.json").write_text(json.dumps(sorted(qrels_train.keys()), ensure_ascii=False, indent=2), encoding="utf-8")
        (dom_out / "valid_qids.json").write_text(json.dumps(sorted(qrels_valid.keys()), ensure_ascii=False, indent=2), encoding="utf-8")

        meta["domains"][dom] = {
            "policy": policy,
            "train_q": len(qrels_train),
            "valid_q": len(qrels_valid),
            "train_path": str(dom_out / "train.tsv"),
            "valid_path": str(dom_out / "valid.tsv"),
        }

        print(f"[OK] {dom}: policy={policy} train_q={len(qrels_train)} valid_q={len(qrels_valid)} -> {dom_out}")

    (out_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAVE] meta -> {out_root/'meta.json'}")


if __name__ == "__main__":
    main()
