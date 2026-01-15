#!/usr/bin/env python3
import csv
from pathlib import Path
from collections import defaultdict

def load_pairs(path: Path):
    pairs = set()
    qids = set()
    with path.open("r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        for row in r:
            if not row: 
                continue
            qid, did = row[0], row[1]
            pairs.add((qid, did))
            qids.add(qid)
    return pairs, qids

def main():
    A = Path("jasper-ft-human-lastturn-s42-r01/splits_used/lastturn")
    B = Path("splits/human_s42_r01/lastturn")

    for dom in ["clapnq","fiqa","govt","cloud"]:
        for split in ["train","valid"]:
            fa = A/dom/f"{split}.tsv"
            fb = B/dom/f"{split}.tsv"
            pa, qa = load_pairs(fa)
            pb, qb = load_pairs(fb)

            print(f"\n== {dom} {split} ==")
            print(f"A: pairs={len(pa)} qids={len(qa)}")
            print(f"B: pairs={len(pb)} qids={len(qb)}")

            only_a = pa - pb
            only_b = pb - pa
            if not only_a and not only_b:
                print("OK: identical (as sets).")
            else:
                print(f"DIFF: only_in_A={len(only_a)} only_in_B={len(only_b)}")
                if only_a:
                    print("  sample only_in_A:", next(iter(only_a)))
                if only_b:
                    print("  sample only_in_B:", next(iter(only_b)))

if __name__ == "__main__":
    main()
