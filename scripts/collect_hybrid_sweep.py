# scripts/collect_hybrid_sweep.py

import argparse
import ast
import re
from pathlib import Path

import pandas as pd


FILENAME_RE = re.compile(
    r"hybrid_rrf_k(?P<rrf_k>\d+)_wd(?P<wd>[0-9.]+)_ws(?P<ws>[0-9.]+)__aggregate\.csv"
)


def parse_list(x):
    """
    Parse strings like:
    "[0.1, 0.2, 0.3, 0.4]"
    """
    if isinstance(x, list):
        return x
    return ast.literal_eval(str(x))


def read_one_csv(path: Path):
    m = FILENAME_RE.match(path.name)
    if not m:
        return None

    rrf_k = int(m.group("rrf_k"))
    wd = float(m.group("wd"))
    ws = float(m.group("ws"))

    df = pd.read_csv(path)

    # In case the csv has no header
    if "nDCG" not in df.columns or "Recall" not in df.columns:
        df = pd.read_csv(
            path,
            header=None,
            names=["nDCG", "Recall", "collection", "count"],
        )

    row = df.iloc[-1]

    ndcg = parse_list(row["nDCG"])
    recall = parse_list(row["Recall"])

    return {
        "file": path.name,
        "rrf_k": rrf_k,
        "wd": wd,
        "ws": ws,
        "collection": row.get("collection", "all"),
        "count": row.get("count", None),

        "nDCG@1": ndcg[0],
        "nDCG@3": ndcg[1],
        "nDCG@5": ndcg[2],
        "nDCG@10": ndcg[3],

        "Recall@1": recall[0],
        "Recall@3": recall[1],
        "Recall@5": recall[2],
        "Recall@10": recall[3],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing hybrid sweep aggregate csv files.",
    )
    parser.add_argument(
        "--output_csv",
        required=True,
        help="Output summary csv path.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    rows = []

    for path in sorted(input_dir.glob("hybrid_rrf_k*_wd*_ws*__aggregate.csv")):
        parsed = read_one_csv(path)
        if parsed is not None:
            rows.append(parsed)

    if not rows:
        raise RuntimeError(f"No matching aggregate files found in {input_dir}")

    out = pd.DataFrame(rows)

    # Sort by main metric first, then parameters
    out = out.sort_values(
        by=["nDCG@10", "Recall@10", "rrf_k", "wd", "ws"],
        ascending=[False, False, True, True, True],
    )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    print(f"Saved {len(out)} rows to {output_csv}")
    print("\nTop 10 by nDCG@10:")
    print(
        out[
            [
                "rrf_k", "wd", "ws",
                "nDCG@1", "nDCG@3", "nDCG@5", "nDCG@10",
                "Recall@1", "Recall@3", "Recall@5", "Recall@10",
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()