# scripts/lang_scan_jsonl.py
import argparse, json, gzip
from collections import Counter, defaultdict
from pathlib import Path

def open_maybe_gz(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return path.open("r", encoding="utf-8", errors="ignore")

def detect_lang_fast(text: str):
    """
    轻量 fallback：不用外部模型，只做一个很粗的启发式。
    强烈建议你在生产里换成 fastText lid.176 或 cld3。
    """
    t = text.strip()
    if not t:
        return "unk", 0.0
    # 统计 ASCII 比例
    ascii_cnt = sum(1 for c in t if ord(c) < 128)
    ratio = ascii_cnt / max(1, len(t))
    # 极粗分类：大多数英文 chunk ASCII 比例会很高
    if ratio > 0.9:
        return "en_like", ratio
    # Cyrillic
    if any("\u0400" <= c <= "\u04FF" for c in t):
        return "ru_like", 1.0 - ratio
    # CJK
    if any("\u4E00" <= c <= "\u9FFF" for c in t):
        return "zh_like", 1.0 - ratio
    return "non_en_like", 1.0 - ratio

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True, help="jsonl or jsonl.gz")
    ap.add_argument("--text_field", default="text")
    ap.add_argument("--max_docs", type=int, default=0, help="0 = all")
    ap.add_argument("--out_report", default="lang_report.json")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    cnt = Counter()
    bytes_cnt = Counter()
    examples = defaultdict(list)

    n = 0
    with open_maybe_gz(in_path) as f:
        for line in f:
            if args.max_docs and n >= args.max_docs:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get(args.text_field) or ""
            lang, score = detect_lang_fast(text)
            cnt[lang] += 1
            bytes_cnt[lang] += len(text.encode("utf-8", errors="ignore"))
            if len(examples[lang]) < 3:
                examples[lang].append({
                    "id": obj.get("_id") or obj.get("id"),
                    "score": float(score),
                    "head": text[:180].replace("\n", " ")
                })
            n += 1

    report = {
        "in_path": str(in_path),
        "total": int(sum(cnt.values())),
        "counts": dict(cnt),
        "bytes": dict(bytes_cnt),
        "examples": dict(examples),
    }
    Path(args.out_report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote {args.out_report}")
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
