import argparse, json, gzip
from pathlib import Path

def open_maybe_gz(path: Path, mode="rt"):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8", errors="ignore")
    return path.open(mode, encoding="utf-8", errors="ignore")

def detect_lang_fast(text: str):
    t = (text or "").strip()
    if not t:
        return "unk", 0.0
    ascii_cnt = sum(1 for c in t if ord(c) < 128)
    ratio = ascii_cnt / max(1, len(t))
    if ratio > 0.9:
        return "en_like", ratio
    if any("\u0400" <= c <= "\u04FF" for c in t):
        return "ru_like", 1.0 - ratio
    if any("\u4E00" <= c <= "\u9FFF" for c in t):
        return "zh_like", 1.0 - ratio
    return "non_en_like", 1.0 - ratio

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--text_field", default="text")
    ap.add_argument("--out_tagged", required=True, help="output jsonl(.gz) with lang field")
    ap.add_argument("--out_non_en_ids", required=True, help="txt file of non-en ids")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_tagged = Path(args.out_tagged)
    out_non_en_ids = Path(args.out_non_en_ids)
    out_tagged.parent.mkdir(parents=True, exist_ok=True)
    out_non_en_ids.parent.mkdir(parents=True, exist_ok=True)

    non_en = 0
    total = 0

    with open_maybe_gz(in_path, "rt") as fin, open_maybe_gz(out_tagged, "wt") as fout, out_non_en_ids.open("w", encoding="utf-8") as fid:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get(args.text_field) or ""
            lang, score = detect_lang_fast(text)
            obj["lang"] = lang
            obj["lang_score"] = float(score)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

            _id = obj.get("_id") or obj.get("id")
            if lang != "en_like":
                non_en += 1
                if _id is not None:
                    fid.write(str(_id) + "\n")
            total += 1

    print(f"[ok] total={total} non_en={non_en} ({non_en/total*100:.2f}%)")
    print(f"[ok] wrote tagged: {out_tagged}")
    print(f"[ok] wrote non-en ids: {out_non_en_ids}")

if __name__ == "__main__":
    main()
