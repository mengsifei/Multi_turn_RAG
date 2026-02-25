import argparse, json, gzip
from pathlib import Path

def open_maybe_gz(path: Path, mode="rt"):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8", errors="ignore")
    return path.open(mode, encoding="utf-8", errors="ignore")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_tagged", required=True, help="*.tagged.jsonl(.gz)")
    ap.add_argument("--out_en", required=True, help="filtered en_like jsonl.gz")
    ap.add_argument("--lang_field", default="lang")
    ap.add_argument("--keep_lang", default="en_like")
    args = ap.parse_args()

    inp = Path(args.in_tagged)
    outp = Path(args.out_en)
    outp.parent.mkdir(parents=True, exist_ok=True)

    total = kept = 0
    with open_maybe_gz(inp, "rt") as fin, open_maybe_gz(outp, "wt") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            total += 1
            if o.get(args.lang_field) == args.keep_lang:
                fout.write(json.dumps(o, ensure_ascii=False) + "\n")
                kept += 1

    print(f"[ok] total={total} kept={kept} ({kept/total*100:.2f}%) -> {outp}")

if __name__ == "__main__":
    main()
