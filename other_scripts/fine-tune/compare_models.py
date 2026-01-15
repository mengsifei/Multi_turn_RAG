# compare_models.py
import argparse, hashlib
from pathlib import Path

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def list_files(root: Path):
    exts = {".safetensors", ".bin", ".json", ".py", ".txt"}
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in exts:
            files.append(p)
    return sorted(files, key=lambda x: str(x.relative_to(root)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    args = ap.parse_args()

    A = Path(args.a)
    B = Path(args.b)

    fa = list_files(A)
    fb = list_files(B)

    ra = [str(p.relative_to(A)) for p in fa]
    rb = [str(p.relative_to(B)) for p in fb]

    only_a = sorted(set(ra) - set(rb))
    only_b = sorted(set(rb) - set(ra))

    if only_a:
        print("[ONLY IN A]")
        for x in only_a[:50]:
            print(" ", x)
        if len(only_a) > 50:
            print(" ...", len(only_a) - 50, "more")

    if only_b:
        print("[ONLY IN B]")
        for x in only_b[:50]:
            print(" ", x)
        if len(only_b) > 50:
            print(" ...", len(only_b) - 50, "more")

    common = sorted(set(ra) & set(rb))
    diff = 0
    for rel in common:
        ha = sha256(A / rel)
        hb = sha256(B / rel)
        if ha != hb:
            diff += 1
            print("[DIFF]", rel)
            print("  A", ha)
            print("  B", hb)

    if diff == 0 and not only_a and not only_b:
        print("\n✅ All tracked files identical (hash match). Models are exactly the same.")
    else:
        print(f"\n⚠️ Differences found. diff_files={diff}, only_a={len(only_a)}, only_b={len(only_b)}")

if __name__ == "__main__":
    main()
