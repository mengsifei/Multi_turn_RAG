import json
from pathlib import Path

domains = ["clapnq", "cloud", "fiqa", "govt"]

def load_counts(p: Path):
    if not p.exists():
        return {}
    o = json.loads(p.read_text(encoding="utf-8"))
    return o.get("counts", {})

def count_lines(p: Path):
    if not p.exists():
        return 0
    t = p.read_text(encoding="utf-8", errors="ignore").strip()
    return 0 if not t else len(t.splitlines())

print("domain\ttotal\tnon_en\tshare_non_en\tqrels_non_en_hits\tsuggest")
for d in domains:
    counts = load_counts(Path(f"reports/lang_official/{d}.json"))
    total = sum(counts.values()) if counts else 0
    non_en = total - counts.get("en_like", 0)
    share = (non_en / total * 100) if total else 0.0
    hits = count_lines(Path(f"reports/overlap_official/{d}_relevant_non_en_ids.txt"))
    suggest = "FILTER(non-en)->en_only" if hits == 0 else "KEEP(non-en): translate/multilingual/dual-index"
    print(f"{d}\t{total}\t{non_en}\t{share:.2f}%\t{hits}\t{suggest}")
