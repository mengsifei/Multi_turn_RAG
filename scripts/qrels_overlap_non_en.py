import argparse, csv, json, gzip
from pathlib import Path
from collections import Counter

def open_maybe_gz(path: Path, mode="rt"):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8", errors="ignore")
    return path.open(mode, encoding="utf-8", errors="ignore")

def load_id_set(txt_path: Path):
    s = set()
    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            t = line.strip()
            if t:
                s.add(t)
    return s

def iter_qrels_docids(qrels_path: Path, docid_col="doc_id", rel_col="relevance", rel_min=1):
    """
    支持：
    1) tsv/csv: 常见 BEIR 格式：query-id, corpus-id, score（可能有 header）
    2) jsonl: 每行一个 dict，包含 doc_id / relevance
    3) json: list[dict] 或 dict[...]（尽量兜底）
    """
    suf = qrels_path.suffix.lower()
    name = qrels_path.name.lower()

    # tsv/csv（含 .tsv.gz / .csv.gz）
    if name.endswith(".tsv") or name.endswith(".tsv.gz") or name.endswith(".csv") or name.endswith(".csv.gz"):
        delimiter = "\t" if "tsv" in name else ","
        with open_maybe_gz(qrels_path, "rt") as f:
            sample = f.readline()
            if not sample:
                return
            f.seek(0)
            # 尝试判断是否有 header
            has_header = any(h in sample.lower() for h in ["query", "qid", "corpus", "doc", "relevance", "score"])
            reader = csv.reader(f, delimiter=delimiter)
            if has_header:
                header = next(reader)
                # 容错：常见列名
                cols = {c.strip().lower(): i for i, c in enumerate(header)}
                # doc id
                for key in ["corpus-id", "corpus_id", "doc_id", "docid", "pid", "passage_id"]:
                    if key in cols:
                        doc_i = cols[key]; break
                else:
                    # BEIR 无 header 时一般是第2列
                    doc_i = 1
                # relevance / score
                rel_i = cols.get("relevance", cols.get("score", 2))
            else:
                doc_i, rel_i = 1, 2

            for row in reader:
                if not row:
                    continue
                if len(row) <= max(doc_i, rel_i):
                    continue
                docid = str(row[doc_i]).strip()
                try:
                    rel = float(row[rel_i])
                except:
                    rel = 0.0
                if rel >= rel_min and docid:
                    yield docid
        return

    # jsonl / jsonl.gz
    if name.endswith(".jsonl") or name.endswith(".jsonl.gz"):
        with open_maybe_gz(qrels_path, "rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                docid = o.get(docid_col) or o.get("corpus-id") or o.get("corpus_id") or o.get("docid") or o.get("pid")
                rel = o.get(rel_col, o.get("score", o.get("relevance", 0)))
                try:
                    rel = float(rel)
                except:
                    rel = 0.0
                if docid is not None and rel >= rel_min:
                    yield str(docid).strip()
        return

    # json / json.gz
    if name.endswith(".json") or name.endswith(".json.gz"):
        with open_maybe_gz(qrels_path, "rt") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            it = obj
        elif isinstance(obj, dict):
            # 可能是 {"qid": {"docid": rel}} 这种
            # 或 {"data":[...]} 这种
            if "data" in obj and isinstance(obj["data"], list):
                it = obj["data"]
            else:
                # dict-of-dict
                for qid, rels in obj.items():
                    if isinstance(rels, dict):
                        for docid, rel in rels.items():
                            try:
                                rel = float(rel)
                            except:
                                rel = 0.0
                            if rel >= rel_min:
                                yield str(docid).strip()
                return
        else:
            return

        for o in it:
            if not isinstance(o, dict):
                continue
            docid = o.get(docid_col) or o.get("corpus-id") or o.get("corpus_id") or o.get("docid") or o.get("pid")
            rel = o.get(rel_col, o.get("score", o.get("relevance", 0)))
            try:
                rel = float(rel)
            except:
                rel = 0.0
            if docid is not None and rel >= rel_min:
                yield str(docid).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qrels_path", required=True)
    ap.add_argument("--non_en_ids", required=True, help="txt from lang_tag_and_split.py")
    ap.add_argument("--rel_min", type=float, default=1.0)
    ap.add_argument("--docid_col", default="doc_id")
    ap.add_argument("--rel_col", default="relevance")
    ap.add_argument("--out_hit_ids", default="")
    args = ap.parse_args()

    qrels_path = Path(args.qrels_path)
    non_en_ids = Path(args.non_en_ids)

    non_en = load_id_set(non_en_ids)

    rel_docids = set()
    for docid in iter_qrels_docids(qrels_path, docid_col=args.docid_col, rel_col=args.rel_col, rel_min=args.rel_min):
        rel_docids.add(docid)

    hit = rel_docids & non_en

    print(f"[qrels] unique relevant docids: {len(rel_docids)}")
    print(f"[non-en] ids: {len(non_en)}")
    print(f"[overlap] relevant ∩ non-en: {len(hit)}")
    if len(rel_docids) > 0:
        print(f"[overlap] share in relevant: {len(hit)/len(rel_docids)*100:.3f}%")
    if len(non_en) > 0:
        print(f"[overlap] share of non-en that are relevant: {len(hit)/len(non_en)*100:.3f}%")

    if args.out_hit_ids:
        outp = Path(args.out_hit_ids)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text("\n".join(sorted(hit)), encoding="utf-8")
        print(f"[ok] wrote hit ids: {outp}")

if __name__ == "__main__":
    main()
