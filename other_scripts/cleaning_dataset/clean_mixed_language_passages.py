#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, gzip, re, unicodedata
from pathlib import Path

CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
CYR_RE = re.compile(r"[\u0400-\u04FF]")
LAT_RE = re.compile(r"[A-Za-z]")
BASE64_RUN_RE = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/=]{120,}(?![A-Za-z0-9+/=])")
HEX_RUN_RE    = re.compile(r"(?i)\b[0-9a-f]{64,}\b")  # 64位以上的hex（hash/指纹）


def open_maybe_gz(path: Path, mode="rt"):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8", errors="ignore")
    return path.open(mode, encoding="utf-8", errors="ignore")

def is_non_english_line(line: str,
                        min_cjk_or_cyr: int = 3,
                        non_ascii_ratio_th: float = 0.40,
                        latin_min: int = 3) -> bool:
    """
    判定一行是不是“明显非英文噪声行”，用于删除。
    """
    s = line.strip()
    if not s:
        return False

    cjk = len(CJK_RE.findall(s))
    cyr = len(CYR_RE.findall(s))
    lat = len(LAT_RE.findall(s))
    L = len(s)
    non_ascii = sum(1 for ch in s if ord(ch) > 127)

    # 规则1：中文/西里尔字符不少，且几乎没有拉丁字母
    if (cjk + cyr) >= min_cjk_or_cyr and lat < latin_min:
        return True

    # 规则2：非 ASCII 比例很高，同时拉丁字母占比很低
    if L > 0 and (non_ascii / L) >= non_ascii_ratio_th and lat < latin_min:
        return True

    return False

def clean_text(text: str,
               drop_non_english_lines: bool = True,
               keep_max_blank: int = 1) -> tuple[str, dict]:
    """
    返回 cleaned_text + 统计信息
    """
    stats = {"orig_len": len(text), "removed_lines": 0, "kept_lines": 0}

    # 1) unicode 归一化：把全角符号等转成更标准的形式
    t = unicodedata.normalize("NFKC", text)
    # 2) NBSP -> space
    t = t.replace("\u00a0", " ")
    # 3) 去掉一些不可见控制字符（保留 \n \t）
    t = "".join(ch for ch in t if (ch == "\n" or ch == "\t" or ord(ch) >= 32))
    # 2.5) remove long base64 / long hex runs *inside* lines (keep the rest)
    t, n_b64 = BASE64_RUN_RE.subn(" ", t)
    t, n_hex = HEX_RUN_RE.subn(" ", t)

    stats["base64_spans"] = n_b64
    stats["hex_spans"] = n_hex

    lines = t.split("\n")
    out_lines = []
    for ln in lines:
        if drop_non_english_lines and is_non_english_line(ln):
            stats["removed_lines"] += 1
            continue
        out_lines.append(ln.rstrip())
        stats["kept_lines"] += 1

    # 4) 压缩空行
    cleaned = []
    blank_run = 0
    for ln in out_lines:
        if ln.strip() == "":
            blank_run += 1
            if blank_run <= keep_max_blank:
                cleaned.append("")
        else:
            blank_run = 0
            cleaned.append(ln)

    out = "\n".join(cleaned).strip()
    stats["clean_len"] = len(out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out, stats

def get_doc_id(o: dict) -> str:
    return str(o.get("document_id") or o.get("_id") or o.get("id"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--text_field", default="text")
    ap.add_argument("--keep_stats", action="store_true", help="attach cleaning stats fields for debugging")
    ap.add_argument("--id_print", default="", help="optional: print before/after for a specific doc id")
    args = ap.parse_args()

    inp = Path(args.in_jsonl)
    outp = Path(args.out_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    changed = 0
    printed = False

    with open_maybe_gz(inp, "rt") as fin, open_maybe_gz(outp, "wt") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            n += 1

            did = get_doc_id(o)
            text = o.get(args.text_field) or ""

            cleaned, stats = clean_text(text)

            if cleaned != text:
                changed += 1

            if args.keep_stats:
                o["_clean_stats"] = stats
            o[args.text_field] = cleaned

            if args.id_print and (did == args.id_print) and (not printed):
                printed = True
                print("===== BEFORE =====")
                print(text)
                print("\n===== AFTER =====")
                print(cleaned)

            fout.write(json.dumps(o, ensure_ascii=False) + "\n")

    print(f"[ok] wrote {outp}")
    print(f"[stat] docs={n} changed={changed} ({changed/max(1,n)*100:.2f}%)")

if __name__ == "__main__":
    main()
