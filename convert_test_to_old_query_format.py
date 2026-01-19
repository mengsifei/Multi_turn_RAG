#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import argparse

def linearize_history(messages, keep=("user",), include_agent=False):
    lines = []
    for m in messages:
        sp = m.get("speaker", "")
        if sp not in keep:
            continue
        role = sp
        if include_agent and sp == "agent":
            role = "assistant"
        lines.append(f"|{role}|: {m.get('text','') or ''}")
    return "\n".join(lines).strip()

def norm_collection(x: str) -> str:
    s = (x or "").strip().lower()
    if s == "ibmcloud":
        return "cloud"
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True, help="test input jsonl (conversation format)")
    ap.add_argument("--out_jsonl", required=True, help="converted jsonl, old query format")
    ap.add_argument("--mode", choices=["user_only", "user_agent"], default="user_only")
    ap.add_argument("--id_field", choices=["task_id", "conversation_id"], default="task_id")
    ap.add_argument("--normalize_domain", action="store_true",
                    help="if set, normalize Collection (e.g., ibmcloud->cloud)")
    args = ap.parse_args()

    if args.mode == "user_only":
        keep = ("user",)
        include_agent = False
    else:
        keep = ("user", "agent")
        include_agent = True

    n = 0
    with open(args.in_jsonl, "r", encoding="utf-8") as fin, \
         open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            _id = item.get(args.id_field)
            if not isinstance(_id, str) or not _id:
                raise ValueError(f"Line {line_no}: missing or invalid {args.id_field}")

            col = item.get("Collection")
            if not isinstance(col, str) or not col:
                raise ValueError(f"Line {line_no}: missing or invalid Collection")

            if args.normalize_domain:
                col = norm_collection(col)

            text = linearize_history(item.get("input", []), keep=keep, include_agent=include_agent)

            out = {"_id": _id, "Collection": col, "text": text}
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1

    print(f"[OK] wrote {n} lines to {args.out_jsonl}")

if __name__ == "__main__":
    main()
