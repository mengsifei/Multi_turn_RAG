# import argparse, json
# from pathlib import Path

# def main():
#     ap=argparse.ArgumentParser()
#     ap.add_argument("--in_jsonl", required=True)
#     ap.add_argument("--out_jsonl", required=True)
#     ap.add_argument("--alpha", type=float, required=True)
#     args=ap.parse_args()

#     a=float(args.alpha)
#     Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)

#     with open(args.in_jsonl,"r",encoding="utf-8") as fin, open(args.out_jsonl,"w",encoding="utf-8") as fout:
#         for line in fin:
#             o=json.loads(line)
#             ctxs=o.get("contexts",[])
#             for c in ctxs:
#                 if "orig_score" not in c or "rerank_score" not in c:
#                     raise SystemExit("missing orig_score/rerank_score; rerank file must store both.")
#                 c["score"]=a*float(c["rerank_score"]) + (1-a)*float(c["orig_score"])
#             ctxs.sort(key=lambda x: float(x["score"]), reverse=True)
#             o["contexts"]=ctxs
#             fout.write(json.dumps(o, ensure_ascii=False) + "\n")

# if __name__=="__main__":
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--keep_topk", type=int, default=None)
    args = ap.parse_args()

    a = float(args.alpha)

    with open(args.in_jsonl, "r", encoding="utf-8") as fin, open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            o = json.loads(line)
            ctxs = o.get("contexts", [])
            if not ctxs:
                fout.write(json.dumps(o, ensure_ascii=False) + "\n")
                continue

            # require orig_score + rerank_score
            if ("orig_score" not in ctxs[0]) or ("rerank_score" not in ctxs[0]):
                raise ValueError("missing orig_score/rerank_score; rerank file must store both.")

            for c in ctxs:
                orig = float(c["orig_score"])
                rr   = float(c["rerank_score"])
                c["score"] = a * rr + (1.0 - a) * orig

            ctxs.sort(key=lambda x: float(x["score"]), reverse=True)
            if args.keep_topk is not None:
                ctxs = ctxs[: args.keep_topk]
            o["contexts"] = ctxs
            fout.write(json.dumps(o, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
