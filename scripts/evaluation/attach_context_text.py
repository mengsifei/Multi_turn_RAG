import json
from pathlib import Path

# ---- inputs ----
REF = Path("human/generation_tasks/reference.jsonl")
CORPUS_FILES = [
    Path("human/retrieval_tasks/clapnq/clapnq.jsonl"),
    Path("human/retrieval_tasks/cloud/cloud.jsonl"),
    Path("human/retrieval_tasks/fiqa/fiqa.jsonl"),
    Path("human/retrieval_tasks/govt/govt.jsonl"),
]
PRED = Path("outputs/taskc/ds_official_concat_lastturn_rewrite_gpt_with_answers.jsonl")

# ---- output ----
OUT = Path("outputs/taskc/ds_official_concat_lastturn_rewrite_gpt_with_answers.with_ctx_text_and_targets.jsonl")

def load_targets(ref_path: Path):
    mp = {}
    with ref_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            tid = o.get("task_id")
            if not tid:
                continue
            # reference 里 targets 是 list[{"speaker","text",...}]
            if "targets" in o:
                mp[tid] = o["targets"]
    return mp

def load_corpus_index(paths):
    idx = {}
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                o = json.loads(line)
                did = o.get("_id") or o.get("id")
                if not did:
                    continue
                idx[did] = {
                    "text": o.get("text"),
                    "title": o.get("title"),
                    "source": o.get("source"),
                    "url": o.get("url"),
                }
    return idx

print("loading reference targets...")
tid_to_targets = load_targets(REF)
print("targets tasks:", len(tid_to_targets))

print("loading corpus index (doc_id -> text)...")
doc_idx = load_corpus_index(CORPUS_FILES)
print("corpus docs:", len(doc_idx))

total = 0
miss_targets = 0
filled_targets = 0
filled_ctx = 0
miss_ctx = 0

with PRED.open("r", encoding="utf-8") as fin, OUT.open("w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip():
            continue
        inst = json.loads(line)
        total += 1

        tid = inst.get("task_id")

        # ---- attach targets ----
        if "targets" not in inst or not inst.get("targets"):
            t = tid_to_targets.get(tid)
            if t is None:
                miss_targets += 1
            else:
                inst["targets"] = t
                filled_targets += 1

        # ---- attach context text ----
        ctxs = inst.get("contexts") or []
        new_ctxs = []
        for c in ctxs:
            if not isinstance(c, dict):
                new_ctxs.append(c)
                continue
            did = c.get("document_id") or c.get("id") or c.get("_id")
            if not did:
                new_ctxs.append(c)
                continue

            hit = doc_idx.get(did)
            if hit and hit.get("text"):
                cc = dict(c)
                cc["text"] = hit["text"]
                if hit.get("title") and "title" not in cc:
                    cc["title"] = hit["title"]
                new_ctxs.append(cc)
                filled_ctx += 1
            else:
                miss_ctx += 1
                new_ctxs.append(c)

        inst["contexts"] = new_ctxs
        fout.write(json.dumps(inst, ensure_ascii=False) + "\n")

print(f"done. total={total}")
print(f"targets: filled={filled_targets}, miss={miss_targets}")
print(f"context text: filled={filled_ctx}, miss={miss_ctx}")
print(f"wrote: {OUT}")