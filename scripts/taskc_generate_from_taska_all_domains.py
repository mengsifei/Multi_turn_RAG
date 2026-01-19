#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path
from xxlimited import Str
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


DOMAINS = ["clapnq", "cloud", "fiqa", "govt"]


# -----------------------------
# helpers
# -----------------------------

def infer_domain(collection_name: str) -> str:
    s = collection_name.lower()
    for d in DOMAINS:
        if d in s:
            return d
    raise ValueError(f"Cannot infer domain from Collection={collection_name}")


def load_doc_map(corpus_jsonl):
    """
    Cleaned corpus loader for MTRAG.
    We always use `id` as the document identifier.
    """
    mp = {}
    with open(corpus_jsonl) as f:
        for line in f:
            j = json.loads(line)

            # cleaned_dataset 标准字段
            if "id" not in j:
                raise KeyError(
                    f"Expected field `id` in corpus, got keys={j.keys()}"
                )

            mp[j["id"]] = j["text"]

    return mp


def load_query_map(rewrite_jsonl):
    """
    Load queries from cleaned_dataset/*/*_rewrite_gpt.jsonl

    Format:
      {
        "_id": "<task_id>",
        "text": "|user|: <query>"
      }
    """
    mp = {}
    with open(rewrite_jsonl) as f:
        for line in f:
            j = json.loads(line)

            if "_id" not in j or "text" not in j:
                raise KeyError(
                    f"Expected _id and text in rewrite file, got keys={j.keys()}"
                )

            tid = j["_id"]
            q = j["text"]

            # 去掉 |user|: 前缀（强烈推荐）
            if q.startswith("|user|:"):
                q = q[len("|user|:"):].strip()

            mp[tid] = q

    return mp




def build_prompt(question, contexts, max_doc_chars=1200):
    docs = []
    for i, c in enumerate(contexts, 1):
        docs.append(f"[Document {i}]\n{c['text'][:max_doc_chars]}")
    docs_str = "\n\n".join(docs)

    return f"""Answer the question using ONLY the provided documents.
If the answer cannot be determined from the documents, say so.

Question:
{question}

Documents:
{docs_str}

Answer:"""


@torch.inference_mode()
def generate_answer(model, tokenizer, prompt, max_new_tokens=128):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text.split("Answer:")[-1].strip()


# -----------------------------
# main
# -----------------------------

def main(args):
    # load all corpora + tasks
    doc_maps = {}
    query_maps = {}

    for d in DOMAINS:
        doc_maps[d] = load_doc_map(
            Path(args.cleaned_root) / d / f"{d}.jsonl"
        )
        query_maps[d] = load_query_map(
            Path(args.cleaned_root) / d / f"{d}_{args.task_name}.jsonl"
        )

    print("[INFO] Loading generation model...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).eval()

    outputs = []

    with open(args.taska_file) as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Generating Task C (all domains)"):
        j = json.loads(line)

        task_id = j["task_id"]
        conv_id = task_id.split("::")[0]

        domain = infer_domain(j["Collection"])
        doc_map = doc_maps[domain]
        query_map = query_maps[domain]

        if task_id not in query_map:
            continue

        question = query_map[task_id]

        contexts = []
        for c in j["contexts"][:args.topk]:
            did = c["document_id"]
            if did not in doc_map:
                continue
            contexts.append({
                "document_id": did,
                "text": doc_map[did],
                "score": c["score"]
            })

        if not contexts:
            continue

        prompt = build_prompt(question, contexts)
        answer = generate_answer(model, tokenizer, prompt)

        outputs.append({
            "conversation_id": conv_id,
            "task_id": task_id,
            "Collection": j["Collection"],
            "input": [
                {"speaker": "user", "text": question}
            ],
            "contexts": contexts,
            "predictions": [
                {"text": answer}
            ]
        })

    print(f"[INFO] Writing {len(outputs)} items to {args.output_file}")
    with open(args.output_file, "w") as f:
        for o in outputs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--taska_file", required=True)
    ap.add_argument("--cleaned_root", default="cleaned_dataset")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--output_file", required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--task_name", type=str, default="rewrite_gpt")
    args = ap.parse_args()
    main(args)
