#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


PROMPT_TEMPLATE = """You are an IDK judge.

Classify whether the response refuses to answer the user's exact inquiry.

Return exactly one label: yes, no, or partial.

Definitions:
- yes: The response ultimately says the exact inquiry cannot be answered, or that there is not enough information to answer it.
- no: The response gives a concrete answer or inference to the exact inquiry. Do not judge whether the answer is factually correct.
- partial: The inquiry asks for multiple distinct requested items, and the response directly answers at least one requested item but explicitly says another requested item cannot be answered.

Strict rules:
- Judge only the response's answerability behavior.
- Do NOT use outside knowledge.
- Do NOT judge factual correctness.
- Do NOT compare against the gold answerability label.
- If the response gives an answer to the exact inquiry, output no, even if the answer may be unsupported or wrong.
- If the response only gives related background but says the exact inquiry cannot be answered, output yes.
- Use partial only when the response both answers one requested part and refuses another requested part.
- Do not output partial just because the response contains caveats, limitations, "however", "not", "different", or "does not explicitly say".
- If unsure between yes and partial, choose yes unless there is a clearly answered subquestion.
- If unsure between no and partial, choose no if the exact inquiry is answered.

Examples:
[Inquiry] What time does the museum close today?
[Response] The document says the museum usually closes at 6 PM, but it does not provide today's closing time.
[Output] yes

[Inquiry] Where do the Arizona Cardinals play this week?
[Response] The documents say the Cardinals played the Rams in London in 2017, but they do not specify where the Cardinals play this week.
[Output] yes

[Inquiry] Why doesn't the government prevent people from living in areas prone to flooding?
[Response] The government does not outright prevent people from living in flood-prone areas; instead, it promotes mitigation, preparedness, building standards, and flood insurance.
[Output] no

[Inquiry] What are sheltered rooms designated for use?
[Response] A safe room is designated for protection from tornadoes, while a shelter-in-place room is used during events involving airborne contaminants.
[Output] no

[Inquiry] What items should I keep in a designated sheltered room?
[Response] You should keep duct tape, plastic sheeting, and disaster supplies in the designated sheltered room.
[Output] no

[Inquiry] Are sheltered room designations and recommended items the same for earthquakes?
[Response] I do not have specific information. The documents discuss tornadoes, windstorms, and shelter-in-place supplies, but they do not address earthquake preparedness, safe room designations, or recommended items for earthquakes.
[Output] yes

[Inquiry] What is NLP and how can I build a sentiment analysis model using Python and NLTK?
[Response] NLP is a subfield of AI focused on human language. However, the documents do not describe how to build a sentiment analysis model using Python and NLTK.
[Output] partial

[Inquiry] Which state has more wildfires?
[Response] The documents only discuss wildfires in California and do not provide data comparing wildfire frequency or size with other states.
[Output] yes

[Inquiry]
{question}

[Response]
{answer}

[Output]"""


def safe_str(x):
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def extract_between(text, start_marker, end_marker):
    text = safe_str(text)
    m = re.search(
        re.escape(start_marker) + r"\s*(.*?)\s*" + re.escape(end_marker),
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return m.group(1).strip() if m else ""


def extract_question_from_prompt(prompt):
    q = extract_between(prompt, "[Question]", "[Answer]")
    if q:
        return q
    q = extract_between(prompt, "[Question]", "[Final Answer]")
    if q:
        return q
    q = extract_between(prompt, "<Question>:", "</Question>")
    if q:
        return q
    return ""


def normalize_label(text):
    s = safe_str(text).strip().lower()
    s = re.sub(r"[^a-z]", "", s)

    if s.startswith("yes"):
        return "yes"
    if s.startswith("no"):
        return "no"
    if s.startswith("partial"):
        return "partial"

    return "parse_error"


def idk_to_score(label):
    if label == "yes":
        return 1.0
    if label == "no":
        return 0.0
    if label == "partial":
        return 0.5
    return 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--model", default="models/gemma-3-4b-it")
    ap.add_argument("--answer_col", default="raw_answer")
    ap.add_argument("--prompt_col", default="prompt")
    ap.add_argument("--question_col", default=None)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--out_jsonl", default="/tmp/gemma_idk_5.jsonl")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--max_new_tokens", type=int, default=5)
    args = ap.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"[INFO] device={device}")
    print(f"[INFO] model={args.model}")

    df = pd.read_csv(args.input_csv, dtype=str, keep_default_na=False).head(args.limit)

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    dtype = torch.bfloat16 if device == "cuda" else torch.float16 if device == "mps" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device).eval()

    rows = []

    for i, row in df.iterrows():
        answer = safe_str(row.get(args.answer_col))

        if args.question_col and args.question_col in row:
            question = safe_str(row.get(args.question_col))
        else:
            question = extract_question_from_prompt(row.get(args.prompt_col))

        judge_prompt = PROMPT_TEMPLATE.format(question=question, answer=answer)

        messages = [{"role": "user", "content": judge_prompt}]
        inputs = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )

        gen = tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True).strip()
        label = normalize_label(gen)

        obj = {
            "row_index": int(i),
            "task_id": row.get("task_id", ""),
            "answerability": row.get("answerability", ""),
            "question": question,
            "raw_answer": answer,
            "idk_raw": gen,
            "idk_label": label,
        }
        rows.append(obj)

        # print("\n==============================")
        # print("task_id:", obj["task_id"])
        # print("answerability:", obj["answerability"])
        # print("question:", question)
        # print("raw_answer:", answer)
        # print("idk_raw:", gen)
        # print("idk_label:", label)

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[DONE] wrote {args.out_jsonl}")


if __name__ == "__main__":
    main()