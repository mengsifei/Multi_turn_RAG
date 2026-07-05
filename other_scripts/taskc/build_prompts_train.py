#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSV-only prompt builder for task C.

Input CSV is expected to contain columns like:
  task_id
  conversation_id
  rewrite_query
  lastturn_query
  questions
  contexts
  answerability
  collection or Collection
  targets

The contexts column should be a JSON string list:
[
  {"document_id": "...", "text": "...", "score": ...},
  ...
]

This version intentionally does NOT support the old JSONL pipeline.
It reads contexts directly from the CSV and builds prompts using:
  - rewrite_query
  - lastturn_query
  - questions
  - concat_lastturn_rewrite
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from tqdm import tqdm


DOMAINS_DEFAULT = ["clapnq", "cloud", "fiqa", "govt"]


# ----------------------- CSV field size fix -----------------------

def increase_csv_field_limit() -> None:
    """
    Python csv module defaults to 131072 bytes per field.
    Our contexts column can easily exceed that because it stores JSON with long passages.
    """
    max_size = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_size)
            break
        except OverflowError:
            max_size = int(max_size / 10)


# ----------------------- utils -----------------------

def is_missing(value: Any) -> bool:
    if value is None:
        return True
    s = str(value).strip()
    return s == "" or s.lower() in {"nan", "none", "null"}


def strip_user_prefix(text: Any) -> str:
    """
    Convert:
      |user|: Where do the Arizona Cardinals play this week?
    to:
      Where do the Arizona Cardinals play this week?

    For multi-line questions, strip |user|: on every line.
    """
    if is_missing(text):
        return ""

    lines: List[str] = []
    for line in str(text).splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("|user|:"):
            line = line[len("|user|:"):].strip()
        lines.append(line)

    return "\n".join(lines).strip()


def parse_jsonish_cell(value: Any, default: Any) -> Any:
    """
    Parse JSON stored inside a CSV cell.

    Handles:
      - valid JSON: [{"document_id": "...", "text": "..."}]
      - Python literal fallback
      - empty cells
    """
    if is_missing(value):
        return default

    s = str(value).strip()

    try:
        return json.loads(s)
    except Exception:
        pass

    try:
        return ast.literal_eval(s)
    except Exception:
        return default


def get_first(row: Dict[str, Any], keys: List[str], default: str = "") -> str:
    for key in keys:
        if key in row and not is_missing(row[key]):
            return str(row[key])
    return default


def infer_domain(collection_name: str, domains: List[str]) -> str:
    s = (collection_name or "").lower()
    for d in domains:
        if d in s:
            return d
    raise ValueError(f"Cannot infer domain from collection={collection_name!r}")


def infer_question_mode(task_name: str) -> str:
    name = (task_name or "").lower()

    if "concat" in name and "lastturn" in name and "rewrite" in name:
        return "concat_lastturn_rewrite"
    if "lastturn" in name and "rewrite" not in name:
        return "lastturn"
    if "questions" in name or "conversation" in name:
        return "questions"
    return "rewrite"


def build_question(row: Dict[str, Any], question_mode: str) -> Tuple[str, str]:
    rewrite = strip_user_prefix(row.get("rewrite_query"))
    lastturn = strip_user_prefix(row.get("lastturn_query"))
    questions = strip_user_prefix(row.get("questions"))

    if question_mode == "rewrite":
        if rewrite:
            return rewrite, "rewrite_query"
        if lastturn:
            return lastturn, "lastturn_query_fallback"
        if questions:
            return questions, "questions_fallback"
        return "", "missing"

    if question_mode == "lastturn":
        if lastturn:
            return lastturn, "lastturn_query"
        if rewrite:
            return rewrite, "rewrite_query_fallback"
        if questions:
            return questions, "questions_fallback"
        return "", "missing"

    if question_mode == "questions":
        if questions:
            return questions, "questions"
        if rewrite:
            return rewrite, "rewrite_query_fallback"
        if lastturn:
            return lastturn, "lastturn_query_fallback"
        return "", "missing"

    if question_mode == "concat_lastturn_rewrite":
        if lastturn and rewrite:
            return (
                f"|user|: {lastturn} {rewrite}", "concat_lastturn_rewrite"
            )
        if rewrite:
            return rewrite, "rewrite_query_fallback"
        if lastturn:
            return lastturn, "lastturn_query_fallback"
        if questions:
            return questions, "questions_fallback"
        return "", "missing"

    raise ValueError(f"Unknown question_mode={question_mode!r}")


# ----------------------- context / prompt builders -----------------------

def normalize_contexts(raw_value: Any, topk: int) -> List[Dict[str, Any]]:
    contexts = parse_jsonish_cell(raw_value, default=[])

    if not isinstance(contexts, list):
        return []

    out: List[Dict[str, Any]] = []
    for c in contexts[:topk]:
        if not isinstance(c, dict):
            continue

        text = c.get("text") or c.get("content") or ""
        if is_missing(text):
            continue

        doc_id = c.get("document_id", c.get("id"))
        out.append(
            {
                "document_id": None if doc_id is None else str(doc_id),
                "text": str(text),
                "score": c.get("score"),
            }
        )

    return out


def docs_to_context(
    docs: List[Dict[str, Any]],
    max_doc_chars: int,
    use_doc_id_in_header: bool = False,
    include_score: bool = False,
) -> str:
    parts: List[str] = ["Documents:"]

    for i, d in enumerate(docs, 1):
        doc_id = d.get("document_id")
        text = str(d.get("text") or "")[:max_doc_chars]

        header = f"[Document {i}]"
        meta: List[str] = []

        if use_doc_id_in_header and doc_id:
            meta.append(f"id={doc_id}")
        if include_score and d.get("score") is not None:
            meta.append(f"score={d.get('score')}")

        if meta:
            header += " " + " ".join(meta)

        parts.append(header)
        parts.append(text)
        parts.append("")

    return "\n".join(parts).rstrip()


def prompt_official_grounded_v1(question: str, context: str) -> str:
    return f"""
Given the following documents and a user question, generate a concise answer
(less than 150 words) that is grounded in the provided documents.

- Use only information that can be supported by the documents.
- You may paraphrase and summarize across documents.
- Do NOT quote verbatim unless necessary.
- If the documents do not contain enough information to answer the question,
  say: "I do not have specific information."

[Documents]
{context}

[Question]
{question}

[Answer]
"""


def prompt_official_typeaware_light_v2(question: str, context: str) -> str:
    return f"""
You are a retrieval-augmented generation answer writer. Given the following documents
and a user question, generate a concise answer (less than 150 words) grounded in the documents.

Rules:
- Use only information that can be supported by the documents.
- Do not use outside knowledge.
- Do not mention "the documents", "the provided documents", "the context", "Document 1",
  "Document 2", or similar source labels in the answer.
- Do not start with phrases like "Based on the documents", "According to the documents",
  "From the documents", or "The documents say".
- You may use exact wording from the evidence when it helps preserve names, numbers, dates,
  commands, technical terms, legal/policy wording, or other key phrases.
- You may paraphrase and summarize across documents.
- Preserve important names, numbers, dates, places, entities, commands, and technical terms.
- Do not add unsupported background, speculation, or unrelated details.

Answer style:
- Silently identify the question type before answering.
- For factoid questions, give the direct answer first, usually in 1 sentence.
- For how-to, why/how, troubleshooting, comparison, summary, or multi-part questions,
  include the main supported steps, reasons, differences, or facts needed to answer fully.
- Be concise, but do not omit important supported information only to make the answer shorter.
- Use complete sentences.

Answerability:
- If the documents contain information that answers the main question, answer it directly.
- If only part of the question is supported, answer the supported part and briefly state what is not specified.
- If the documents do not contain enough information to answer the main question,
  say exactly: I do not have specific information.
- Do not infer an answer from merely related information.

[Documents]
{context}

[Question]
{question}

[Answer]
""".strip()




def prompt_grounded_rb_alg_v1(question: str, context: str) -> str:
    return f"""
Given the following documents and a user question, generate a concise reference-style answer
that is grounded only in the provided documents.

Your goal is to maximize answer precision and recall for automatic evaluation.

Core rules:
- Use only information supported by the documents.
- Do not use outside knowledge.
- Do not mention "the documents", "the provided documents", "Document 1", "Document 2", or similar source labels.
- Do not add background information, examples, caveats, or related facts unless they are necessary to answer the question.
- Preserve exact names, numbers, dates, entities, and key phrases from the documents.
- Prefer wording that closely matches the evidence.
- Do not quote long passages verbatim.
- Do not start with phrases like "Based on the documents" or "According to the documents".
- Stop immediately after answering the question.

Length rules:
- For factoid questions asking who, what, when, where, how many, or how much:
  answer in exactly one sentence and under 20 words when possible.
- For yes/no questions:
  start with "Yes" or "No", then give only the minimal supporting fact.
- For definition questions:
  answer in one concise sentence when possible.
- For comparison questions:
  answer in 1-2 concise sentences.
- For why/how questions:
  answer in 2-3 concise sentences only if needed.
- The answer must be under 150 words.

Answerability rules:
- If the documents fully answer the question, answer directly.
- If the documents answer only part of the question, answer the supported part first,
  then briefly state what is not specified.
- If the documents do not contain enough information to answer any part of the question,
  output exactly: I do not have specific information.
- Do not refuse the whole question if any part can be answered from the documents.

Candidate selection instruction:
Internally draft three candidate answers:
A. Minimal answer
B. Natural one-sentence answer
C. Evidence-wording answer

Then choose the single best final answer using these criteria:
1. It fully answers the question.
2. It contains no extra facts.
3. It is the shortest complete answer.
4. It preserves key wording from the evidence.
5. It avoids unnecessary explanation.

Output only the final answer. Do not show the candidates.

[Documents]
{context}

[Question]
{question}

[Answer]
""".strip()



def prompt_grounded_balanced_v1(question: str, context: str) -> str:
    return f"""
Given the following documents and a user question, generate a compact but complete
reference-style answer grounded only in the provided documents.

Your goal is to maximize both factual coverage and precision for automatic evaluation.

Core rules:
- Use only information that is directly supported by the documents.
- Do not use outside knowledge.
- Do not mention "the documents", "the provided documents", "Document 1", "Document 2",
  or similar source labels.
- Preserve exact names, numbers, dates, places, entities, technical terms, and key phrases
  from the evidence whenever they are part of the answer.
- Prefer wording that closely matches the evidence, but do not copy long passages verbatim.
- Do not add unrelated background, examples, or caveats.
- Do not start with phrases like "Based on the documents" or "According to the documents".
- Output only the final answer.

Coverage rules:
- Answer the full information need of the question, not just the first matching fact.
- Include all directly supported answer-bearing facts needed for a complete reference answer.
- If the evidence gives multiple required items, reasons, steps, conditions, dates, or examples
  that answer the question, include them compactly.
- For list questions, include the requested items and any short distinguishing details that
  help identify them.
- For explanation, why, how, or process questions, give the main supported reasons or steps,
  using concise complete sentences.
- For comparison questions, state the comparison directly and include the key supported basis.
- For yes/no questions, answer "Yes" or "No" only when the evidence directly supports that
  conclusion; then add the minimal supported explanation or qualification.

Length rules:
- Be concise, but do not make the answer so short that important supported facts are omitted.
- Use complete sentences.
- Most answers should be 1-4 sentences.
- Longer answers are acceptable when the question asks for steps, reasons, procedures,
  comparisons, or multiple facts.
- The answer must be under 150 words.

Answerability rules:
- Before answering, check whether the evidence directly and unambiguously answers the main
  information need of the question.
- The evidence must match the key entity, relation, time, condition, comparison, and constraint
  in the question.
- Do not answer from merely related information.
- Do not infer a yes/no answer unless it is directly supported.
- If the evidence fully answers the question, answer directly.
- If the evidence answers only a clearly separable part of the question, answer the supported
  part and briefly state what is not specified.
- If the evidence does not directly answer the main information need, output exactly:
  I do not have specific information.
- If the question asks for a specific fact but the evidence only provides general background,
  output exactly:
  I do not have specific information.
- If the evidence is about a similar but different entity, situation, time, or condition,
  output exactly:
  I do not have specific information.

Internal decision process:
1. Identify the main information need of the question.
2. Identify all evidence spans that directly answer it.
3. Decide whether the answer is full, partial, or unsupported.
4. Write the shortest answer that still includes all important supported answer-bearing facts.
5. Remove unsupported or unrelated facts, but do not remove facts that are needed for coverage.

[Documents]
{context}

[Question]
{question}

[Answer]
""".strip()



def prompt_grounded_typeaware_balanced_v1(question: str, context: str) -> str:
    return f"""
Given the following documents and a user question, generate a compact but sufficiently complete
reference-style answer grounded only in the provided documents.

Your goal is to maximize both factual coverage and precision for automatic evaluation.

Core rules:
- Use only information supported by the documents.
- Do not use outside knowledge.
- Do not mention "the documents", "the provided documents", "Document 1", "Document 2",
  or similar source labels.
- Preserve exact names, numbers, dates, places, entities, technical terms, and key phrases
  from the evidence when they are part of the answer.
- Prefer wording close to the evidence, but do not copy long passages verbatim.
- Include relevant explanatory context when it directly helps answer the question.
- Do not add unrelated background, examples, or speculation.
- Do not start with phrases like "Based on the documents" or "According to the documents".
- Output only the final answer.

Question-type decision:
Internally classify the user question into one of these types:
factoid, how-to, troubleshooting, explanation, summarization, comparative, opinion/recommendation,
keyword/non-question, or composite/multi-part.

Do not output the question type. Use it only to choose the answer shape.

Answer-shape rules by question type:
- Factoid questions:
  Give the direct answer first. Use 1 concise sentence if the answer is simple.
  Use 2 sentences if needed to include important dates, qualifiers, distinctions, or exceptions.
  Do not force the answer to be extremely short if the reference answer requires context.

- How-to or procedural questions:
  Give the main steps, requirements, commands, conditions, or constraints that are directly supported.
  A compact numbered list or 3-6 concise sentences is acceptable.
  Include prerequisites or warnings when they are part of the supported answer.

- Troubleshooting questions:
  State the likely issue or limitation if supported, then give the relevant checks or steps.
  Include error conditions, access requirements, configuration details, or fixes when supported.

- Explanation, why, or how-it-works questions:
  Explain the main cause, mechanism, or reasoning chain.
  Use 2-5 concise sentences and include key supporting facts needed to make the explanation complete.

- Summarization or overview questions:
  Cover the main answer-bearing points, not just one fact.
  Use 3-6 concise sentences or a compact paragraph.
  Include major entities, events, dates, outcomes, or examples when supported.

- Comparative questions:
  Compare all requested items directly.
  State the key similarities, differences, advantages, disadvantages, or basis for the comparison.
  Do not discuss only one side unless the other side is not specified.

- Opinion, recommendation, or should-I questions:
  Give a grounded recommendation or conclusion only if the evidence supports it.
  Include the main supported factors, tradeoffs, pros/cons, or conditions.
  Do not present unsupported personal opinion.

- Keyword or non-question queries:
  Infer the likely information need from the query and context.
  Provide a short definition, description, or answer with the key supported facts.
  Do not reject the query only because it is not phrased as a full question.

- Composite or multi-part questions:
  Answer each separable part in order.
  Do not omit a subquestion when the evidence supports it.
  If only some parts are supported, answer those parts and briefly say what is not specified.

Coverage rules:
- Answer the full information need, not just the first matching fact.
- Include all directly supported answer-bearing facts that a reference answer would likely contain.
- If the evidence gives multiple required items, reasons, steps, conditions, dates, examples, or caveats,
  include them compactly.
- Be concise, but do not remove important supported facts only to make the answer shorter.
- Most answers should be 1-5 sentences.
- Longer answers are acceptable for how-to, troubleshooting, summarization, explanation, comparison,
  recommendation, or multi-part questions.
- The answer should normally be under 150 words.

Answerability rules:
- If the evidence contains directly relevant information that answers the main information need,
  answer using that evidence.
- Do not output an unsupported answer from merely related information.
- Do not infer a yes/no answer unless the evidence supports the conclusion.
- If the evidence answers only a clearly separable part of the question, answer the supported part
  and briefly state what is not specified.
- If the evidence does not contain answer-bearing information for the main question, output exactly:
  I do not have specific information.
- If the evidence is about a different entity, different time, different condition, or different comparison
  than the question asks about, output exactly:
  I do not have specific information.

Internal decision process:
1. Identify the question type.
2. Identify the main information need and any subquestions.
3. Find all evidence spans that directly answer the question.
4. Decide whether the answer is supported, partially supported, or unsupported.
5. Write a compact reference-style answer with enough coverage for that question type.
6. Remove unsupported or unrelated facts, but keep supported facts needed for coverage.

[Documents]
{context}

[Question]
{question}

[Answer]
""".strip()


def prompt_grounded_typeaware_cot_v1(question: str, context: str) -> str:
    return f"""
Given the following documents and a user question, generate a compact but sufficiently complete
reference-style answer grounded only in the provided documents.

Your goal is to maximize both factual coverage and precision for automatic evaluation.

Core rules:
- Use only information supported by the documents.
- Do not use outside knowledge.
- Do not mention "the documents", "the provided documents", "Document 1", "Document 2",
  or similar source labels in the final answer.
- Preserve exact names, numbers, dates, places, entities, technical terms, and key phrases
  from the evidence when they are part of the answer.
- Prefer wording close to the evidence, but do not copy long passages verbatim.
- Include relevant explanatory context when it directly helps answer the question.
- Do not add unrelated background, examples, or speculation.

Question-type decision:
Internally classify the user question into one of these types:
factoid, how-to, troubleshooting, explanation, summarization, comparative, opinion/recommendation,
keyword/non-question, or composite/multi-part.

Use the question type only to choose the answer shape.

Answer-shape rules by question type:
- Factoid questions:
  Give the direct answer first. Use 1 concise sentence if the answer is simple.
  Use 2 sentences if needed to include important dates, qualifiers, distinctions, or exceptions.

- How-to or procedural questions:
  Give the main steps, requirements, commands, conditions, or constraints that are directly supported.
  A compact numbered list or 3-6 concise sentences is acceptable.
  Include prerequisites or warnings when they are part of the supported answer.

- Troubleshooting questions:
  State the likely issue or limitation if supported, then give the relevant checks or steps.
  Include error conditions, access requirements, configuration details, or fixes when supported.

- Explanation, why, or how-it-works questions:
  Explain the main cause, mechanism, or reasoning chain.
  Use 2-5 concise sentences and include key supporting facts needed to make the explanation complete.

- Summarization or overview questions:
  Cover the main answer-bearing points, not just one fact.
  Use 3-6 concise sentences or a compact paragraph.
  Include major entities, events, dates, outcomes, or examples when supported.

- Comparative questions:
  Compare all requested items directly.
  State the key similarities, differences, advantages, disadvantages, or basis for the comparison.
  Do not discuss only one side unless the other side is not specified.

- Opinion, recommendation, or should-I questions:
  Give a grounded recommendation or conclusion only if the evidence supports it.
  Include the main supported factors, tradeoffs, pros/cons, or conditions.
  Do not present unsupported personal opinion.

- Keyword or non-question queries:
  Infer the likely information need from the query and context.
  Provide a short definition, description, or answer with the key supported facts.
  Do not reject the query only because it is not phrased as a full question.

- Composite or multi-part questions:
  Answer each separable part in order.
  Do not omit a subquestion when the evidence supports it.
  If only some parts are supported, answer those parts and briefly say what is not specified.

Answerability rules:
- If the evidence contains directly relevant information that answers the main information need,
  answer using that evidence.
- Do not output an unsupported answer from merely related information.
- Do not infer a yes/no answer unless the evidence supports the conclusion.
- If the evidence answers only a clearly separable part of the question, answer the supported part
  and briefly state what is not specified.
- If the evidence does not contain answer-bearing information for the main question, the final answer
  must be exactly:
  I do not have specific information.
- If the evidence is about a different entity, different time, different condition, or different comparison
  than the question asks about, the final answer must be exactly:
  I do not have specific information.

Before the final answer, write a short grounded analysis in this exact format:

[COT]
Question type: <one type>
Answerability: <supported / partially supported / unsupported>
Evidence notes:
1. <answer-bearing fact from the documents, or "None">
2. <answer-bearing fact from the documents, if needed>
3. <answer-bearing fact from the documents, if needed>
4. <answer-bearing fact from the documents, if needed>
5. <answer-bearing fact from the documents, if needed>

Rules for [COT]:
- Keep it short.
- Evidence notes must be copied or closely paraphrased from the documents.
- Use at most 5 evidence notes.
- Do not include unsupported assumptions.
- If unsupported, write Evidence notes: 1. None

Then write the final answer in this exact format:

[ANSWER]
<final answer only>

Rules for [ANSWER]:
- The final answer should be compact but sufficiently complete.
- Include all important supported answer-bearing facts from the evidence notes.
- Do not mention the question type, answerability label, evidence notes, or documents.
- Most answers should be 1-5 sentences.
- Longer answers are acceptable for how-to, troubleshooting, summarization, explanation, comparison,
  recommendation, or multi-part questions.
- The final answer should normally be under 150 words.

[Documents]
{context}

[Question]
{question}

[COT]
""".strip()


def prompt_grounded_typeaware_internalcheck_v1(question: str, context: str) -> str:
    return f"""
Given the following documents and a user question, generate a compact but sufficiently complete
reference-style answer grounded only in the provided documents.

Your goal is to maximize both factual coverage and precision for automatic evaluation.

Core rules:
- Use only information supported by the documents.
- Do not use outside knowledge.
- Do not mention "the documents", "the provided documents", "Document 1", "Document 2",
  or similar source labels.
- Preserve exact names, numbers, dates, places, entities, technical terms, and key phrases
  from the evidence when they are part of the answer.
- Prefer wording close to the evidence, but do not copy long passages verbatim.
- Include relevant explanatory context when it directly helps answer the question.
- Do not add unrelated background, examples, or speculation.
- Do not start with phrases like "Based on the documents" or "According to the documents".
- Output only the final answer.

Question-type decision:
Silently classify the user question into one of these types:
factoid, how-to, troubleshooting, explanation, summarization, comparative, opinion/recommendation,
keyword/non-question, or composite/multi-part.

Do not output the question type. Use it only to choose the answer shape.

Answer-shape rules by question type:
- Factoid questions:
  Give the direct answer first. Use 1 concise sentence if the answer is simple.
  Use 2 sentences if needed to include important dates, qualifiers, distinctions, or exceptions.
  Do not force the answer to be extremely short if the reference answer requires context.

- How-to or procedural questions:
  Give the main steps, requirements, commands, conditions, or constraints that are directly supported.
  A compact numbered list or 3-6 concise sentences is acceptable.
  Include prerequisites or warnings when they are part of the supported answer.

- Troubleshooting questions:
  State the likely issue or limitation if supported, then give the relevant checks or steps.
  Include error conditions, access requirements, configuration details, or fixes when supported.

- Explanation, why, or how-it-works questions:
  Explain the main cause, mechanism, or reasoning chain.
  Use 2-5 concise sentences and include key supporting facts needed to make the explanation complete.

- Summarization or overview questions:
  Cover the main answer-bearing points, not just one fact.
  Use 3-6 concise sentences or a compact paragraph.
  Include major entities, events, dates, outcomes, or examples when supported.

- Comparative questions:
  Compare all requested items directly.
  State the key similarities, differences, advantages, disadvantages, or basis for the comparison.
  Do not discuss only one side unless the other side is not specified.

- Opinion, recommendation, or should-I questions:
  Give a grounded recommendation or conclusion only if the evidence supports it.
  Include the main supported factors, tradeoffs, pros/cons, or conditions.
  Do not present unsupported personal opinion.

- Keyword or non-question queries:
  Infer the likely information need from the query and context.
  Provide a short definition, description, or answer with the key supported facts.
  Do not reject the query only because it is not phrased as a full question.

- Composite or multi-part questions:
  Answer each separable part in order.
  Do not omit a subquestion when the evidence supports it.
  If only some parts are supported, answer those parts and briefly say what is not specified.

Coverage rules:
- Answer the full information need, not just the first matching fact.
- Include all directly supported answer-bearing facts that a reference answer would likely contain.
- If the evidence gives multiple required items, reasons, steps, conditions, dates, examples, or caveats,
  include them compactly.
- Be concise, but do not remove important supported facts only to make the answer shorter.
- Most answers should be 1-5 sentences.
- Longer answers are acceptable for how-to, troubleshooting, summarization, explanation, comparison,
  recommendation, or multi-part questions.
- The answer should normally be under 150 words.

Answerability rules:
- If the evidence contains directly relevant information that answers the main information need,
  answer using that evidence.
- Do not output an unsupported answer from merely related information.
- Do not infer a yes/no answer unless the evidence supports the conclusion.
- If the evidence answers only a clearly separable part of the question, answer the supported part
  and briefly state what is not specified.
- If the evidence does not contain answer-bearing information for the main question, output exactly:
  I do not have specific information.
- If the evidence is mostly about a different entity, different time, different condition, or different comparison
  than the question asks about, output exactly:
  I do not have specific information.

Silent internal checklist:
Before writing the final answer, silently do the following:
1. Identify the question type.
2. Identify the main information need and any subquestions.
3. Select up to five directly supported answer-bearing facts that are most likely to appear
   in a reference answer.
4. Check whether the evidence supports a full answer, a partial answer, or no answer.
5. Write the final answer using the selected facts.

Do not output the checklist.
Do not output reasoning.
Do not output labels such as [COT], [ANALYSIS], or [ANSWER].
Output only the final answer.

[Documents]
{context}

[Question]
{question}

[Answer]
""".strip()


def prompt_grounded_typeaware_exactgate_examples_v1(question: str, context: str) -> str:
    return f"""
Given the following documents and a user question, generate a compact but sufficiently complete
reference-style answer grounded only in the provided documents.

Your goal is to maximize factual coverage, grounding, and answer correctness for automatic evaluation.

Core rules:
- Use only information supported by the documents.
- Do not use outside knowledge.
- Do not mention "the documents", "the provided documents", "Document 1", "Document 2",
  or similar source labels.
- Preserve exact names, numbers, dates, places, entities, technical terms, and key phrases
  from the evidence when they are part of the answer.
- Prefer wording close to the evidence, but do not copy long passages verbatim.
- Include relevant explanatory context when it directly helps answer the question.
- Do not add unrelated background, examples, speculation, or unsupported implications.
- Do not start with phrases like "Based on the documents" or "According to the documents".
- Output only the final answer.

Silent exact-evidence gate:
Before answering, silently check whether the documents answer the exact question being asked.

The answer is supported only if the evidence matches the key:
- entity or entities
- relation or event
- time period or date, if any
- location, if any
- condition, constraint, comparison, or requested aspect
- requested answer type, such as name, number, date, reason, step, definition, or yes/no conclusion

Use this gate carefully:
- If the evidence directly answers the exact question, answer it.
- If the evidence gives a general answer that clearly applies to the question, answer it.
- If the evidence answers only a clearly separable part of a multi-part question, answer the supported part
  and briefly say what is not specified.
- If the evidence is only topically related but does not answer the exact information need, output exactly:
  I do not have specific information.
- If the evidence is about a similar but different entity, time, condition, comparison, or requested aspect,
  output exactly:
  I do not have specific information.
- For names, dates, pronunciations, legal/medical/technical requirements, commands, prices, counts,
  and yes/no questions, do not infer the answer from related background. Answer only when directly supported.
- Do not force an answer from weak or partial evidence.

Gate examples:
These examples show how to apply the exact-evidence gate. They are only behavior examples;
do not use their content to answer the current question.

Example 1 — supported:
Question: What year did Alpha launch Product X?
Evidence: Alpha launched Product X in 2019 after a two-year pilot.
Correct behavior: Answer directly.
Final answer: Alpha launched Product X in 2019 after a two-year pilot.

Example 2 — partially supported:
Question: What are the setup steps and supported regions for Feature A?
Evidence: To set up Feature A, open Settings, enable Feature A, and save the configuration.
Correct behavior: Answer the supported part and say what is not specified.
Final answer: To set up Feature A, open Settings, enable Feature A, and save the configuration. The supported regions are not specified.

Example 3 — unsupported:
Question: Did Company B acquire Company C in 2021?
Evidence: Company B announced a partnership with Company C in 2021.
Correct behavior: Do not infer acquisition from a related partnership.
Final answer: I do not have specific information.

Question-type decision:
Silently classify the user question into one of these types:
factoid, how-to, troubleshooting, explanation, summarization, comparative, opinion/recommendation,
keyword/non-question, or composite/multi-part.

Do not output the question type. Use it only to choose the answer shape.

Answer-shape rules by question type:
- Factoid questions:
  Give the direct answer first. Use 1 concise sentence if the answer is simple.
  Use 2 sentences if needed to include important dates, qualifiers, distinctions, or exceptions.
  Do not make the answer extremely short if the reference answer likely requires context.

- How-to or procedural questions:
  Give the main steps, requirements, commands, conditions, or constraints that are directly supported.
  A compact numbered list or 3-6 concise sentences is acceptable.
  Include prerequisites, limitations, or warnings when they are part of the supported answer.

- Troubleshooting questions:
  State the likely issue or limitation if supported, then give the relevant checks or steps.
  Include error conditions, access requirements, configuration details, or fixes when supported.

- Explanation, why, or how-it-works questions:
  Explain the main cause, mechanism, or reasoning chain.
  Use 2-5 concise sentences and include key supporting facts needed to make the explanation complete.

- Summarization or overview questions:
  Cover the main answer-bearing points, not just one fact.
  Use 3-6 concise sentences or a compact paragraph.
  Include major entities, events, dates, outcomes, or examples when supported.

- Comparative questions:
  Compare all requested items directly.
  State the key similarities, differences, advantages, disadvantages, or basis for the comparison.
  Do not discuss only one side unless the other side is not specified.

- Opinion, recommendation, or should-I questions:
  Give a grounded recommendation or conclusion only if the evidence supports it.
  Include the main supported factors, tradeoffs, pros/cons, or conditions.
  Do not present unsupported personal opinion.

- Keyword or non-question queries:
  Infer the likely information need from the query and context.
  Provide a short definition, description, or answer with the key supported facts.
  Do not reject the query only because it is not phrased as a full question.

- Composite or multi-part questions:
  Answer each separable part in order.
  Do not omit a subquestion when the evidence supports it.
  If only some parts are supported, answer those parts and briefly say what is not specified.

Coverage rules:
- Answer the full information need, not just the first matching fact.
- Include all directly supported answer-bearing facts that a reference answer would likely contain.
- If the evidence gives multiple required items, reasons, steps, conditions, dates, examples, or caveats,
  include them compactly.
- Be concise, but do not remove important supported facts only to make the answer shorter.
- Most answers should be 1-5 sentences.
- Longer answers are acceptable for how-to, troubleshooting, summarization, explanation, comparison,
  recommendation, or multi-part questions.
- The answer should normally be under 150 words.

Faithfulness rules:
- Every factual claim in the final answer must be supported by the documents.
- Avoid vague claims that are difficult to verify.
- Do not combine facts from different documents into a stronger conclusion unless that conclusion is directly supported.
- If a useful detail is not clearly supported, omit it.
- If omitting unsupported details makes the question unanswerable, output exactly:
  I do not have specific information.

Silent internal checklist:
Before writing the final answer, silently do the following:
1. Identify the question type.
2. Identify the exact entity, relation, time, condition, and requested answer type.
3. Select the directly supported answer-bearing facts that are most likely to appear in a reference answer.
4. Check whether those facts answer the exact question, only a separable part, or none of it.
5. Write the final answer using only those facts.

Do not output the checklist.
Do not output reasoning.
Do not output labels such as [COT], [ANALYSIS], or [ANSWER].
Output only the final answer.

[Documents]
{context}

[Question]
{question}

[Answer]
""".strip()


def build_prompt(prompt_style: str, question: str, context: str) -> str:
    if prompt_style == "official":
        return prompt_official_grounded_v1(question, context)
    if prompt_style == "official_typeaware_light_v2":
        return prompt_official_typeaware_light_v2(question, context)
    if prompt_style == "concise":
        return prompt_grounded_rb_alg_v1(question, context)
    if prompt_style == "balanced":
        return prompt_grounded_balanced_v1(question, context)
    if prompt_style == "typeaware_balanced":
        return prompt_grounded_typeaware_balanced_v1(question, context)
    if prompt_style == "typeaware_cot":
        return prompt_grounded_typeaware_cot_v1(question, context)
    if prompt_style == "typeaware_internalcheck":
        return prompt_grounded_typeaware_internalcheck_v1(question, context)
    if prompt_style == "typeaware_exactgate_examples":
        return prompt_grounded_typeaware_exactgate_examples_v1(question, context)
    raise ValueError(f"Unknown prompt_style={prompt_style!r}")


# ----------------------- main -----------------------

def main(args: argparse.Namespace) -> None:
    increase_csv_field_limit()

    domains = [d.strip() for d in args.domains.split(",") if d.strip()] or DOMAINS_DEFAULT

    question_mode = args.question_mode
    if question_mode == "auto":
        question_mode = infer_question_mode(args.task_name)

    in_path = Path(args.taska_file)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] input_format=csv_only")
    print(f"[INFO] question_mode={question_mode}")
    print(f"[INFO] taska_file={in_path}")
    print(f"[INFO] out_csv={out_path}")

    n_total = 0
    n_written = 0
    n_skip_no_id_collection = 0
    n_skip_bad_domain = 0
    n_skip_no_query = 0
    n_skip_no_ctx = 0
    n_written_empty_ctx = 0
    question_source_counts: Dict[str, int] = {}

    fieldnames = ["task_id", "conversation_id", "Collection", "domain", "prompt"]

    if args.add_ctx_count:
        fieldnames.append("n_contexts")
    if args.add_question_source:
        fieldnames.append("question_source")
    if args.add_answerability:
        fieldnames.append("answerability")

    with in_path.open("r", encoding="utf-8-sig", newline="") as fin, \
         out_path.open("w", encoding="utf-8-sig", newline="") as fout:

        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in tqdm(reader, desc="Building prompts_train.csv"):
            n_total += 1

            task_id = get_first(row, ["task_id", "_id", "id"])
            collection = get_first(row, ["Collection", "collection"])
            conversation_id = get_first(row, ["conversation_id"], default=task_id.split("<::>")[0])

            if not task_id or not collection:
                n_skip_no_id_collection += 1
                continue

            try:
                domain = infer_domain(collection, domains)
            except Exception:
                n_skip_bad_domain += 1
                continue

            question, question_source = build_question(row, question_mode)
            if not question:
                n_skip_no_query += 1
                continue

            question_source_counts[question_source] = question_source_counts.get(question_source, 0) + 1

            contexts = normalize_contexts(row.get("contexts"), topk=args.topk)
            if not contexts:
                if args.skip_no_ctx:
                    n_skip_no_ctx += 1
                    continue
                n_written_empty_ctx += 1

            context_text = docs_to_context(
                docs=contexts,
                max_doc_chars=args.max_doc_chars,
                use_doc_id_in_header=args.doc_id_in_header,
                include_score=args.include_score,
            )

            prompt = build_prompt(args.prompt_style, question, context_text)

            out_row = {
                "task_id": task_id,
                "conversation_id": conversation_id,
                "Collection": collection,
                "domain": domain,
                "prompt": prompt,
            }

            if args.add_ctx_count:
                out_row["n_contexts"] = len(contexts)
            if args.add_question_source:
                out_row["question_source"] = question_source
            if args.add_answerability:
                out_row["answerability"] = get_first(row, ["answerability", "Answerability"])

            writer.writerow(out_row)
            n_written += 1

    print("========== PROMPTS CSV SUMMARY ==========")
    print(f"Input rows                  : {n_total}")
    print(f"Rows written                : {n_written}")
    print(f"Skipped missing id/collection: {n_skip_no_id_collection}")
    print(f"Skipped bad domain           : {n_skip_bad_domain}")
    print(f"Skipped no query             : {n_skip_no_query}")
    print(f"Skipped no contexts          : {n_skip_no_ctx}")
    print(f"Written with empty contexts  : {n_written_empty_ctx}")
    if args.add_question_source:
        print(f"Question source counts       : {json.dumps(question_source_counts, ensure_ascii=False)}")
    print(f"Output file                  : {out_path}")
    print("========================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--taska_file", required=True, help="Input task C CSV")
    parser.add_argument("--out_csv", required=True, help="Output prompt CSV")

    # Kept for compatibility with your old command. Not used in CSV-only mode.
    parser.add_argument("--cleaned_root", default=None, help="Ignored in CSV-only mode")

    parser.add_argument(
        "--task_name",
        default="rewrite_gpt",
        help="Used only to infer question_mode when --question_mode auto.",
    )
    parser.add_argument(
        "--question_mode",
        default="auto",
        choices=["auto", "rewrite", "lastturn", "questions", "concat_lastturn_rewrite"],
        help=(
            "Which CSV query columns to use. "
            "auto infers from --task_name; concat_lastturn_rewrite_gpt -> concat_lastturn_rewrite."
        ),
    )
    parser.add_argument(
        "--prompt_style",
        default="concise",
        help="Prompt template.",
    )

    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max_doc_chars", type=int, default=1200)
    parser.add_argument("--domains", default="clapnq,cloud,fiqa,govt")
    parser.add_argument("--include_score", action="store_true")
    parser.add_argument("--doc_id_in_header", action="store_true")
    parser.add_argument("--add_ctx_count", action="store_true")
    parser.add_argument("--add_question_source", action="store_true")
    parser.add_argument("--add_answerability", action="store_true")

    parser.add_argument(
        "--skip_no_ctx",
        action="store_true",
        help="Skip rows with empty contexts. Default: keep them.",
    )

    args = parser.parse_args()
    main(args)
