import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def iter_conversations(path: Path) -> Iterable[Dict]:
    """Assume the file is a JSON array: [{...}, {...}, ...]."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for conv in data:
        yield conv


def format_dialogue(history: List[Tuple[str, str]]) -> str:
    """history: list of (speaker, text) where speaker in {"user","agent"}."""
    lines = []
    for spk, txt in history:
        prefix = "User" if spk == "user" else "Assistant"
        lines.append(f"{prefix}: {txt}")
    return "\n".join(lines).strip()


def write_record(out_f, query: str, passage: str, rec_type: str):
    out_f.write(
        json.dumps(
            {"query": query, "passage": passage, "type": rec_type},
            ensure_ascii=False,
        )
        + "\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("synthetic/conversations/conversations.json"))
    ap.add_argument("--output", type=Path, default=Path("jasper_train_pairs.jsonl"))
    ap.add_argument("--include-last-user", action="store_true", default=True,
                    help="Also write pairs where query is only the current user block (default: True).")
    ap.add_argument("--include-dialogue", action="store_true", default=True,
                    help="Also write pairs where query is dialogue history + current user block (default: True).")
    ap.add_argument("--no-answers", action="store_true",
                    help="If set, do NOT write query<->answer pairs (only query<->ctx).")
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    n_pairs = 0
    n_turns = 0

    with args.output.open("w", encoding="utf-8") as out_f:
        for conv in iter_conversations(args.input):
            messages = conv.get("messages", [])

            # history stores completed turns only (user+agent that already happened)
            history: List[Tuple[str, str]] = []
            # pending_user stores consecutive user utterances before the next agent
            pending_user: List[str] = []

            for msg in messages:
                speaker = msg.get("speaker")
                text = (msg.get("text") or "").strip()
                if not text:
                    continue

                if speaker == "user":
                    pending_user.append(text)
                    continue

                if speaker == "agent":
                    if not pending_user:
                        # agent without preceding user block; skip defensively
                        continue

                    # Build two query variants
                    user_block = "\n".join(pending_user).strip()

                    q_last_user = user_block if args.include_last_user else None

                    q_dialogue = None
                    if args.include_dialogue:
                        # history + current pending user block (as "User:" lines)
                        tmp_hist = history + [("user", u) for u in pending_user]
                        q_dialogue = format_dialogue(tmp_hist)

                    # Context pairs
                    contexts = msg.get("contexts") or []
                    for ctx in contexts:
                        ctx_text = (ctx.get("text") or "").strip()
                        if not ctx_text:
                            continue

                        if q_last_user:
                            write_record(out_f, q_last_user, ctx_text, "last_user__ctx")
                            n_pairs += 1

                        if q_dialogue:
                            write_record(out_f, q_dialogue, ctx_text, "dialogue__ctx")
                            n_pairs += 1

                    # Optional: query-answer pairs
                    if not args.no_answers:
                        if q_last_user:
                            write_record(out_f, q_last_user, text, "last_user__answer")
                            n_pairs += 1
                        if q_dialogue:
                            write_record(out_f, q_dialogue, text, "dialogue__answer")
                            n_pairs += 1

                    # Commit this turn into history, clear pending_user
                    for u in pending_user:
                        history.append(("user", u))
                    history.append(("agent", text))
                    pending_user = []
                    n_turns += 1

    print(f"Saved {n_pairs} training pairs to {args.output}")
    print(f"Total turns (agent responses paired): {n_turns}")


if __name__ == "__main__":
    main()

# import json
# from pathlib import Path

# INPUT_PATH = Path("synthetic/conversations/conversations.json")
# OUTPUT_PATH = Path("jasper_train_pairs.jsonl")

# def iter_conversations(path: Path):
#     """假设文件是一个 JSON 数组：[{...}, {...}, ...]"""
#     with path.open("r", encoding="utf-8") as f:
#         data = json.load(f)
#     for conv in data:
#         yield conv

# def main():
#     n_pairs = 0
#     with OUTPUT_PATH.open("w", encoding="utf-8") as out_f:
#         for conv in iter_conversations(INPUT_PATH):
#             messages = conv.get("messages", [])
#             last_user_text = None

#             for msg in messages:
#                 speaker = msg.get("speaker")
#                 text = (msg.get("text") or "").strip()
#                 if not text:
#                     continue

#                 if speaker == "user":
#                     # 当前轮的检索 query
#                     last_user_text = text

#                 elif speaker == "agent" and last_user_text:
#                     # 该轮的答案 + 检索到的 contexts
#                     contexts = msg.get("contexts") or []

#                     # 1) user 问题 <-> 每个检索文本
#                     for ctx in contexts:
#                         ctx_text = (ctx.get("text") or "").strip()
#                         if not ctx_text:
#                             continue
#                         record = {
#                             "query": last_user_text,
#                             "passage": ctx_text,
#                             "type": "query_ctx"
#                         }
#                         out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
#                         n_pairs += 1

#                     # 2) 可选：user 问题 <-> agent 回答（也可注释掉）
#                     if text:
#                         record = {
#                             "query": last_user_text,
#                             "passage": text,
#                             "type": "query_answer"
#                         }
#                         out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
#                         n_pairs += 1

#     print(f"Saved {n_pairs} training pairs to {OUTPUT_PATH}")

# if __name__ == "__main__":
#     main()
