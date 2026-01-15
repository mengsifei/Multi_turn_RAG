#!/bin/bash
#SBATCH --account=proj_1782
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH -t 24:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
mkdir -p logs cache/doc_ctx

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

QWEN_PATH="Qwen3-4B-Instruct-2507"
BATCH=2
SNIPPET=1600
NEW_TOK=140

run_one () {
  DOM="$1"
  IN="cleaned_dataset/${DOM}/${DOM}.jsonl"
  OUT="cache/doc_ctx/${DOM}.qwen_kw5.fixed.jsonl"

  echo "================================================================"
  echo "[DOMAIN] ${DOM}"
  echo "[IN]  ${IN}"
  echo "[OUT] ${OUT}"
  echo "================================================================"

  CMD="python3 qwen_doc_keywords_with_tfidf_hint.py \
    --domain ${DOM} \
    --in_jsonl ${IN} \
    --qwen_path ${QWEN_PATH} \
    --out_doc_ctx ${OUT} \
    --batch_size ${BATCH} \
    --snippet_chars ${SNIPPET} \
    --max_new_tokens ${NEW_TOK}"

  echo "[RUN] $CMD"
  # 失败重试一次（防止偶发显存/IO抖动）
  bash -lc "$CMD" || (echo "[WARN] ${DOM} failed, retry once..." && sleep 10 && bash -lc "$CMD")
}

# 串行跑四个域
run_one fiqa
run_one govt
run_one cloud
run_one clapnq

echo "[DONE] all domains finished."
