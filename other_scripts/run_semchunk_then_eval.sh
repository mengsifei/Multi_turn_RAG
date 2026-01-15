#!/usr/bin/env bash
set -euo pipefail

# -----------------------
# Config: semantic chunking
# -----------------------
ST_MODEL="Jasper-Token-Compression-600M"
HF_TOKENIZER="Jasper-Token-Compression-600M"
MAX_TOKENS=384
OVERLAP_TOKENS=64
SIM_THRESHOLD=0.55
BATCH_SIZE=32
BUF_SENTS=512

IN_DIR="corpora/passage_level"
OUT_DIR="corpora/chunk_level"
LOG_DIR="logs/semchunk_384"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

domains=(clapnq fiqa govt cloud)

# -----------------------
# Step 1: Build semantic chunks (sequential)
# -----------------------
for d in "${domains[@]}"; do
  in_zip="${IN_DIR}/${d}.jsonl.zip"
  jsonl_name="${d}.jsonl"
  out_corpus="${OUT_DIR}/${d}_passage_semantic.jsonl.gz"
  out_map="${OUT_DIR}/${d}_passage_semantic_map.jsonl.gz"
  log_file="${LOG_DIR}/${d}.log"

  echo "=============================="
  echo "[START CHUNK] ${d}  $(date)"
  echo "in_zip=${in_zip}"
  echo "out_corpus=${out_corpus}"
  echo "log=${log_file}"
  echo "=============================="

  # Skip if already exists
  if [[ -f "${out_corpus}" && -f "${out_map}" ]]; then
    echo "[SKIP CHUNK] ${d} already exists: ${out_corpus}" | tee -a "${log_file}"
    continue
  fi

  python3 make_semantic_chunks_batched.py \
      --in_zip "${in_zip}" \
      --jsonl_name "${jsonl_name}" \
      --out_corpus "${out_corpus}" \
      --out_map "${out_map}" \
      --st_model "${ST_MODEL}" \
      --hf_tokenizer "${HF_TOKENIZER}" \
      --max_tokens "${MAX_TOKENS}" \
      --overlap_tokens "${OVERLAP_TOKENS}" \
      --sim_threshold "${SIM_THRESHOLD}" \
      --embed_bs "${BATCH_SIZE}" \
      --buf_sents "${BUF_SENTS}" \
      --gzip_level 1 \
      2>&1 | tee "${log_file}"


  echo "[DONE CHUNK] ${d}  $(date)"
done

echo "=============================="
echo "[ALL CHUNK DONE] $(date)"
echo "=============================="

# -----------------------
# Step 2: Run official eval (semantic chunks)
# -----------------------
EVAL_LOG="logs/eval_semchunk_384_lastturn.log"
mkdir -p "$(dirname "${EVAL_LOG}")"

echo "=============================="
echo "[START EVAL] $(date)"
echo "log=${EVAL_LOG}"
echo "=============================="

python3 eval_jasper_ft_cached_semchunk_official.py \
  --task lastturn \
  --model_dir Jasper-Token-Compression-600M \
  --base_dir Jasper-Token-Compression-600M \
  --model_name jasper_base_semchunk_384 \
  --use_semantic_chunks \
  --chunk_corpus_dir corpora/chunk_level \
  --cache_dir cache/doc_emb_jasper_base_semchunk_384 \
  --top_k 50 \
  --top_k_chunks 1000 \
  2>&1 | tee "${EVAL_LOG}"

echo "=============================="
echo "[DONE EVAL] $(date)"
echo "=============================="
