#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ST_MODEL="Jasper-Token-Compression-600M"
HF_TOKENIZER="Jasper-Token-Compression-600M"

# final chunks (match official)
MAX_TOKENS=512
OVERLAP_TOKENS=100
SIM_THRESHOLD=0.55

# embedding safety (Govt)
SENT_MAX_TOKENS=128
SENT_OVERLAP_TOKENS=16

# batching
EMBED_BS=64
BUF_UNITS=4096

IN_DOC_DIR="corpora/document_level"
IN_PASS_DIR="corpora/passage_level"

OUT_DOCSEM_DIR="corpora/chunk_level_docsem512_100"
OUT_SPAN_DIR="corpora/passage_level"   # keep indices here to be easy

LOG_DIR="logs/docsem512_100"
mkdir -p "${OUT_DOCSEM_DIR}" "${LOG_DIR}"

domains=(clapnq fiqa govt cloud)

echo "[START] $(date)"

# -----------------------
# Step 1: doc-level semantic chunks (512/100)
# -----------------------
for d in "${domains[@]}"; do
  in_zip="${IN_DOC_DIR}/${d}.jsonl.zip"
  jsonl_name="${d}.jsonl"
  out_corpus="${OUT_DOCSEM_DIR}/${d}_docsem512_100.jsonl.gz"
  out_map="${OUT_DOCSEM_DIR}/${d}_docsem512_100_map.jsonl.gz"
  log_file="${LOG_DIR}/${d}_chunk.log"

  echo "=============================="
  echo "[DOCSEM CHUNK] ${d} $(date)"
  echo "in_zip=${in_zip}"
  echo "out=${out_corpus}"
  echo "log=${log_file}"
  echo "=============================="

  if [[ -f "${out_corpus}" && -f "${out_map}" ]]; then
    echo "[SKIP] exists: ${out_corpus}" | tee -a "${log_file}"
  else
    python3 other_scripts/make_doc_semantic_chunks.py \
      --in_zip "${in_zip}" \
      --jsonl_name "${jsonl_name}" \
      --out_corpus "${out_corpus}" \
      --out_map "${out_map}" \
      --st_model "${ST_MODEL}" \
      --hf_tokenizer "${HF_TOKENIZER}" \
      --max_tokens "${MAX_TOKENS}" \
      --overlap_tokens "${OVERLAP_TOKENS}" \
      --sim_threshold "${SIM_THRESHOLD}" \
      --sent_max_tokens "${SENT_MAX_TOKENS}" \
      --sent_overlap_tokens "${SENT_OVERLAP_TOKENS}" \
      --embed_bs "${EMBED_BS}" \
      --buf_units "${BUF_UNITS}" \
      --gzip_level 1 \
      2>&1 | tee "${log_file}"
  fi
done

echo "=============================="
echo "[DOCSEM DONE] $(date)"
echo "=============================="

# -----------------------
# Step 2: build official passage span indices
# -----------------------
# for d in "${domains[@]}"; do
#   in_zip="${IN_PASS_DIR}/${d}.jsonl.zip"
#   jsonl_name="${d}.jsonl"
#   out_index="${OUT_SPAN_DIR}/${d}_passage_spans.jsonl.gz"
#   log_file="${LOG_DIR}/${d}_span.log"

#   echo "=============================="
#   echo "[SPAN INDEX] ${d} $(date)"
#   echo "in_zip=${in_zip}"
#   echo "out=${out_index}"
#   echo "log=${log_file}"
#   echo "=============================="

#   if [[ -f "${out_index}" ]]; then
#     echo "[SKIP] exists: ${out_index}" | tee -a "${log_file}"
#   else
#     python3 other_scripts/build_passage_span_index.py \
#       --in_zip "${in_zip}" \
#       --jsonl_name "${jsonl_name}" \
#       --out_index "${out_index}" \
#       --gzip_level 1 \
#       2>&1 | tee "${log_file}"
#   fi
# done



# -----------------------
# Step 2: build official passage span indices
# -----------------------
for d in "${domains[@]}"; do
  in_zip="${IN_PASS_DIR}/${d}.jsonl.zip"
  jsonl_name="${d}.jsonl"
  out_index="${OUT_SPAN_DIR}/${d}_passage_spans.jsonl.gz"
  log_file="${LOG_DIR}/${d}_span.log"

  echo "=============================="
  echo "[SPAN INDEX] ${d} $(date)"
  echo "in_zip=${in_zip}"
  echo "out=${out_index}"
  echo "log=${log_file}"
  echo "=============================="

  if [[ -f "${out_index}" ]]; then
    echo "[SKIP] exists: ${out_index}" | tee -a "${log_file}"
  else
    python3 other_scripts/build_passage_span_index.py \
      --in_path "${in_zip}" \
      --jsonl_name "${jsonl_name}" \
      --out_path "${out_index}" \
      --max_warn 5 \
      2>&1 | tee "${log_file}"
  fi
done

echo "=============================="
echo "[SPAN INDEX DONE] $(date)"
echo "=============================="


# -----------------------
# Step 3: eval (docsem -> aggregate -> official eval)
# -----------------------
EVAL_LOG="${LOG_DIR}/eval_lastturn.log"
mkdir -p "$(dirname "${EVAL_LOG}")"

python3 other_scripts/eval_jasper_docsemchunk_to_passage_official.py \
  --task lastturn \
  --model_dir "${ST_MODEL}" \
  --base_dir "${ST_MODEL}" \
  --model_name jasper_docsem512_100 \
  --docsem_corpus_dir "${OUT_DOCSEM_DIR}" \
  --passage_span_dir "${OUT_SPAN_DIR}" \
  --cache_dir cache/doc_emb_jasper_docsem512_100 \
  --top_k_passages 50 \
  --top_k_chunks 1000 \
  --agg max \
  2>&1 | tee "${EVAL_LOG}"

echo "[DONE] $(date)"
