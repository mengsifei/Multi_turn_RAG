#!/usr/bin/env bash
set -euo pipefail

IN_JSONL="outputs/hybrid_rrf10_denseJasper_spladev3_top200_rerank_bge_m3_cand100_ktop100.jsonl"
TASK="rewrite_gpt"
KEEP=100
ALPHAS=(0.05 0.08 0.1 0.12 0.15 0.2)

mkdir -p reports logs/alpha_sweep

OUT_TSV="reports/alpha_sweep_rrf10.tsv"
echo -e "alpha\tndcg10\tout_jsonl\tscore_jsonl\tlog" > "$OUT_TSV"

# ---- helper python: extract last weighted avg nDCG list's last value (nDCG@10)
EXTRACT_PY="logs/alpha_sweep/_extract_ndcg10.py"
cat > "$EXTRACT_PY" <<'PY'
import re, sys
path = sys.argv[1]
s = open(path, "r", encoding="utf-8", errors="replace").read()
m = re.findall(r"Weighted average nDCG:\s*\[([^\]]+)\]", s)
if not m:
    print("NA"); raise SystemExit(0)
nums = [float(x.strip()) for x in m[-1].split(",")]
print(nums[-1])
PY

best_alpha=""
best_ndcg="-1"

for a in "${ALPHAS[@]}"; do
  tag="hybrid_rrf10_denseJasper_spladev3_rerank_bge_m3_alpha${a}_top${KEEP}"
  OUT_JSONL="outputs/${tag}.jsonl"
  SCORE_JSONL="outputs/${tag}.score.jsonl"
  LOG="logs/alpha_sweep/${tag}.eval.out"

  echo "==== alpha=${a} ===="

  # 1) rescore
  python3 alpha_rescore.py \
    --in_jsonl "$IN_JSONL" \
    --out_jsonl "$OUT_JSONL" \
    --alpha "$a" \
    --keep_topk "$KEEP"

  # 2) eval (save stdout to log)
  python3 scripts/evaluation/run_retrieval_eval.py \
    --input_file "$OUT_JSONL" \
    --output_file "$SCORE_JSONL" \
    --model_name "$tag" \
    --task_name "$TASK" \
    | tee "$LOG" >/dev/null

  # 3) extract nDCG@10
  ndcg10="$(python3 "$EXTRACT_PY" "$LOG")"

  echo -e "${a}\t${ndcg10}\t${OUT_JSONL}\t${SCORE_JSONL}\t${LOG}" >> "$OUT_TSV"

  # 4) update best (ignore NA)
  if [[ "$ndcg10" != "NA" ]]; then
    is_better="$(awk -v x="$ndcg10" -v y="$best_ndcg" 'BEGIN{print (x>y)?1:0}')"
    if [[ "$is_better" == "1" ]]; then
      best_ndcg="$ndcg10"
      best_alpha="$a"
    fi
  fi

  echo "[alpha=${a}] nDCG@10=${ndcg10}  (best_alpha=${best_alpha} best_nDCG@10=${best_ndcg})"
done

echo
echo "===================="
echo "BEST alpha = ${best_alpha}"
echo "BEST nDCG@10 = ${best_ndcg}"
echo "Report TSV: ${OUT_TSV}"
echo "Best jsonl: outputs/hybrid_rrf10_denseJasper_spladev3_rerank_bge_m3_alpha${best_alpha}_top${KEEP}.jsonl"
echo "===================="
