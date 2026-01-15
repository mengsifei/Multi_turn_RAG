#!/bin/bash
#SBATCH --job-name=Jasper_rescorePassage
#SBATCH --account=proj_1782
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --output=logs/Jasper_original_cleaned_packed512_top2000_expand20k_KEEP20k_rescorePassage_top1000_%j.out
#SBATCH --error=logs/Jasper_original_cleaned_packed512_top2000_expand20k_KEEP20k_rescorePassage_top1000_%j.err
#SBATCH --time=08:00:00

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# 如果你需要激活环境，把下面两行取消注释并改成你的env名：
# source ~/.bashrc
# conda activate mtrag310

srun python3 rescore_with_passage_emb.py \
  --task lastturn \
  --in_jsonl outputs/Jasper_original_cleaned_packed512_top2000_expand20k_KEEP20k_lastturn.jsonl \
  --out_jsonl outputs/Jasper_original_cleaned_packed512_top2000_expand20k_KEEP20k_rescorePassage_top1000_lastturn.jsonl \
  --model_dir Jasper-Token-Compression-600M \
  --cache_dir cache/doc_emb_Jasper_original_cleaned_passageRescore_KEEP20k \
  --retrieval_tasks_root cleaned_dataset \
  --corpus_override_dir human/retrieval_tasks_derived \
  --corpus_override_suffix .cleaned.jsonl \
  --max_len 512 \
  --compression_ratio 0.3333 \
  --top_k_out 1000 \
  --local_files_only
