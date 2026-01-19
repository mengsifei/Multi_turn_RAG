#!/bin/bash
#SBATCH -J elser-4d
#SBATCH -p normal
#SBATCH -c 16
#SBATCH -t 30-00:00:00
#SBATCH -o logs/elser-4d.%j.out
#SBATCH -e logs/elser-4d.%j.err

set -euo pipefail
cd $HOME/mt-rag/mt-rag-benchmark
mkdir -p logs

PORT=$((9200 + (${SLURM_JOB_ID:-0} % 1000)))
export ESDATA=${SCRATCH:-$PWD}/esdata_810_${SLURM_JOB_ID}
export ESLOG=${SCRATCH:-$PWD}/eslogs_810_${SLURM_JOB_ID}
mkdir -p "$ESDATA" "$ESLOG"

export ES_JAVA_OPTS="-Xms4g -Xmx4g"

./elasticsearch-8.10.0/bin/elasticsearch -d -p es.pid \
  -Ediscovery.type=single-node \
  -Expack.security.enabled=false \
  -Enetwork.host=127.0.0.1 \
  -Ehttp.port=$PORT \
  -Epath.data="$ESDATA" \
  -Epath.logs="$ESLOG"

trap 'test -f es.pid && kill "$(cat es.pid)" 2>/dev/null || true' EXIT

for i in $(seq 1 120); do
  curl -sS http://127.0.0.1:$PORT >/dev/null && break
  sleep 5
done
if ! curl -sS http://127.0.0.1:$PORT >/dev/null; then
  echo "[FATAL] ES did not start"
  tail -n 200 "$ESLOG"/* 2>/dev/null || true
  exit 1
fi
curl -sS http://127.0.0.1:$PORT | head

python3 other_scripts/elser_mtrag_auto.py \
  --es_url http://127.0.0.1:$PORT \
  --task lastturn \
  --domains "clapnq,fiqa,cloud,govt" \
  --index_tag cleaned \
  --corpus_override_dir human/retrieval_tasks_derived \
  --corpus_override_suffix .cleaned.jsonl \
  --blacklist_path reports/blacklist.valid.txt \
  --fast_bulk \
  --allocations 4 \
  --do_eval
