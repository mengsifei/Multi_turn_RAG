#!/bin/bash
#SBATCH -J elser-4d
#SBATCH -c 4
#SBATCH -o logs/elser-4d.%j.out
#SBATCH -e logs/elser-4d.%j.err

mkdir -p logs

export ESDATA=${SCRATCH:-$PWD}/esdata_810
export ESLOG=${SCRATCH:-$PWD}/eslogs_810
mkdir -p "$ESDATA" "$ESLOG"

export ES_JAVA_OPTS="-Xms2g -Xmx2g"

./elasticsearch-8.10.0/bin/elasticsearch -d -p es.pid \
  -Ediscovery.type=single-node \
  -Expack.security.enabled=false \
  -Enetwork.host=127.0.0.1 \
  -Ehttp.port=9200 \
  -Epath.data="$ESDATA" \
  -Epath.logs="$ESLOG"

trap 'kill "$(cat es.pid)" 2>/dev/null || true' EXIT

for i in $(seq 1 120); do
  curl -sS http://127.0.0.1:9200 >/dev/null && break
  sleep 5
done
curl -sS http://127.0.0.1:9200 | head

python3 scripts/elser_mtrag_auto.py \
  --es_url http://127.0.0.1:9200 \
  --task lastturn \
  --domains "clapnq,fiqa,cloud,govt" \
  --index_tag cleaned \
  --corpus_override_dir human/retrieval_tasks_derived \
  --corpus_override_suffix .cleaned.jsonl \
  --blacklist_path reports/blacklist.valid.txt \
  --fast_bulk \
  --allocations 4 \
  --do_eval
