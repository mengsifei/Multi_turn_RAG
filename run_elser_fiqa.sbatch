#!/bin/bash
#SBATCH -J elser-4d
#SBATCH -p normal
#SBATCH -c 16
#SBATCH -t 24:00:00
#SBATCH -o logs/elser-4d.%j.out
#SBATCH -e logs/elser-4d.%j.err

set -euo pipefail

ROOT="$HOME/mt-rag/mt-rag-benchmark"
cd "$ROOT"

echo "[INFO] JOBID=${SLURM_JOB_ID:-NA} HOST=$(hostname) PWD=$PWD"
echo "[INFO] SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-NA}"

# --- 端口：保证 < 65535，且每个 job 不冲突 ---
PORT=$((30000 + (${SLURM_JOB_ID:-1} % 20000)))   # 30000-49999
TPORT=$((PORT + 10000))                         # 40000-59999
export ES_URL="http://127.0.0.1:${PORT}"
echo "[INFO] ES_URL=$ES_URL transport_port=$TPORT"

# --- 持久化：断点续跑不重复 passage 推理（索引会保留） ---
export ESDATA="${SCRATCH:-$ROOT}/esdata_810_persist_cleaned"
export ESLOG="${SCRATCH:-$ROOT}/eslogs_810_${SLURM_JOB_ID}"
mkdir -p "$ESDATA" "$ESLOG"

# 强烈建议：单实例锁（防并发）
LOCKDIR="${ESDATA}/.lockdir"
if mkdir "$LOCKDIR" 2>/dev/null; then
  echo "[INFO] acquired lock $LOCKDIR"
else
  echo "[FATAL] ESDATA is locked by another job: $LOCKDIR"
  exit 1
fi

cleanup() {
  echo "[INFO] cleanup: stopping ES (if running)"
  if [[ -f "$ESLOG/es.pid" ]]; then
    kill "$(cat "$ESLOG/es.pid")" 2>/dev/null || true
  fi
  rmdir "$LOCKDIR" 2>/dev/null || true
}
trap cleanup EXIT

# JVM
export ES_JAVA_OPTS="-Xms4g -Xmx4g"
export PYTHONUNBUFFERED=1
ulimit -n 65535 || true

# --- 起 ES（显式端口 + 固定 cluster.name，适配 persist data） ---
"$ROOT/elasticsearch-8.10.0/bin/elasticsearch" -d -p "$ESLOG/es.pid" \
  -Ecluster.name="elser_persist_cleaned" \
  -Enode.name="node_${SLURM_JOB_ID}" \
  -Ediscovery.type=single-node \
  -Expack.security.enabled=false \
  -Enetwork.host=127.0.0.1 \
  -Ehttp.port="$PORT" \
  -Etransport.port="$TPORT" \
  -Epath.data="$ESDATA" \
  -Epath.logs="$ESLOG"

# --- 等 ES ready ---
echo "[INFO] waiting for ES to be ready..."
for i in $(seq 1 240); do
  if curl -fsS "$ES_URL" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

echo "[INFO] ES is up:"
curl -sS "$ES_URL" | head -n 30

# --- 开 trial（忽略失败） ---
curl -sS -X POST "$ES_URL/_license/start_trial?acknowledge=true" >/dev/null 2>&1 || true

# --- 注册/触发下载 ELSER（忽略已存在） ---
curl -sS -X PUT "$ES_URL/_ml/trained_models/.elser_model_1" \
  -H "Content-Type: application/json" \
  -d '{"input":{"field_names":["text_field"]}}' >/dev/null 2>&1 || true

# --- 启动 deployment（不要带 wait_for/timeout；你的 ES 会报未知参数） ---
echo "[INFO] starting ELSER deployment..."
curl -sS -X POST \
  "$ES_URL/_ml/trained_models/.elser_model_1/deployment/_start?deployment_id=for_search" \
  -H "Content-Type: application/json" \
  -d '{"number_of_allocations":4,"threads_per_allocation":1}' \
  >/dev/null

# --- 轮询直到 fully_allocated（关键：避免 bulk 直接全 409） ---
echo "[INFO] waiting for deployment to be ready..."
for i in $(seq 1 360); do  # 最多等 60min（360*10s）
  if curl -fsS "$ES_URL/_ml/trained_models/.elser_model_1/_stats" \
    | python3 - <<'PY'
import sys, json
j = json.load(sys.stdin)
stats = j.get("trained_model_stats") or []
ok = False
for s in stats:
  for d in (s.get("deployment_stats") or []):
    if d.get("deployment_id") == "for_search":
      state = (d.get("state") or "").lower()
      alloc = (d.get("allocation_status") or "").lower()
      # 兼容：有的版本只有 state/alloc 其中之一
      if state == "started" and (alloc in ("fully_allocated","started","") ):
        ok = True
print("OK" if ok else "NO")
sys.exit(0 if ok else 1)
PY
  then
    echo "[INFO] deployment is ready."
    break
  fi
  sleep 10
done

# 再确认一次，不然就直接 fail（避免 bulk 白跑）
if ! curl -fsS "$ES_URL/_ml/trained_models/.elser_model_1/_stats" | grep -q '"deployment_stats"'; then
  echo "[FATAL] deployment_stats missing -> ELSER not deployed"
  exit 1
fi

echo "[INFO] running 4-domain indexing + eval..."
python3 other_scripts/elser_mtrag_auto.py \
  --es_url "$ES_URL" \
  --task lastturn \
  --domains "clapnq,fiqa,cloud,govt" \
  --index_tag cleaned \
  --corpus_override_dir human/retrieval_tasks_derived \
  --corpus_override_suffix .cleaned.jsonl \
  --blacklist_path reports/blacklist.valid.txt \
  --fast_bulk \
  --allocations 4 \
  --do_eval
