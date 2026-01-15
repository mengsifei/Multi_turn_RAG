#!/usr/bin/env bash
set -euo pipefail

BASE=${1:-Jasper-Token-Compression-600M}
FT=${2:-jasper-ft-lastturn}

echo "[INFO] Syncing Jasper remote code"
echo "  BASE = $BASE"
echo "  FT   = $FT"

if [[ ! -d "$BASE" ]]; then
  echo "[ERROR] BASE dir not found: $BASE"
  exit 1
fi

if [[ ! -d "$FT" ]]; then
  echo "[ERROR] FT dir not found: $FT"
  exit 1
fi

# 复制所有可能的 remote-code 文件（安全：存在才拷）
for f in modeling_*.py configuration_*.py tokenization_*.py; do
  if ls "$BASE"/$f >/dev/null 2>&1; then
    cp -v "$BASE"/$f "$FT"/
  fi
done

echo "[OK] Remote code synced to $FT"
