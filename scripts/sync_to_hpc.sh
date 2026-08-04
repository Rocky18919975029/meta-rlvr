#!/usr/bin/env bash
set -euo pipefail

HPC_HOST="${META_RLVR_HPC_HOST:-zhongal@hpc3login.hpc.hkust-gz.edu.cn}"
HPC_DIR="${META_RLVR_HPC_DIR:-meta-rlvr}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LOCAL_PROJECT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

echo "Creating offline artifact directories on ${HPC_HOST}:${HPC_DIR}"
ssh "${HPC_HOST}" \
  "mkdir -p '${HPC_DIR}/artifacts/data' '${HPC_DIR}/artifacts/models' '${HPC_DIR}/artifacts/wheelhouse' '${HPC_DIR}/logs' '${HPC_DIR}/outputs'"

echo "Synchronizing project code from ${LOCAL_PROJECT}"
rsync -azP \
  --exclude '.pytest_cache/' \
  --exclude '__pycache__/' \
  --exclude '*.egg-info/' \
  --exclude 'outputs/' \
  --exclude 'logs/' \
  "${LOCAL_PROJECT}/" \
  "${HPC_HOST}:${HPC_DIR}/"

if [[ -n "${META_RLVR_LOCAL_TRAIN_PARQUET:-}" ]]; then
  echo "Synchronizing DAPO training parquet"
  rsync -ahP \
    "${META_RLVR_LOCAL_TRAIN_PARQUET}" \
    "${HPC_HOST}:${HPC_DIR}/artifacts/data/DAPO-17k.parquet"
fi

if [[ -n "${META_RLVR_LOCAL_VALIDATION_PARQUET:-}" ]]; then
  echo "Synchronizing AIME24 validation parquet"
  rsync -ahP \
    "${META_RLVR_LOCAL_VALIDATION_PARQUET}" \
    "${HPC_HOST}:${HPC_DIR}/artifacts/data/AIME24.parquet"
fi

if [[ -n "${META_RLVR_LOCAL_MODEL_PATH:-}" ]]; then
  echo "Synchronizing offline Qwen model snapshot"
  rsync -ahP \
    "${META_RLVR_LOCAL_MODEL_PATH%/}/" \
    "${HPC_HOST}:${HPC_DIR}/artifacts/models/Qwen2.5-Math-7B/"
fi

if [[ -n "${META_RLVR_LOCAL_WHEELHOUSE:-}" ]]; then
  echo "Synchronizing Linux wheelhouse"
  rsync -ahP \
    "${META_RLVR_LOCAL_WHEELHOUSE%/}/" \
    "${HPC_HOST}:${HPC_DIR}/artifacts/wheelhouse/"
fi

echo "Synchronization complete: ${HPC_HOST}:${HPC_DIR}"
