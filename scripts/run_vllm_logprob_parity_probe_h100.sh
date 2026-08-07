#!/usr/bin/env bash
set -euo pipefail

: "${META_RLVR_PROJECT_DIR:?}"
: "${META_RLVR_MODEL_PATH:?}"
: "${META_RLVR_TRAIN_PARQUET:?}"

cleanup() {
  if declare -F stop_meta_rlvr_vllm_servers >/dev/null 2>&1; then
    stop_meta_rlvr_vllm_servers
  fi
}
trap cleanup EXIT
trap 'echo "[$(date --iso-8601=seconds)] vLLM log-probability parity probe failed" >&2' ERR

cd "${META_RLVR_PROJECT_DIR}"
export PYTHONPATH="${META_RLVR_PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL="${TQDM_MININTERVAL:-5}"
export SMOKE_GPUS=1

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:?}}"
fi

source scripts/vllm_hybrid_servers.sh
start_meta_rlvr_vllm_servers

python -m meta_rlvr.probe_vllm_logprob_parity \
  --model "${META_RLVR_MODEL_PATH}" \
  --parquet "${META_RLVR_TRAIN_PARQUET}" \
  --vllm-base-url "${META_RLVR_VLLM_BASE_URLS}" \
  --group-size "${PARITY_GROUP_SIZE:-16}" \
  --max-new-tokens "${PARITY_MAX_NEW_TOKENS:-3072}" \
  --response-micro-batch-size "${PARITY_RESPONSE_MICRO_BATCH_SIZE:-4}" \
  --max-tokens-per-micro-batch "${PARITY_MAX_TOKENS_PER_MICRO_BATCH:-16384}" \
  --logprob-position-chunk-size "${PARITY_LOGPROB_POSITION_CHUNK_SIZE:-256}" \
  --adapter-root "/dev/shm/meta-rlvr-parity-${SLURM_JOB_ID}" \
  --request-timeout "${VLLM_REQUEST_TIMEOUT:-1800}" \
  --control-timeout "${VLLM_CONTROL_TIMEOUT:-120}"

echo "[$(date --iso-8601=seconds)] vLLM log-probability parity probe completed successfully"
