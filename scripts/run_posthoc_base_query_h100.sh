#!/usr/bin/env bash
set -euo pipefail

: "${META_RLVR_PROJECT_DIR:?}"
: "${META_RLVR_MODEL_PATH:?}"
: "${META_RLVR_REFERENCE_RUN_DIR:?}"
: "${SMOKE_GPUS:?}"

META_RLVR_POSTHOC_OUTPUT_DIR="${META_RLVR_POSTHOC_OUTPUT_DIR:-${META_RLVR_PROJECT_DIR}/outputs/posthoc-base-query-k32-${SLURM_JOB_ID}}"
export META_RLVR_POSTHOC_OUTPUT_DIR

cleanup() {
  if declare -F stop_meta_rlvr_vllm_servers >/dev/null 2>&1; then
    stop_meta_rlvr_vllm_servers
  fi
}
trap cleanup EXIT

cd "${META_RLVR_PROJECT_DIR}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

source scripts/vllm_hybrid_servers.sh
start_meta_rlvr_vllm_servers

python -m meta_rlvr.posthoc_base_query \
  --reference-run-dir "${META_RLVR_REFERENCE_RUN_DIR}" \
  --reference-step 3 \
  --output-dir "${META_RLVR_POSTHOC_OUTPUT_DIR}" \
  --model "${META_RLVR_MODEL_PATH}" \
  --vllm-base-urls "${META_RLVR_VLLM_BASE_URLS}" \
  --request-timeout "${VLLM_REQUEST_TIMEOUT:-1800}"

echo "[$(date --iso-8601=seconds)] post-hoc base-query K=32 completed successfully"
echo "output=${META_RLVR_POSTHOC_OUTPUT_DIR}"
