#!/usr/bin/env bash
set -euo pipefail

: "${META_RLVR_PROJECT_DIR:?}"
: "${META_RLVR_MODEL_PATH:?}"
: "${META_RLVR_EVAL_CHECKPOINT:?}"
: "${META_RLVR_EVAL_DATASET:?}"
: "${META_RLVR_EVAL_LABEL:?}"
: "${EVAL_GPUS:?}"

META_RLVR_EVAL_OUTPUT_DIR="${META_RLVR_PROJECT_DIR}/outputs/${META_RLVR_EVAL_LABEL}-${SLURM_JOB_ID}"
export META_RLVR_EVAL_OUTPUT_DIR

case "${EVAL_GPUS}" in
  1) ACCELERATE_CONFIG="configs/accelerate_single_h100.yaml" ;;
  4) ACCELERATE_CONFIG="configs/accelerate_fsdp_4xh100.yaml" ;;
  8) ACCELERATE_CONFIG="configs/accelerate_fsdp_8xh100.yaml" ;;
  *) echo "EVAL_GPUS must be 1, 4, or 8." >&2; exit 2 ;;
esac

cleanup() {
  if declare -F stop_meta_rlvr_vllm_servers >/dev/null 2>&1; then
    stop_meta_rlvr_vllm_servers
  fi
}
trap cleanup EXIT
trap 'echo "[$(date --iso-8601=seconds)] checkpoint evaluation failed" >&2' ERR

cd "${META_RLVR_PROJECT_DIR}"
export PYTHONPATH="${META_RLVR_PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL="${TQDM_MININTERVAL:-5}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:?}}"
fi

source scripts/vllm_hybrid_servers.sh
start_meta_rlvr_vllm_servers

EXTRA_ARGS=()
if [[ -n "${EVAL_MAX_PROBLEMS:-}" ]]; then
  EXTRA_ARGS+=(--max-problems "${EVAL_MAX_PROBLEMS}")
fi
if [[ -n "${EVAL_MAX_NEW_TOKENS:-}" ]]; then
  EXTRA_ARGS+=(--max-new-tokens "${EVAL_MAX_NEW_TOKENS}")
fi
if [[ -n "${EVAL_INNER_ITERATIONS:-}" ]]; then
  EXTRA_ARGS+=(--inner-iterations "${EVAL_INNER_ITERATIONS}")
fi
if [[ -n "${EVAL_INNER_LEARNING_RATE:-}" ]]; then
  EXTRA_ARGS+=(--inner-learning-rate "${EVAL_INNER_LEARNING_RATE}")
fi

setsid accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  -m meta_rlvr.evaluate_checkpoint \
  --checkpoint "${META_RLVR_EVAL_CHECKPOINT}" \
  --dataset-parquet "${META_RLVR_EVAL_DATASET}" \
  --output-dir "${META_RLVR_EVAL_OUTPUT_DIR}" \
  --model "${META_RLVR_MODEL_PATH}" \
  --vllm-base-urls "${META_RLVR_VLLM_BASE_URLS}" \
  --support-group-size "${EVAL_SUPPORT_GROUP_SIZE:-16}" \
  --base-query-group-size "${EVAL_BASE_QUERY_GROUP_SIZE:-32}" \
  --adapted-query-group-size "${EVAL_ADAPTED_QUERY_GROUP_SIZE:-32}" \
  --seed "${EVAL_SEED:-42}" \
  --local-rollout-batch-size "${EVAL_LOCAL_ROLLOUT_BATCH_SIZE:-8}" \
  --local-adaptation-batch-size "${EVAL_LOCAL_ADAPTATION_BATCH_SIZE:-2}" \
  --request-timeout "${VLLM_REQUEST_TIMEOUT:-1800}" \
  --control-timeout "${VLLM_CONTROL_TIMEOUT:-120}" \
  "${EXTRA_ARGS[@]}"

echo "[$(date --iso-8601=seconds)] checkpoint evaluation completed successfully"
echo "output=${META_RLVR_EVAL_OUTPUT_DIR}"
