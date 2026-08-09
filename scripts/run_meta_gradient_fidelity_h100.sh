#!/usr/bin/env bash
set -euo pipefail

: "${META_RLVR_PROJECT_DIR:?}"
: "${META_RLVR_MODEL_PATH:?}"
: "${META_RLVR_TRAIN_PARQUET:?}"
: "${META_RLVR_FIDELITY_CHECKPOINT:?}"

export META_RLVR_FIDELITY_OUTPUT_DIR="${META_RLVR_FIDELITY_OUTPUT_DIR:-${META_RLVR_PROJECT_DIR}/outputs/${META_RLVR_RUN_LABEL:-token-credit-fidelity}-${SLURM_JOB_ID:?}}"

cleanup() {
  if declare -F stop_meta_rlvr_vllm_servers >/dev/null 2>&1; then
    stop_meta_rlvr_vllm_servers
  fi
}
trap cleanup EXIT
trap 'echo "[$(date --iso-8601=seconds)] meta-gradient fidelity probe failed" >&2' ERR

cd "${META_RLVR_PROJECT_DIR}"
export PYTHONPATH="${META_RLVR_PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL="${TQDM_MININTERVAL:-5}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export SMOKE_GPUS=8

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:?}}"
fi
IFS=',' read -r -a visible_gpus <<<"${CUDA_VISIBLE_DEVICES}"
if [[ "${#visible_gpus[@]}" -ne 8 ]]; then
  echo "The fidelity probe requires exactly 8 visible GPUs." >&2
  exit 2
fi

source scripts/vllm_hybrid_servers.sh
start_meta_rlvr_vllm_servers

extra_probe_args=()
if [[ -n "${FIDELITY_INNER_OPTIMIZER:-}" ]]; then
  extra_probe_args+=(--inner-optimizer "${FIDELITY_INNER_OPTIMIZER}")
fi
if [[ -n "${FIDELITY_INNER_LEARNING_RATE:-}" ]]; then
  extra_probe_args+=(--inner-learning-rate "${FIDELITY_INNER_LEARNING_RATE}")
fi

echo "[$(date --iso-8601=seconds)] launching 8-rank fidelity probe"
accelerate launch \
  --config_file configs/accelerate_fsdp_8xh100.yaml \
  -m meta_rlvr.probe_meta_gradient_fidelity \
  --checkpoint "${META_RLVR_FIDELITY_CHECKPOINT}" \
  --dataset-parquet "${META_RLVR_TRAIN_PARQUET}" \
  --output-dir "${META_RLVR_FIDELITY_OUTPUT_DIR}" \
  --model "${META_RLVR_MODEL_PATH}" \
  --vllm-base-urls "${META_RLVR_VLLM_BASE_URLS}" \
  --problem-batch-size 8 \
  --group-size "${FIDELITY_GROUP_SIZE:-16}" \
  --max-new-tokens "${FIDELITY_MAX_NEW_TOKENS:-3072}" \
  --token-credit-max "${FIDELITY_TOKEN_CREDIT_MAX:-1.0}" \
  --seed "${FIDELITY_SEED:-42}" \
  --policy-micro-batch-size "${FIDELITY_POLICY_MICRO_BATCH_SIZE:-16}" \
  --confidence-micro-batch-size "${FIDELITY_CONFIDENCE_MICRO_BATCH_SIZE:-16}" \
  --policy-max-tokens-per-micro-batch "${FIDELITY_POLICY_MAX_TOKENS_PER_MICRO_BATCH:-4096}" \
  --confidence-max-tokens-per-micro-batch "${FIDELITY_CONFIDENCE_MAX_TOKENS_PER_MICRO_BATCH:-4096}" \
  --token-jvp-response-micro-batch-size "${FIDELITY_TOKEN_JVP_RESPONSE_MICRO_BATCH_SIZE:-4}" \
  --token-jvp-logprob-position-chunk-size "${FIDELITY_TOKEN_JVP_LOGPROB_POSITION_CHUNK_SIZE:-256}" \
  --request-timeout "${VLLM_REQUEST_TIMEOUT:-1800}" \
  --control-timeout "${VLLM_CONTROL_TIMEOUT:-120}" \
  --max-mean-absolute-logprob-delta "${FIDELITY_MAX_MEAN_ABSOLUTE_LOGPROB_DELTA:-0.03}" \
  "${extra_probe_args[@]}" \
  "$@"

echo "[$(date --iso-8601=seconds)] meta-gradient fidelity probe completed successfully"
echo "output=${META_RLVR_FIDELITY_OUTPUT_DIR}"
