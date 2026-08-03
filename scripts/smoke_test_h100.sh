#!/usr/bin/env bash
set -euo pipefail

META_RLVR_TRAINER_PID=""

cleanup_all() {
  if [[ -n "${META_RLVR_TRAINER_PID:-}" ]] && \
    { kill -0 -- "-${META_RLVR_TRAINER_PID}" 2>/dev/null || kill -0 "${META_RLVR_TRAINER_PID}" 2>/dev/null; }; then
    kill -TERM -- "-${META_RLVR_TRAINER_PID}" 2>/dev/null || \
      kill -TERM "${META_RLVR_TRAINER_PID}" 2>/dev/null || true
    wait "${META_RLVR_TRAINER_PID}" 2>/dev/null || true
  fi
  if declare -F stop_meta_rlvr_vllm_servers >/dev/null 2>&1; then
    stop_meta_rlvr_vllm_servers
  fi
}

on_error() {
  local exit_code=$?
  echo "[$(date --iso-8601=seconds)] smoke test failed at line ${BASH_LINENO[0]} (exit ${exit_code})" >&2
  exit "${exit_code}"
}
trap on_error ERR
trap cleanup_all EXIT
trap 'echo "[$(date --iso-8601=seconds)] smoke test terminated" >&2; exit 143' TERM
trap 'echo "[$(date --iso-8601=seconds)] smoke test interrupted" >&2; exit 130' INT

: "${META_RLVR_PROJECT_DIR:?Set META_RLVR_PROJECT_DIR to the HPC project directory}"
: "${META_RLVR_MODEL_PATH:?Set META_RLVR_MODEL_PATH to the offline Qwen2.5-Math-7B directory}"
: "${META_RLVR_TRAIN_PARQUET:?Set META_RLVR_TRAIN_PARQUET to DAPO-17k.parquet}"
: "${META_RLVR_VALIDATION_PARQUET:?Set META_RLVR_VALIDATION_PARQUET to AIME24.parquet}"
: "${META_RLVR_OUTPUT_DIR:?Set META_RLVR_OUTPUT_DIR to an output base directory}"

SMOKE_GPUS="${SMOKE_GPUS:-4}"
case "${SMOKE_GPUS}" in
  1) ACCELERATE_CONFIG="configs/accelerate_single_h100.yaml" ;;
  4) ACCELERATE_CONFIG="configs/accelerate_fsdp_4xh100.yaml" ;;
  8) ACCELERATE_CONFIG="configs/accelerate_fsdp_8xh100.yaml" ;;
  *) echo "SMOKE_GPUS must be 1, 4, or 8, got ${SMOKE_GPUS}" >&2; exit 2 ;;
esac

for required_path in \
  "${META_RLVR_MODEL_PATH}" \
  "${META_RLVR_TRAIN_PARQUET}" \
  "${META_RLVR_VALIDATION_PARQUET}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Required offline artifact does not exist: ${required_path}" >&2
    exit 2
  fi
done

cd "${META_RLVR_PROJECT_DIR}"
if [[ ! -f "${ACCELERATE_CONFIG}" ]]; then
  echo "Missing Accelerate config: ${META_RLVR_PROJECT_DIR}/${ACCELERATE_CONFIG}" >&2
  exit 2
fi

RUN_LABEL="${META_RLVR_RUN_LABEL:-smoke}"
RUN_TAG="${SLURM_JOB_ID:-manual-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${META_RLVR_OUTPUT_DIR%/}/${RUN_LABEL}-${RUN_TAG}"
mkdir -p "${RUN_DIR}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL="${TQDM_MININTERVAL:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "[$(date --iso-8601=seconds)] starting Meta-RLVR ${RUN_LABEL} test"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "host=$(hostname)"
echo "project=${META_RLVR_PROJECT_DIR}"
echo "model=${META_RLVR_MODEL_PATH}"
echo "train_data=${META_RLVR_TRAIN_PARQUET}"
echo "validation_data=${META_RLVR_VALIDATION_PARQUET}"
echo "output=${RUN_DIR}"
echo "gpus=${SMOKE_GPUS}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

nvidia-smi
python - <<'PY'
import platform
import os
import sys
from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version

import accelerate
import datasets
import peft
import torch
import transformers

print(f"python={sys.version}")
print(f"platform={platform.platform()}")
print(f"torch={torch.__version__}")
print(f"cuda_runtime={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
print(f"transformers={transformers.__version__}")
print(f"peft={peft.__version__}")
print(f"accelerate={accelerate.__version__}")
print(f"datasets={datasets.__version__}")
if os.environ.get("SMOKE_ROLLOUT_BACKEND", "transformers") == "vllm":
    from meta_rlvr.vllm_preflight import validate_vllm_environment

    for package, package_version in validate_vllm_environment().items():
        print(f"{package}={package_version}")
try:
    torchao_version = version("torchao")
except PackageNotFoundError:
    print("torchao=not installed (expected for BF16 LoRA)")
else:
    print(f"torchao={torchao_version}")
    if Version(torchao_version) < Version("0.16.0"):
        raise RuntimeError(
            "PEFT 0.19 requires torchao >= 0.16 when torchao is installed. "
            "This BF16 LoRA run does not use torchao; remove it from the "
            "dedicated Meta-RLVR environment."
        )
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available inside the Slurm allocation")
PY

echo "[$(date --iso-8601=seconds)] launching ${SMOKE_GPUS} distributed workers"
EXTRA_TRAIN_ARGS=()
if [[ "${SMOKE_ROLLOUT_BACKEND:-transformers}" == "vllm" ]]; then
  source scripts/vllm_hybrid_servers.sh
  start_meta_rlvr_vllm_servers
  EXTRA_TRAIN_ARGS+=(
    --rollout-backend vllm
    --vllm-base-urls "${META_RLVR_VLLM_BASE_URLS}"
    --vllm-adapter-root "${VLLM_ADAPTER_ROOT:-/dev/shm}"
    --vllm-request-timeout "${VLLM_REQUEST_TIMEOUT:-900}"
    --vllm-control-timeout "${VLLM_CONTROL_TIMEOUT:-120}"
  )
elif [[ "${SMOKE_ROLLOUT_BACKEND:-transformers}" != "transformers" ]]; then
  echo "SMOKE_ROLLOUT_BACKEND must be transformers or vllm." >&2
  exit 2
fi
if [[ "${SMOKE_LOG_ROLLOUTS:-0}" == "1" ]]; then
  EXTRA_TRAIN_ARGS+=(--log-rollouts)
fi
if [[ "${SMOKE_ROLLOUT_ONLY:-0}" == "1" ]]; then
  EXTRA_TRAIN_ARGS+=(--rollout-only)
fi
if ! command -v setsid >/dev/null 2>&1; then
  echo "setsid is required to supervise the distributed trainer." >&2
  exit 2
fi
setsid accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  -m meta_rlvr.train \
  --train-parquet "${META_RLVR_TRAIN_PARQUET}" \
  --validation-parquet "${META_RLVR_VALIDATION_PARQUET}" \
  --validation-max-problems "${SMOKE_VALIDATION_PROBLEMS:-${SMOKE_GPUS}}" \
  --output-dir "${RUN_DIR}" \
  --policy-model "${META_RLVR_MODEL_PATH}" \
  --confidence-model "${META_RLVR_MODEL_PATH}" \
  --attn-implementation sdpa \
  --max-steps "${SMOKE_MAX_STEPS:-1}" \
  --save-steps "${SMOKE_SAVE_STEPS:-1}" \
  --support-group-size "${SMOKE_SUPPORT_GROUP_SIZE:-2}" \
  --query-group-size "${SMOKE_QUERY_GROUP_SIZE:-2}" \
  --generation-micro-batch-size "${SMOKE_GENERATION_MICRO_BATCH_SIZE:-1}" \
  --policy-micro-batch-size "${SMOKE_POLICY_MICRO_BATCH_SIZE:-1}" \
  --confidence-micro-batch-size "${SMOKE_CONFIDENCE_MICRO_BATCH_SIZE:-1}" \
  --max-new-tokens "${SMOKE_MAX_NEW_TOKENS:-128}" \
  --inner-iterations "${SMOKE_INNER_ITERATIONS:-1}" \
  --outer-iterations "${SMOKE_OUTER_ITERATIONS:-1}" \
  "${EXTRA_TRAIN_ARGS[@]}" \
  "$@" &
META_RLVR_TRAINER_PID="$!"
set +e
wait "${META_RLVR_TRAINER_PID}"
trainer_status=$?
set -e
META_RLVR_TRAINER_PID=""
if [[ "${trainer_status}" -ne 0 ]]; then
  echo "Distributed trainer exited with status ${trainer_status}." >&2
  exit "${trainer_status}"
fi

echo "[$(date --iso-8601=seconds)] ${RUN_LABEL} test completed successfully"
echo "output=${RUN_DIR}"
