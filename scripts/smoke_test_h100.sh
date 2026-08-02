#!/usr/bin/env bash
set -euo pipefail

on_error() {
  local exit_code=$?
  echo "[$(date --iso-8601=seconds)] smoke test failed at line ${BASH_LINENO[0]} (exit ${exit_code})" >&2
  exit "${exit_code}"
}
trap on_error ERR

: "${META_RLVR_PROJECT_DIR:?Set META_RLVR_PROJECT_DIR to the HPC project directory}"
: "${META_RLVR_MODEL_PATH:?Set META_RLVR_MODEL_PATH to the offline Qwen2.5-Math-7B directory}"
: "${META_RLVR_TRAIN_PARQUET:?Set META_RLVR_TRAIN_PARQUET to DAPO-17k.parquet}"
: "${META_RLVR_VALIDATION_PARQUET:?Set META_RLVR_VALIDATION_PARQUET to AIME24.parquet}"
: "${META_RLVR_OUTPUT_DIR:?Set META_RLVR_OUTPUT_DIR to an output base directory}"

SMOKE_GPUS="${SMOKE_GPUS:-4}"
case "${SMOKE_GPUS}" in
  4) ACCELERATE_CONFIG="configs/accelerate_fsdp_4xh100.yaml" ;;
  8) ACCELERATE_CONFIG="configs/accelerate_fsdp_8xh100.yaml" ;;
  *) echo "SMOKE_GPUS must be 4 or 8, got ${SMOKE_GPUS}" >&2; exit 2 ;;
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

RUN_TAG="${SLURM_JOB_ID:-manual-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${META_RLVR_OUTPUT_DIR%/}/smoke-${RUN_TAG}"
mkdir -p "${RUN_DIR}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL="${TQDM_MININTERVAL:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "[$(date --iso-8601=seconds)] starting Meta-RLVR distributed smoke test"
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
import sys

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
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available inside the Slurm allocation")
PY

echo "[$(date --iso-8601=seconds)] launching ${SMOKE_GPUS} distributed workers"
accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  -m meta_rlvr.train \
  --train-parquet "${META_RLVR_TRAIN_PARQUET}" \
  --validation-parquet "${META_RLVR_VALIDATION_PARQUET}" \
  --validation-max-problems "${SMOKE_VALIDATION_PROBLEMS:-${SMOKE_GPUS}}" \
  --output-dir "${RUN_DIR}" \
  --policy-model "${META_RLVR_MODEL_PATH}" \
  --confidence-model "${META_RLVR_MODEL_PATH}" \
  --attn-implementation sdpa \
  --max-steps 1 \
  --save-steps 1 \
  --support-group-size 2 \
  --query-group-size 2 \
  --generation-micro-batch-size 1 \
  --policy-micro-batch-size 1 \
  --confidence-micro-batch-size 1 \
  --max-new-tokens "${SMOKE_MAX_NEW_TOKENS:-128}" \
  --inner-iterations 1 \
  --outer-iterations 1 \
  "$@"

echo "[$(date --iso-8601=seconds)] smoke test completed successfully"
echo "output=${RUN_DIR}"
