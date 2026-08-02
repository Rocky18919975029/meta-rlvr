#!/usr/bin/env bash
set -euo pipefail

export META_RLVR_PROJECT_DIR="${META_RLVR_PROJECT_DIR:-${HOME}/meta-rlvr}"
export META_RLVR_CONDA_ENV="${META_RLVR_CONDA_ENV:-verl}"
export META_RLVR_MODEL_PATH="${META_RLVR_MODEL_PATH:-/data/user/zhongal/.cache/qwen2.5-math-7b-local}"
export META_RLVR_TRAIN_PARQUET="${META_RLVR_TRAIN_PARQUET:-/data/user/zhongal/data/reschedule/DAPO-Math-17k.filtered.seed42.sample1536.parquet}"
export META_RLVR_VALIDATION_PARQUET="${META_RLVR_VALIDATION_PARQUET:-/data/user/zhongal/data/reschedule/aime24.parquet}"
export META_RLVR_OUTPUT_DIR="${META_RLVR_OUTPUT_DIR:-${META_RLVR_PROJECT_DIR}/outputs}"
export SLURM_PARTITION="${SLURM_PARTITION:-acd_u}"

export META_RLVR_RUN_LABEL="${META_RLVR_RUN_LABEL:-meta-meaningful}"
export SMOKE_MAX_STEPS=1
export SMOKE_SAVE_STEPS=1
export SMOKE_MAX_NEW_TOKENS=3072
export SMOKE_SUPPORT_GROUP_SIZE=16
export SMOKE_QUERY_GROUP_SIZE=16
export SMOKE_INNER_ITERATIONS=2
export SMOKE_OUTER_ITERATIONS=2
export SMOKE_GENERATION_MICRO_BATCH_SIZE=16
export SMOKE_POLICY_MICRO_BATCH_SIZE=1
export SMOKE_CONFIDENCE_MICRO_BATCH_SIZE=1
export SMOKE_LOG_ROLLOUTS=1
export SMOKE_ROLLOUT_BACKEND="${SMOKE_ROLLOUT_BACKEND:-vllm}"
export SMOKE_LORA_RANK="${SMOKE_LORA_RANK:-8}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.42}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
export VLLM_MAX_LORAS="${VLLM_MAX_LORAS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

if [[ -z "${CONDA_EXE:-}" ]]; then
  CONDA_EXE=$(type -P conda || true)
  if [[ -z "${CONDA_EXE}" || ! -x "${CONDA_EXE}" ]]; then
    echo "Cannot locate the conda executable in the submission shell." >&2
    exit 2
  fi
  export CONDA_EXE
fi

SMOKE_GPUS="${SMOKE_GPUS:-4}"
if [[ "${SMOKE_GPUS}" != "4" && "${SMOKE_GPUS}" != "8" ]]; then
  echo "SMOKE_GPUS must be 4 or 8, got ${SMOKE_GPUS}" >&2
  exit 2
fi
export SMOKE_GPUS

LOG_DIR="${META_RLVR_PROJECT_DIR%/}/logs"
mkdir -p "${LOG_DIR}" "${META_RLVR_OUTPUT_DIR}"

SBATCH_ARGS=(
  --job-name="meta-rlvr-${META_RLVR_RUN_LABEL}"
  --partition="${SLURM_PARTITION}"
  --gres="${SLURM_GRES:-gpu:${SMOKE_GPUS}}"
  --time="${SLURM_TIME:-08:00:00}"
  --output="${LOG_DIR}/${META_RLVR_RUN_LABEL}-%j.out"
  --error="${LOG_DIR}/${META_RLVR_RUN_LABEL}-%j.err"
  --export=ALL
)
if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
  SBATCH_ARGS+=(--account="${SLURM_ACCOUNT}")
fi
if [[ -n "${SLURM_QOS:-}" ]]; then
  SBATCH_ARGS+=(--qos="${SLURM_QOS}")
fi

submission=$(sbatch "${SBATCH_ARGS[@]}" scripts/slurm_smoke_test.sbatch)
job_id="${submission##* }"
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo "Could not parse Slurm job id from: ${submission}" >&2
  exit 2
fi

echo "${submission}"
echo "configuration: max_new_tokens=3072 K=16 inner=2 outer=2 rollout=${SMOKE_ROLLOUT_BACKEND} gpus=${SMOKE_GPUS}"
echo "stdout: ${LOG_DIR}/${META_RLVR_RUN_LABEL}-${job_id}.out"
echo "stderr/progress: ${LOG_DIR}/${META_RLVR_RUN_LABEL}-${job_id}.err"
echo "vLLM throughput: ${LOG_DIR}/vllm-${job_id}/gpu-*.log"
echo "queue: squeue -j ${job_id}"
echo "progress: tail -F ${LOG_DIR}/${META_RLVR_RUN_LABEL}-${job_id}.err"
echo "metrics: tail -F ${LOG_DIR}/${META_RLVR_RUN_LABEL}-${job_id}.out"
