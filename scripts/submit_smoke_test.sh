#!/usr/bin/env bash
set -euo pipefail

export META_RLVR_PROJECT_DIR="${META_RLVR_PROJECT_DIR:-${HOME}/meta-rlvr}"
export META_RLVR_VENV="${META_RLVR_VENV:-${HOME}/venvs/meta-rlvr}"
export META_RLVR_MODEL_PATH="${META_RLVR_MODEL_PATH:-/data/user/zhongal/.cache/qwen2.5-math-7b-local}"
export META_RLVR_TRAIN_PARQUET="${META_RLVR_TRAIN_PARQUET:-/data/user/zhongal/data/reschedule/DAPO-Math-17k.filtered.seed42.sample1536.parquet}"
export META_RLVR_VALIDATION_PARQUET="${META_RLVR_VALIDATION_PARQUET:-/data/user/zhongal/data/reschedule/aime24.parquet}"
export META_RLVR_OUTPUT_DIR="${META_RLVR_OUTPUT_DIR:-${META_RLVR_PROJECT_DIR}/outputs}"
export SLURM_PARTITION="${SLURM_PARTITION:-acd_u}"

SMOKE_GPUS="${SMOKE_GPUS:-4}"
if [[ "${SMOKE_GPUS}" != "4" && "${SMOKE_GPUS}" != "8" ]]; then
  echo "SMOKE_GPUS must be 4 or 8, got ${SMOKE_GPUS}" >&2
  exit 2
fi
export SMOKE_GPUS

LOG_DIR="${META_RLVR_PROJECT_DIR%/}/logs"
mkdir -p "${LOG_DIR}" "${META_RLVR_OUTPUT_DIR}"

SBATCH_ARGS=(
  --partition="${SLURM_PARTITION}"
  --gres="${SLURM_GRES:-gpu:${SMOKE_GPUS}}"
  --output="${LOG_DIR}/smoke-%j.out"
  --error="${LOG_DIR}/smoke-%j.err"
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
echo "stdout: ${LOG_DIR}/smoke-${job_id}.out"
echo "stderr/progress: ${LOG_DIR}/smoke-${job_id}.err"
echo "queue: squeue -j ${job_id}"
echo "progress: tail -f ${LOG_DIR}/smoke-${job_id}.err"
echo "metrics: tail -f ${LOG_DIR}/smoke-${job_id}.out"
