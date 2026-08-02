#!/usr/bin/env bash
set -euo pipefail

: "${META_RLVR_PROJECT_DIR:?Set META_RLVR_PROJECT_DIR to the absolute HPC project directory}"
: "${META_RLVR_VENV:?Set META_RLVR_VENV to the absolute Linux virtual environment directory}"
: "${META_RLVR_MODEL_PATH:?Set META_RLVR_MODEL_PATH to the offline model directory}"
: "${META_RLVR_TRAIN_PARQUET:?Set META_RLVR_TRAIN_PARQUET to DAPO-17k.parquet}"
: "${META_RLVR_VALIDATION_PARQUET:?Set META_RLVR_VALIDATION_PARQUET to AIME24.parquet}"
: "${META_RLVR_OUTPUT_DIR:?Set META_RLVR_OUTPUT_DIR to an output base directory}"
: "${SLURM_PARTITION:?Set SLURM_PARTITION to the cluster's H100 partition}"

SMOKE_GPUS="${SMOKE_GPUS:-4}"
if [[ "${SMOKE_GPUS}" != "4" && "${SMOKE_GPUS}" != "8" ]]; then
  echo "SMOKE_GPUS must be 4 or 8, got ${SMOKE_GPUS}" >&2
  exit 2
fi

LOG_DIR="${META_RLVR_PROJECT_DIR%/}/logs"
mkdir -p "${LOG_DIR}" "${META_RLVR_OUTPUT_DIR}"

SBATCH_ARGS=(
  --partition="${SLURM_PARTITION}"
  --gres="${SLURM_GRES:-gpu:h100:${SMOKE_GPUS}}"
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
