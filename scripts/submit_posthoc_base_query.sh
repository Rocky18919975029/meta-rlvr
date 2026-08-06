#!/usr/bin/env bash
set -euo pipefail

unset PYTORCH_CUDA_ALLOC_CONF
unset PYTORCH_ALLOC_CONF

export META_RLVR_PROJECT_DIR="${HOME}/meta-rlvr"
export META_RLVR_CONDA_ENV="verl"
export META_RLVR_MODEL_PATH="/data/user/zhongal/.cache/qwen2.5-math-7b-local"
export META_RLVR_REFERENCE_RUN_DIR="${HOME}/meta-rlvr/outputs/tiny-meta-only-b512-472864"
export META_RLVR_RUN_LABEL="posthoc-base-query-k32"
export SMOKE_GPUS=4
export SMOKE_LORA_RANK=8
export VLLM_GPU_MEMORY_UTILIZATION=0.30
export VLLM_MAX_MODEL_LEN=4096
export VLLM_MAX_NUM_SEQS=64
export VLLM_MAX_LORAS=1
export VLLM_MAX_CPU_LORAS=1
export VLLM_REQUEST_TIMEOUT=1800
export VLLM_CONTROL_TIMEOUT=120
export VLLM_STARTUP_TIMEOUT=600
export NCCL_DEBUG=WARN

export SLURM_PARTITION=acd_u
export SLURM_GRES=gpu:4
export SLURM_CPUS_PER_TASK=32
export SLURM_MEM=120G
export SLURM_TIME=01:00:00
export SLURM_EXCLUDE=ACD1-1

if [[ -z "${CONDA_EXE:-}" ]]; then
  CONDA_EXE=$(type -P conda)
  export CONDA_EXE
fi

cd "${META_RLVR_PROJECT_DIR}"
python src/meta_rlvr/vllm_preflight.py

mkdir -p logs outputs
sbatch_args=(
  --job-name="meta-rlvr-posthoc-base"
  --partition="${SLURM_PARTITION}"
  --gres="${SLURM_GRES}"
  --cpus-per-task="${SLURM_CPUS_PER_TASK}"
  --mem="${SLURM_MEM}"
  --time="${SLURM_TIME}"
  --exclude="${SLURM_EXCLUDE}"
  --output="${META_RLVR_PROJECT_DIR}/logs/${META_RLVR_RUN_LABEL}-%j.out"
  --error="${META_RLVR_PROJECT_DIR}/logs/${META_RLVR_RUN_LABEL}-%j.err"
  --export=ALL
)
submission=$(sbatch "${sbatch_args[@]}" scripts/slurm_posthoc_base_query.sbatch)
job_id="${submission##* }"
result_dir="${META_RLVR_PROJECT_DIR}/outputs/${META_RLVR_RUN_LABEL}-${job_id}"

echo "${submission}"
echo "stdout: ${META_RLVR_PROJECT_DIR}/logs/${META_RLVR_RUN_LABEL}-${job_id}.out"
echo "stderr/progress: ${META_RLVR_PROJECT_DIR}/logs/${META_RLVR_RUN_LABEL}-${job_id}.err"
echo "result: ${result_dir}/summary.json"
