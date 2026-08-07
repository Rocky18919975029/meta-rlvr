#!/usr/bin/env bash
set -euo pipefail

export META_RLVR_PROJECT_DIR="${HOME}/meta-rlvr"
export META_RLVR_CONDA_ENV="verl"
export META_RLVR_MODEL_PATH="/data/user/zhongal/.cache/qwen2.5-math-7b-local"
export META_RLVR_TRAIN_PARQUET="/data/user/zhongal/data/reschedule/DAPO-Math-17k.filtered.seed42.sample1536.parquet"

export PARITY_GROUP_SIZE="${PARITY_GROUP_SIZE:-16}"
export PARITY_MAX_NEW_TOKENS="${PARITY_MAX_NEW_TOKENS:-3072}"
export PARITY_RESPONSE_MICRO_BATCH_SIZE="${PARITY_RESPONSE_MICRO_BATCH_SIZE:-4}"
export PARITY_MAX_TOKENS_PER_MICRO_BATCH="${PARITY_MAX_TOKENS_PER_MICRO_BATCH:-16384}"
export PARITY_LOGPROB_POSITION_CHUNK_SIZE="${PARITY_LOGPROB_POSITION_CHUNK_SIZE:-256}"

export SMOKE_GPUS=1
export SMOKE_LORA_RANK=8
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.30}"
export VLLM_MAX_MODEL_LEN=4096
export VLLM_MAX_NUM_SEQS=16
export VLLM_MAX_LORAS=1
export VLLM_MAX_CPU_LORAS=1
export VLLM_REQUEST_TIMEOUT="${VLLM_REQUEST_TIMEOUT:-1800}"
export VLLM_CONTROL_TIMEOUT="${VLLM_CONTROL_TIMEOUT:-120}"
export VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-600}"
export NCCL_DEBUG=WARN

export SLURM_PARTITION="${SLURM_PARTITION:-acd_u}"
export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-12}"
export SLURM_MEM="${SLURM_MEM:-120G}"
export SLURM_TIME="${SLURM_TIME:-01:00:00}"
export SLURM_EXCLUDE="${SLURM_EXCLUDE:-ACD1-1}"

if [[ -z "${CONDA_EXE:-}" ]]; then
  CONDA_EXE=$(type -P conda)
  export CONDA_EXE
fi

cd "${META_RLVR_PROJECT_DIR}"
python src/meta_rlvr/vllm_preflight.py
mkdir -p logs

submission=$(sbatch \
  --job-name="meta-rlvr-logprob-parity" \
  --partition="${SLURM_PARTITION}" \
  --gres="gpu:1" \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  --mem="${SLURM_MEM}" \
  --time="${SLURM_TIME}" \
  --exclude="${SLURM_EXCLUDE}" \
  --output="${META_RLVR_PROJECT_DIR}/logs/vllm-logprob-parity-%j.out" \
  --error="${META_RLVR_PROJECT_DIR}/logs/vllm-logprob-parity-%j.err" \
  --export=ALL \
  scripts/slurm_vllm_logprob_parity_probe.sbatch)
job_id="${submission##* }"

echo "${submission}"
echo "configuration: K=${PARITY_GROUP_SIZE} max_new_tokens=${PARITY_MAX_NEW_TOKENS} response_micro_batch=${PARITY_RESPONSE_MICRO_BATCH_SIZE}"
echo "stdout: ${META_RLVR_PROJECT_DIR}/logs/vllm-logprob-parity-${job_id}.out"
echo "stderr/progress: ${META_RLVR_PROJECT_DIR}/logs/vllm-logprob-parity-${job_id}.err"
echo "vLLM: ${META_RLVR_PROJECT_DIR}/logs/vllm-${job_id}/gpu-0.log"
echo "queue: squeue -j ${job_id}"
