#!/usr/bin/env bash
set -euo pipefail

export META_RLVR_PROJECT_DIR="${META_RLVR_PROJECT_DIR:-${HOME}/meta-rlvr}"
export META_RLVR_CONDA_ENV="${META_RLVR_CONDA_ENV:-verl}"
export META_RLVR_MODEL_PATH="${META_RLVR_MODEL_PATH:-/data/user/zhongal/.cache/qwen2.5-math-7b-local}"
export META_RLVR_TRAIN_PARQUET="${META_RLVR_TRAIN_PARQUET:-/data/user/zhongal/data/reschedule/DAPO-Math-17k.filtered.seed42.sample1536.parquet}"
: "${META_RLVR_FIDELITY_CHECKPOINT:?Set the corrected token-confidence checkpoint.}"

export META_RLVR_RUN_LABEL="${META_RLVR_RUN_LABEL:-token-credit-fidelity}"
export FIDELITY_GROUP_SIZE="${FIDELITY_GROUP_SIZE:-16}"
export FIDELITY_MAX_NEW_TOKENS="${FIDELITY_MAX_NEW_TOKENS:-3072}"
export FIDELITY_TOKEN_CREDIT_MAX="${FIDELITY_TOKEN_CREDIT_MAX:-1.0}"
export FIDELITY_SEED="${FIDELITY_SEED:-42}"
export FIDELITY_POLICY_MICRO_BATCH_SIZE="${FIDELITY_POLICY_MICRO_BATCH_SIZE:-16}"
export FIDELITY_CONFIDENCE_MICRO_BATCH_SIZE="${FIDELITY_CONFIDENCE_MICRO_BATCH_SIZE:-16}"
export FIDELITY_POLICY_MAX_TOKENS_PER_MICRO_BATCH="${FIDELITY_POLICY_MAX_TOKENS_PER_MICRO_BATCH:-4096}"
export FIDELITY_CONFIDENCE_MAX_TOKENS_PER_MICRO_BATCH="${FIDELITY_CONFIDENCE_MAX_TOKENS_PER_MICRO_BATCH:-4096}"
export FIDELITY_TOKEN_JVP_RESPONSE_MICRO_BATCH_SIZE="${FIDELITY_TOKEN_JVP_RESPONSE_MICRO_BATCH_SIZE:-4}"
export FIDELITY_TOKEN_JVP_LOGPROB_POSITION_CHUNK_SIZE="${FIDELITY_TOKEN_JVP_LOGPROB_POSITION_CHUNK_SIZE:-256}"
export FIDELITY_MAX_MEAN_ABSOLUTE_LOGPROB_DELTA="${FIDELITY_MAX_MEAN_ABSOLUTE_LOGPROB_DELTA:-0.03}"
export FIDELITY_INNER_OPTIMIZER="${FIDELITY_INNER_OPTIMIZER:-}"
export FIDELITY_INNER_LEARNING_RATE="${FIDELITY_INNER_LEARNING_RATE:-}"

export SMOKE_GPUS=8
export SMOKE_LORA_RANK=8
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.30}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
export VLLM_MAX_LORAS=1
export VLLM_MAX_CPU_LORAS=1
export VLLM_REQUEST_TIMEOUT="${VLLM_REQUEST_TIMEOUT:-1800}"
export VLLM_CONTROL_TIMEOUT="${VLLM_CONTROL_TIMEOUT:-120}"
export VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-900}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

export SLURM_PARTITION="${SLURM_PARTITION:-acd_u}"
export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-64}"
export SLURM_MEM="${SLURM_MEM:-400G}"
export SLURM_TIME="${SLURM_TIME:-03:00:00}"
export SLURM_EXCLUDE="${SLURM_EXCLUDE:-ACD1-1}"

if [[ -z "${CONDA_EXE:-}" ]]; then
  CONDA_EXE=$(type -P conda)
  export CONDA_EXE
fi

cd "${META_RLVR_PROJECT_DIR}"
python src/meta_rlvr/vllm_preflight.py
python src/meta_rlvr/fidelity_preflight.py \
  --checkpoint "${META_RLVR_FIDELITY_CHECKPOINT}" \
  --expected-world-size "${SMOKE_GPUS}"
mkdir -p logs outputs

submission=$(sbatch \
  --job-name="meta-rlvr-fidelity" \
  --partition="${SLURM_PARTITION}" \
  --gres="gpu:8" \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  --mem="${SLURM_MEM}" \
  --time="${SLURM_TIME}" \
  --exclude="${SLURM_EXCLUDE}" \
  --output="${META_RLVR_PROJECT_DIR}/logs/${META_RLVR_RUN_LABEL}-%j.out" \
  --error="${META_RLVR_PROJECT_DIR}/logs/${META_RLVR_RUN_LABEL}-%j.err" \
  --export=ALL \
  scripts/slurm_meta_gradient_fidelity.sbatch)
job_id="${submission##* }"

echo "${submission}"
echo "configuration: problems=8 support_K=${FIDELITY_GROUP_SIZE} query_K=${FIDELITY_GROUP_SIZE} max_new_tokens=${FIDELITY_MAX_NEW_TOKENS} credit_max=${FIDELITY_TOKEN_CREDIT_MAX} inner_optimizer=${FIDELITY_INNER_OPTIMIZER:-checkpoint-default} inner_lr=${FIDELITY_INNER_LEARNING_RATE:-checkpoint-default} checkpoint=${META_RLVR_FIDELITY_CHECKPOINT}"
echo "stdout: ${META_RLVR_PROJECT_DIR}/logs/${META_RLVR_RUN_LABEL}-${job_id}.out"
echo "stderr/progress: ${META_RLVR_PROJECT_DIR}/logs/${META_RLVR_RUN_LABEL}-${job_id}.err"
echo "vLLM: ${META_RLVR_PROJECT_DIR}/logs/vllm-${job_id}/gpu-*.log"
echo "output: ${META_RLVR_PROJECT_DIR}/outputs/${META_RLVR_RUN_LABEL}-${job_id}"
echo "queue: squeue -j ${job_id}"
echo "progress: tail -F ${META_RLVR_PROJECT_DIR}/logs/${META_RLVR_RUN_LABEL}-${job_id}.err"
echo "metrics: tail -F ${META_RLVR_PROJECT_DIR}/logs/${META_RLVR_RUN_LABEL}-${job_id}.out"
