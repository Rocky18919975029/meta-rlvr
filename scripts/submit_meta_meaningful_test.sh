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
export SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-1}"
export SMOKE_SAVE_STEPS="${SMOKE_SAVE_STEPS:-1}"
export SMOKE_MAX_NEW_TOKENS=3072
export SMOKE_SUPPORT_GROUP_SIZE=16
export SMOKE_QUERY_GROUP_SIZE=16
export SMOKE_VALIDATION_SUPPORT_GROUP_SIZE="${SMOKE_VALIDATION_SUPPORT_GROUP_SIZE:-${SMOKE_SUPPORT_GROUP_SIZE}}"
export SMOKE_VALIDATION_QUERY_GROUP_SIZE="${SMOKE_VALIDATION_QUERY_GROUP_SIZE:-${SMOKE_QUERY_GROUP_SIZE}}"
export SMOKE_INNER_ITERATIONS=2
export SMOKE_OUTER_ITERATIONS=2
export SMOKE_GENERATION_MICRO_BATCH_SIZE=16
export SMOKE_POLICY_MICRO_BATCH_SIZE="${SMOKE_POLICY_MICRO_BATCH_SIZE:-16}"
export SMOKE_CONFIDENCE_MICRO_BATCH_SIZE="${SMOKE_CONFIDENCE_MICRO_BATCH_SIZE:-16}"
export SMOKE_POLICY_MAX_TOKENS_PER_MICRO_BATCH="${SMOKE_POLICY_MAX_TOKENS_PER_MICRO_BATCH:-4096}"
export SMOKE_CONFIDENCE_MAX_TOKENS_PER_MICRO_BATCH="${SMOKE_CONFIDENCE_MAX_TOKENS_PER_MICRO_BATCH:-4096}"
export SMOKE_LOG_ROLLOUTS=1
export SMOKE_ROLLOUT_BACKEND="${SMOKE_ROLLOUT_BACKEND:-vllm}"
export SMOKE_LORA_RANK="${SMOKE_LORA_RANK:-8}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.30}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
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
export SMOKE_PROBLEM_BATCH_SIZE="${SMOKE_PROBLEM_BATCH_SIZE:-${SMOKE_GPUS}}"
if [[ ! "${SMOKE_PROBLEM_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || \
  (( SMOKE_PROBLEM_BATCH_SIZE < SMOKE_GPUS || SMOKE_PROBLEM_BATCH_SIZE % SMOKE_GPUS != 0 )); then
  echo "SMOKE_PROBLEM_BATCH_SIZE must be at least and divisible by SMOKE_GPUS." >&2
  exit 2
fi
export SMOKE_PROBLEM_MICRO_BATCH_SIZE="${SMOKE_PROBLEM_MICRO_BATCH_SIZE:-${SMOKE_PROBLEM_BATCH_SIZE}}"
if [[ ! "${SMOKE_PROBLEM_MICRO_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || \
  (( SMOKE_PROBLEM_MICRO_BATCH_SIZE < SMOKE_GPUS || \
     SMOKE_PROBLEM_MICRO_BATCH_SIZE > SMOKE_PROBLEM_BATCH_SIZE || \
     SMOKE_PROBLEM_MICRO_BATCH_SIZE % SMOKE_GPUS != 0 || \
     SMOKE_PROBLEM_BATCH_SIZE % SMOKE_PROBLEM_MICRO_BATCH_SIZE != 0 )); then
  echo "SMOKE_PROBLEM_MICRO_BATCH_SIZE must divide the problem batch and be divisible by SMOKE_GPUS." >&2
  exit 2
fi
export SMOKE_ROLLOUT_PROBLEM_BATCH_SIZE="${SMOKE_ROLLOUT_PROBLEM_BATCH_SIZE:-${SMOKE_PROBLEM_BATCH_SIZE}}"
if [[ ! "${SMOKE_ROLLOUT_PROBLEM_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || \
  (( SMOKE_ROLLOUT_PROBLEM_BATCH_SIZE < SMOKE_GPUS || \
     SMOKE_ROLLOUT_PROBLEM_BATCH_SIZE > SMOKE_PROBLEM_BATCH_SIZE || \
     SMOKE_ROLLOUT_PROBLEM_BATCH_SIZE % SMOKE_GPUS != 0 || \
     SMOKE_PROBLEM_BATCH_SIZE % SMOKE_ROLLOUT_PROBLEM_BATCH_SIZE != 0 )); then
  echo "SMOKE_ROLLOUT_PROBLEM_BATCH_SIZE must divide the problem batch and be divisible by SMOKE_GPUS." >&2
  exit 2
fi
local_rollout_problem_batch_size=$((SMOKE_ROLLOUT_PROBLEM_BATCH_SIZE / SMOKE_GPUS))
export VLLM_MAX_LORAS="${VLLM_MAX_LORAS:-${local_rollout_problem_batch_size}}"
export VLLM_MAX_CPU_LORAS="${VLLM_MAX_CPU_LORAS:-${VLLM_MAX_LORAS}}"
if [[ ! "${VLLM_MAX_LORAS}" =~ ^[1-9][0-9]*$ ]] || \
  [[ ! "${VLLM_MAX_CPU_LORAS}" =~ ^[1-9][0-9]*$ ]] || \
  (( VLLM_MAX_LORAS < local_rollout_problem_batch_size )) || \
  (( VLLM_MAX_CPU_LORAS < VLLM_MAX_LORAS )); then
  echo "vLLM LoRA capacities must cover the per-rank rollout problem batch ${local_rollout_problem_batch_size}." >&2
  exit 2
fi

SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-32}"
if [[ ! "${SLURM_CPUS_PER_TASK}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SLURM_CPUS_PER_TASK must be a positive integer." >&2
  exit 2
fi
if ((SLURM_CPUS_PER_TASK > 12 * SMOKE_GPUS)); then
  echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK} exceeds the cluster limit of 12 CPUs per GPU." >&2
  exit 2
fi
export SLURM_CPUS_PER_TASK

cd "${META_RLVR_PROJECT_DIR}"
if [[ "${SMOKE_ROLLOUT_BACKEND}" == "vllm" ]]; then
  export SLURM_EXCLUDE="${SLURM_EXCLUDE-ACD1-1}"
  echo "Checking vLLM environment before requesting GPUs..."
  python "${META_RLVR_PROJECT_DIR%/}/src/meta_rlvr/vllm_preflight.py"
fi

LOG_DIR="${META_RLVR_PROJECT_DIR%/}/logs"
mkdir -p "${LOG_DIR}" "${META_RLVR_OUTPUT_DIR}"

SBATCH_ARGS=(
  --job-name="meta-rlvr-${META_RLVR_RUN_LABEL}"
  --partition="${SLURM_PARTITION}"
  --gres="${SLURM_GRES:-gpu:${SMOKE_GPUS}}"
  --cpus-per-task="${SLURM_CPUS_PER_TASK}"
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
if [[ -n "${SLURM_EXCLUDE:-}" ]]; then
  SBATCH_ARGS+=(--exclude="${SLURM_EXCLUDE}")
fi
if [[ -n "${SLURM_MEM:-}" ]]; then
  SBATCH_ARGS+=(--mem="${SLURM_MEM}")
fi

submission=$(sbatch "${SBATCH_ARGS[@]}" scripts/slurm_smoke_test.sbatch)
job_id="${submission##* }"
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo "Could not parse Slurm job id from: ${submission}" >&2
  exit 2
fi

echo "${submission}"
echo "configuration: problems=${SMOKE_PROBLEM_BATCH_SIZE} rollout_problem_batch=${SMOKE_ROLLOUT_PROBLEM_BATCH_SIZE} gradient_problem_microbatch=${SMOKE_PROBLEM_MICRO_BATCH_SIZE} vjp_forward_batch=${SMOKE_FIRST_ORDER_VJP_FORWARD_BATCH_SIZE:-1} max_new_tokens=3072 train_K=${SMOKE_SUPPORT_GROUP_SIZE}/${SMOKE_QUERY_GROUP_SIZE} validation_K=${SMOKE_VALIDATION_SUPPORT_GROUP_SIZE}/${SMOKE_VALIDATION_QUERY_GROUP_SIZE} losses=${SMOKE_META_COEFFICIENT:-1.0}/${SMOKE_BCE_COEFFICIENT:-1.0}/${SMOKE_RANKING_COEFFICIENT:-1.0} inner=2 outer=2 rollout=${SMOKE_ROLLOUT_BACKEND} gpus=${SMOKE_GPUS} policy_batch=${SMOKE_POLICY_MICRO_BATCH_SIZE}/${SMOKE_POLICY_MAX_TOKENS_PER_MICRO_BATCH}tokens confidence_batch=${SMOKE_CONFIDENCE_MICRO_BATCH_SIZE}/${SMOKE_CONFIDENCE_MAX_TOKENS_PER_MICRO_BATCH}tokens deferred_sync=${SMOKE_DEFER_CONFIDENCE_GRADIENT_SYNC:-0} optimizer_offload=${SMOKE_OFFLOAD_CONFIDENCE_OPTIMIZER:-0} component_gradient_norms=${SMOKE_LOG_COMPONENT_GRADIENT_NORMS:-0} resume=${SMOKE_RESUME_FROM_CHECKPOINT:-none}"
echo "stdout: ${LOG_DIR}/${META_RLVR_RUN_LABEL}-${job_id}.out"
echo "stderr/progress: ${LOG_DIR}/${META_RLVR_RUN_LABEL}-${job_id}.err"
echo "vLLM throughput: ${LOG_DIR}/vllm-${job_id}/gpu-*.log"
echo "queue: squeue -j ${job_id}"
echo "progress: tail -F ${LOG_DIR}/${META_RLVR_RUN_LABEL}-${job_id}.err"
echo "metrics: tail -F ${LOG_DIR}/${META_RLVR_RUN_LABEL}-${job_id}.out"
