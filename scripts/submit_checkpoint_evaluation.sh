#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --checkpoint PATH --label NAME [--dataset PARQUET] [options]" >&2
  exit 2
}

CHECKPOINT=""
DATASET="/data/user/zhongal/data/reschedule/aime24.parquet"
LABEL=""
while (($#)); do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    --model) META_RLVR_MODEL_PATH="$2"; shift 2 ;;
    --support-group-size) EVAL_SUPPORT_GROUP_SIZE="$2"; shift 2 ;;
    --base-query-group-size) EVAL_BASE_QUERY_GROUP_SIZE="$2"; shift 2 ;;
    --adapted-query-group-size) EVAL_ADAPTED_QUERY_GROUP_SIZE="$2"; shift 2 ;;
    --seed) EVAL_SEED="$2"; shift 2 ;;
    --max-problems) EVAL_MAX_PROBLEMS="$2"; shift 2 ;;
    --max-new-tokens) EVAL_MAX_NEW_TOKENS="$2"; shift 2 ;;
    --inner-iterations) EVAL_INNER_ITERATIONS="$2"; shift 2 ;;
    --adaptation-rounds) EVAL_ADAPTATION_ROUNDS="$2"; shift 2 ;;
    --inner-learning-rate) EVAL_INNER_LEARNING_RATE="$2"; shift 2 ;;
    --local-rollout-batch-size) EVAL_LOCAL_ROLLOUT_BATCH_SIZE="$2"; shift 2 ;;
    --local-adaptation-batch-size) EVAL_LOCAL_ADAPTATION_BATCH_SIZE="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[[ -n "${CHECKPOINT}" && -n "${LABEL}" ]] || usage

export META_RLVR_PROJECT_DIR="${HOME}/meta-rlvr"
export META_RLVR_CONDA_ENV="verl"
export META_RLVR_MODEL_PATH="${META_RLVR_MODEL_PATH:-/data/user/zhongal/.cache/qwen2.5-math-7b-local}"
export META_RLVR_EVAL_CHECKPOINT="${CHECKPOINT}"
export META_RLVR_EVAL_DATASET="${DATASET}"
export META_RLVR_EVAL_LABEL="${LABEL}"

read -r EVAL_GPUS SMOKE_LORA_RANK < <(python - \
  "${CHECKPOINT}/trainer_state.json" \
  "$(dirname "${CHECKPOINT}")/run_config.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as stream:
    world_size = int(json.load(stream)["world_size"])
with open(sys.argv[2], encoding="utf-8") as stream:
    lora_rank = int(json.load(stream)["lora_rank"])
print(world_size, lora_rank)
PY
)
export EVAL_GPUS SMOKE_LORA_RANK
case "${EVAL_GPUS}" in
  1|4|8) ;;
  *) echo "Unsupported checkpoint world size: ${EVAL_GPUS}" >&2; exit 2 ;;
esac

export EVAL_SUPPORT_GROUP_SIZE="${EVAL_SUPPORT_GROUP_SIZE:-16}"
export EVAL_BASE_QUERY_GROUP_SIZE="${EVAL_BASE_QUERY_GROUP_SIZE:-32}"
export EVAL_ADAPTED_QUERY_GROUP_SIZE="${EVAL_ADAPTED_QUERY_GROUP_SIZE:-32}"
export EVAL_SEED="${EVAL_SEED:-42}"
export EVAL_LOCAL_ROLLOUT_BATCH_SIZE="${EVAL_LOCAL_ROLLOUT_BATCH_SIZE:-8}"
export EVAL_LOCAL_ADAPTATION_BATCH_SIZE="${EVAL_LOCAL_ADAPTATION_BATCH_SIZE:-2}"
export EVAL_MAX_PROBLEMS="${EVAL_MAX_PROBLEMS:-}"
export EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-}"
export EVAL_INNER_ITERATIONS="${EVAL_INNER_ITERATIONS:-}"
export EVAL_ADAPTATION_ROUNDS="${EVAL_ADAPTATION_ROUNDS:-1}"
export EVAL_INNER_LEARNING_RATE="${EVAL_INNER_LEARNING_RATE:-}"
export SMOKE_GPUS="${EVAL_GPUS}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.30}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
export VLLM_MAX_LORAS="${VLLM_MAX_LORAS:-${EVAL_LOCAL_ROLLOUT_BATCH_SIZE}}"
export VLLM_MAX_CPU_LORAS="${VLLM_MAX_CPU_LORAS:-${VLLM_MAX_LORAS}}"
export VLLM_REQUEST_TIMEOUT="${VLLM_REQUEST_TIMEOUT:-1800}"
export VLLM_CONTROL_TIMEOUT="${VLLM_CONTROL_TIMEOUT:-120}"
export VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-600}"
export TQDM_MININTERVAL="${TQDM_MININTERVAL:-5}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

export SLURM_PARTITION="${SLURM_PARTITION:-acd_u}"
export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-$((EVAL_GPUS * 8))}"
export SLURM_MEM="${SLURM_MEM:-400G}"
export SLURM_TIME="${SLURM_TIME:-08:00:00}"
export SLURM_EXCLUDE="${SLURM_EXCLUDE:-ACD1-1}"

if [[ -z "${CONDA_EXE:-}" ]]; then
  CONDA_EXE=$(type -P conda)
  export CONDA_EXE
fi

cd "${META_RLVR_PROJECT_DIR}"
python src/meta_rlvr/vllm_preflight.py
mkdir -p logs outputs

submission=$(sbatch \
  --job-name="meta-rlvr-eval" \
  --partition="${SLURM_PARTITION}" \
  --gres="gpu:${EVAL_GPUS}" \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  --mem="${SLURM_MEM}" \
  --time="${SLURM_TIME}" \
  --exclude="${SLURM_EXCLUDE}" \
  --output="${META_RLVR_PROJECT_DIR}/logs/${LABEL}-%j.out" \
  --error="${META_RLVR_PROJECT_DIR}/logs/${LABEL}-%j.err" \
  --export=ALL \
  scripts/slurm_checkpoint_evaluation.sbatch)
job_id="${submission##* }"
result_dir="${META_RLVR_PROJECT_DIR}/outputs/${LABEL}-${job_id}"

echo "${submission}"
echo "configuration: checkpoint=${CHECKPOINT} dataset=${DATASET} gpus=${EVAL_GPUS} support=${EVAL_SUPPORT_GROUP_SIZE} adaptation_rounds=${EVAL_ADAPTATION_ROUNDS} inner_iterations=${EVAL_INNER_ITERATIONS:-checkpoint-default} base_query=${EVAL_BASE_QUERY_GROUP_SIZE} adapted_query=${EVAL_ADAPTED_QUERY_GROUP_SIZE} seed=${EVAL_SEED}"
echo "stdout: ${META_RLVR_PROJECT_DIR}/logs/${LABEL}-${job_id}.out"
echo "stderr/progress: ${META_RLVR_PROJECT_DIR}/logs/${LABEL}-${job_id}.err"
echo "vLLM: ${META_RLVR_PROJECT_DIR}/logs/vllm-${job_id}/gpu-*.log"
echo "result: ${result_dir}/summary.json"
