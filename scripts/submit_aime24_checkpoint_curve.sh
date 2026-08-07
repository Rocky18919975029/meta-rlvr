#!/usr/bin/env bash
set -euo pipefail

export META_RLVR_PROJECT_DIR="${HOME}/meta-rlvr"
export META_RLVR_CONDA_ENV="verl"
export META_RLVR_MODEL_PATH="/data/user/zhongal/.cache/qwen2.5-math-7b-local"
export META_RLVR_EVAL_DATASET="${META_RLVR_EVAL_DATASET:-/data/user/zhongal/data/reschedule/aime24.parquet}"
export SEQUENCE_RUN_DIR="${SEQUENCE_RUN_DIR:-${META_RLVR_PROJECT_DIR}/outputs/tiny-meta-only-b512-472864}"
export TOKEN_RUN_DIR="${TOKEN_RUN_DIR:-${META_RLVR_PROJECT_DIR}/outputs/tiny-token-meta-only-b512-474799}"
export CURVE_PREFIX="${CURVE_PREFIX:-aime24-curve-$(date +%Y%m%d-%H%M%S)}"

export EVAL_GPUS=8
export SMOKE_GPUS=8
export EVAL_SUPPORT_GROUP_SIZE=16
export EVAL_BASE_QUERY_GROUP_SIZE=32
export EVAL_ADAPTED_QUERY_GROUP_SIZE=32
export EVAL_SEED=42
export EVAL_MAX_PROBLEMS=30
export EVAL_MAX_NEW_TOKENS=3072
export EVAL_INNER_ITERATIONS=2
export EVAL_ADAPTATION_ROUNDS=1
export EVAL_LOCAL_ROLLOUT_BATCH_SIZE=8
export EVAL_LOCAL_ADAPTATION_BATCH_SIZE=2

export VLLM_GPU_MEMORY_UTILIZATION=0.30
export VLLM_MAX_MODEL_LEN=4096
export VLLM_MAX_NUM_SEQS=64
export VLLM_MAX_LORAS=8
export VLLM_MAX_CPU_LORAS=8
export VLLM_REQUEST_TIMEOUT=1800
export VLLM_CONTROL_TIMEOUT=120
export VLLM_STARTUP_TIMEOUT=600
export TQDM_MININTERVAL=5
export NCCL_DEBUG=WARN

export SLURM_PARTITION=acd_u
export SLURM_CPUS_PER_TASK=64
export SLURM_MEM=400G
export SLURM_TIME=08:00:00
export SLURM_EXCLUDE=ACD1-1

cd "${META_RLVR_PROJECT_DIR}"
test -f "${META_RLVR_EVAL_DATASET}"
for step in 1 2 3 4 5 6; do
  test -f "${SEQUENCE_RUN_DIR}/checkpoint-${step}/trainer_state.json"
done
for step in 1 2 3; do
  test -f "${TOKEN_RUN_DIR}/checkpoint-${step}/trainer_state.json"
done

read -r CHECKPOINT_WORLD_SIZE SMOKE_LORA_RANK < <(python - \
  "${SEQUENCE_RUN_DIR}" "${TOKEN_RUN_DIR}" <<'PY'
import json
import sys
from pathlib import Path

runs = [Path(item) for item in sys.argv[1:]]
world_sizes = set()
lora_ranks = set()
for run in runs:
    config = json.loads((run / "run_config.json").read_text())
    world_sizes.add(int(config["distributed_world_size"]))
    lora_ranks.add(int(config["lora_rank"]))
if world_sizes != {8}:
    raise ValueError(f"Curve checkpoints must use world size 8: {world_sizes}")
if len(lora_ranks) != 1:
    raise ValueError(f"Curve checkpoints must share one LoRA rank: {lora_ranks}")
print(world_sizes.pop(), lora_ranks.pop())
PY
)
export CHECKPOINT_WORLD_SIZE SMOKE_LORA_RANK
echo "checkpoint curve preflight: world_size=${CHECKPOINT_WORLD_SIZE} lora_rank=${SMOKE_LORA_RANK}"

if [[ -z "${CONDA_EXE:-}" ]]; then
  CONDA_EXE=$(type -P conda)
  export CONDA_EXE
fi

python src/meta_rlvr/vllm_preflight.py
mkdir -p logs outputs

submission=$(sbatch \
  --job-name="meta-rlvr-curve" \
  --partition="${SLURM_PARTITION}" \
  --gres="gpu:${EVAL_GPUS}" \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  --mem="${SLURM_MEM}" \
  --time="${SLURM_TIME}" \
  --exclude="${SLURM_EXCLUDE}" \
  --output="${META_RLVR_PROJECT_DIR}/logs/${CURVE_PREFIX}-%j.out" \
  --error="${META_RLVR_PROJECT_DIR}/logs/${CURVE_PREFIX}-%j.err" \
  --export=ALL \
  scripts/slurm_aime24_checkpoint_curve.sbatch)
job_id="${submission##* }"
output_root="${META_RLVR_PROJECT_DIR}/outputs/${CURVE_PREFIX}-${job_id}"

echo "${submission}"
echo "configuration: one_job=1 gpus=8 dataset=${META_RLVR_EVAL_DATASET} sequence_steps=1-6 token_steps=1-3 support=16 base_query=32 adapted_query=32 seed=42"
echo "stdout: ${META_RLVR_PROJECT_DIR}/logs/${CURVE_PREFIX}-${job_id}.out"
echo "stderr/progress: ${META_RLVR_PROJECT_DIR}/logs/${CURVE_PREFIX}-${job_id}.err"
echo "vLLM: ${META_RLVR_PROJECT_DIR}/logs/vllm-${job_id}/gpu-*.log"
echo "output: ${output_root}"
echo "curve: ${output_root}/plot/checkpoint_curve.png"
