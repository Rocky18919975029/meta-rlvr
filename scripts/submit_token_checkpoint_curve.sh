#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --run-dir PATH --dataset PARQUET --label NAME [--steps 1,2,3]" >&2
  exit 2
}

TOKEN_RUN_DIR=""
META_RLVR_EVAL_DATASET=""
CURVE_PREFIX=""
TOKEN_CURVE_STEPS="1,2,3"
while (($#)); do
  case "$1" in
    --run-dir) TOKEN_RUN_DIR="$2"; shift 2 ;;
    --dataset) META_RLVR_EVAL_DATASET="$2"; shift 2 ;;
    --label) CURVE_PREFIX="$2"; shift 2 ;;
    --steps) TOKEN_CURVE_STEPS="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[[ -n "${TOKEN_RUN_DIR}" && -n "${META_RLVR_EVAL_DATASET}" && -n "${CURVE_PREFIX}" ]] || usage

export TOKEN_RUN_DIR META_RLVR_EVAL_DATASET CURVE_PREFIX TOKEN_CURVE_STEPS
export META_RLVR_PROJECT_DIR="${HOME}/meta-rlvr"
export META_RLVR_CONDA_ENV="verl"
export META_RLVR_MODEL_PATH="${META_RLVR_MODEL_PATH:-/data/user/zhongal/.cache/qwen2.5-math-7b-local}"

export EVAL_SUPPORT_GROUP_SIZE="${EVAL_SUPPORT_GROUP_SIZE:-16}"
export EVAL_BASE_QUERY_GROUP_SIZE="${EVAL_BASE_QUERY_GROUP_SIZE:-32}"
export EVAL_ADAPTED_QUERY_GROUP_SIZE="${EVAL_ADAPTED_QUERY_GROUP_SIZE:-32}"
export EVAL_SEED="${EVAL_SEED:-42}"
export EVAL_MAX_PROBLEMS="${EVAL_MAX_PROBLEMS:-}"
export EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-3072}"
export EVAL_INNER_ITERATIONS="${EVAL_INNER_ITERATIONS:-2}"
export EVAL_ADAPTATION_ROUNDS="${EVAL_ADAPTATION_ROUNDS:-1}"
export EVAL_LOCAL_ROLLOUT_BATCH_SIZE="${EVAL_LOCAL_ROLLOUT_BATCH_SIZE:-8}"
export EVAL_LOCAL_ADAPTATION_BATCH_SIZE="${EVAL_LOCAL_ADAPTATION_BATCH_SIZE:-2}"
export EVAL_ADAPTATION_TEMPERATURE="${EVAL_ADAPTATION_TEMPERATURE:-1.0}"
export EVAL_ADAPTATION_TOP_P="${EVAL_ADAPTATION_TOP_P:-1.0}"
export EVAL_ADAPTATION_TOP_K="${EVAL_ADAPTATION_TOP_K:-0}"
export EVAL_QUERY_TEMPERATURE="${EVAL_QUERY_TEMPERATURE:-1.0}"
export EVAL_QUERY_TOP_P="${EVAL_QUERY_TOP_P:-0.7}"
export EVAL_QUERY_TOP_K="${EVAL_QUERY_TOP_K:-0}"

export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.30}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
export VLLM_MAX_LORAS="${VLLM_MAX_LORAS:-8}"
export VLLM_MAX_CPU_LORAS="${VLLM_MAX_CPU_LORAS:-8}"
export VLLM_REQUEST_TIMEOUT="${VLLM_REQUEST_TIMEOUT:-1800}"
export VLLM_CONTROL_TIMEOUT="${VLLM_CONTROL_TIMEOUT:-120}"
export VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-600}"
export TQDM_MININTERVAL="${TQDM_MININTERVAL:-5}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

export SLURM_PARTITION="${SLURM_PARTITION:-acd_u}"
export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-64}"
export SLURM_MEM="${SLURM_MEM:-400G}"
export SLURM_TIME="${SLURM_TIME:-04:00:00}"
export SLURM_EXCLUDE="${SLURM_EXCLUDE:-ACD1-1,ACD1-52}"

cd "${META_RLVR_PROJECT_DIR}"
test -f "${META_RLVR_EVAL_DATASET}"

read -r CHECKPOINT_WORLD_SIZE SMOKE_LORA_RANK < <(python - \
  "${TOKEN_RUN_DIR}" "${TOKEN_CURVE_STEPS}" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
steps = [int(item) for item in sys.argv[2].split(",")]
if not steps or len(steps) != len(set(steps)) or any(step <= 0 for step in steps):
    raise ValueError(f"Invalid token checkpoint steps: {steps}")
config = json.loads((run_dir / "run_config.json").read_text())
if float(config.get("token_meta_coefficient", 0.0)) <= 0:
    raise ValueError("Token curve requires token-confidence checkpoints.")
if float(config.get("meta_coefficient", 0.0)) != 0:
    raise ValueError("Token curve requires the sequence meta branch to be disabled.")
if config.get("attn_implementation") != "sdpa":
    raise ValueError("Token curve requires corrected SDPA checkpoints.")
world_size = int(config["distributed_world_size"])
for step in steps:
    state = json.loads(
        (run_dir / f"checkpoint-{step}" / "trainer_state.json").read_text()
    )
    if int(state["completed_steps"]) != step:
        raise ValueError(f"Checkpoint state mismatch at step {step}.")
    if int(state["world_size"]) != world_size:
        raise ValueError(f"Checkpoint world-size mismatch at step {step}.")
print(world_size, int(config["lora_rank"]))
PY
)
export CHECKPOINT_WORLD_SIZE SMOKE_LORA_RANK
case "${CHECKPOINT_WORLD_SIZE}" in
  8) ;;
  *) echo "Token checkpoint curve currently requires world size 8." >&2; exit 2 ;;
esac
export EVAL_GPUS="${CHECKPOINT_WORLD_SIZE}"
export SMOKE_GPUS="${EVAL_GPUS}"

if [[ -z "${CONDA_EXE:-}" ]]; then
  CONDA_EXE=$(type -P conda)
  export CONDA_EXE
fi

python src/meta_rlvr/vllm_preflight.py
mkdir -p logs outputs

submission=$(sbatch \
  --job-name="meta-rlvr-token-curve" \
  --partition="${SLURM_PARTITION}" \
  --gres="gpu:${EVAL_GPUS}" \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  --mem="${SLURM_MEM}" \
  --time="${SLURM_TIME}" \
  --exclude="${SLURM_EXCLUDE}" \
  --output="${META_RLVR_PROJECT_DIR}/logs/${CURVE_PREFIX}-%j.out" \
  --error="${META_RLVR_PROJECT_DIR}/logs/${CURVE_PREFIX}-%j.err" \
  --export=ALL \
  scripts/slurm_token_checkpoint_curve.sbatch)
job_id="${submission##* }"
output_root="${META_RLVR_PROJECT_DIR}/outputs/${CURVE_PREFIX}-${job_id}"

echo "${submission}"
echo "configuration: one_job=1 gpus=${EVAL_GPUS} dataset=${META_RLVR_EVAL_DATASET} token_steps=${TOKEN_CURVE_STEPS} support=${EVAL_SUPPORT_GROUP_SIZE} base_query=${EVAL_BASE_QUERY_GROUP_SIZE} adapted_query=${EVAL_ADAPTED_QUERY_GROUP_SIZE} seed=${EVAL_SEED} adaptation_sampling=${EVAL_ADAPTATION_TEMPERATURE}/${EVAL_ADAPTATION_TOP_P}/${EVAL_ADAPTATION_TOP_K} query_sampling=${EVAL_QUERY_TEMPERATURE}/${EVAL_QUERY_TOP_P}/${EVAL_QUERY_TOP_K} max_problems=${EVAL_MAX_PROBLEMS:-all} nccl_nvls=${NCCL_NVLS_ENABLE}"
echo "stdout: ${META_RLVR_PROJECT_DIR}/logs/${CURVE_PREFIX}-${job_id}.out"
echo "stderr/progress: ${META_RLVR_PROJECT_DIR}/logs/${CURVE_PREFIX}-${job_id}.err"
echo "vLLM: ${META_RLVR_PROJECT_DIR}/logs/vllm-${job_id}/gpu-*.log"
echo "output: ${output_root}"
echo "curve: ${output_root}/plot/checkpoint_curve.png"
