#!/usr/bin/env bash
set -euo pipefail

: "${META_RLVR_PROJECT_DIR:?}"
: "${META_RLVR_MODEL_PATH:?}"
: "${META_RLVR_EVAL_DATASET:?}"
: "${SEQUENCE_EVAL_ROOT:?}"
: "${TOKEN_RUN_DIR:?}"
: "${CURVE_PREFIX:?}"
: "${EVAL_GPUS:?}"

case "${EVAL_GPUS}" in
  8) ACCELERATE_CONFIG="configs/accelerate_fsdp_8xh100.yaml" ;;
  *) echo "AIME24 checkpoint curves require 8 GPUs." >&2; exit 2 ;;
esac

OUTPUT_ROOT="${META_RLVR_PROJECT_DIR}/outputs/${CURVE_PREFIX}-${SLURM_JOB_ID}"
MANIFEST="${OUTPUT_ROOT}/submission.json"

cleanup() {
  if declare -F stop_meta_rlvr_vllm_servers >/dev/null 2>&1; then
    stop_meta_rlvr_vllm_servers
  fi
}
trap cleanup EXIT
trap 'echo "[$(date --iso-8601=seconds)] checkpoint curve failed" >&2' ERR

cd "${META_RLVR_PROJECT_DIR}"
export PYTHONPATH="${META_RLVR_PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL="${TQDM_MININTERVAL:-5}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:?}}"
fi

mkdir -p "${OUTPUT_ROOT}"
python - "${MANIFEST}" "${OUTPUT_ROOT}" "${SEQUENCE_EVAL_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
root = Path(sys.argv[2])
sequence_root = Path(sys.argv[3])
evaluations = []
for method, steps in (("sequence", range(1, 7)), ("token", range(1, 4))):
    for step in steps:
        result_root = sequence_root if method == "sequence" else root
        evaluations.append(
            {
                "method": method,
                "step": step,
                "result_dir": str(result_root / f"{method}-step{step}"),
            }
        )
manifest.write_text(
    json.dumps({"evaluations": evaluations}, indent=2) + "\n",
    encoding="utf-8",
)
PY

source scripts/vllm_hybrid_servers.sh
start_meta_rlvr_vllm_servers

EXTRA_ARGS=()
if [[ -n "${EVAL_MAX_PROBLEMS:-}" ]]; then
  EXTRA_ARGS+=(--max-problems "${EVAL_MAX_PROBLEMS}")
fi
if [[ -n "${EVAL_MAX_NEW_TOKENS:-}" ]]; then
  EXTRA_ARGS+=(--max-new-tokens "${EVAL_MAX_NEW_TOKENS}")
fi
if [[ -n "${EVAL_INNER_ITERATIONS:-}" ]]; then
  EXTRA_ARGS+=(--inner-iterations "${EVAL_INNER_ITERATIONS}")
fi
if [[ -n "${EVAL_INNER_LEARNING_RATE:-}" ]]; then
  EXTRA_ARGS+=(--inner-learning-rate "${EVAL_INNER_LEARNING_RATE}")
fi

run_evaluation() {
  local method="$1"
  local step="$2"
  local run_dir="$3"
  local checkpoint="${run_dir}/checkpoint-${step}"
  local output_dir="${OUTPUT_ROOT}/${method}-step${step}"

  echo "[$(date --iso-8601=seconds)] evaluating ${method} checkpoint ${step}"
  setsid accelerate launch \
    --config_file "${ACCELERATE_CONFIG}" \
    -m meta_rlvr.evaluate_checkpoint \
    --checkpoint "${checkpoint}" \
    --dataset-parquet "${META_RLVR_EVAL_DATASET}" \
    --output-dir "${output_dir}" \
    --model "${META_RLVR_MODEL_PATH}" \
    --vllm-base-urls "${META_RLVR_VLLM_BASE_URLS}" \
    --support-group-size "${EVAL_SUPPORT_GROUP_SIZE:-16}" \
    --adaptation-rounds "${EVAL_ADAPTATION_ROUNDS:-1}" \
    --base-query-group-size "${EVAL_BASE_QUERY_GROUP_SIZE:-32}" \
    --adapted-query-group-size "${EVAL_ADAPTED_QUERY_GROUP_SIZE:-32}" \
    --seed "${EVAL_SEED:-42}" \
    --local-rollout-batch-size "${EVAL_LOCAL_ROLLOUT_BATCH_SIZE:-8}" \
    --local-adaptation-batch-size "${EVAL_LOCAL_ADAPTATION_BATCH_SIZE:-2}" \
    --adaptation-temperature "${EVAL_ADAPTATION_TEMPERATURE:-1.0}" \
    --adaptation-top-p "${EVAL_ADAPTATION_TOP_P:-1.0}" \
    --adaptation-top-k "${EVAL_ADAPTATION_TOP_K:-0}" \
    --query-temperature "${EVAL_QUERY_TEMPERATURE:-1.0}" \
    --query-top-p "${EVAL_QUERY_TOP_P:-0.7}" \
    --query-top-k "${EVAL_QUERY_TOP_K:-0}" \
    --request-timeout "${VLLM_REQUEST_TIMEOUT:-1800}" \
    --control-timeout "${VLLM_CONTROL_TIMEOUT:-120}" \
    "${EXTRA_ARGS[@]}"
  echo "[$(date --iso-8601=seconds)] completed ${method} checkpoint ${step}"
}

echo "[$(date --iso-8601=seconds)] reusing sequence checkpoints 1-6 from ${SEQUENCE_EVAL_ROOT}"
for step in 1 2 3; do
  run_evaluation token "${step}" "${TOKEN_RUN_DIR}"
done

python -m meta_rlvr.plot_checkpoint_curve \
  --submission-manifest "${MANIFEST}" \
  --output-dir "${OUTPUT_ROOT}/plot"

echo "[$(date --iso-8601=seconds)] AIME24 checkpoint curve completed successfully"
echo "output=${OUTPUT_ROOT}"
echo "curve=${OUTPUT_ROOT}/plot/checkpoint_curve.png"
