#!/usr/bin/env bash
set -euo pipefail

: "${META_RLVR_PROJECT_DIR:?}"
: "${META_RLVR_MODEL_PATH:?}"
: "${META_RLVR_EVAL_DATASET:?}"
: "${TOKEN_RUN_DIR:?}"
: "${TOKEN_CURVE_STEPS:?}"
: "${CURVE_PREFIX:?}"
: "${EVAL_GPUS:?}"

case "${EVAL_GPUS}" in
  8) ACCELERATE_CONFIG="configs/accelerate_fsdp_8xh100.yaml" ;;
  *) echo "Token checkpoint curves require 8 GPUs." >&2; exit 2 ;;
esac

OUTPUT_ROOT="${META_RLVR_PROJECT_DIR}/outputs/${CURVE_PREFIX}-${SLURM_JOB_ID}"
MANIFEST="${OUTPUT_ROOT}/submission.json"

cleanup() {
  if declare -F stop_meta_rlvr_vllm_servers >/dev/null 2>&1; then
    stop_meta_rlvr_vllm_servers
  fi
}
trap cleanup EXIT
trap 'echo "[$(date --iso-8601=seconds)] token checkpoint curve failed" >&2' ERR

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
python - "${MANIFEST}" "${OUTPUT_ROOT}" "${META_RLVR_EVAL_DATASET}" \
  "${TOKEN_CURVE_STEPS}" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
root = Path(sys.argv[2])
dataset = Path(sys.argv[3])
steps = [int(item) for item in sys.argv[4].split(",")]
manifest.write_text(
    json.dumps(
        {
            "dataset_label": dataset.stem,
            "evaluations": [
                {
                    "method": "token",
                    "step": step,
                    "result_dir": str(root / f"token-step{step}"),
                }
                for step in steps
            ],
        },
        indent=2,
    )
    + "\n",
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

run_evaluation() {
  local step="$1"
  local checkpoint="${TOKEN_RUN_DIR}/checkpoint-${step}"
  local output_dir="${OUTPUT_ROOT}/token-step${step}"

  echo "[$(date --iso-8601=seconds)] evaluating token checkpoint ${step}"
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
    --request-timeout "${VLLM_REQUEST_TIMEOUT:-1800}" \
    --control-timeout "${VLLM_CONTROL_TIMEOUT:-120}" \
    "${EXTRA_ARGS[@]}"
  echo "[$(date --iso-8601=seconds)] completed token checkpoint ${step}"
}

IFS=',' read -r -a TOKEN_STEPS <<<"${TOKEN_CURVE_STEPS}"
for step in "${TOKEN_STEPS[@]}"; do
  run_evaluation "${step}"
done

python -m meta_rlvr.plot_checkpoint_curve \
  --submission-manifest "${MANIFEST}" \
  --output-dir "${OUTPUT_ROOT}/plot"

echo "[$(date --iso-8601=seconds)] token checkpoint curve completed successfully"
echo "output=${OUTPUT_ROOT}"
echo "curve=${OUTPUT_ROOT}/plot/checkpoint_curve.png"
