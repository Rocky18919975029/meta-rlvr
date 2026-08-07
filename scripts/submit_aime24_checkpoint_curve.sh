#!/usr/bin/env bash
set -euo pipefail

export META_RLVR_PROJECT_DIR="${HOME}/meta-rlvr"
export META_RLVR_CONDA_ENV="verl"
export META_RLVR_MODEL_PATH="/data/user/zhongal/.cache/qwen2.5-math-7b-local"
export AIME24_PARQUET="${AIME24_PARQUET:-/data/user/zhongal/data/reschedule/aime24.parquet}"
export SEQUENCE_RUN_DIR="${SEQUENCE_RUN_DIR:-${META_RLVR_PROJECT_DIR}/outputs/tiny-meta-only-b512-472864}"
export TOKEN_RUN_DIR="${TOKEN_RUN_DIR:-${META_RLVR_PROJECT_DIR}/outputs/tiny-token-meta-only-b512-474799}"
export CURVE_PREFIX="${CURVE_PREFIX:-aime24-curve-$(date +%Y%m%d-%H%M%S)}"

export SLURM_PARTITION=acd_u
export SLURM_MEM=400G
export SLURM_TIME=02:00:00
export SLURM_EXCLUDE=ACD1-1
export NCCL_DEBUG=WARN

cd "${META_RLVR_PROJECT_DIR}"

for step in 1 2 3 4 5 6; do
  test -f "${SEQUENCE_RUN_DIR}/checkpoint-${step}/trainer_state.json"
done
for step in 1 2 3; do
  test -f "${TOKEN_RUN_DIR}/checkpoint-${step}/trainer_state.json"
done
test -f "${AIME24_PARQUET}"

SUBMISSIONS=()
LAST_JOB=""
submit_one() {
  local method="$1"
  local step="$2"
  local run_dir="$3"
  local dependency="$4"
  local label="${CURVE_PREFIX}-${method}-step${step}"
  local output
  output=$(SLURM_DEPENDENCY="${dependency}" \
    bash scripts/submit_checkpoint_evaluation.sh \
      --checkpoint "${run_dir}/checkpoint-${step}" \
      --dataset "${AIME24_PARQUET}" \
      --label "${label}" \
      --support-group-size 16 \
      --base-query-group-size 32 \
      --adapted-query-group-size 32 \
      --seed 42 \
      --max-problems 30 \
      --max-new-tokens 3072 \
      --inner-iterations 2 \
      --adaptation-rounds 1 \
      --local-rollout-batch-size 8 \
      --local-adaptation-batch-size 2)
  printf '%s\n' "${output}"
  LAST_JOB=$(awk '/Submitted batch job/{print $4}' <<<"${output}")
  if [[ ! "${LAST_JOB}" =~ ^[0-9]+$ ]]; then
    echo "Could not parse evaluation job id." >&2
    exit 2
  fi
  SUBMISSIONS+=(
    "${method}|${step}|${LAST_JOB}|${META_RLVR_PROJECT_DIR}/outputs/${label}-${LAST_JOB}"
  )
}

dependency=""
for step in 1 2 3 4 5 6; do
  submit_one sequence "${step}" "${SEQUENCE_RUN_DIR}" "${dependency}"
  dependency="afterok:${LAST_JOB}"
done
SEQUENCE_LAST_JOB="${LAST_JOB}"

dependency=""
for step in 1 2 3; do
  submit_one token "${step}" "${TOKEN_RUN_DIR}" "${dependency}"
  dependency="afterok:${LAST_JOB}"
done
TOKEN_LAST_JOB="${LAST_JOB}"

python - "${CURVE_PREFIX}" "${SUBMISSIONS[@]}" <<'PY'
import json
import sys
from pathlib import Path

prefix = sys.argv[1]
records = []
for item in sys.argv[2:]:
    method, step, job_id, result_dir = item.split("|", 3)
    records.append(
        {
            "method": method,
            "step": int(step),
            "job_id": int(job_id),
            "result_dir": result_dir,
        }
    )
path = Path.home() / "meta-rlvr/outputs" / f"{prefix}-submission.json"
path.write_text(
    json.dumps({"prefix": prefix, "evaluations": records}, indent=2) + "\n"
)
print(f"submission_manifest={path}")
PY

echo "curve_prefix=${CURVE_PREFIX}"
echo "sequence_chain_final_job=${SEQUENCE_LAST_JOB}"
echo "token_chain_final_job=${TOKEN_LAST_JOB}"
echo "Both chains run concurrently; jobs within each chain run sequentially."
echo "After both final jobs complete, run:"
echo "python -m meta_rlvr.plot_checkpoint_curve --submission-manifest ${META_RLVR_PROJECT_DIR}/outputs/${CURVE_PREFIX}-submission.json --output-dir ${META_RLVR_PROJECT_DIR}/outputs/${CURVE_PREFIX}-plot"
