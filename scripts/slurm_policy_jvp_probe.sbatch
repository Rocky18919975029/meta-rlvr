#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

eval "$("${CONDA_EXE}" shell.bash hook)"
conda activate "${META_RLVR_CONDA_ENV}"
cd "${META_RLVR_PROJECT_DIR}"
export PYTHONPATH="${META_RLVR_PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

PROBE_ARGS=(
  --model "${META_RLVR_MODEL_PATH}"
  --run-config "${META_RLVR_RUN_CONFIG}"
  --attn-implementation "${JVP_ATTN_IMPLEMENTATION}"
  --max-sequence-length "${JVP_MAX_SEQUENCE_LENGTH}"
  --group-size "${JVP_GROUP_SIZE}"
  --response-micro-batch-size "${JVP_RESPONSE_MICRO_BATCH_SIZE}"
  --seed "${JVP_SEED}"
)
if [[ -n "${JVP_LOGPROB_POSITION_CHUNK_SIZE}" ]]; then
  PROBE_ARGS+=(
    --logprob-position-chunk-size "${JVP_LOGPROB_POSITION_CHUNK_SIZE}"
  )
fi
if [[ "${JVP_SKIP_DUALITY_CHECK}" == "1" ]]; then
  PROBE_ARGS+=(--skip-duality-check)
elif [[ "${JVP_SKIP_DUALITY_CHECK}" != "0" ]]; then
  echo "JVP_SKIP_DUALITY_CHECK must be 0 or 1." >&2
  exit 2
fi

python -m meta_rlvr.probe_policy_jvp "${PROBE_ARGS[@]}"

echo "[$(date --iso-8601=seconds)] policy JVP probe completed successfully"
