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

python -m meta_rlvr.probe_policy_jvp \
  --model "${META_RLVR_MODEL_PATH}" \
  --run-config "${META_RLVR_RUN_CONFIG}" \
  --attn-implementation "${JVP_ATTN_IMPLEMENTATION}" \
  --max-sequence-length "${JVP_MAX_SEQUENCE_LENGTH}" \
  --seed "${JVP_SEED}"

echo "[$(date --iso-8601=seconds)] policy JVP probe completed successfully"
