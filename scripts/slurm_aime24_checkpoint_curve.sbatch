#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

eval "$("${CONDA_EXE}" shell.bash hook)"
conda activate "${META_RLVR_CONDA_ENV}"
cd "${META_RLVR_PROJECT_DIR}"
export PYTHONPATH="${META_RLVR_PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

bash scripts/run_aime24_checkpoint_curve_h100.sh
