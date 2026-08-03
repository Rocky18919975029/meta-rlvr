#!/usr/bin/env bash
#SBATCH --job-name=meta-rlvr-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00

set -euo pipefail

: "${META_RLVR_PROJECT_DIR:?META_RLVR_PROJECT_DIR was not exported by the submitter}"

if [[ -n "${META_RLVR_CONDA_ENV:-}" ]]; then
  if [[ -z "${CONDA_EXE:-}" || ! -x "${CONDA_EXE}" ]]; then
    echo "CONDA_EXE is unavailable in the Slurm job environment." >&2
    echo "Submit from a shell in which conda has been initialized." >&2
    exit 2
  fi
  eval "$("${CONDA_EXE}" shell.bash hook)"
  conda activate "${META_RLVR_CONDA_ENV}"
  echo "[$(date --iso-8601=seconds)] activated conda environment ${META_RLVR_CONDA_ENV}"
elif [[ -n "${META_RLVR_VENV:-}" ]]; then
  if [[ ! -f "${META_RLVR_VENV}/bin/activate" ]]; then
    echo "Missing virtual environment: ${META_RLVR_VENV}" >&2
    exit 2
  fi
  source "${META_RLVR_VENV}/bin/activate"
  echo "[$(date --iso-8601=seconds)] activated venv ${META_RLVR_VENV}"
else
  echo "Set META_RLVR_CONDA_ENV or META_RLVR_VENV." >&2
  exit 2
fi
cd "${META_RLVR_PROJECT_DIR}"

echo "python=$(command -v python)"
echo "conda_prefix=${CONDA_PREFIX:-none}"

bash scripts/smoke_test_h100.sh
