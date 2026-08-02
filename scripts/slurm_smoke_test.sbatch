#!/usr/bin/env bash
#SBATCH --job-name=meta-rlvr-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00

set -euo pipefail

: "${META_RLVR_PROJECT_DIR:?META_RLVR_PROJECT_DIR was not exported by the submitter}"
: "${META_RLVR_VENV:?Set META_RLVR_VENV to the Linux virtual environment directory}"

if [[ ! -f "${META_RLVR_VENV}/bin/activate" ]]; then
  echo "Missing virtual environment: ${META_RLVR_VENV}" >&2
  exit 2
fi

source "${META_RLVR_VENV}/bin/activate"
cd "${META_RLVR_PROJECT_DIR}"

echo "[$(date --iso-8601=seconds)] activated ${META_RLVR_VENV}"
echo "python=$(command -v python)"

bash scripts/smoke_test_h100.sh
