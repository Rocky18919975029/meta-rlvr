#!/usr/bin/env bash
set -euo pipefail

export META_RLVR_RUN_LABEL="rollout-test"
export SMOKE_ROLLOUT_ONLY=1
export SLURM_TIME="${SLURM_TIME:-02:00:00}"

exec bash scripts/submit_meta_meaningful_test.sh
