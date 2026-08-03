#!/usr/bin/env bash
set -euo pipefail

export SMOKE_ROLLOUT_BACKEND=vllm
export SMOKE_LOG_ROLLOUTS=1
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.30}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
export VLLM_MAX_LORAS=1
export VLLM_MAX_CPU_LORAS=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

exec bash scripts/submit_smoke_test.sh
