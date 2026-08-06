#!/usr/bin/env bash
set -euo pipefail

export META_RLVR_PROJECT_DIR="${HOME}/meta-rlvr"
export META_RLVR_CONDA_ENV="verl"
export META_RLVR_MODEL_PATH="/data/user/zhongal/.cache/qwen2.5-math-7b-local"
export META_RLVR_RUN_CONFIG="${HOME}/meta-rlvr/outputs/tiny-meta-only-b512-472864/run_config.json"
export JVP_ATTN_IMPLEMENTATION="${JVP_ATTN_IMPLEMENTATION:-sdpa}"
export JVP_MAX_SEQUENCE_LENGTH="${JVP_MAX_SEQUENCE_LENGTH:-64}"
export JVP_GROUP_SIZE="${JVP_GROUP_SIZE:-2}"
export JVP_RESPONSE_MICRO_BATCH_SIZE="${JVP_RESPONSE_MICRO_BATCH_SIZE:-2}"
export JVP_LOGPROB_POSITION_CHUNK_SIZE="${JVP_LOGPROB_POSITION_CHUNK_SIZE:-}"
export JVP_SKIP_DUALITY_CHECK="${JVP_SKIP_DUALITY_CHECK:-0}"
export JVP_SEED="${JVP_SEED:-42}"

export SLURM_PARTITION="${SLURM_PARTITION:-acd_u}"
export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-12}"
export SLURM_MEM="${SLURM_MEM:-120G}"
export SLURM_TIME="${SLURM_TIME:-00:30:00}"
export SLURM_EXCLUDE="${SLURM_EXCLUDE:-ACD1-1}"

if [[ -z "${CONDA_EXE:-}" ]]; then
  CONDA_EXE=$(type -P conda)
  export CONDA_EXE
fi

cd "${META_RLVR_PROJECT_DIR}"
mkdir -p logs
submission=$(sbatch \
  --job-name="meta-rlvr-jvp-probe" \
  --partition="${SLURM_PARTITION}" \
  --gres="gpu:1" \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  --mem="${SLURM_MEM}" \
  --time="${SLURM_TIME}" \
  --exclude="${SLURM_EXCLUDE}" \
  --output="${META_RLVR_PROJECT_DIR}/logs/policy-jvp-probe-%j.out" \
  --error="${META_RLVR_PROJECT_DIR}/logs/policy-jvp-probe-%j.err" \
  --export=ALL \
  scripts/slurm_policy_jvp_probe.sbatch)
job_id="${submission##* }"

echo "${submission}"
echo "configuration: gpu=1 attn=${JVP_ATTN_IMPLEMENTATION} sequence_length=${JVP_MAX_SEQUENCE_LENGTH} group_size=${JVP_GROUP_SIZE} response_micro_batch=${JVP_RESPONSE_MICRO_BATCH_SIZE} logprob_position_chunk=${JVP_LOGPROB_POSITION_CHUNK_SIZE:-none} skip_duality=${JVP_SKIP_DUALITY_CHECK}"
echo "stdout: ${META_RLVR_PROJECT_DIR}/logs/policy-jvp-probe-${job_id}.out"
echo "stderr: ${META_RLVR_PROJECT_DIR}/logs/policy-jvp-probe-${job_id}.err"
echo "queue: squeue -j ${job_id}"
