#!/usr/bin/env bash
set -euo pipefail

: "${META_RLVR_TRAIN_PARQUET:?Set META_RLVR_TRAIN_PARQUET to DAPO-17k.parquet}"
: "${META_RLVR_VALIDATION_PARQUET:?Set META_RLVR_VALIDATION_PARQUET to AIME24.parquet}"
: "${META_RLVR_OUTPUT_DIR:?Set META_RLVR_OUTPUT_DIR}"
: "${META_RLVR_MAX_NEW_TOKENS:?Set META_RLVR_MAX_NEW_TOKENS explicitly}"

accelerate launch \
  --config_file configs/accelerate_fsdp_8xh100.yaml \
  -m meta_rlvr.train \
  --train-parquet "${META_RLVR_TRAIN_PARQUET}" \
  --validation-parquet "${META_RLVR_VALIDATION_PARQUET}" \
  --output-dir "${META_RLVR_OUTPUT_DIR}" \
  --max-new-tokens "${META_RLVR_MAX_NEW_TOKENS}" \
  "$@"
