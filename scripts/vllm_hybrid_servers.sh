#!/usr/bin/env bash

# Source this file from a Slurm job. It launches one TP=1 vLLM replica per GPU,
# puts every replica into level-1 sleep, and exports META_RLVR_VLLM_BASE_URLS.

META_RLVR_VLLM_PIDS=()

start_meta_rlvr_vllm_servers() {
  : "${META_RLVR_MODEL_PATH:?META_RLVR_MODEL_PATH is required}"
  : "${META_RLVR_PROJECT_DIR:?META_RLVR_PROJECT_DIR is required}"

  local gpu_count="${SMOKE_GPUS:?SMOKE_GPUS is required}"
  local first_port
  if [[ -n "${VLLM_FIRST_PORT:-}" ]]; then
    first_port="${VLLM_FIRST_PORT}"
  elif [[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]; then
    first_port=$((20000 + SLURM_JOB_ID % 20000))
  else
    first_port=18100
  fi
  local log_dir="${META_RLVR_PROJECT_DIR%/}/logs/vllm-${SLURM_JOB_ID:-manual}"
  local gpu
  local port
  local url
  local urls=()
  local visible_devices=()
  local cuda_device
  mkdir -p "${log_dir}"

  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a visible_devices <<<"${CUDA_VISIBLE_DEVICES}"
    if [[ "${#visible_devices[@]}" -ne "${gpu_count}" ]]; then
      echo "CUDA_VISIBLE_DEVICES exposes ${#visible_devices[@]} devices, expected ${gpu_count}." >&2
      return 2
    fi
  fi

  export VLLM_USE_V1=1
  export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

  for ((gpu = 0; gpu < gpu_count; gpu++)); do
    port=$((first_port + gpu))
    url="http://127.0.0.1:${port}"
    urls+=("${url}")
    cuda_device="${visible_devices[${gpu}]:-${gpu}}"
    echo "[$(date --iso-8601=seconds)] starting vLLM replica gpu=${cuda_device} url=${url}"
    CUDA_VISIBLE_DEVICES="${cuda_device}" \
      python -m vllm.entrypoints.openai.api_server \
        --model "${META_RLVR_MODEL_PATH}" \
        --served-model-name meta-rlvr-base \
        --host 127.0.0.1 \
        --port "${port}" \
        --dtype bfloat16 \
        --load-format safetensors \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.42}" \
        --max-model-len "${VLLM_MAX_MODEL_LEN:-8192}" \
        --max-num-seqs "${VLLM_MAX_NUM_SEQS:-64}" \
        --enable-prefix-caching \
        --enable-sleep-mode \
        --enable-lora \
        --max-lora-rank "${SMOKE_LORA_RANK:-8}" \
        --max-loras "${VLLM_MAX_LORAS:-1}" \
        >"${log_dir}/gpu-${gpu}.log" 2>&1 &
    META_RLVR_VLLM_PIDS+=("$!")
  done

  META_RLVR_VLLM_BASE_URLS=$(IFS=,; echo "${urls[*]}")
  export META_RLVR_VLLM_BASE_URLS
  local pid_csv
  pid_csv=$(IFS=,; echo "${META_RLVR_VLLM_PIDS[*]}")

  python - "${META_RLVR_VLLM_BASE_URLS}" "${pid_csv}" <<'PY'
import json
import os
import sys
import time
import urllib.error
import urllib.request

urls = sys.argv[1].split(",")
pids = [int(pid) for pid in sys.argv[2].split(",")]
if len(urls) != len(pids):
    raise RuntimeError("vLLM URL/PID count mismatch")
deadline = time.monotonic() + 900
pending = set(urls)
while pending and time.monotonic() < deadline:
    for url, pid in zip(urls, pids, strict=True):
        if url not in pending:
            continue
        try:
            os.kill(pid, 0)
        except ProcessLookupError as error:
            raise RuntimeError(f"vLLM replica exited before becoming ready: {url}") from error
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as response:
                if response.status == 200:
                    pending.remove(url)
                    print(f"vLLM ready: {url}", flush=True)
        except (urllib.error.URLError, TimeoutError):
            pass
    if pending:
        time.sleep(2)
if pending:
    raise RuntimeError(f"vLLM replicas did not become ready: {sorted(pending)}")

for url in urls:
    request = urllib.request.Request(f"{url}/sleep?level=1", method="POST")
    with urllib.request.urlopen(request, timeout=900) as response:
        if response.status != 200:
            raise RuntimeError(f"Failed to sleep vLLM replica {url}")
    with urllib.request.urlopen(f"{url}/is_sleeping", timeout=10) as response:
        status = json.load(response)
    if status != {"is_sleeping": True}:
        raise RuntimeError(f"vLLM replica did not enter sleep mode: {url}: {status}")
    print(f"vLLM sleeping: {url}", flush=True)
PY
}

stop_meta_rlvr_vllm_servers() {
  local pid
  for pid in "${META_RLVR_VLLM_PIDS[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}"
    fi
  done
  for pid in "${META_RLVR_VLLM_PIDS[@]:-}"; do
    wait "${pid}" 2>/dev/null || true
  done
}
