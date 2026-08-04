#!/usr/bin/env bash

# Source this file from a Slurm job. It launches one isolated TP=1 vLLM
# process group per GPU, verifies every required API endpoint, enters level-1
# sleep, and exports META_RLVR_VLLM_BASE_URLS.

META_RLVR_VLLM_PIDS=()
META_RLVR_VLLM_WATCHDOG_PID=""

_meta_rlvr_show_vllm_logs() {
  local log_dir="${META_RLVR_PROJECT_DIR%/}/logs/vllm-${SLURM_JOB_ID:-manual}"
  local log_file
  for log_file in "${log_dir}"/gpu-*.log; do
    if [[ -f "${log_file}" ]]; then
      echo "===== ${log_file} (last 120 lines) =====" >&2
      tail -n 120 "${log_file}" >&2
    fi
  done
}

_meta_rlvr_terminate_process_group() {
  local leader_pid="$1"
  local signal_name="$2"
  if kill -0 -- "-${leader_pid}" 2>/dev/null; then
    kill -"${signal_name}" -- "-${leader_pid}" 2>/dev/null || true
  elif kill -0 "${leader_pid}" 2>/dev/null; then
    kill -"${signal_name}" "${leader_pid}" 2>/dev/null || true
  fi
}

start_meta_rlvr_vllm_servers() {
  : "${META_RLVR_MODEL_PATH:?META_RLVR_MODEL_PATH is required}"
  : "${META_RLVR_PROJECT_DIR:?META_RLVR_PROJECT_DIR is required}"

  if ! command -v setsid >/dev/null 2>&1; then
    echo "setsid is required to isolate and clean up vLLM process groups." >&2
    return 2
  fi

  META_RLVR_VLLM_PIDS=()
  local gpu_count="${SMOKE_GPUS:?SMOKE_GPUS is required}"
  local startup_timeout="${VLLM_STARTUP_TIMEOUT:-300}"
  local control_timeout="${VLLM_CONTROL_TIMEOUT:-120}"
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
  local replica_cache
  mkdir -p "${log_dir}"

  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "CUDA_VISIBLE_DEVICES must identify the GPUs allocated by Slurm." >&2
    return 2
  fi
  IFS=',' read -r -a visible_devices <<<"${CUDA_VISIBLE_DEVICES}"
  if [[ "${#visible_devices[@]}" -ne "${gpu_count}" ]]; then
    echo "CUDA_VISIBLE_DEVICES exposes ${#visible_devices[@]} devices, expected ${gpu_count}." >&2
    return 2
  fi

  python - "${first_port}" "${gpu_count}" <<'PY'
import socket
import sys

first_port = int(sys.argv[1])
gpu_count = int(sys.argv[2])
for port in range(first_port, first_port + gpu_count):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as error:
            raise RuntimeError(f"vLLM port is already in use: {port}") from error
PY

  export VLLM_USE_V1=1
  export VLLM_SERVER_DEV_MODE=1
  export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

  if [[ -n "${PYTORCH_CUDA_ALLOC_CONF:-}" ]] || \
    [[ -n "${PYTORCH_ALLOC_CONF:-}" ]]; then
    echo "vLLM replicas will ignore the trainer PyTorch allocator configuration;" \
      "sleep-mode memory pools require expandable_segments to be disabled."
  fi

  for ((gpu = 0; gpu < gpu_count; gpu++)); do
    port=$((first_port + gpu))
    url="http://127.0.0.1:${port}"
    urls+=("${url}")
    cuda_device="${visible_devices[${gpu}]}"
    replica_cache="${TMPDIR:-/tmp}/meta-rlvr-vllm-${SLURM_JOB_ID:-manual}/gpu-${gpu}"
    mkdir -p "${replica_cache}"
    echo "[$(date --iso-8601=seconds)] starting vLLM replica gpu=${cuda_device} url=${url}"
    setsid env \
      -u PYTORCH_CUDA_ALLOC_CONF \
      -u PYTORCH_ALLOC_CONF \
      CUDA_VISIBLE_DEVICES="${cuda_device}" \
      VLLM_CACHE_ROOT="${replica_cache}/vllm" \
      TORCHINDUCTOR_CACHE_DIR="${replica_cache}/torchinductor" \
      TRITON_CACHE_DIR="${replica_cache}/triton" \
      CUDA_CACHE_PATH="${replica_cache}/cuda" \
      VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}" \
      python -m vllm.entrypoints.openai.api_server \
        --model "${META_RLVR_MODEL_PATH}" \
        --served-model-name meta-rlvr-base \
        --host 127.0.0.1 \
        --port "${port}" \
        --dtype bfloat16 \
        --logprobs-mode raw_logprobs \
        --load-format safetensors \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.30}" \
        --max-model-len "${VLLM_MAX_MODEL_LEN:-4096}" \
        --max-num-seqs "${VLLM_MAX_NUM_SEQS:-64}" \
        --enable-prefix-caching \
        --enable-sleep-mode \
        --enable-lora \
        --lora-dtype bfloat16 \
        --max-lora-rank "${SMOKE_LORA_RANK:-8}" \
        --max-loras "${VLLM_MAX_LORAS:-1}" \
        --max-cpu-loras "${VLLM_MAX_CPU_LORAS:-${VLLM_MAX_LORAS:-1}}" \
        >"${log_dir}/gpu-${gpu}.log" 2>&1 &
    META_RLVR_VLLM_PIDS+=("$!")
  done

  META_RLVR_VLLM_BASE_URLS=$(IFS=,; echo "${urls[*]}")
  export META_RLVR_VLLM_BASE_URLS
  local pid_csv
  pid_csv=$(IFS=,; echo "${META_RLVR_VLLM_PIDS[*]}")

  if ! python - \
    "${META_RLVR_VLLM_BASE_URLS}" \
    "${pid_csv}" \
    "${startup_timeout}" \
    "${control_timeout}" <<'PY'
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

urls = sys.argv[1].split(",")
pids = [int(pid) for pid in sys.argv[2].split(",")]
startup_timeout = float(sys.argv[3])
control_timeout = float(sys.argv[4])
if startup_timeout <= 0 or control_timeout <= 0:
    raise ValueError("vLLM startup/control timeouts must be positive")
if len(urls) != len(pids):
    raise RuntimeError("vLLM URL/PID count mismatch")


def request_json(method, url, path, timeout):
    request = urllib.request.Request(f"{url}{path}", method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"vLLM {method} {url}{path} returned HTTP {error.code}: {body}"
        ) from error
    if not body:
        return None
    return json.loads(body)


deadline = time.monotonic() + startup_timeout
pending = set(urls)
while pending and time.monotonic() < deadline:
    for url, pid in zip(urls, pids, strict=True):
        if url not in pending:
            continue
        try:
            os.kill(pid, 0)
        except ProcessLookupError as error:
            raise RuntimeError(
                f"vLLM replica exited before becoming ready: {url}"
            ) from error
        stat_path = f"/proc/{pid}/stat"
        if os.path.exists(stat_path):
            with open(stat_path, encoding="utf-8") as stream:
                process_state = stream.read().split()[2]
            if process_state == "Z":
                raise RuntimeError(
                    f"vLLM replica became a zombie before becoming ready: {url}"
                )
        try:
            request_json("GET", url, "/health", 2)
        except urllib.error.URLError:
            continue
        pending.remove(url)
        print(f"vLLM ready: {url}", flush=True)
    if pending:
        time.sleep(1)
if pending:
    raise RuntimeError(f"vLLM replicas did not become ready: {sorted(pending)}")

def validate_and_sleep(url):
    models = request_json("GET", url, "/v1/models", control_timeout)
    model_ids = {item["id"] for item in models["data"]}
    if model_ids != {"meta-rlvr-base"}:
        raise RuntimeError(f"Unexpected vLLM models at {url}: {sorted(model_ids)}")
    sleeping = request_json("GET", url, "/is_sleeping", control_timeout)
    if sleeping != {"is_sleeping": False}:
        raise RuntimeError(f"Unexpected initial sleep state at {url}: {sleeping}")
    request_json("POST", url, "/sleep?level=1", control_timeout)
    sleeping = request_json("GET", url, "/is_sleeping", control_timeout)
    if sleeping != {"is_sleeping": True}:
        raise RuntimeError(f"vLLM replica did not enter sleep mode: {url}: {sleeping}")
    return url


with ThreadPoolExecutor(max_workers=len(urls)) as executor:
    futures = {executor.submit(validate_and_sleep, url): url for url in urls}
    for future in as_completed(futures):
        url = future.result()
        print(f"vLLM sleeping: {url}", flush=True)
PY
  then
    echo "vLLM startup/lifecycle validation failed; terminating all replicas." >&2
    _meta_rlvr_show_vllm_logs
    stop_meta_rlvr_vllm_servers
    return 1
  fi

  local owner_pid="$$"
  (
    while kill -0 "${owner_pid}" 2>/dev/null; do
      local replica_pid
      for replica_pid in "${META_RLVR_VLLM_PIDS[@]}"; do
        local process_state
        process_state=$(ps -o stat= -p "${replica_pid}" 2>/dev/null || true)
        if ! kill -0 "${replica_pid}" 2>/dev/null || [[ -z "${process_state}" || "${process_state}" == Z* ]]; then
          echo "[$(date --iso-8601=seconds)] vLLM replica ${replica_pid} exited; terminating job shell ${owner_pid}." >&2
          _meta_rlvr_show_vllm_logs
          kill -TERM "${owner_pid}" 2>/dev/null || true
          exit 1
        fi
      done
      sleep 2
    done
  ) &
  META_RLVR_VLLM_WATCHDOG_PID="$!"
}

stop_meta_rlvr_vllm_servers() {
  local pid
  if [[ -n "${META_RLVR_VLLM_WATCHDOG_PID:-}" ]]; then
    kill "${META_RLVR_VLLM_WATCHDOG_PID}" 2>/dev/null || true
    wait "${META_RLVR_VLLM_WATCHDOG_PID}" 2>/dev/null || true
    META_RLVR_VLLM_WATCHDOG_PID=""
  fi
  for pid in "${META_RLVR_VLLM_PIDS[@]:-}"; do
    _meta_rlvr_terminate_process_group "${pid}" TERM
  done
  local deadline=$((SECONDS + 15))
  while ((SECONDS < deadline)); do
    local any_alive=0
    for pid in "${META_RLVR_VLLM_PIDS[@]:-}"; do
      if kill -0 -- "-${pid}" 2>/dev/null || kill -0 "${pid}" 2>/dev/null; then
        any_alive=1
      fi
    done
    if [[ "${any_alive}" == "0" ]]; then
      break
    fi
    sleep 1
  done
  for pid in "${META_RLVR_VLLM_PIDS[@]:-}"; do
    _meta_rlvr_terminate_process_group "${pid}" KILL
    wait "${pid}" 2>/dev/null || true
  done
  META_RLVR_VLLM_PIDS=()
}
