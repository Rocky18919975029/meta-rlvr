# Meta-RLVR

Research implementation of verifier-free, confidence-guided test-time policy
adaptation for mathematical RLVR.

The policy and confidence model both initialize from
`Qwen/Qwen2.5-Math-7B`, but they are independent model instances:

- the policy backbone is frozen and only an ephemeral, per-problem LoRA
  adapter is updated in the inner loop;
- the confidence model is a trainable Qwen backbone with the released Qwen
  reward-model head structure: `Linear -> ReLU -> Linear -> scalar` on the
  last non-padding token; both Linear layers use Qwen's normal initialization
  with `config.initializer_range` and zero biases;
- there is no separate confidence pre-training stage.

## Objective

For a support group from one problem, the confidence model produces

```text
c(q, y) = sigmoid(s(q, y)).
```

The inner loop applies configurable GRPO updates to a task-specific adapter
using `c(q, y)` as the sequence reward. A fresh query group is sampled from the
adapted policy and scored by the official DAPO verifier. The outer objective is

```text
lambda_meta * query_verifier_GRPO
+ lambda_bce * support_correctness_BCE
+ lambda_rank * support_Qwen_pairwise_ranking.
```

Support verifier labels are used only by the two confidence supervision terms.
The inner update never consumes them. `adapt_task(...,
supervise_confidence=False)` is the inference path and requires no verifier.
The verifier's official `-1/+1` score is kept separately from its `0/1`
correctness label: outer GRPO consumes the former, while BCE, ranking and
accuracy metrics consume the latter.

## Code structure

- `src/meta_rlvr/confidence.py`: Qwen-style sequence confidence model.
- `src/meta_rlvr/losses.py`: strict BCE, Qwen ranking, advantage variants and
  clipped GRPO objective.
- `src/meta_rlvr/bilevel.py`: differentiable multi-step task adaptation and
  first/second-order outer meta-gradients.
- `src/meta_rlvr/functional.py`: functional fast weights and token log-probs.
- `src/meta_rlvr/rollout.py`: correctness-first Transformers rollout backend.
- `src/meta_rlvr/data.py`: strict DAPO parquet parsing and deduplication.
- `src/meta_rlvr/verifier.py`: binary wrapper around the vendored official DAPO
  verifier.
- `src/meta_rlvr/train.py`: one-problem-per-rank Accelerate/FSDP training loop.

The vendored verifier is pinned to
`BytedTsinghua-SIA/DAPO@33fe3176f0bb212588e84fc8ccf50dd554975144`.

The implementation stays in the Hugging Face/TRL ecosystem
(`Transformers`, `PEFT`, `Accelerate`, FSDP) and mirrors TRL's standard GRPO
knobs. It intentionally does not call `GRPOTrainer` for the bilevel step:
TRL's normal reward boundary is detached and its optimizer owns one persistent
policy, whereas this method must preserve gradients through confidence rewards
and create a fresh functional LoRA state per problem. Ordinary GRPO comparison
runs can still use TRL directly with the same base model and generation/loss
settings.

## Strict behavior

This is research code. Invalid schemas, missing tokenizer fields, incompatible
tensor shapes, unsupported options and context overflow raise immediately. The
code does not silently replace missing values or switch algorithms.

## Local verification

Local development does not require a GPU or model checkpoint. Install a CPU
PyTorch environment and run:

```bash
pip install -e '.[test]'
pytest
```

The tests use toy causal models to verify loss values, masks, PPO clipping,
Qwen ranking pairs, independent task fast weights and the gradient path

```text
query meta loss -> adapted LoRA -> confidence model.
```

They do not download Qwen weights.

## HPC environment

Use Python 3.10 or 3.11 with a CUDA build of PyTorch suitable for the server,
then install the project:

```bash
pip install -e '.[test]'
```

`flash-attn` must be installed because the training entry point defaults to
`--attn-implementation flash_attention_2`. To use PyTorch SDPA explicitly,
pass `--attn-implementation sdpa`.

The default `--meta-gradient-mode first_order` preserves the complete
confidence -> normalized advantage -> adapter -> query-loss path while
stopping the inner policy-gradient Jacobian. The numerical inner update is
still exactly the configured GRPO update. This avoids attention double
backward and is the practical setting for long Qwen trajectories on 4--8
H100s. `--meta-gradient-mode second_order` is provided as an exact ablation,
requires `--attn-implementation eager`, and is expected to be substantially
more memory intensive.

The provided Accelerate configurations use FSDP full sharding for the trainable
confidence model. The frozen 7B policy is replicated on each GPU. By default,
the global meta-batch contains one problem per rank; `--problem-batch-size`
increases it to any positive multiple of the world size. Each rank accumulates
the mean gradient over its local problem batch and performs one confidence
optimizer step only after all local problems finish. Confidence forwards are
packed across the responses of several problems, while each task-dependent
adapter is still constructed and released independently.

Large logical batches use independent rollout and backward batch sizes.
`--rollout-problem-batch-size` controls how many prompts vLLM receives per
continuous-batching transaction, while `--problem-micro-batch-size` controls
how many cached problems are moved to the GPU for one outer backward. All
three values are global, divisible by the distributed world size, and each
sub-batch must divide the logical problem batch. For example,
`--problem-batch-size 512`, `--rollout-problem-batch-size 32`, and
`--problem-micro-batch-size 8` use 16 rollout batches but 64 gradient
accumulation microbatches. This preserves rollout concurrency without making
the meta-gradient working set equally large.

Support and query groups are detached and cached on CPU. The trainer completes
all support rollouts, then all confidence-guided query rollouts, before the
first outer update. During each outer iteration it transfers one cached
microbatch to the GPU, computes its task losses together, backpropagates their
sum scaled by the complete per-rank problem count, releases that microbatch,
and calls the confidence optimizer exactly once after all logical problems.
`--defer-confidence-gradient-sync` additionally suppresses FSDP gradient
collectives until the final accumulation microbatch when memory permits. Thus,
microbatching changes the memory schedule, not the objective, rollout data,
PPO old log-probabilities, number of
outer updates, or effective batch size. Host RAM and output storage still grow
with the complete logical batch. Full response text and the redundant vLLM
diagnostic log-probabilities are written before caching and are not retained in
the training cache.

## Offline HPC synchronization

The target login node is `zhongal@hpc3login.hpc.hkust-gz.edu.cn`. From the
local machine, synchronize code and offline artifacts without deleting
anything already present on the server:

```bash
export META_RLVR_LOCAL=/Users/zeshenghong/.codex/.chatgpt-projects/g-p-6a6eba58defc8191bd0fd7798dc974b8
export META_RLVR_HPC=zhongal@hpc3login.hpc.hkust-gz.edu.cn

ssh "${META_RLVR_HPC}" \
  'mkdir -p meta-rlvr/artifacts/data meta-rlvr/artifacts/models meta-rlvr/artifacts/wheelhouse meta-rlvr/logs meta-rlvr/outputs'

rsync -azP \
  --exclude '.pytest_cache/' \
  --exclude '__pycache__/' \
  --exclude '*.egg-info/' \
  --exclude 'outputs/' \
  "${META_RLVR_LOCAL}/" \
  "${META_RLVR_HPC}:meta-rlvr/"

rsync -ahP /local/path/DAPO-17k.parquet \
  "${META_RLVR_HPC}:meta-rlvr/artifacts/data/DAPO-17k.parquet"
rsync -ahP /local/path/AIME24.parquet \
  "${META_RLVR_HPC}:meta-rlvr/artifacts/data/AIME24.parquet"
rsync -ahP /local/path/Qwen2.5-Math-7B/ \
  "${META_RLVR_HPC}:meta-rlvr/artifacts/models/Qwen2.5-Math-7B/"
```

A macOS virtual environment cannot be copied to the Linux HPC. Either use a
cluster-provided PyTorch/CUDA environment, or build a wheelhouse on an
internet-connected Linux x86_64 machine with matching Python/CUDA versions,
then synchronize it to `meta-rlvr/artifacts/wheelhouse`.

The existing conda environment `verl` has been verified with Python 3.12,
PyTorch 2.8.0+cu128, Transformers 4.56.1, PEFT 0.19.1, Accelerate 1.14.0,
Datasets 5.0.0 and tqdm 4.68.3. Its old torchao 0.9.0 installation is
incompatible with PEFT 0.19.1 during LoRA injection and is unused by this BF16
experiment. After confirming that no installed package requires torchao, remove
it and install the local project directly in `verl`:

```bash
conda activate verl
python -m pip uninstall -y torchao
cd "$HOME/meta-rlvr"
python -m pip install --no-deps --no-build-isolation -e .
```

Confirm that the conflicting package is gone:

```bash
python - <<'PY'
from importlib.util import find_spec
print("torchao spec:", find_spec("torchao"))
PY
```

It must print `torchao spec: None`. The optional TRL baseline extra is not
needed by this custom bilevel trainer. To use an offline venv instead:

```bash
module load python cuda  # replace with the cluster's actual module names
python -m venv --system-site-packages "$HOME/venvs/meta-rlvr"
source "$HOME/venvs/meta-rlvr/bin/activate"
python -m pip install --no-index \
  --no-build-isolation \
  --find-links "$HOME/meta-rlvr/artifacts/wheelhouse" \
  -e "$HOME/meta-rlvr[test]"
```

The smoke test uses PyTorch SDPA, so `flash-attn` is not required for this
first distributed check.

Inspect the exact dataset messages, tokenizer-owned Jinja template and rendered
prompt used by training with:

```bash
python -m meta_rlvr.inspect_prompt \
  --parquet /data/user/zhongal/data/reschedule/DAPO-Math-17k.filtered.seed42.sample1536.parquet \
  --model /data/user/zhongal/.cache/qwen2.5-math-7b-local \
  --row 0
```

Use `--uid <extra_info.index>` instead of `--row` to inspect the exact problem
identifier printed in a training progress line.

## Slurm smoke test

The configured HPC artifacts are:

```text
full training: /data/user/zhongal/data/reschedule/DAPO-Math-17k.parquet
smoke subset:  /data/user/zhongal/data/reschedule/DAPO-Math-17k.filtered.seed42.sample1536.parquet
validation:    /data/user/zhongal/data/reschedule/aime24.parquet
model:         /data/user/zhongal/.cache/qwen2.5-math-7b-local
```

The smoke submitter uses the 1,536-problem subset by default. These paths,
`$HOME/meta-rlvr`, conda environment `verl` and the output directory are already
encoded as overridable defaults. The detected Slurm defaults are partition
`acd_u` and generic GRES `gpu:4`; `debug` is deliberately not used because its
30-minute limit is too short for a reliable first run. `SLURM_ACCOUNT` and
`SLURM_QOS` remain optional.

```bash
cd "$HOME/meta-rlvr"
export META_RLVR_CONDA_ENV=verl
export SMOKE_GPUS=4

bash scripts/submit_smoke_test.sh
```

Training defaults to the official DAPO strict-box verifier. The `minerva` path
is retained only as an explicit `--verifier-mode minerva` ablation.

Before testing outer optimization, run the two-rollout path by itself:

```bash
cd "$HOME/meta-rlvr"
conda activate verl
export SMOKE_GPUS=4
bash scripts/submit_rollout_test.sh
```

This generates support K=16, performs two confidence-guided inner updates and
generates query K=16, then exits before outer backward, checkpointing and
validation. The HPC meaningful and rollout-only jobs default to four colocated
TP=1 vLLM replicas. Each replica continuously batches a full K=16 group, uses
the exact task-dependent LoRA for both support and query, and returns generated
token IDs to PyTorch for old-log-probability computation. Full responses and
strict-box verifier predictions are written to per-rank JSONL files under
`outputs/rollout-test-$SLURM_JOB_ID`.

The vLLM lifecycle follows verl's hybrid engine ordering. Servers start first
and enter level-1 sleep. A complete support or query phase wakes `weights`
once, dynamically swaps the required detached LoRAs from node-local `/dev/shm`
at rollout-problem-batch boundaries, wakes `kv_cache` once, and sleeps before
PyTorch performs log-probability or gradient work. Support exports and loads
the shared initial adapter once per phase; query keeps only the current rollout
batch's task adapters registered. The
launcher explicitly enables vLLM's development lifecycle endpoints and
supervises every server process. An HTTP 500, a missing endpoint or a dead
replica terminates the trainer and the Slurm job instead of waiting through a
long timeout. The Transformers backend remains available for CPU tests via
`SMOKE_ROLLOUT_BACKEND=transformers`; it is not the default for the H100
meaningful tests. Per-replica vLLM logs and throughput metrics are under
`logs/vllm-$SLURM_JOB_ID/gpu-*.log`.

Recent FastAPI releases require `prometheus-fastapi-instrumentator>=8.0.1`.
The submitters validate this on the login node before requesting any GPU. Since
the HPC has no network, download the universal wheel on a networked machine,
copy it with the project wheelhouse, and install it into `verl` once:

```bash
# Networked local machine
python -m pip download --no-deps \
  prometheus-fastapi-instrumentator==8.0.1 \
  -d /tmp/meta-rlvr-wheelhouse
META_RLVR_LOCAL_WHEELHOUSE=/tmp/meta-rlvr-wheelhouse \
  bash scripts/sync_to_hpc.sh

# HPC login node
conda activate verl
python -m pip install --no-deps \
  "$HOME/meta-rlvr/artifacts/wheelhouse/prometheus_fastapi_instrumentator-8.0.1-"*.whl
```

Before any four-GPU job, run the capped 30-minute, one-H100 lifecycle test. It
uses K=2 and 32 generated tokens but still executes support rollout, confidence
adaptation and query rollout:

```bash
cd "$HOME/meta-rlvr"
conda activate verl
bash scripts/submit_vllm_lifecycle_test.sh
```

The conservative colocated allocation is 30% of each H100 for vLLM. This leaves
room for the FSDP confidence model and AdamW state after the first outer update:

```bash
export SMOKE_GPUS=4
bash scripts/submit_vllm_smoke_test.sh

# Only after the short hybrid lifecycle test succeeds:
export VLLM_GPU_MEMORY_UTILIZATION=0.30
export VLLM_MAX_NUM_SEQS=64
export VLLM_MAX_LORAS=1
bash scripts/submit_rollout_test.sh
```

After the one-problem-per-rank smoke test passes, test a global problem batch of
32 using short K=2, 128-token rollouts:

```bash
export SMOKE_GPUS=4
bash scripts/submit_problem_batch_test.sh
```

This gives each rank eight independent problems. Support prompts share one
initial LoRA and enter one vLLM phase transaction. Query prompts use eight
task-specific LoRAs in the same phase transaction. The differentiable outer
pass batches confidence scoring and performs one backward per problem
microbatch, clips once and performs one optimizer step. To test B=64 on eight
GPUs with the same script:

```bash
export SMOKE_GPUS=8
export SMOKE_PROBLEM_BATCH_SIZE=64
export SMOKE_PROBLEM_MICRO_BATCH_SIZE=32
export SMOKE_ROLLOUT_PROBLEM_BATCH_SIZE=64
bash scripts/submit_problem_batch_test.sh
```

This generates all eight local problems together while exercising two logical
gradient-accumulation microbatches of four problems per rank. The optimizer
still averages all eight local problems. The launcher derives
`VLLM_MAX_LORAS` from the per-rank rollout problem batch and fails before
requesting generation if the configured vLLM adapter capacity is too small.

The B=512 layout is supported without requiring 64 or 128 simultaneous task
adapters per rank. Validate the memory schedule with short rollout-only data
before attempting an expensive full-length run:

```bash
export SMOKE_GPUS=8
export SMOKE_PROBLEM_BATCH_SIZE=512
export SMOKE_ROLLOUT_PROBLEM_BATCH_SIZE=32
export SMOKE_PROBLEM_MICRO_BATCH_SIZE=32
export SMOKE_ROLLOUT_ONLY=1
export SLURM_MEM=300G
bash scripts/submit_problem_batch_test.sh
```

This produces 16 rollout batches and 16 gradient microbatches globally, with
four problems per rank in each. They can be tuned independently; reducing only
`SMOKE_PROBLEM_MICRO_BATCH_SIZE` lowers backward memory without reducing vLLM
concurrency.
Unset `SMOKE_ROLLOUT_ONLY` to test the accumulated outer backward and single
optimizer step. On four GPUs the same global microbatch means eight resident
problems per rank; use `SMOKE_OFFLOAD_CONFIDENCE_OPTIMIZER=1` for multi-outer
training after the optimizer state has been created. Full K=16, 3,072-token
B=512 runs are intrinsically very expensive, so scale through B=64 and B=128
first and size `SLURM_MEM` from observed host-memory use.

The local Qwen checkpoint declares `max_position_embeddings=4096`, so the
launcher deliberately fixes vLLM `max_model_len=4096`. Do not set
`VLLM_ALLOW_LONG_MAX_MODEL_LEN`; the training code separately rejects any
prompt whose prompt tokens plus `max_new_tokens` exceed the same limit.

Each colocated replica receives an independent node-local vLLM, TorchInductor,
Triton and CUDA cache. This is required because every TP=1 server otherwise
writes the same `rank_0_0` compile-cache paths concurrently. The launcher also
defaults `VLLM_USE_FLASHINFER_SAMPLER=0`: the cluster's default host GCC is too
old to JIT-build FlashInfer's sampling extension, while vLLM's native PyTorch
sampler requires no such build. This changes only the sampling kernel, not the
temperature/top-p distribution. FlashInfer can be re-enabled later after a
GCC 9+ toolchain is loaded and its kernels are precompiled.

After the rollout-only test passes, submit a one-step test that can carry a
nonzero meta signal with 3,072 generated tokens, support/query group size 16,
two inner updates and two outer updates:

```bash
cd "$HOME/meta-rlvr"
conda activate verl
export SMOKE_GPUS=4
bash scripts/submit_meta_meaningful_test.sh
```

For the first full B=32 check, use eight GPUs without optimizer offload. This
separates correctness of the complete two-outer-update run from the 4-GPU
memory optimization:

```bash
conda activate verl
cd "$HOME/meta-rlvr"
export SMOKE_GPUS=8
export SMOKE_PROBLEM_BATCH_SIZE=32
export SMOKE_PROBLEM_MICRO_BATCH_SIZE=32
export SMOKE_ROLLOUT_PROBLEM_BATCH_SIZE=32
export META_RLVR_RUN_LABEL=meta-meaningful-b32-8gpu
export VLLM_MAX_LORAS=4
export VLLM_MAX_CPU_LORAS=4
export VLLM_MAX_NUM_SEQS=64
export SLURM_TIME=08:00:00
export SLURM_MEM=200G
unset SMOKE_OFFLOAD_CONFIDENCE_OPTIMIZER
bash scripts/submit_meta_meaningful_test.sh
```

On four GPUs, the first outer AdamW step creates FP32 `exp_avg` and
`exp_avg_sq` tensors. Keeping those states resident during the second outer
forward/backward can exhaust an 80 GB H100. Enable the targeted optimizer-state
offload as follows:

```bash
conda activate verl
cd "$HOME/meta-rlvr"
export SMOKE_GPUS=4
export SMOKE_PROBLEM_BATCH_SIZE=32
export SMOKE_PROBLEM_MICRO_BATCH_SIZE=32
export SMOKE_ROLLOUT_PROBLEM_BATCH_SIZE=32
export META_RLVR_RUN_LABEL=meta-meaningful-b32-4gpu-offload
export SMOKE_OFFLOAD_CONFIDENCE_OPTIMIZER=1
export SLURM_MEM=200G
export VLLM_MAX_LORAS=8
export VLLM_MAX_CPU_LORAS=8
export VLLM_MAX_NUM_SEQS=64
export SLURM_TIME=08:00:00
bash scripts/submit_meta_meaningful_test.sh
```

`--offload-confidence-optimizer` moves only AdamW moment tensors. They remain
on CPU during the outer forward/backward, move to the local GPU immediately
before `optimizer.step()`, and return to CPU immediately afterward. Parameters,
gradients, the confidence forward, and all bilevel losses are unchanged. Each
transfer prints its per-rank state size, duration, and current CUDA allocated
and reserved memory. Checkpoint save/resume preserves the same lifecycle.
Because the CPU copy is sharded across ranks but colocated vLLM replicas also
sleep their weights in host memory, request sufficient node RAM; `SLURM_MEM`
is passed directly to `sbatch` by both smoke submitters.

Do not enable expandable segments by default for this hybrid job. vLLM 0.11
sleep mode uses `CuMemAllocator`, whose memory pool is incompatible with that
setting. As a defensive measure, the hybrid launcher explicitly removes both
`PYTORCH_CUDA_ALLOC_CONF` and `PYTORCH_ALLOC_CONF` from each vLLM server process
if a custom allocator setting is intentionally enabled for the separate
Accelerate/FSDP trainer.

This job defaults to an eight-hour Slurm limit and writes to
`outputs/meta-meaningful-$SLURM_JOB_ID`, with progress and metrics in matching
`logs/meta-meaningful-$SLURM_JOB_ID.{err,out}` files. It still performs only one
meta-step; its purpose is to observe mixed verifier rewards, nonzero ranking and
meta losses, two finite confidence-gradient updates, checkpoint saving and
post-adaptation validation before launching a long training run.

The meaningful test also writes one JSONL file per distributed rank:
`rollouts-rank-$RANK.jsonl`. Every record contains the full response, completion
length, length-limit flag, verifier-extracted prediction, official `-1/+1`
reward and binary correctness label. vLLM is pinned to return raw sampled-token
log-probabilities, and the records contain their deltas against PyTorch's raw
recomputation; an aggregate mean/maximum diagnostic is printed after every
rollout. These values
make rollout-policy mismatch observable before a long run. Checkpointing drops
completed rollout tensors and confidence gradients, synchronizes CUDA and
empties the caching allocator before the FSDP save; stdout reports allocated,
reserved and peak GPU memory at that boundary.

The equivalent explicit settings are:

```bash
export SLURM_PARTITION=acd_u
export SLURM_GRES=gpu:4
```

For a full training launch, use the complete DAPO file explicitly:

```bash
export META_RLVR_TRAIN_PARQUET=/data/user/zhongal/data/reschedule/DAPO-Math-17k.parquet
export META_RLVR_VALIDATION_PARQUET=/data/user/zhongal/data/reschedule/aime24.parquet
export META_RLVR_MODEL_PATH=/data/user/zhongal/.cache/qwen2.5-math-7b-local
```

The submitter prints the job id and exact monitoring commands. Slurm stdout
contains environment diagnostics, NCCL information and JSON training metrics;
stderr contains nested progress bars for generation, old/reference
log-probabilities, inner adapter updates, query forwards, outer updates and
validation. Monitor them with:

```bash
squeue -j JOB_ID
tail -f "$HOME/meta-rlvr/logs/smoke-JOB_ID.err"
tail -f "$HOME/meta-rlvr/logs/smoke-JOB_ID.out"
```

## Launch

The generation length is deliberately required rather than inferred. It must
satisfy

```text
tokenized prompt length + max_new_tokens <= model max_position_embeddings.
```

Example for four H100s:

```bash
export META_RLVR_TRAIN_PARQUET=/path/to/DAPO-17k.parquet
export META_RLVR_VALIDATION_PARQUET=/path/to/AIME24.parquet
export META_RLVR_MODEL_PATH=/path/to/Qwen2.5-Math-7B
export META_RLVR_OUTPUT_DIR=/path/to/output
export META_RLVR_MAX_NEW_TOKENS=3072

bash scripts/launch_4xh100.sh \
  --max-steps 1000 \
  --inner-iterations 4 \
  --outer-iterations 16
```

AIME24's replicated rollout rows are collapsed by prompt, answer and data
source before evaluation. Validation adaptation receives no correctness
labels: the verifier is called only afterward to report base/adapted accuracy
and pass-at-group. `--eval-steps 0` (the default) evaluates once at the end;
positive values enable periodic evaluation.

Before a long run, use a one-step distributed smoke test that exercises model
loading, FSDP, rollout, verifier, inner/outer gradients, checkpoint saving and
validation while keeping generation short:

```bash
bash scripts/launch_4xh100.sh \
  --max-steps 1 \
  --save-steps 1 \
  --validation-max-problems 4 \
  --support-group-size 2 \
  --query-group-size 2 \
  --generation-micro-batch-size 1 \
  --max-new-tokens 128 \
  --inner-iterations 1 \
  --outer-iterations 1
```

Resume only from a completed step checkpoint, for example
`--resume-from-checkpoint /path/to/output/checkpoint-50`; the numeric suffix is
used as the next meta-step index. The frozen policy is reloaded from the base
checkpoint, while Accelerate restores the sharded confidence model, optimizer
and RNG state.

Use `scripts/launch_8xh100.sh` for eight H100s.

Policy and confidence forwards accept both a maximum sequence count and a
dense-token budget through `--policy-micro-batch-size`,
`--confidence-micro-batch-size`, `--policy-max-tokens-per-micro-batch`, and
`--confidence-max-tokens-per-micro-batch`. Rows are packed longest-first and
right padding is trimmed. Qwen receives `logits_to_keep`, so policy forwards
project only completion positions to the vocabulary. Token log-probabilities
are still reduced in float32 even when model weights are bfloat16.

Important configurable ablations include:

```text
--inner-advantage-scale group_std|center_only|floored_group_std|none
--inner-baseline group_mean|leave_one_out|none
--inner-group-gate none|max_confidence|probability_any
--disable-importance-ratio
--disable-clipping
--token-normalization per_response|global_tokens|sequence_sum
--bce-coefficient 0
--ranking-coefficient 0
--meta-coefficient 0
--meta-gradient-mode first_order|second_order
--offload-confidence-optimizer
--log-component-gradient-norms
--component-gradient-norm-interval N
```

For `floored_group_std`, `--inner-std-floor` is mandatory. Incompatible
combinations raise during configuration construction.

`--log-component-gradient-norms` measures the exact parameter-gradient norm of
the meta, BCE and ranking objectives with respect to the confidence model at
the start of the selected outer update. The JSON metrics contain both raw
norms (`gradient_norm/meta_raw`, `gradient_norm/bce_raw`, and
`gradient_norm/ranking_raw`) and coefficient-weighted norms with the matching
`_weighted` suffix. The existing `confidence_gradient_norm` remains the norm
of their summed, coefficient-weighted gradient immediately before clipping.
The diagnostic clears its gradients before the real total-loss backward and
therefore does not alter the update. It is intentionally opt-in: an exact meta
norm repeats one complete bilevel forward/backward, while BCE and ranking each
repeat a confidence-only forward/backward. Use
`--component-gradient-norm-interval N` to sample every N outer updates.
The trainer restores the per-rank CPU/CUDA RNG state after diagnostics, so the
extra forwards do not change the random stream used by the real update.

The HPC smoke launcher exposes the same controls as environment variables:

```bash
export SMOKE_LOG_COMPONENT_GRADIENT_NORMS=1
export SMOKE_COMPONENT_GRADIENT_NORM_INTERVAL=1
```

## Multi-iteration semantics

Support and query rollouts are generated once per meta-step and problem. The
task adapter is reset to the shared zero-LoRA initialization and re-adapted
under the current confidence parameters before every outer optimizer iteration.
Query old log-probabilities remain fixed, so PPO ratios and clipping become
active as the confidence model changes. For a problem batch, all task losses are
scaled by the per-rank problem count and accumulated before the single outer
optimizer step; FSDP averaging then yields the exact global problem mean.

The first query generation uses a non-differentiable copy of the inner update;
the differentiable inner loop is then recomputed for the outer loss. This
avoids retaining the unrolled graph during autoregressive generation.

The functional AdamW update uses the standard forward formula. For bilevel
backpropagation only, `sqrt(0)` is assigned the zero subgradient; this prevents
the exact-zero initial confidence advantage from producing `0 * inf = NaN`.
The trainer also rejects a non-finite confidence gradient norm before the
optimizer step, so it cannot save a newly corrupted checkpoint.
