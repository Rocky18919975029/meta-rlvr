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
confidence model. The frozen 7B policy is replicated on each GPU so that every
rank can maintain one independent fast adapter. With 4 or 8 GPUs, the global
meta-batch therefore contains 4 or 8 different problems without keeping
multiple task adapters on one device.

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

After the minimal smoke test passes, submit a one-step test that can carry a
nonzero meta signal with 3,072 generated tokens, support/query group size 16,
two inner updates and two outer updates:

```bash
cd "$HOME/meta-rlvr"
conda activate verl
export SMOKE_GPUS=4
bash scripts/submit_meta_meaningful_test.sh
```

This job defaults to an eight-hour Slurm limit and writes to
`outputs/meta-meaningful-$SLURM_JOB_ID`, with progress and metrics in matching
`logs/meta-meaningful-$SLURM_JOB_ID.{err,out}` files. It still performs only one
meta-step; its purpose is to observe mixed verifier rewards, nonzero ranking and
meta losses, two finite confidence-gradient updates, checkpoint saving and
post-adaptation validation before launching a long training run.

The meaningful test also writes one JSONL file per distributed rank:
`rollouts-rank-$RANK.jsonl`. Every record contains the full response, completion
length, length-limit flag, verifier-extracted prediction, official `-1/+1`
reward and binary correctness label. Checkpointing drops completed rollout
tensors and confidence gradients, synchronizes CUDA and empties the caching
allocator before the FSDP save; stdout reports allocated, reserved and peak GPU
memory at that boundary.

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

Policy log-probability forwards default to one response at a time via
`--policy-micro-batch-size 1`, with full-forward activation recomputation so
the `[batch, sequence, vocabulary]` logits are not retained across the whole
group. Confidence forwards default to micro-batches of two. Increase these
only after a server smoke test establishes sufficient memory headroom. Token
log-probabilities are reduced in float32 even when model weights are bfloat16.

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
```

For `floored_group_std`, `--inner-std-floor` is mandatory. Incompatible
combinations raise during configuration construction.

## Multi-iteration semantics

Support and query rollouts are generated once per meta-step. The task adapter
is reset to the shared zero-LoRA initialization and re-adapted under the current
confidence parameters before every outer optimizer iteration. Query old
log-probabilities remain fixed, so PPO ratios and clipping become active as the
confidence model changes.

The first query generation uses a non-differentiable copy of the inner update;
the differentiable inner loop is then recomputed for the outer loss. This
avoids retaining the unrolled graph during autoregressive generation.

The functional AdamW update uses the standard forward formula. For bilevel
backpropagation only, `sqrt(0)` is assigned the zero subgradient; this prevents
the exact-zero initial confidence advantage from producing `0 * inf = NaN`.
The trainer also rejects a non-finite confidence gradient norm before the
optimizer step, so it cannot save a newly corrupted checkpoint.
