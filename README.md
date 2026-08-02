# Meta-RLVR

Research implementation of verifier-free, confidence-guided test-time policy
adaptation for mathematical RLVR.

The policy and confidence model both initialize from
`Qwen/Qwen2.5-Math-7B`, but they are independent model instances:

- the policy backbone is frozen and only an ephemeral, per-problem LoRA
  adapter is updated in the inner loop;
- the confidence model is a trainable Qwen backbone with the released Qwen
  reward-model head structure: `Linear -> ReLU -> Linear -> scalar` on the
  last non-padding token;
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
