from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .bilevel import BilevelGRPO
from .config import (
    AdvantageConfig,
    ConfidenceLossConfig,
    FastOptimizerConfig,
    GRPOLossConfig,
    InnerLoopConfig,
    MetaLossConfig,
)
from .data import (
    MathProblem,
    load_semantically_unique_dapo_problems,
    load_unique_dapo_problems,
    rank_shard,
)
from .models import load_confidence_model, load_policy_with_lora
from .rollout import TransformersRolloutEngine, VLLMHybridRolloutEngine
from .types import RolloutGroup
from .verifier import DAPOMathVerifier, VerificationBatch


@dataclass(frozen=True)
class CachedRolloutMicrobatch:
    problems: tuple[MathProblem, ...]
    supports: tuple[RolloutGroup, ...]
    queries: tuple[RolloutGroup, ...] | None = None

    def __post_init__(self) -> None:
        if not self.problems or len(self.problems) != len(self.supports):
            raise ValueError(
                "A cached rollout microbatch requires equal non-empty problem "
                "and support groups."
            )
        if self.queries is not None and len(self.queries) != len(self.problems):
            raise ValueError(
                "Cached query groups must match the problem microbatch size."
            )
        groups = self.supports
        if self.queries is not None:
            groups += self.queries
        if any(group.device.type != "cpu" for group in groups):
            raise ValueError("Cached rollout microbatches must reside on CPU.")


@dataclass
class StageTimings:
    values: dict[str, float]

    @classmethod
    def create(cls) -> StageTimings:
        return cls(values={})

    @contextmanager
    def measure(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.values[name] = self.values.get(name, 0.0) + (
                time.perf_counter() - started
            )


def _cache_rollout_group(group: RolloutGroup) -> RolloutGroup:
    # Text and vLLM/PyTorch delta diagnostics have already been written before
    # caching and are not inputs to either bilevel loss. Dropping them avoids
    # retaining and repeatedly transferring large response strings and a
    # second dense [K, L-1] log-probability tensor for every logical problem.
    training_group = replace(
        group,
        texts=("",) * group.group_size,
        rollout_logprobs=None,
    )
    return training_group.to("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bilevel confidence-guided GRPO for Qwen2.5-Math-7B."
    )
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--validation-parquet", type=Path, required=True)
    parser.add_argument("--validation-max-problems", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-model", default="Qwen/Qwen2.5-Math-7B")
    parser.add_argument("--confidence-model", default="Qwen/Qwen2.5-Math-7B")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated PEFT target module names.",
    )

    parser.add_argument("--support-group-size", type=int, default=32)
    parser.add_argument("--query-group-size", type=int, default=32)
    parser.add_argument(
        "--problem-batch-size",
        type=int,
        help=(
            "Global number of independent problems per confidence-model update. "
            "Defaults to one problem per distributed rank."
        ),
    )
    parser.add_argument(
        "--problem-micro-batch-size",
        type=int,
        help=(
            "Global number of problems generated and moved to GPU at once. "
            "Defaults to problem_batch_size; gradients still accumulate over "
            "the complete problem batch before each outer optimizer step."
        ),
    )
    parser.add_argument("--generation-micro-batch-size", type=int, default=4)
    parser.add_argument("--policy-micro-batch-size", type=int, default=1)
    parser.add_argument("--confidence-micro-batch-size", type=int, default=2)
    parser.add_argument("--policy-max-tokens-per-micro-batch", type=int)
    parser.add_argument("--confidence-max-tokens-per-micro-batch", type=int)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument(
        "--rollout-backend",
        choices=["transformers", "vllm"],
        default="transformers",
    )
    parser.add_argument(
        "--vllm-base-urls",
        help="Comma-separated colocated vLLM server URLs, one per process rank.",
    )
    parser.add_argument("--vllm-adapter-root", type=Path, default=Path("/dev/shm"))
    parser.add_argument("--vllm-request-timeout", type=float, default=900.0)
    parser.add_argument("--vllm-control-timeout", type=float, default=120.0)

    parser.add_argument("--inner-iterations", type=int, default=4)
    parser.add_argument(
        "--meta-gradient-mode",
        choices=["first_order", "second_order"],
        default="first_order",
        help=(
            "first_order retains confidence-to-adapter gradients without "
            "attention double backward; second_order is an expensive exact ablation"
        ),
    )
    parser.add_argument("--outer-iterations", type=int, default=16)
    parser.add_argument("--inner-learning-rate", type=float, default=1e-5)
    parser.add_argument("--inner-optimizer", choices=["sgd", "adamw"], default="adamw")
    parser.add_argument("--inner-weight-decay", type=float, default=0.0)
    parser.add_argument("--confidence-learning-rate", type=float, default=1e-6)
    parser.add_argument("--confidence-weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--offload-confidence-optimizer",
        action="store_true",
        help=(
            "Keep AdamW moment tensors on CPU during outer forward/backward and "
            "move them to the GPU only for optimizer.step()."
        ),
    )
    parser.add_argument(
        "--defer-confidence-gradient-sync",
        action="store_true",
        help=(
            "Use FSDP no_sync for all but the final problem accumulation "
            "microbatch. This reduces collectives but retains full gradients "
            "until the final synchronized backward."
        ),
    )
    parser.add_argument(
        "--log-component-gradient-norms",
        action="store_true",
        help=(
            "Measure exact confidence-parameter gradient norms for the meta, BCE, "
            "and ranking losses using diagnostic backward passes."
        ),
    )
    parser.add_argument(
        "--component-gradient-norm-interval",
        type=int,
        default=1,
        help="Measure component gradient norms every N outer updates.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument(
        "--inner-advantage-scale",
        choices=["group_std", "center_only", "floored_group_std", "none"],
        default="group_std",
    )
    parser.add_argument(
        "--inner-baseline",
        choices=["group_mean", "leave_one_out", "none"],
        default="group_mean",
    )
    parser.add_argument(
        "--inner-group-gate",
        choices=["none", "max_confidence", "probability_any"],
        default="none",
    )
    parser.add_argument("--inner-std-floor", type=float)
    parser.add_argument("--std-epsilon", type=float, default=1e-4)
    parser.add_argument(
        "--detach-inner-group-stats",
        action="store_true",
        help="Ablation only: stop gradients through confidence group statistics.",
    )

    parser.add_argument("--clip-epsilon-low", type=float, default=0.2)
    parser.add_argument("--clip-epsilon-high", type=float, default=0.2)
    parser.add_argument("--disable-importance-ratio", action="store_true")
    parser.add_argument("--disable-clipping", action="store_true")
    parser.add_argument("--inner-kl-coefficient", type=float, default=0.0)
    parser.add_argument("--outer-kl-coefficient", type=float, default=0.0)
    parser.add_argument(
        "--token-normalization",
        choices=["per_response", "global_tokens", "sequence_sum"],
        default="per_response",
    )

    parser.add_argument("--meta-coefficient", type=float, default=1.0)
    parser.add_argument("--bce-coefficient", type=float, default=1.0)
    parser.add_argument("--ranking-coefficient", type=float, default=1.0)
    parser.add_argument(
        "--verifier-mode",
        choices=["strict_box", "minerva"],
        default="strict_box",
    )
    parser.add_argument("--log-rollouts", action="store_true")
    parser.add_argument("--rollout-only", action="store_true")

    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=0,
        help="Evaluate every N meta-steps; zero evaluates only at the end.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _grpo_config(args: argparse.Namespace, *, kl_coefficient: float) -> GRPOLossConfig:
    use_ratio = not args.disable_importance_ratio
    use_clipping = not args.disable_clipping
    return GRPOLossConfig(
        use_importance_ratio=use_ratio,
        use_clipping=use_clipping,
        clip_epsilon_low=args.clip_epsilon_low,
        clip_epsilon_high=args.clip_epsilon_high,
        kl_coefficient=kl_coefficient,
        token_normalization=args.token_normalization,
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.outer_iterations <= 0:
        raise ValueError("outer_iterations must be positive.")
    if args.max_steps <= 0 or args.save_steps <= 0:
        raise ValueError("max_steps and save_steps must be positive.")
    if args.eval_steps < 0:
        raise ValueError("eval_steps must be non-negative.")
    if args.validation_max_problems is not None and args.validation_max_problems <= 0:
        raise ValueError("validation_max_problems must be positive.")
    if args.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive.")
    if args.confidence_learning_rate <= 0:
        raise ValueError("confidence_learning_rate must be positive.")
    if args.component_gradient_norm_interval <= 0:
        raise ValueError("component_gradient_norm_interval must be positive.")
    if args.problem_batch_size is not None and args.problem_batch_size <= 0:
        raise ValueError("problem_batch_size must be positive.")
    if args.problem_micro_batch_size is not None and args.problem_micro_batch_size <= 0:
        raise ValueError("problem_micro_batch_size must be positive.")
    if (
        args.policy_max_tokens_per_micro_batch is not None
        and args.policy_max_tokens_per_micro_batch <= 0
    ):
        raise ValueError("policy_max_tokens_per_micro_batch must be positive.")
    if (
        args.confidence_max_tokens_per_micro_batch is not None
        and args.confidence_max_tokens_per_micro_batch <= 0
    ):
        raise ValueError("confidence_max_tokens_per_micro_batch must be positive.")
    if args.rollout_backend == "vllm" and not args.vllm_base_urls:
        raise ValueError("--vllm-base-urls is required for the vLLM backend.")
    if args.vllm_request_timeout <= 0:
        raise ValueError("--vllm-request-timeout must be positive.")
    if args.vllm_control_timeout <= 0:
        raise ValueError("--vllm-control-timeout must be positive.")
    if args.top_k < 0:
        raise ValueError("--top-k must be non-negative.")
    if args.confidence_weight_decay < 0:
        raise ValueError("confidence_weight_decay must be non-negative.")
    if (
        args.meta_gradient_mode == "second_order"
        and args.attn_implementation != "eager"
    ):
        raise ValueError(
            "second_order meta-gradients require --attn-implementation eager; "
            "fused attention kernels do not provide the required double backward."
        )
    target_modules = tuple(
        module.strip() for module in args.lora_target_modules.split(",")
    )
    if not target_modules or any(not module for module in target_modules):
        raise ValueError(
            "lora_target_modules must be a non-empty comma-separated list."
        )


def _configs(args: argparse.Namespace):
    inner_advantage = AdvantageConfig(
        baseline=args.inner_baseline,
        scale=args.inner_advantage_scale,
        std_epsilon=args.std_epsilon,
        std_floor=args.inner_std_floor,
        group_gate=args.inner_group_gate,
        differentiate_group_stats=not args.detach_inner_group_stats,
    )
    inner = InnerLoopConfig(
        num_iterations=args.inner_iterations,
        meta_gradient_mode=args.meta_gradient_mode,
        advantage=inner_advantage,
        grpo=_grpo_config(args, kl_coefficient=args.inner_kl_coefficient),
        optimizer=FastOptimizerConfig(
            name=args.inner_optimizer,
            learning_rate=args.inner_learning_rate,
            weight_decay=args.inner_weight_decay,
        ),
    )
    meta = MetaLossConfig(
        meta_coefficient=args.meta_coefficient,
        confidence=ConfidenceLossConfig(
            bce_coefficient=args.bce_coefficient,
            ranking_coefficient=args.ranking_coefficient,
        ),
    )
    query_advantage = AdvantageConfig(
        baseline="group_mean",
        scale="group_std",
        std_epsilon=args.std_epsilon,
        group_gate="none",
        differentiate_group_stats=False,
    )
    query_grpo = _grpo_config(args, kl_coefficient=args.outer_kl_coefficient)
    return inner, meta, query_advantage, query_grpo


def _next_problem(
    shard: list[MathProblem],
    *,
    step: int,
    seed: int,
) -> MathProblem:
    epoch, offset = divmod(step, len(shard))
    order = list(range(len(shard)))
    random.Random(seed + epoch).shuffle(order)
    return shard[order[offset]]


def _problem_batch(
    shard: list[MathProblem],
    *,
    step: int,
    batch_size: int,
    seed: int,
) -> list[MathProblem]:
    if not shard:
        raise ValueError("Problem shard cannot be empty.")
    if step < 0:
        raise ValueError("step must be non-negative.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    first_position = step * batch_size
    return [
        _next_problem(shard, step=first_position + offset, seed=seed)
        for offset in range(batch_size)
    ]


def _local_problem_batch_size(
    global_batch_size: int | None,
    *,
    world_size: int,
) -> tuple[int, int]:
    if world_size <= 0:
        raise ValueError("world_size must be positive.")
    effective_global_batch = (
        world_size if global_batch_size is None else global_batch_size
    )
    if effective_global_batch < world_size:
        raise ValueError(
            "problem_batch_size must be at least the distributed world size."
        )
    if effective_global_batch % world_size != 0:
        raise ValueError(
            "problem_batch_size must be divisible by the distributed world size."
        )
    return effective_global_batch, effective_global_batch // world_size


def _problem_batch_layout(
    global_batch_size: int | None,
    global_micro_batch_size: int | None,
    *,
    world_size: int,
) -> tuple[int, int, int, int, int]:
    global_batch, local_batch = _local_problem_batch_size(
        global_batch_size,
        world_size=world_size,
    )
    global_micro_batch = (
        global_batch if global_micro_batch_size is None else global_micro_batch_size
    )
    if global_micro_batch < world_size:
        raise ValueError(
            "problem_micro_batch_size must be at least the distributed world size."
        )
    if global_micro_batch > global_batch:
        raise ValueError("problem_micro_batch_size cannot exceed problem_batch_size.")
    if global_micro_batch % world_size != 0:
        raise ValueError(
            "problem_micro_batch_size must be divisible by the distributed world size."
        )
    if global_batch % global_micro_batch != 0:
        raise ValueError(
            "problem_batch_size must be divisible by problem_micro_batch_size."
        )
    local_micro_batch = global_micro_batch // world_size
    num_micro_batches = global_batch // global_micro_batch
    if local_batch != local_micro_batch * num_micro_batches:
        raise RuntimeError("Inconsistent distributed problem microbatch layout.")
    return (
        global_batch,
        local_batch,
        global_micro_batch,
        local_micro_batch,
        num_micro_batches,
    )


def _write_run_config(args: argparse.Namespace, output_dir: Path, accelerator) -> None:
    if not accelerator.is_main_process:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    path = output_dir / "run_config.json"
    path.write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n")


def _append_rollout_diagnostics(
    path: Path | None,
    *,
    step: int,
    phase: str,
    problem: MathProblem,
    group: RolloutGroup,
    verification: VerificationBatch,
    max_new_tokens: int,
    eos_token_id: int,
) -> None:
    if path is None:
        return
    completion_lengths = group.completion_mask.sum(dim=1).tolist()
    if len(verification.predictions) != group.group_size:
        raise ValueError("Verifier predictions must match rollout group size.")
    with path.open("a", encoding="utf-8") as stream:
        for response_index, response in enumerate(group.texts):
            completion_tokens = int(completion_lengths[response_index])
            completion_ids = group.input_ids[response_index, 1:][
                group.completion_mask[response_index]
            ]
            ended_with_eos = int(completion_ids[-1].item()) == eos_token_id
            record = {
                "step": step,
                "phase": phase,
                "problem_uid": problem.uid,
                "data_source": problem.data_source,
                "ground_truth": problem.ground_truth,
                "response_index": response_index,
                "completion_tokens": completion_tokens,
                "ended_with_eos": ended_with_eos,
                "hit_max_new_tokens": (
                    completion_tokens == max_new_tokens and not ended_with_eos
                ),
                "prediction": verification.predictions[response_index],
                "reward": verification.rewards[response_index].item(),
                "correct": verification.correctness[response_index].item(),
                "response": response,
            }
            if group.rollout_logprobs is not None:
                selected = group.completion_mask[response_index]
                delta = (
                    group.rollout_logprobs[response_index, selected]
                    - group.old_logprobs[response_index, selected]
                )
                record.update(
                    {
                        "vllm_raw_vs_pytorch_raw_mean_delta": (delta.mean().item()),
                        "vllm_raw_vs_pytorch_raw_mean_absolute_delta": (
                            delta.abs().mean().item()
                        ),
                        "vllm_raw_vs_pytorch_raw_max_absolute_delta": (
                            delta.abs().max().item()
                        ),
                    }
                )
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()


def _prepare_for_checkpoint(confidence_optimizer, accelerator) -> None:
    confidence_optimizer.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.synchronize(accelerator.device)
    torch.cuda.empty_cache()
    accelerator.wait_for_everyone()

    memory = torch.tensor(
        (
            torch.cuda.memory_allocated(accelerator.device),
            torch.cuda.memory_reserved(accelerator.device),
            torch.cuda.max_memory_allocated(accelerator.device),
        ),
        dtype=torch.float64,
        device=accelerator.device,
    )
    memory = accelerator.reduce(memory, reduction="max") / 1024**3
    if accelerator.is_main_process:
        print(
            json.dumps(
                {
                    "checkpoint/max_allocated_gib": memory[0].item(),
                    "checkpoint/max_reserved_gib": memory[1].item(),
                    "checkpoint/peak_allocated_gib": memory[2].item(),
                },
                sort_keys=True,
            ),
            flush=True,
        )


def _adamw_optimizer(optimizer) -> torch.optim.AdamW:
    raw_optimizer = getattr(optimizer, "optimizer", optimizer)
    if not isinstance(raw_optimizer, torch.optim.AdamW):
        raise TypeError("Confidence optimizer offload requires torch.optim.AdamW.")
    return raw_optimizer


def _move_adamw_moments(
    optimizer,
    target_device: torch.device,
) -> int:
    raw_optimizer = _adamw_optimizer(optimizer)
    allowed_state_keys = {
        "step",
        "exp_avg",
        "exp_avg_sq",
        "max_exp_avg_sq",
    }
    moment_keys = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")
    initialized_parameters = 0
    moment_bytes = 0
    for state in raw_optimizer.state.values():
        if not state:
            continue
        unexpected = set(state).difference(allowed_state_keys)
        if unexpected:
            raise ValueError(
                f"Unexpected AdamW optimizer state keys: {sorted(unexpected)}"
            )
        if "exp_avg" not in state or "exp_avg_sq" not in state:
            raise ValueError(
                "Initialized AdamW state must contain exp_avg and exp_avg_sq."
            )
        initialized_parameters += 1
        for key in moment_keys:
            if key not in state:
                continue
            moment = state[key]
            if not isinstance(moment, torch.Tensor):
                raise TypeError(f"AdamW state {key!r} must be a tensor.")
            moment_bytes += moment.numel() * moment.element_size()
            if moment.device != target_device:
                state[key] = moment.to(device=target_device)
    if initialized_parameters == 0:
        return 0
    if moment_bytes == 0:
        raise RuntimeError("Initialized AdamW state contains no moment tensors.")
    return moment_bytes


def _transfer_confidence_optimizer_moments(
    optimizer,
    *,
    target_device: torch.device,
    accelerator,
    outer_iteration: int | str,
) -> int:
    if accelerator.device.type == "cuda":
        torch.cuda.synchronize(accelerator.device)
        if target_device.type == "cuda":
            torch.cuda.empty_cache()
    started = time.perf_counter()
    moment_bytes = _move_adamw_moments(optimizer, target_device)
    if accelerator.device.type == "cuda":
        torch.cuda.synchronize(accelerator.device)
        if target_device.type == "cpu":
            torch.cuda.empty_cache()
    elapsed = time.perf_counter() - started
    if moment_bytes <= 0:
        raise RuntimeError(
            "Confidence optimizer state is empty when a transfer was required."
        )
    if accelerator.is_main_process:
        cuda_memory = {}
        if accelerator.device.type == "cuda":
            cuda_memory = {
                "optimizer_moments/cuda_allocated_gib": (
                    torch.cuda.memory_allocated(accelerator.device) / 1024**3
                ),
                "optimizer_moments/cuda_reserved_gib": (
                    torch.cuda.memory_reserved(accelerator.device) / 1024**3
                ),
            }
        print(
            json.dumps(
                {
                    "outer_iteration": outer_iteration,
                    "optimizer_moments/device": target_device.type,
                    "optimizer_moments/gib_per_rank": moment_bytes / 1024**3,
                    "optimizer_moments/transfer_seconds": elapsed,
                    **cuda_memory,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return moment_bytes


def _measure_component_gradient_norms(
    *,
    algorithm: BilevelGRPO,
    rollout_microbatches: list[CachedRolloutMicrobatch],
    local_problem_batch_size: int,
    initial_fast,
    confidence_model,
    confidence_optimizer,
    accelerator,
    progress_prefix: str,
) -> dict[str, float]:
    cached_problem_count = sum(
        len(microbatch.problems) for microbatch in rollout_microbatches
    )
    if cached_problem_count != local_problem_batch_size:
        raise ValueError(
            "Component gradient norms require one cached support/query pair per "
            "local problem."
        )
    if any(microbatch.queries is None for microbatch in rollout_microbatches):
        raise ValueError("Component gradient norms require cached query groups.")
    coefficients = {
        "meta": algorithm.meta_config.meta_coefficient,
        "bce": algorithm.meta_config.confidence.bce_coefficient,
        "ranking": algorithm.meta_config.confidence.ranking_coefficient,
    }
    norms: dict[str, float] = {}
    measurement_start = time.perf_counter()
    components = tqdm(
        tuple(coefficients),
        desc=f"{progress_prefix}: component gradient norms",
        unit="component",
        leave=True,
        disable=not accelerator.is_main_process,
    )
    for component in components:
        confidence_optimizer.zero_grad(set_to_none=True)
        problem_offset = 0
        for microbatch in rollout_microbatches:
            supports = tuple(
                group.to(accelerator.device) for group in microbatch.supports
            )
            queries = (
                tuple(group.to(accelerator.device) for group in microbatch.queries)
                if component == "meta"
                else None
            )
            for problem_index, support in enumerate(supports):
                if component == "meta":
                    output = algorithm.outer_loss(
                        support,
                        queries[problem_index],
                        initial_fast,
                        show_progress=False,
                        progress_prefix=(
                            f"{progress_prefix} diagnostic problem "
                            f"{problem_offset + problem_index + 1}"
                        ),
                    )
                    component_loss = output.meta_grpo.loss
                else:
                    confidence_loss = algorithm.confidence_supervision_loss(support)
                    component_loss = getattr(confidence_loss, component)
                accelerator.backward(component_loss / local_problem_batch_size)
                del component_loss
                if component == "meta":
                    del output
                else:
                    del confidence_loss
            problem_offset += len(supports)
            del support, supports, queries

        raw_norm = accelerator.clip_grad_norm_(
            confidence_model.parameters(),
            math.inf,
        )
        if not torch.isfinite(raw_norm):
            raise FloatingPointError(
                f"{component} confidence gradient norm is non-finite."
            )
        raw_norm = accelerator.reduce(raw_norm.detach(), reduction="mean")
        raw_value = raw_norm.item()
        norms[f"gradient_norm/{component}_raw"] = raw_value
        norms[f"gradient_norm/{component}_weighted"] = (
            coefficients[component] * raw_value
        )
        confidence_optimizer.zero_grad(set_to_none=True)
        del raw_norm
        if accelerator.device.type == "cuda":
            torch.cuda.synchronize(accelerator.device)
            torch.cuda.empty_cache()

    accelerator.wait_for_everyone()
    elapsed = time.perf_counter() - measurement_start
    norms["gradient_norm/measurement_seconds"] = elapsed
    return norms


def _accumulate_outer_batch(
    *,
    algorithm: BilevelGRPO,
    rollout_microbatches: list[CachedRolloutMicrobatch],
    local_problem_batch_size: int,
    initial_fast,
    accelerator,
    progress_description: str,
    confidence_model=None,
    defer_gradient_sync: bool = False,
) -> torch.Tensor:
    cached_problem_count = sum(
        len(microbatch.problems) for microbatch in rollout_microbatches
    )
    if cached_problem_count != local_problem_batch_size:
        raise ValueError(
            "Outer gradient accumulation requires exactly one cached rollout "
            "pair per local problem."
        )
    if any(microbatch.queries is None for microbatch in rollout_microbatches):
        raise ValueError("Outer gradient accumulation requires cached query groups.")
    if defer_gradient_sync and confidence_model is None:
        raise ValueError(
            "confidence_model is required when gradient synchronization is deferred."
        )

    local_metric_sums = torch.zeros(10, dtype=torch.float32, device=accelerator.device)
    problem_progress = tqdm(
        total=local_problem_batch_size,
        desc=progress_description,
        unit="problem",
        leave=True,
        disable=not accelerator.is_main_process,
    )
    problem_offset = 0
    for microbatch_index, microbatch in enumerate(rollout_microbatches):
        is_final_microbatch = microbatch_index == len(rollout_microbatches) - 1
        sync_context = (
            accelerator.no_sync(confidence_model)
            if defer_gradient_sync and not is_final_microbatch
            else nullcontext()
        )
        with sync_context:
            supports = tuple(
                group.to(accelerator.device) for group in microbatch.supports
            )
            queries = tuple(
                group.to(accelerator.device) for group in microbatch.queries
            )
            outputs = algorithm.outer_losses_batch(
                supports,
                queries,
                initial_fast,
                show_progress=False,
                progress_prefix=(
                    f"{progress_description} problems "
                    f"{problem_offset + 1}-"
                    f"{problem_offset + len(supports)}"
                ),
            )
            accumulated_loss = torch.stack(
                tuple(output.loss for output in outputs)
            ).sum() / local_problem_batch_size
            accelerator.backward(accumulated_loss)
        for support, query, output in zip(
            supports, queries, outputs, strict=True
        ):
            local_metric_sums += torch.stack(
                (
                    output.loss.detach(),
                    output.meta_grpo.loss.detach(),
                    output.adaptation.confidence_loss.bce.detach(),
                    output.adaptation.confidence_loss.ranking.detach(),
                    output.meta_grpo.clip_fraction.detach(),
                    output.adaptation.inner_losses[-1].clip_fraction.detach(),
                    support.verifier_rewards.mean(),
                    query.verifier_rewards.mean(),
                    support.correctness_labels.mean(),
                    query.correctness_labels.mean(),
                )
            )
            problem_progress.update(1)
        problem_offset += len(supports)
        del accumulated_loss, output, outputs, support, query, supports, queries
    problem_progress.close()
    return local_metric_sums


def _evaluate(
    *,
    step: int,
    problems: list[MathProblem],
    algorithm: BilevelGRPO,
    support_rollouts: TransformersRolloutEngine,
    query_rollouts: TransformersRolloutEngine,
    verifier: DAPOMathVerifier,
    initial_fast,
    confidence_model,
    accelerator,
    rollout_log_path: Path | None,
) -> None:
    if not problems:
        raise ValueError("Validation problems cannot be empty.")
    rounds = math.ceil(len(problems) / accelerator.num_processes)
    was_training = confidence_model.training
    confidence_model.eval()
    local_metrics = torch.zeros(6, device=accelerator.device, dtype=torch.float64)
    try:
        round_indices = tqdm(
            range(rounds),
            desc=f"validation step {step}",
            unit="round",
            leave=True,
            disable=not accelerator.is_main_process,
        )
        for round_index in round_indices:
            problem_index = (
                round_index * accelerator.num_processes + accelerator.process_index
            )
            valid = problem_index < len(problems)
            problem = problems[problem_index] if valid else problems[0]

            progress_prefix = f"validation {problem.uid}"
            support = support_rollouts.generate(
                problem,
                initial_fast,
                show_progress=accelerator.is_main_process,
                progress_description=f"{progress_prefix} base",
            )
            generation_adaptation = algorithm.adapt_task(
                support,
                initial_fast,
                differentiable=False,
                supervise_confidence=False,
                show_progress=accelerator.is_main_process,
                progress_prefix=f"{progress_prefix} adaptation",
            )
            query = query_rollouts.generate(
                problem,
                generation_adaptation.fast_parameters,
                show_progress=accelerator.is_main_process,
                progress_description=f"{progress_prefix} adapted",
            )
            del generation_adaptation
            base_verification = verifier(
                support.texts,
                problem.ground_truth,
                device=accelerator.device,
            )
            adapted_verification = verifier(
                query.texts,
                problem.ground_truth,
                device=accelerator.device,
            )
            if valid:
                _append_rollout_diagnostics(
                    rollout_log_path,
                    step=step,
                    phase="validation_base",
                    problem=problem,
                    group=support,
                    verification=base_verification,
                    max_new_tokens=support_rollouts.max_new_tokens,
                    eos_token_id=support_rollouts.tokenizer.eos_token_id,
                )
                _append_rollout_diagnostics(
                    rollout_log_path,
                    step=step,
                    phase="validation_adapted",
                    problem=problem,
                    group=query,
                    verification=adapted_verification,
                    max_new_tokens=query_rollouts.max_new_tokens,
                    eos_token_id=query_rollouts.tokenizer.eos_token_id,
                )
                local_metrics += torch.tensor(
                    (
                        base_verification.correctness.sum().item(),
                        float(base_verification.correctness.numel()),
                        adapted_verification.correctness.sum().item(),
                        float(adapted_verification.correctness.numel()),
                        float(torch.any(base_verification.correctness == 1).item()),
                        float(torch.any(adapted_verification.correctness == 1).item()),
                    ),
                    device=accelerator.device,
                    dtype=torch.float64,
                )
    finally:
        confidence_model.train(was_training)

    totals = accelerator.reduce(local_metrics, reduction="sum")
    if accelerator.is_main_process:
        base_accuracy = (totals[0] / totals[1]).item()
        adapted_accuracy = (totals[2] / totals[3]).item()
        print(
            json.dumps(
                {
                    "step": step,
                    "validation/base_accuracy": base_accuracy,
                    "validation/adapted_accuracy": adapted_accuracy,
                    "validation/accuracy_delta": adapted_accuracy - base_accuracy,
                    "validation/base_pass_at_group": (totals[4] / len(problems)).item(),
                    "validation/adapted_pass_at_group": (
                        totals[5] / len(problems)
                    ).item(),
                    "validation/num_unique_problems": len(problems),
                },
                sort_keys=True,
            ),
            flush=True,
        )


def main() -> None:
    args = parse_args()
    _validate_args(args)
    inner_config, meta_config, query_advantage_config, query_grpo_config = _configs(
        args
    )

    from accelerate import Accelerator
    from accelerate.utils import set_seed

    accelerator = Accelerator()
    if accelerator.mixed_precision != "bf16":
        raise ValueError(
            "The HPC launch must configure Accelerate mixed_precision='bf16'."
        )
    (
        global_problem_batch_size,
        local_problem_batch_size,
        global_problem_micro_batch_size,
        local_problem_micro_batch_size,
        num_problem_micro_batches,
    ) = _problem_batch_layout(
        args.problem_batch_size,
        args.problem_micro_batch_size,
        world_size=accelerator.num_processes,
    )
    args.problem_batch_size = global_problem_batch_size
    args.problem_micro_batch_size = global_problem_micro_batch_size
    set_seed(args.seed, device_specific=True)
    _write_run_config(args, args.output_dir, accelerator)
    accelerator.wait_for_everyone()
    rollout_log_path = (
        args.output_dir / f"rollouts-rank-{accelerator.process_index}.jsonl"
        if args.log_rollouts
        else None
    )

    model_kwargs = {"attn_implementation": args.attn_implementation}
    target_modules = tuple(
        module.strip() for module in args.lora_target_modules.split(",")
    )
    policy_bundle = load_policy_with_lora(
        args.policy_model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        model_kwargs=model_kwargs,
    )
    policy = policy_bundle.model.to(accelerator.device)
    confidence_model = load_confidence_model(
        args.confidence_model,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        model_kwargs=model_kwargs,
    )
    confidence_optimizer = torch.optim.AdamW(
        confidence_model.parameters(),
        lr=args.confidence_learning_rate,
        weight_decay=args.confidence_weight_decay,
    )
    confidence_model, confidence_optimizer = accelerator.prepare(
        confidence_model,
        confidence_optimizer,
    )
    confidence_model.train()
    start_step = 0
    optimizer_moments_offloaded = False
    if args.resume_from_checkpoint is not None:
        if not args.resume_from_checkpoint.is_dir():
            raise FileNotFoundError(args.resume_from_checkpoint)
        match = re.fullmatch(r"checkpoint-(\d+)", args.resume_from_checkpoint.name)
        if match is None:
            raise ValueError(
                "resume_from_checkpoint must end in checkpoint-<completed_steps>."
            )
        start_step = int(match.group(1))
        if start_step >= args.max_steps:
            raise ValueError(
                "Checkpoint already reached or exceeded the requested max_steps."
            )
        accelerator.load_state(args.resume_from_checkpoint)
        if args.offload_confidence_optimizer:
            _transfer_confidence_optimizer_moments(
                confidence_optimizer,
                target_device=torch.device("cpu"),
                accelerator=accelerator,
                outer_iteration="resume",
            )
            optimizer_moments_offloaded = True

    algorithm = BilevelGRPO(
        policy=policy,
        confidence_model=confidence_model,
        inner_config=inner_config,
        meta_config=meta_config,
        query_advantage_config=query_advantage_config,
        query_grpo_config=query_grpo_config,
        policy_micro_batch_size=args.policy_micro_batch_size,
        confidence_micro_batch_size=args.confidence_micro_batch_size,
        policy_max_tokens_per_micro_batch=(
            args.policy_max_tokens_per_micro_batch
        ),
        confidence_max_tokens_per_micro_batch=(
            args.confidence_max_tokens_per_micro_batch
        ),
    )
    rollout_kwargs: dict[str, object] = {}
    rollout_engine_type = TransformersRolloutEngine
    if args.rollout_backend == "vllm":
        base_urls = [url.strip() for url in args.vllm_base_urls.split(",")]
        if any(not url for url in base_urls):
            raise ValueError("--vllm-base-urls contains an empty URL.")
        if len(base_urls) != accelerator.num_processes:
            raise ValueError(
                f"Expected {accelerator.num_processes} vLLM URLs, got {len(base_urls)}."
            )
        rollout_engine_type = VLLMHybridRolloutEngine
        rollout_kwargs = {
            "base_url": base_urls[accelerator.process_index],
            "adapter_root": (
                args.vllm_adapter_root
                / f"meta-rlvr-{args.output_dir.name}"
                / f"rank-{accelerator.process_index}"
            ),
            "request_timeout": args.vllm_request_timeout,
            "control_timeout": args.vllm_control_timeout,
        }

    def build_rollout_engine(group_size: int):
        return rollout_engine_type(
            policy,
            policy_bundle.tokenizer,
            group_size=group_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            generation_micro_batch_size=args.generation_micro_batch_size,
            logprob_micro_batch_size=args.policy_micro_batch_size,
            logprob_max_tokens_per_micro_batch=(
                args.policy_max_tokens_per_micro_batch
            ),
            **rollout_kwargs,
        )

    support_rollouts = build_rollout_engine(args.support_group_size)
    query_rollouts = build_rollout_engine(args.query_group_size)
    verifier = DAPOMathVerifier(strict_box_verify=args.verifier_mode == "strict_box")

    all_problems = load_unique_dapo_problems(args.train_parquet)
    validation_problems = load_semantically_unique_dapo_problems(
        args.validation_parquet
    )
    if args.validation_max_problems is not None:
        validation_problems = validation_problems[: args.validation_max_problems]
    local_problems = rank_shard(
        all_problems,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    if len(local_problems) < local_problem_batch_size:
        raise ValueError(
            f"Rank {accelerator.process_index} owns {len(local_problems)} "
            f"problems, fewer than its local batch {local_problem_batch_size}."
        )
    accelerator.print(
        f"Loaded {len(all_problems)} unique problems; "
        f"rank {accelerator.process_index} owns {len(local_problems)}; "
        f"validation has {len(validation_problems)} semantically unique problems; "
        f"global problem batch is {global_problem_batch_size} "
        f"({local_problem_batch_size} per rank); global rollout microbatch is "
        f"{global_problem_micro_batch_size} "
        f"({local_problem_micro_batch_size} per rank, "
        f"{num_problem_micro_batches} accumulation microbatches)."
    )

    initial_fast = {
        name: value.to(accelerator.device)
        for name, value in policy_bundle.initial_fast_parameters.items()
    }

    step_indices = tqdm(
        range(start_step, args.max_steps),
        total=args.max_steps - start_step,
        desc="meta-training",
        unit="step",
        leave=True,
        disable=not accelerator.is_main_process,
    )
    for step in step_indices:
        problems = _problem_batch(
            local_problems,
            step=step,
            batch_size=local_problem_batch_size,
            seed=args.seed,
        )
        batch_prefix = (
            f"train step {step + 1} local problems={local_problem_batch_size}"
        )
        timings = StageTimings.create()
        rollout_microbatches: list[CachedRolloutMicrobatch] = []
        rollout_totals = torch.zeros(6, dtype=torch.float32, device=accelerator.device)
        problem_batches = [
            list(
                problems[
                    microbatch_index * local_problem_micro_batch_size :
                    (microbatch_index + 1) * local_problem_micro_batch_size
                ]
            )
            for microbatch_index in range(num_problem_micro_batches)
        ]

        step_indices.set_postfix_str(
            f"problems={global_problem_batch_size} stage=support-rollout"
        )
        with timings.measure("support_rollout"):
            support_batches = support_rollouts.generate_batches(
                problem_batches,
                [
                    [initial_fast] * local_problem_micro_batch_size
                    for _ in problem_batches
                ],
                show_progress=accelerator.is_main_process,
                progress_description=f"{batch_prefix} support",
            )

        # Finish and cache every support rollout before computing any query
        # adapter. This preserves the full-batch rollout order while bounding
        # GPU-resident rollout state by the configured problem microbatch.
        for microbatch_index in range(num_problem_micro_batches):
            start = microbatch_index * local_problem_micro_batch_size
            end = start + local_problem_micro_batch_size
            microbatch_problems = tuple(problems[start:end])
            microbatch_prefix = (
                f"{batch_prefix} rollout microbatch "
                f"{microbatch_index + 1}/{num_problem_micro_batches}"
            )
            step_indices.set_postfix_str(
                f"problems={global_problem_batch_size} "
                f"microbatch={microbatch_index + 1}/{num_problem_micro_batches} "
                "stage=support-rollout"
            )
            supports = list(support_batches[microbatch_index])
            step_indices.set_postfix_str(
                f"problems={global_problem_batch_size} "
                f"microbatch={microbatch_index + 1}/{num_problem_micro_batches} "
                "stage=support-verifier"
            )
            with timings.measure("support_verifier_and_cache"):
                for problem_index, (problem, support) in enumerate(
                    zip(microbatch_problems, supports, strict=True)
                ):
                    support_verification = verifier(
                        support.texts,
                        problem.ground_truth,
                        device=support.device,
                    )
                    support = support.with_verification(
                        support_verification.rewards,
                        support_verification.correctness,
                    )
                    if args.inner_kl_coefficient > 0:
                        support = support.with_reference_logprobs(
                            support.old_logprobs
                        )
                    _append_rollout_diagnostics(
                        rollout_log_path,
                        step=step,
                        phase="train_support",
                        problem=problem,
                        group=support,
                        verification=support_verification,
                        max_new_tokens=support_rollouts.max_new_tokens,
                        eos_token_id=support_rollouts.tokenizer.eos_token_id,
                    )
                    rollout_totals[0] += support.verifier_rewards.sum().to(
                        accelerator.device
                    )
                    rollout_totals[1] += support.correctness_labels.sum().to(
                        accelerator.device
                    )
                    rollout_totals[2] += support.group_size
                    supports[problem_index] = support
            cached_supports = tuple(
                _cache_rollout_group(support) for support in supports
            )
            rollout_microbatches.append(
                CachedRolloutMicrobatch(
                    problems=microbatch_problems,
                    supports=cached_supports,
                )
            )
            del supports, cached_supports, support, support_verification
            if accelerator.device.type == "cuda":
                torch.cuda.empty_cache()
        del support_batches

        # Compute task adapters in problem batches, then run the complete query
        # phase through one vLLM wake/sleep lifecycle. Fast parameters and
        # rollout caches remain CPU-backed between these two phases.
        generation_fast_parameter_batches = []
        step_indices.set_postfix_str(
            f"problems={global_problem_batch_size} stage=generation-adaptation"
        )
        with timings.measure("generation_adaptation"):
            adaptation_progress = tqdm(
                rollout_microbatches,
                total=len(rollout_microbatches),
                desc=f"{batch_prefix} batched generation adaptation",
                unit="microbatch",
                leave=True,
                disable=not accelerator.is_main_process,
            )
            for microbatch_index, cached_microbatch in enumerate(
                adaptation_progress
            ):
                supports = tuple(
                    support.to(accelerator.device)
                    for support in cached_microbatch.supports
                )
                generation_adaptations = algorithm.adapt_tasks(
                    supports,
                    initial_fast,
                    differentiable=False,
                    supervise_confidence=False,
                    show_progress=False,
                    progress_prefix=(
                        f"{batch_prefix} rollout microbatch "
                        f"{microbatch_index + 1}/{num_problem_micro_batches}"
                    ),
                )
                generation_fast_parameter_batches.append(
                    [
                        {
                            name: value.detach().to("cpu")
                            for name, value in adaptation.fast_parameters.items()
                        }
                        for adaptation in generation_adaptations
                    ]
                )
                del supports, generation_adaptations

        step_indices.set_postfix_str(
            f"problems={global_problem_batch_size} stage=query-rollout"
        )
        with timings.measure("query_rollout"):
            query_batches = query_rollouts.generate_batches(
                problem_batches,
                generation_fast_parameter_batches,
                show_progress=accelerator.is_main_process,
                progress_description=f"{batch_prefix} query",
            )
        del generation_fast_parameter_batches

        # All queries are fixed before the first confidence-model update, so
        # every outer iteration reuses exactly the same on-policy data.
        for microbatch_index, cached_microbatch in enumerate(rollout_microbatches):
            microbatch_prefix = (
                f"{batch_prefix} rollout microbatch "
                f"{microbatch_index + 1}/{num_problem_micro_batches}"
            )
            queries = list(query_batches[microbatch_index])
            step_indices.set_postfix_str(
                f"problems={global_problem_batch_size} "
                f"microbatch={microbatch_index + 1}/{num_problem_micro_batches} "
                "stage=query-verifier"
            )
            with timings.measure("query_verifier_and_cache"):
                for problem_index, (problem, query) in enumerate(
                    zip(cached_microbatch.problems, queries, strict=True)
                ):
                    query_verification = verifier(
                        query.texts,
                        problem.ground_truth,
                        device=query.device,
                    )
                    query = query.with_verification(
                        query_verification.rewards,
                        query_verification.correctness,
                    )
                    if args.outer_kl_coefficient > 0:
                        query = query_rollouts.add_reference_logprobs(
                            query,
                            initial_fast,
                            show_progress=False,
                            progress_description=(
                                f"{microbatch_prefix} query reference "
                                f"{problem_index + 1}"
                            ),
                        )
                    _append_rollout_diagnostics(
                        rollout_log_path,
                        step=step,
                        phase="train_query",
                        problem=problem,
                        group=query,
                        verification=query_verification,
                        max_new_tokens=query_rollouts.max_new_tokens,
                        eos_token_id=query_rollouts.tokenizer.eos_token_id,
                    )
                    rollout_totals[3] += query.verifier_rewards.sum().to(
                        accelerator.device
                    )
                    rollout_totals[4] += query.correctness_labels.sum().to(
                        accelerator.device
                    )
                    rollout_totals[5] += query.group_size
                    queries[problem_index] = query
                cached_queries = tuple(
                    _cache_rollout_group(query) for query in queries
                )
            rollout_microbatches[microbatch_index] = replace(
                cached_microbatch,
                queries=cached_queries,
            )
            del queries, cached_queries, query, query_verification
            if accelerator.device.type == "cuda":
                torch.cuda.empty_cache()
        del query_batches

        if any(microbatch.queries is None for microbatch in rollout_microbatches):
            raise RuntimeError("Query rollout cache is incomplete.")
        if accelerator.is_main_process:
            print(
                json.dumps(
                    {
                        "diagnostic": "pipeline_timing",
                        "step": step,
                        "timing_seconds": dict(timings.values),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if args.rollout_only:
            rollout_totals = accelerator.reduce(rollout_totals, reduction="sum")
            if accelerator.is_main_process:
                print(
                    json.dumps(
                        {
                            "rollout_only": True,
                            "step": step,
                            "support_mean_reward": (
                                rollout_totals[0] / rollout_totals[2]
                            ).item(),
                            "support_accuracy": (
                                rollout_totals[1] / rollout_totals[2]
                            ).item(),
                            "query_mean_reward": (
                                rollout_totals[3] / rollout_totals[5]
                            ).item(),
                            "query_accuracy": (
                                rollout_totals[4] / rollout_totals[5]
                            ).item(),
                            "num_problems": global_problem_batch_size,
                            "problem_micro_batch_size": (
                                global_problem_micro_batch_size
                            ),
                            "support_responses": int(rollout_totals[2].item()),
                            "query_responses": int(rollout_totals[5].item()),
                            "timing_seconds": dict(timings.values),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            accelerator.wait_for_everyone()
            accelerator.end_training()
            return

        outer_iterations = tqdm(
            range(args.outer_iterations),
            desc=f"train step {step + 1}: outer updates",
            unit="update",
            leave=True,
            disable=not accelerator.is_main_process,
        )
        for outer_iteration in outer_iterations:
            confidence_optimizer.zero_grad(set_to_none=True)
            component_gradient_norms: dict[str, float] = {}

            global_outer_iteration = step * args.outer_iterations + outer_iteration
            if (
                args.log_component_gradient_norms
                and global_outer_iteration % args.component_gradient_norm_interval == 0
            ):
                outer_iterations.set_postfix_str("stage=component-gradient-norms")
                rng_devices = (
                    [accelerator.device] if accelerator.device.type == "cuda" else []
                )
                with torch.random.fork_rng(devices=rng_devices):
                    with timings.measure("component_gradient_norm_diagnostics"):
                        component_gradient_norms = _measure_component_gradient_norms(
                            algorithm=algorithm,
                            rollout_microbatches=rollout_microbatches,
                            local_problem_batch_size=local_problem_batch_size,
                            initial_fast=initial_fast,
                            confidence_model=confidence_model,
                            confidence_optimizer=confidence_optimizer,
                            accelerator=accelerator,
                            progress_prefix=(
                                f"train step {step + 1} outer "
                                f"{outer_iteration + 1}"
                            ),
                        )
                if accelerator.is_main_process:
                    print(
                        json.dumps(
                            {
                                "diagnostic": "component_gradient_norms",
                                "step": step,
                                "outer_iteration": outer_iteration,
                                **component_gradient_norms,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

            outer_iterations.set_postfix_str("stage=meta-forward")
            outer_timing_prefix = f"outer_{outer_iteration + 1}"
            with timings.measure(f"{outer_timing_prefix}/forward_backward"):
                local_metric_sums = _accumulate_outer_batch(
                    algorithm=algorithm,
                    rollout_microbatches=rollout_microbatches,
                    local_problem_batch_size=local_problem_batch_size,
                    initial_fast=initial_fast,
                    accelerator=accelerator,
                    confidence_model=confidence_model,
                    defer_gradient_sync=args.defer_confidence_gradient_sync,
                    progress_description=(
                        f"train step {step + 1} outer {outer_iteration + 1}: "
                        "problem gradients"
                    ),
                )
            outer_iterations.set_postfix_str("stage=meta-step")
            with timings.measure(f"{outer_timing_prefix}/gradient_clip"):
                gradient_norm = accelerator.clip_grad_norm_(
                    confidence_model.parameters(), args.max_grad_norm
                )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(
                    "Confidence gradient norm is non-finite; refusing to update "
                    "or save a corrupted checkpoint."
                )
            if args.offload_confidence_optimizer and optimizer_moments_offloaded:
                outer_iterations.set_postfix_str("stage=optimizer-moments-to-gpu")
                with timings.measure(
                    f"{outer_timing_prefix}/optimizer_moments_to_gpu"
                ):
                    _transfer_confidence_optimizer_moments(
                        confidence_optimizer,
                        target_device=accelerator.device,
                        accelerator=accelerator,
                        outer_iteration=outer_iteration,
                    )
                optimizer_moments_offloaded = False
            outer_iterations.set_postfix_str("stage=optimizer-step")
            with timings.measure(f"{outer_timing_prefix}/optimizer_step"):
                confidence_optimizer.step()
            if args.offload_confidence_optimizer:
                outer_iterations.set_postfix_str("stage=optimizer-moments-to-cpu")
                with timings.measure(
                    f"{outer_timing_prefix}/optimizer_moments_to_cpu"
                ):
                    _transfer_confidence_optimizer_moments(
                        confidence_optimizer,
                        target_device=torch.device("cpu"),
                        accelerator=accelerator,
                        outer_iteration=outer_iteration,
                    )
                optimizer_moments_offloaded = True
            outer_iterations.set_postfix_str("stage=metrics")

            metrics = accelerator.reduce(local_metric_sums, reduction="sum")
            metrics = metrics / global_problem_batch_size
            reduced_gradient_norm = accelerator.reduce(
                gradient_norm.detach(), reduction="mean"
            )
            if accelerator.is_main_process:
                print(
                    json.dumps(
                        {
                            "step": step,
                            "outer_iteration": outer_iteration,
                            "problem_batch_size": global_problem_batch_size,
                            "problem_micro_batch_size": (
                                global_problem_micro_batch_size
                            ),
                            "loss": metrics[0].item(),
                            "meta_loss": metrics[1].item(),
                            "bce": metrics[2].item(),
                            "ranking": metrics[3].item(),
                            "outer_clip_fraction": metrics[4].item(),
                            "inner_clip_fraction": metrics[5].item(),
                            "support_reward": metrics[6].item(),
                            "query_reward": metrics[7].item(),
                            "support_accuracy": metrics[8].item(),
                            "query_accuracy": metrics[9].item(),
                            "confidence_gradient_norm": (reduced_gradient_norm.item()),
                            "timing_seconds": dict(timings.values),
                            **component_gradient_norms,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        del problems, rollout_microbatches, rollout_totals

        if (step + 1) % args.save_steps == 0:
            _prepare_for_checkpoint(confidence_optimizer, accelerator)
            accelerator.save_state(args.output_dir / f"checkpoint-{step + 1}")

        if args.eval_steps > 0 and (step + 1) % args.eval_steps == 0:
            _evaluate(
                step=step + 1,
                problems=validation_problems,
                algorithm=algorithm,
                support_rollouts=support_rollouts,
                query_rollouts=query_rollouts,
                verifier=verifier,
                initial_fast=initial_fast,
                confidence_model=confidence_model,
                accelerator=accelerator,
                rollout_log_path=rollout_log_path,
            )

    if args.eval_steps == 0 or args.max_steps % args.eval_steps != 0:
        _evaluate(
            step=args.max_steps,
            problems=validation_problems,
            algorithm=algorithm,
            support_rollouts=support_rollouts,
            query_rollouts=query_rollouts,
            verifier=verifier,
            initial_fast=initial_fast,
            confidence_model=confidence_model,
            accelerator=accelerator,
            rollout_log_path=rollout_log_path,
        )

    _prepare_for_checkpoint(confidence_optimizer, accelerator)
    accelerator.save_state(args.output_dir / "final")


if __name__ == "__main__":
    main()
