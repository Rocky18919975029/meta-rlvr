from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import re
import shutil
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .bilevel import BilevelGRPO, TokenGradientAlignmentContext
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
from .losses import (
    TOKEN_CREDIT_CROSS_TRAJECTORY_NORMALIZATION,
    TOKEN_CREDIT_PARAMETERIZATION,
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
    token_queries: tuple[RolloutGroup, ...] | None = None
    token_alignment_contexts: tuple[TokenGradientAlignmentContext, ...] | None = None

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
        if self.token_queries is not None and len(self.token_queries) != len(
            self.problems
        ):
            raise ValueError(
                "Cached token query groups must match the problem microbatch size."
            )
        if self.token_alignment_contexts is not None and len(
            self.token_alignment_contexts
        ) != len(self.problems):
            raise ValueError(
                "Cached token alignment contexts must match the problem microbatch "
                "size."
            )
        groups = self.supports
        if self.queries is not None:
            groups += self.queries
        if self.token_queries is not None:
            groups += self.token_queries
        if any(group.device.type != "cpu" for group in groups):
            raise ValueError("Cached rollout microbatches must reside on CPU.")
        if self.token_alignment_contexts is not None:
            for context in self.token_alignment_contexts:
                if context.support_logprobs.device.type != "cpu":
                    raise ValueError("Cached token alignment contexts must be on CPU.")


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
        "--validation-support-group-size",
        type=int,
        help="Validation support responses per problem; defaults to support-group-size.",
    )
    parser.add_argument(
        "--validation-query-group-size",
        type=int,
        help="Validation adapted responses per problem; defaults to query-group-size.",
    )
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
            "Global number of problems moved to GPU for one outer backward. "
            "Defaults to problem_batch_size; gradients accumulate over the "
            "complete problem batch before each outer optimizer step."
        ),
    )
    parser.add_argument(
        "--rollout-problem-batch-size",
        type=int,
        help=(
            "Global number of problems submitted to vLLM in one rollout batch. "
            "Defaults to problem_batch_size and is independent of the outer "
            "gradient problem microbatch."
        ),
    )
    parser.add_argument("--generation-micro-batch-size", type=int, default=4)
    parser.add_argument("--policy-micro-batch-size", type=int, default=1)
    parser.add_argument(
        "--first-order-vjp-forward-batch-size",
        type=int,
        default=1,
        help=(
            "Maximum responses sharing one policy forward before sequential "
            "first-order VJPs. Defaults to 1."
        ),
    )
    parser.add_argument(
        "--token-jvp-response-micro-batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--token-jvp-logprob-position-chunk-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--token-credit-max",
        type=float,
        default=1.0,
        help="Maximum absolute token credit in A=max*tanh(logit).",
    )
    parser.add_argument(
        "--token-meta-gradient-mode",
        choices=["gradient_alignment", "unrolled"],
        default="gradient_alignment",
        help=(
            "gradient_alignment uses the single-layer exact-JVP surrogate and "
            "caches its policy-side direction; unrolled retains differentiable "
            "task-adapter updates."
        ),
    )
    parser.add_argument("--confidence-micro-batch-size", type=int, default=2)
    parser.add_argument("--policy-max-tokens-per-micro-batch", type=int)
    parser.add_argument("--confidence-max-tokens-per-micro-batch", type=int)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--validation-temperature", type=float, default=1.0)
    parser.add_argument("--validation-top-p", type=float, default=0.7)
    parser.add_argument("--validation-top-k", type=int, default=0)
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
    parser.add_argument("--token-meta-coefficient", type=float, default=0.0)
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
    parser.add_argument(
        "--save-steps",
        type=int,
        default=1,
        help="Save one committed checkpoint every N completed global steps.",
    )
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument(
        "--resume-preflight-only",
        action="store_true",
        help="Restore and validate a checkpoint, then exit before policy/vLLM setup.",
    )
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
    objective_coefficients = (
        args.meta_coefficient,
        args.token_meta_coefficient,
        args.bce_coefficient,
        args.ranking_coefficient,
    )
    if any(coefficient < 0 for coefficient in objective_coefficients):
        raise ValueError("All objective coefficients must be non-negative.")
    if not any(coefficient > 0 for coefficient in objective_coefficients):
        raise ValueError("At least one training objective must be enabled.")
    if args.outer_iterations <= 0:
        raise ValueError("outer_iterations must be positive.")
    if args.max_steps <= 0 or args.save_steps <= 0:
        raise ValueError("max_steps and save_steps must be positive.")
    if args.eval_steps < 0:
        raise ValueError("eval_steps must be non-negative.")
    if args.validation_max_problems is not None and args.validation_max_problems <= 0:
        raise ValueError("validation_max_problems must be positive.")
    if args.support_group_size <= 0 or args.query_group_size <= 0:
        raise ValueError("Training group sizes must be positive.")
    if (
        args.validation_support_group_size is not None
        and args.validation_support_group_size <= 0
    ):
        raise ValueError("validation_support_group_size must be positive.")
    if (
        args.validation_query_group_size is not None
        and args.validation_query_group_size <= 0
    ):
        raise ValueError("validation_query_group_size must be positive.")
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
        args.rollout_problem_batch_size is not None
        and args.rollout_problem_batch_size <= 0
    ):
        raise ValueError("rollout_problem_batch_size must be positive.")
    if (
        args.first_order_vjp_forward_batch_size is not None
        and args.first_order_vjp_forward_batch_size <= 0
    ):
        raise ValueError("first_order_vjp_forward_batch_size must be positive.")
    if args.token_jvp_response_micro_batch_size <= 0:
        raise ValueError("token_jvp_response_micro_batch_size must be positive.")
    if args.token_jvp_logprob_position_chunk_size <= 0:
        raise ValueError("token_jvp_logprob_position_chunk_size must be positive.")
    if args.token_credit_max <= 0:
        raise ValueError("token_credit_max must be positive.")
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
    if args.resume_preflight_only and args.resume_from_checkpoint is None:
        raise ValueError("--resume-preflight-only requires --resume-from-checkpoint.")
    if args.vllm_request_timeout <= 0:
        raise ValueError("--vllm-request-timeout must be positive.")
    if args.vllm_control_timeout <= 0:
        raise ValueError("--vllm-control-timeout must be positive.")
    if args.temperature <= 0 or args.validation_temperature <= 0:
        raise ValueError("Sampling temperatures must be positive.")
    if not 0 < args.top_p <= 1 or not 0 < args.validation_top_p <= 1:
        raise ValueError("Sampling top-p values must be in (0, 1].")
    if args.top_k < 0:
        raise ValueError("--top-k must be non-negative.")
    if args.validation_top_k < 0:
        raise ValueError("--validation-top-k must be non-negative.")
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
    if args.token_meta_coefficient > 0:
        if args.meta_gradient_mode != "first_order":
            raise ValueError("Token confidence requires first_order meta-gradients.")
        if args.attn_implementation != "sdpa":
            raise ValueError(
                "Token confidence requires --attn-implementation sdpa; policy "
                "forwards are forced to the SDPA math backend for exact JVP."
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
        token_meta_coefficient=args.token_meta_coefficient,
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
    sub_batch_name: str = "problem_micro_batch_size",
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
            f"{sub_batch_name} must be at least the distributed world size."
        )
    if global_micro_batch > global_batch:
        raise ValueError(f"{sub_batch_name} cannot exceed problem_batch_size.")
    if global_micro_batch % world_size != 0:
        raise ValueError(
            f"{sub_batch_name} must be divisible by the distributed world size."
        )
    if global_batch % global_micro_batch != 0:
        raise ValueError(f"problem_batch_size must be divisible by {sub_batch_name}.")
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


def _fixed_size_batches(items, batch_size: int) -> list[list]:
    if not items:
        raise ValueError("Cannot partition an empty sequence.")
    if batch_size <= 0 or len(items) % batch_size != 0:
        raise ValueError("Batch size must be positive and divide the sequence length.")
    return [
        list(items[start : start + batch_size])
        for start in range(0, len(items), batch_size)
    ]


def _generation_seed(
    *,
    base_seed: int,
    step: int,
    phase: str,
    rank: int,
    problem_uid: str,
) -> int:
    payload = f"{base_seed}|{step}|{phase}|{rank}|{problem_uid}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _generation_seed_batches(
    problem_batches: list[list[MathProblem]],
    *,
    base_seed: int,
    step: int,
    phase: str,
    rank: int,
) -> list[list[int]]:
    return [
        [
            _generation_seed(
                base_seed=base_seed,
                step=step,
                phase=phase,
                rank=rank,
                problem_uid=problem.uid,
            )
            for problem in problems
        ]
        for problems in problem_batches
    ]


def _serializable_run_config(args: argparse.Namespace, *, world_size: int) -> dict:
    serializable = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    if float(serializable.get("token_meta_coefficient", 0.0)) <= 0:
        serializable.pop("token_credit_max", None)
        serializable.pop("token_meta_gradient_mode", None)
    else:
        serializable.setdefault("token_meta_gradient_mode", "gradient_alignment")
        serializable["token_credit_parameterization"] = TOKEN_CREDIT_PARAMETERIZATION
        serializable["token_credit_cross_trajectory_normalization"] = (
            TOKEN_CREDIT_CROSS_TRAJECTORY_NORMALIZATION
        )
        if serializable["token_meta_gradient_mode"] == "gradient_alignment":
            serializable["token_meta_gradient_objective"] = (
                "base_query_grpo_support_logprob_jvp_v1"
            )
            serializable["token_meta_gradient_sampling_gradient"] = False
            serializable["token_meta_gradient_inner_optimizer_approximation"] = (
                "sgd_step_size_times_inner_iterations"
            )
    serializable["distributed_world_size"] = world_size
    return serializable


def _atomic_write_json(path: Path, record: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_metric(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()


_RESUME_MUTABLE_CONFIG_KEYS = {
    "max_steps",
    "save_steps",
    "eval_steps",
    "resume_from_checkpoint",
    "resume_preflight_only",
    "vllm_base_urls",
}


def _initialize_or_validate_run(
    args: argparse.Namespace,
    *,
    accelerator,
) -> None:
    output_dir = args.output_dir.resolve()
    args.output_dir = output_dir
    current = _serializable_run_config(
        args,
        world_size=accelerator.num_processes,
    )
    config_path = output_dir / "run_config.json"
    if args.resume_from_checkpoint is None:
        if accelerator.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
            existing_training_artifacts = [
                path
                for path in output_dir.iterdir()
                if path.name == "run_config.json"
                or path.name == "metrics.jsonl"
                or path.name.startswith("rollouts-rank-")
                or path.name.startswith("checkpoint-")
                or path.name == "final"
            ]
            if existing_training_artifacts:
                raise FileExistsError(
                    "Refusing to start a fresh run in a directory containing "
                    f"training artifacts: {existing_training_artifacts}"
                )
            _atomic_write_json(config_path, current)
        return

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Resume requires the original run config: {config_path}"
        )
    original = json.loads(config_path.read_text(encoding="utf-8"))
    for key, value in {
        "token_meta_coefficient": 0.0,
        "token_jvp_response_micro_batch_size": 4,
        "token_jvp_logprob_position_chunk_size": 256,
    }.items():
        original.setdefault(key, value)
        current.setdefault(key, value)
    if float(original.get("token_meta_coefficient", 0.0)) > 0:
        # Token checkpoints predating this field used the unrolled objective.
        original.setdefault("token_meta_gradient_mode", "unrolled")
    compared_keys = set(original) | set(current)
    mismatches = {
        key: {"checkpoint_run": original.get(key), "resume_run": current.get(key)}
        for key in sorted(compared_keys - _RESUME_MUTABLE_CONFIG_KEYS)
        if original.get(key) != current.get(key)
    }
    if mismatches:
        raise ValueError(
            "Resume configuration differs from the original run: "
            + json.dumps(mismatches, sort_keys=True)
        )


def _checkpoint_number(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if match is None:
        raise ValueError(
            "resume_from_checkpoint must end in checkpoint-<completed_steps>."
        )
    return int(match.group(1))


def _resolve_resume_checkpoint(args: argparse.Namespace) -> tuple[Path | None, int]:
    if args.resume_from_checkpoint is None:
        return None, 0
    checkpoint = args.resume_from_checkpoint.resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    if checkpoint.parent != args.output_dir.resolve():
        raise ValueError(
            "resume_from_checkpoint must be a direct child of output_dir so "
            "metrics and rollout logs resume in one run directory."
        )
    completed_steps = _checkpoint_number(checkpoint)
    committed = []
    for candidate in checkpoint.parent.glob("checkpoint-*"):
        if candidate.is_dir() and (candidate / "trainer_state.json").is_file():
            committed.append((_checkpoint_number(candidate), candidate))
    if not committed:
        raise ValueError("No committed checkpoint exists in output_dir.")
    latest_steps, latest_checkpoint = max(committed)
    if checkpoint != latest_checkpoint:
        raise ValueError(
            "Perfect resume is only supported from the latest committed "
            f"checkpoint ({latest_checkpoint}), not {checkpoint}."
        )
    if completed_steps != latest_steps:
        raise RuntimeError("Checkpoint directory numbering is inconsistent.")
    return checkpoint, completed_steps


def _truncate_file(path: Path, size: int) -> None:
    if size < 0:
        raise ValueError("Committed file size cannot be negative.")
    if not path.is_file():
        if size == 0:
            return
        raise FileNotFoundError(path)
    current_size = path.stat().st_size
    if current_size < size:
        raise RuntimeError(
            f"Append-only log {path} is shorter than its committed size "
            f"({current_size} < {size})."
        )
    if current_size > size:
        with path.open("r+b") as stream:
            stream.truncate(size)


def _restore_committed_log_boundaries(
    checkpoint: Path,
    *,
    output_dir: Path,
    accelerator,
) -> dict:
    trainer_state = json.loads(
        (checkpoint / "trainer_state.json").read_text(encoding="utf-8")
    )
    if trainer_state.get("schema_version") != 1:
        raise ValueError("Unsupported trainer checkpoint schema.")
    if trainer_state.get("world_size") != accelerator.num_processes:
        raise ValueError(
            "Perfect resume requires the same distributed world size as the "
            "checkpoint."
        )
    rank_state_path = checkpoint / f"rank-{accelerator.process_index}.json"
    if not rank_state_path.is_file():
        raise FileNotFoundError(rank_state_path)
    rank_state = json.loads(rank_state_path.read_text(encoding="utf-8"))
    if rank_state.get("rank") != accelerator.process_index:
        raise ValueError("Checkpoint rank metadata is inconsistent.")
    rollout_path = output_dir / f"rollouts-rank-{accelerator.process_index}.jsonl"
    _truncate_file(rollout_path, int(rank_state["rollout_log_bytes"]))
    if accelerator.is_main_process:
        _truncate_file(
            output_dir / "metrics.jsonl",
            int(trainer_state["metrics_log_bytes"]),
        )
    return trainer_state


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


def _prepare_for_checkpoint(confidence_optimizer, accelerator) -> dict[str, float]:
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
    return {
        "checkpoint/max_allocated_gib": memory[0].item(),
        "checkpoint/max_reserved_gib": memory[1].item(),
        "checkpoint/peak_allocated_gib": memory[2].item(),
    }


def _optimizer_step_values(optimizer) -> set[int]:
    values: set[int] = set()
    for state in _adamw_optimizer(optimizer).state.values():
        if not state:
            continue
        step = state.get("step")
        if isinstance(step, torch.Tensor):
            if step.numel() != 1:
                raise ValueError("AdamW step tensors must be scalar.")
            step = step.item()
        if not isinstance(step, (int, float)) or int(step) != step:
            raise TypeError("AdamW optimizer step must be an integer scalar.")
        values.add(int(step))
    return values


def _validate_optimizer_steps(optimizer, *, expected_steps: int) -> None:
    values = _optimizer_step_values(optimizer)
    if values != {expected_steps}:
        raise RuntimeError(
            "AdamW optimizer state was not restored exactly: expected every "
            f"initialized parameter at step {expected_steps}, got {sorted(values)}."
        )


def _initialize_adamw_state_for_fsdp_load(
    optimizer,
    *,
    moment_device: torch.device,
) -> None:
    """Materialize AdamW state so FSDP DCP has tensors to restore into."""
    raw_optimizer = _adamw_optimizer(optimizer)
    if raw_optimizer.state:
        raise RuntimeError("AdamW state must be empty before checkpoint restore.")
    for group in raw_optimizer.param_groups:
        for parameter in group["params"]:
            state = raw_optimizer.state[parameter]
            state["step"] = torch.zeros((), dtype=torch.float32)
            state["exp_avg"] = torch.zeros_like(
                parameter,
                device=moment_device,
                memory_format=torch.preserve_format,
            )
            state["exp_avg_sq"] = torch.zeros_like(
                parameter,
                device=moment_device,
                memory_format=torch.preserve_format,
            )
            if group["amsgrad"]:
                state["max_exp_avg_sq"] = torch.zeros_like(
                    parameter,
                    device=moment_device,
                    memory_format=torch.preserve_format,
                )


def _load_accelerator_state_without_optimizer(
    accelerator,
    checkpoint: Path,
) -> None:
    registered_optimizers = accelerator._optimizers
    accelerator._optimizers = []
    try:
        accelerator.load_state(checkpoint)
    finally:
        accelerator._optimizers = registered_optimizers


def _load_fsdp_optimizer_state(
    *,
    checkpoint: Path,
    optimizer,
    model,
    accelerator,
) -> None:
    import torch.distributed.checkpoint as dist_cp
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
    from torch.distributed.fsdp.fully_sharded_data_parallel import (
        FullyShardedDataParallel as FSDP,
    )

    fsdp_plugin = accelerator.state.fsdp_plugin
    with FSDP.state_dict_type(
        model,
        fsdp_plugin.state_dict_type,
        fsdp_plugin.state_dict_config,
        fsdp_plugin.optim_state_dict_config,
    ):
        optimizer_state = FSDP.optim_state_dict(model, optimizer)
        checkpoint_state = {"optimizer": optimizer_state}
        dist_cp.load(
            state_dict=checkpoint_state,
            storage_reader=dist_cp.FileSystemReader(checkpoint / "optimizer_0"),
            planner=DefaultLoadPlanner(allow_partial_load=True),
        )
        local_optimizer_state = FSDP.optim_state_dict_to_load(
            model=model,
            optim=optimizer,
            optim_state_dict=checkpoint_state["optimizer"],
        )
        optimizer.load_state_dict(local_optimizer_state)
    accelerator.wait_for_everyone()


def _save_committed_checkpoint(
    *,
    checkpoint: Path,
    completed_steps: int,
    expected_optimizer_steps: int,
    confidence_optimizer,
    accelerator,
    metrics_path: Path,
    rollout_log_path: Path | None,
) -> None:
    checkpoint_started = time.perf_counter()
    if accelerator.is_main_process and checkpoint.exists():
        if (checkpoint / "trainer_state.json").is_file():
            raise FileExistsError(
                f"Refusing to overwrite committed checkpoint: {checkpoint}"
            )
        else:
            shutil.rmtree(checkpoint)
    accelerator.wait_for_everyone()
    _validate_optimizer_steps(
        confidence_optimizer,
        expected_steps=expected_optimizer_steps,
    )
    memory = _prepare_for_checkpoint(confidence_optimizer, accelerator)
    accelerator.save_state(checkpoint)
    accelerator.wait_for_everyone()

    checkpoint_record = {
        "event": "checkpoint_committed",
        "completed_steps": completed_steps,
        "optimizer_steps": expected_optimizer_steps,
        "checkpoint": str(checkpoint),
        "checkpoint/save_seconds": time.perf_counter() - checkpoint_started,
        **memory,
    }
    if accelerator.is_main_process:
        _append_metric(metrics_path, checkpoint_record)
        print(json.dumps(checkpoint_record, sort_keys=True), flush=True)
    accelerator.wait_for_everyone()

    rank_state = {
        "rank": accelerator.process_index,
        "rollout_log_bytes": (
            rollout_log_path.stat().st_size
            if rollout_log_path is not None and rollout_log_path.is_file()
            else 0
        ),
    }
    _atomic_write_json(
        checkpoint / f"rank-{accelerator.process_index}.json",
        rank_state,
    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        trainer_state = {
            "schema_version": 1,
            "completed_steps": completed_steps,
            "optimizer_steps": expected_optimizer_steps,
            "world_size": accelerator.num_processes,
            "metrics_log_bytes": metrics_path.stat().st_size,
        }
        _atomic_write_json(checkpoint / "trainer_state.json", trainer_state)
    accelerator.wait_for_everyone()


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
    metrics_path: Path | None = None,
    step: int | None = None,
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
        record = {
            "event": "optimizer_moment_transfer",
            "step": step,
            "completed_steps": step,
            "outer_iteration": outer_iteration,
            "optimizer_moments/device": target_device.type,
            "optimizer_moments/gib_per_rank": moment_bytes / 1024**3,
            "optimizer_moments/transfer_seconds": elapsed,
            **cuda_memory,
        }
        if metrics_path is not None:
            _append_metric(metrics_path, record)
        print(json.dumps(record, sort_keys=True), flush=True)
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
    coefficients = {
        "meta": algorithm.meta_config.meta_coefficient,
        "token_meta": algorithm.meta_config.token_meta_coefficient,
        "bce": algorithm.meta_config.confidence.bce_coefficient,
        "ranking": algorithm.meta_config.confidence.ranking_coefficient,
    }
    enabled_coefficients = {
        name: coefficient
        for name, coefficient in coefficients.items()
        if coefficient > 0
    }
    norms: dict[str, float] = {}
    measurement_start = time.perf_counter()
    components = tqdm(
        tuple(enabled_coefficients),
        desc=f"{progress_prefix}: component gradient norms",
        unit="component",
        leave=True,
        disable=not accelerator.is_main_process,
    )
    for component in components:
        confidence_optimizer.zero_grad(set_to_none=True)
        problem_offset = 0
        for microbatch_index, microbatch in enumerate(rollout_microbatches):
            supports = tuple(
                group.to(accelerator.device) for group in microbatch.supports
            )
            queries = (
                tuple(group.to(accelerator.device) for group in microbatch.queries)
                if component == "meta"
                else None
            )
            token_queries = (
                tuple(
                    group.to(accelerator.device) for group in microbatch.token_queries
                )
                if component == "token_meta"
                else None
            )
            token_alignment_contexts = None
            if (
                component == "token_meta"
                and algorithm.token_meta_gradient_mode == "gradient_alignment"
            ):
                if microbatch.token_alignment_contexts is None:
                    token_alignment_contexts = (
                        algorithm.token_gradient_alignment_contexts_batch(
                            supports,
                            token_queries,
                            initial_fast,
                        )
                    )
                    rollout_microbatches[microbatch_index] = replace(
                        microbatch,
                        token_alignment_contexts=tuple(
                            context.to("cpu") for context in token_alignment_contexts
                        ),
                    )
                else:
                    token_alignment_contexts = tuple(
                        context.to(accelerator.device)
                        for context in microbatch.token_alignment_contexts
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
                elif component == "token_meta":
                    output = (
                        algorithm.token_gradient_alignment_losses_batch(
                            (support,),
                            (token_alignment_contexts[problem_index],),
                        )[0]
                        if algorithm.token_meta_gradient_mode == "gradient_alignment"
                        else algorithm.token_outer_losses_batch(
                            (support,),
                            (token_queries[problem_index],),
                            initial_fast,
                        )[0]
                    )
                    component_loss = output.meta_objective
                else:
                    confidence_loss = algorithm.confidence_supervision_loss(support)
                    component_loss = getattr(confidence_loss, component)
                accelerator.backward(component_loss / local_problem_batch_size)
                del component_loss
                if component in ("meta", "token_meta"):
                    del output
                else:
                    del confidence_loss
            problem_offset += len(supports)
            del (
                support,
                supports,
                queries,
                token_queries,
                token_alignment_contexts,
            )

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
            enabled_coefficients[component] * raw_value
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
    sequence_enabled = algorithm.meta_config.meta_coefficient > 0
    token_enabled = algorithm.meta_config.token_meta_coefficient > 0
    supervision_enabled = (
        algorithm.meta_config.confidence.bce_coefficient > 0
        or algorithm.meta_config.confidence.ranking_coefficient > 0
    )
    if sequence_enabled and any(
        microbatch.queries is None for microbatch in rollout_microbatches
    ):
        raise ValueError("Sequence meta loss requires cached sequence queries.")
    if token_enabled and any(
        microbatch.token_queries is None for microbatch in rollout_microbatches
    ):
        raise ValueError("Token meta loss requires cached token queries.")
    if defer_gradient_sync and confidence_model is None:
        raise ValueError(
            "confidence_model is required when gradient synchronization is deferred."
        )

    local_metric_sums = torch.zeros(28, dtype=torch.float32, device=accelerator.device)
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
            queries = (
                tuple(group.to(accelerator.device) for group in microbatch.queries)
                if sequence_enabled
                else None
            )
            token_queries = (
                tuple(
                    group.to(accelerator.device) for group in microbatch.token_queries
                )
                if token_enabled
                else None
            )
            token_alignment_contexts = None
            if (
                token_enabled
                and algorithm.token_meta_gradient_mode == "gradient_alignment"
            ):
                if microbatch.token_alignment_contexts is None:
                    token_alignment_contexts = (
                        algorithm.token_gradient_alignment_contexts_batch(
                            supports,
                            token_queries,
                            initial_fast,
                        )
                    )
                    rollout_microbatches[microbatch_index] = replace(
                        microbatch,
                        token_alignment_contexts=tuple(
                            context.to("cpu") for context in token_alignment_contexts
                        ),
                    )
                else:
                    token_alignment_contexts = tuple(
                        context.to(accelerator.device)
                        for context in microbatch.token_alignment_contexts
                    )
            sequence_outputs = (
                algorithm.outer_losses_batch(
                    supports,
                    queries,
                    initial_fast,
                    show_progress=False,
                    progress_prefix=progress_description,
                )
                if sequence_enabled
                else None
            )
            token_outputs = (
                (
                    algorithm.token_gradient_alignment_losses_batch(
                        supports,
                        token_alignment_contexts,
                    )
                    if algorithm.token_meta_gradient_mode == "gradient_alignment"
                    else algorithm.token_outer_losses_batch(
                        supports,
                        token_queries,
                        initial_fast,
                        show_progress=False,
                        progress_prefix=progress_description,
                    )
                )
                if token_enabled
                else None
            )
            confidence_outputs = (
                tuple(
                    algorithm.confidence_supervision_loss(support)
                    for support in supports
                )
                if supervision_enabled and not sequence_enabled
                else None
            )
            problem_losses = []
            for index, support in enumerate(supports):
                loss = support.old_logprobs.new_zeros(())
                if sequence_outputs is not None:
                    loss = loss + sequence_outputs[index].loss
                elif confidence_outputs is not None:
                    loss = loss + confidence_outputs[index].loss
                if token_outputs is not None:
                    loss = loss + token_outputs[index].loss
                problem_losses.append(loss)
            accumulated_loss = (
                torch.stack(problem_losses).sum() / local_problem_batch_size
            )
            accelerator.backward(accumulated_loss)
        for index, support in enumerate(supports):
            sequence_output = (
                None if sequence_outputs is None else sequence_outputs[index]
            )
            token_output = None if token_outputs is None else token_outputs[index]
            query = None if queries is None else queries[index]
            token_query = None if token_queries is None else token_queries[index]
            confidence_loss = (
                sequence_output.adaptation.confidence_loss
                if sequence_output is not None
                else (None if confidence_outputs is None else confidence_outputs[index])
            )
            zero = support.old_logprobs.new_zeros(())
            sequence_probability = (
                zero
                if sequence_output is None
                else sequence_output.adaptation.confidence_probabilities.detach().mean()
            )
            token_credit = (
                zero
                if token_output is None
                else token_output.token_credits.detach()[support.completion_mask].mean()
            )
            sequence_probability_square = (
                zero
                if sequence_output is None
                else sequence_output.adaptation.confidence_probabilities.detach()
                .square()
                .mean()
            )
            token_credit_square = (
                zero
                if token_output is None
                else token_output.token_credits.detach()[support.completion_mask]
                .square()
                .mean()
            )
            token_credit_absolute = (
                zero
                if token_output is None
                else token_output.token_credits.detach()[support.completion_mask]
                .abs()
                .mean()
            )
            token_credit_saturation = (
                zero
                if token_output is None
                else (
                    token_output.token_credits.detach()[support.completion_mask].abs()
                    >= 0.95 * algorithm.token_credit_max
                )
                .float()
                .mean()
            )
            local_metric_sums += torch.stack(
                (
                    problem_losses[index].detach(),
                    (
                        zero
                        if sequence_output is None
                        else sequence_output.meta_grpo.loss.detach()
                    ),
                    (
                        zero
                        if token_output is None
                        else token_output.meta_objective.detach()
                    ),
                    zero if confidence_loss is None else confidence_loss.bce.detach(),
                    (
                        zero
                        if confidence_loss is None
                        else confidence_loss.ranking.detach()
                    ),
                    (
                        zero
                        if sequence_output is None
                        else sequence_output.meta_grpo.clip_fraction.detach()
                    ),
                    (
                        zero
                        if token_output is None
                        else token_output.meta_grpo.clip_fraction.detach()
                    ),
                    (
                        zero
                        if sequence_output is None
                        else sequence_output.adaptation.inner_losses[
                            -1
                        ].clip_fraction.detach()
                    ),
                    (
                        zero
                        if token_output is None
                        else token_output.inner_grpo.clip_fraction.detach()
                    ),
                    support.verifier_rewards.mean(),
                    zero if query is None else query.verifier_rewards.mean(),
                    (
                        zero
                        if token_query is None
                        else token_query.verifier_rewards.mean()
                    ),
                    support.correctness_labels.mean(),
                    zero if query is None else query.correctness_labels.mean(),
                    (
                        zero
                        if token_query is None
                        else token_query.correctness_labels.mean()
                    ),
                    (
                        zero
                        if sequence_output is None
                        else sequence_output.meta_grpo.mean_kl.detach()
                    ),
                    (
                        zero
                        if token_output is None
                        else token_output.meta_grpo.mean_kl.detach()
                    ),
                    (
                        zero
                        if sequence_output is None
                        else sequence_output.adaptation.inner_losses[
                            -1
                        ].mean_kl.detach()
                    ),
                    (
                        zero
                        if token_output is None
                        else token_output.inner_grpo.mean_kl.detach()
                    ),
                    sequence_probability,
                    sequence_probability_square,
                    token_credit,
                    token_credit_square,
                    torch.any(support.correctness_labels == 1).float(),
                    (
                        zero
                        if query is None
                        else torch.any(query.correctness_labels == 1).float()
                    ),
                    (
                        zero
                        if token_query is None
                        else torch.any(token_query.correctness_labels == 1).float()
                    ),
                    token_credit_absolute,
                    token_credit_saturation,
                )
            )
            problem_progress.update(1)
        problem_offset += len(supports)
        del (
            accumulated_loss,
            problem_losses,
            support,
            supports,
            queries,
            token_queries,
            token_alignment_contexts,
            sequence_outputs,
            token_outputs,
            confidence_outputs,
        )
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
    metrics_path: Path,
    base_seed: int,
) -> dict[str, float | int | str]:
    if not problems:
        raise ValueError("Validation problems cannot be empty.")
    evaluation_started = time.perf_counter()
    rounds = math.ceil(len(problems) / accelerator.num_processes)
    was_training = confidence_model.training
    confidence_model.eval()
    sequence_enabled = algorithm.meta_config.meta_coefficient > 0
    token_enabled = algorithm.meta_config.token_meta_coefficient > 0
    local_metrics = torch.zeros(9, device=accelerator.device, dtype=torch.float64)
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
                seed=_generation_seed(
                    base_seed=base_seed,
                    step=step,
                    phase="validation_base",
                    rank=accelerator.process_index,
                    problem_uid=problem.uid,
                ),
            )
            base_verification = verifier(
                support.texts,
                problem.ground_truth,
                device=accelerator.device,
            )
            query = None
            adapted_verification = None
            if sequence_enabled:
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
                    seed=_generation_seed(
                        base_seed=base_seed,
                        step=step,
                        phase="validation_adapted",
                        rank=accelerator.process_index,
                        problem_uid=problem.uid,
                    ),
                )
                adapted_verification = verifier(
                    query.texts,
                    problem.ground_truth,
                    device=accelerator.device,
                )
                del generation_adaptation
            token_query = None
            token_verification = None
            if token_enabled:
                token_adaptation = algorithm.adapt_token_task(
                    support,
                    initial_fast,
                    differentiable=False,
                )
                token_query = query_rollouts.generate(
                    problem,
                    token_adaptation.fast_parameters,
                    show_progress=accelerator.is_main_process,
                    progress_description=f"{progress_prefix} token adapted",
                    seed=_generation_seed(
                        base_seed=base_seed,
                        step=step,
                        phase="validation_token_adapted",
                        rank=accelerator.process_index,
                        problem_uid=problem.uid,
                    ),
                )
                token_verification = verifier(
                    token_query.texts,
                    problem.ground_truth,
                    device=accelerator.device,
                )
                del token_adaptation
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
                if sequence_enabled:
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
                if token_enabled:
                    _append_rollout_diagnostics(
                        rollout_log_path,
                        step=step,
                        phase="validation_token_adapted",
                        problem=problem,
                        group=token_query,
                        verification=token_verification,
                        max_new_tokens=query_rollouts.max_new_tokens,
                        eos_token_id=query_rollouts.tokenizer.eos_token_id,
                    )
                local_metrics += torch.tensor(
                    (
                        base_verification.correctness.sum().item(),
                        float(base_verification.correctness.numel()),
                        float(torch.any(base_verification.correctness == 1).item()),
                        (
                            0.0
                            if adapted_verification is None
                            else adapted_verification.correctness.sum().item()
                        ),
                        (
                            0.0
                            if adapted_verification is None
                            else float(adapted_verification.correctness.numel())
                        ),
                        (
                            0.0
                            if adapted_verification is None
                            else float(
                                torch.any(adapted_verification.correctness == 1).item()
                            )
                        ),
                        (
                            0.0
                            if token_verification is None
                            else token_verification.correctness.sum().item()
                        ),
                        (
                            0.0
                            if token_verification is None
                            else float(token_verification.correctness.numel())
                        ),
                        (
                            0.0
                            if token_verification is None
                            else float(
                                torch.any(token_verification.correctness == 1).item()
                            )
                        ),
                    ),
                    device=accelerator.device,
                    dtype=torch.float64,
                )
    finally:
        confidence_model.train(was_training)

    totals = accelerator.reduce(local_metrics, reduction="sum")
    record = {}
    if accelerator.is_main_process:
        base_accuracy = (totals[0] / totals[1]).item()
        sequence_accuracy = (totals[3] / totals[4]).item() if sequence_enabled else 0.0
        token_accuracy = (totals[6] / totals[7]).item() if token_enabled else 0.0
        adapted_accuracy = sequence_accuracy if sequence_enabled else token_accuracy
        adapted_pass = (
            (totals[5] / len(problems)).item()
            if sequence_enabled
            else (totals[8] / len(problems)).item()
        )
        record = {
            "event": "validation",
            "completed_steps": step,
            "validation/base_accuracy": base_accuracy,
            "validation/adapted_accuracy": adapted_accuracy,
            "validation/accuracy_delta": adapted_accuracy - base_accuracy,
            "validation/base_pass_at_group": (totals[2] / len(problems)).item(),
            "validation/adapted_pass_at_group": adapted_pass,
            "validation/sequence_adapted_accuracy": sequence_accuracy,
            "validation/sequence_adapted_pass_at_group": (
                (totals[5] / len(problems)).item() if sequence_enabled else 0.0
            ),
            "validation/token_adapted_accuracy": token_accuracy,
            "validation/token_adapted_pass_at_group": (
                (totals[8] / len(problems)).item() if token_enabled else 0.0
            ),
            "validation/num_unique_problems": len(problems),
            "validation/support_group_size": support_rollouts.group_size,
            "validation/query_group_size": query_rollouts.group_size,
            "validation/support_responses": int(totals[1].item()),
            "validation/query_responses": int(totals[4].item()),
            "validation/token_query_responses": int(totals[7].item()),
            "validation/seconds": time.perf_counter() - evaluation_started,
        }
        _append_metric(metrics_path, record)
        print(json.dumps(record, sort_keys=True), flush=True)
    return record


def main() -> None:
    args = parse_args()
    _validate_args(args)
    if args.validation_support_group_size is None:
        args.validation_support_group_size = args.support_group_size
    if args.validation_query_group_size is None:
        args.validation_query_group_size = args.query_group_size
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
    (
        _,
        _,
        global_rollout_problem_batch_size,
        local_rollout_problem_batch_size,
        num_rollout_problem_batches,
    ) = _problem_batch_layout(
        global_problem_batch_size,
        args.rollout_problem_batch_size,
        world_size=accelerator.num_processes,
        sub_batch_name="rollout_problem_batch_size",
    )
    args.problem_batch_size = global_problem_batch_size
    args.problem_micro_batch_size = global_problem_micro_batch_size
    args.rollout_problem_batch_size = global_rollout_problem_batch_size
    args.output_dir = args.output_dir.resolve()
    if args.resume_from_checkpoint is not None:
        args.resume_from_checkpoint = args.resume_from_checkpoint.resolve()
    resume_checkpoint, start_step = _resolve_resume_checkpoint(args)
    set_seed(args.seed, device_specific=True)
    _initialize_or_validate_run(args, accelerator=accelerator)
    accelerator.wait_for_everyone()
    metrics_path = args.output_dir / "metrics.jsonl"
    rollout_log_path = (
        args.output_dir / f"rollouts-rank-{accelerator.process_index}.jsonl"
        if args.log_rollouts
        else None
    )
    checkpoint_trainer_state = None
    if resume_checkpoint is not None:
        checkpoint_trainer_state = _restore_committed_log_boundaries(
            resume_checkpoint,
            output_dir=args.output_dir,
            accelerator=accelerator,
        )
        if checkpoint_trainer_state["completed_steps"] != start_step:
            raise ValueError("Checkpoint completed-step metadata is inconsistent.")
        if start_step >= args.max_steps:
            raise ValueError(
                "Checkpoint already reached or exceeded the requested max_steps."
            )
    accelerator.wait_for_everyone()

    model_kwargs = {"attn_implementation": args.attn_implementation}
    target_modules = tuple(
        module.strip() for module in args.lora_target_modules.split(",")
    )
    confidence_model = load_confidence_model(
        args.confidence_model,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        model_kwargs=model_kwargs,
        enable_sequence_head=(
            args.meta_coefficient > 0
            or args.bce_coefficient > 0
            or args.ranking_coefficient > 0
        ),
        enable_token_head=args.token_meta_coefficient > 0,
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
    optimizer_moments_offloaded = False
    if resume_checkpoint is not None:
        _initialize_adamw_state_for_fsdp_load(
            confidence_optimizer,
            moment_device=(
                torch.device("cpu")
                if args.offload_confidence_optimizer
                else accelerator.device
            ),
        )
        _load_accelerator_state_without_optimizer(accelerator, resume_checkpoint)
        _load_fsdp_optimizer_state(
            checkpoint=resume_checkpoint,
            optimizer=confidence_optimizer,
            model=confidence_model,
            accelerator=accelerator,
        )
        expected_optimizer_steps = start_step * args.outer_iterations
        if checkpoint_trainer_state["optimizer_steps"] != expected_optimizer_steps:
            raise ValueError("Checkpoint optimizer-step metadata is inconsistent.")
        _validate_optimizer_steps(
            confidence_optimizer,
            expected_steps=expected_optimizer_steps,
        )
        if args.offload_confidence_optimizer:
            _transfer_confidence_optimizer_moments(
                confidence_optimizer,
                target_device=torch.device("cpu"),
                accelerator=accelerator,
                outer_iteration="resume",
                metrics_path=metrics_path,
                step=start_step,
            )
            optimizer_moments_offloaded = True
        if accelerator.is_main_process:
            resume_record = {
                "event": "resumed",
                "completed_steps": start_step,
                "optimizer_steps": expected_optimizer_steps,
                "checkpoint": str(resume_checkpoint),
            }
            _append_metric(metrics_path, resume_record)
            print(json.dumps(resume_record, sort_keys=True), flush=True)
        if args.resume_preflight_only:
            if accelerator.is_main_process:
                print(
                    json.dumps(
                        {
                            "event": "resume_preflight_passed",
                            "completed_steps": start_step,
                            "optimizer_steps": expected_optimizer_steps,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            accelerator.wait_for_everyone()
            return

    policy_bundle = load_policy_with_lora(
        args.policy_model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        dtype=args.dtype,
        force_sdpa_math=args.token_meta_coefficient > 0,
        trust_remote_code=args.trust_remote_code,
        model_kwargs=model_kwargs,
    )
    policy = policy_bundle.model.to(accelerator.device)

    algorithm = BilevelGRPO(
        policy=policy,
        confidence_model=confidence_model,
        inner_config=inner_config,
        meta_config=meta_config,
        query_advantage_config=query_advantage_config,
        query_grpo_config=query_grpo_config,
        policy_micro_batch_size=args.policy_micro_batch_size,
        first_order_vjp_forward_batch_size=(args.first_order_vjp_forward_batch_size),
        confidence_micro_batch_size=args.confidence_micro_batch_size,
        policy_max_tokens_per_micro_batch=(args.policy_max_tokens_per_micro_batch),
        confidence_max_tokens_per_micro_batch=(
            args.confidence_max_tokens_per_micro_batch
        ),
        token_jvp_response_micro_batch_size=(args.token_jvp_response_micro_batch_size),
        token_jvp_logprob_position_chunk_size=(
            args.token_jvp_logprob_position_chunk_size
        ),
        token_credit_max=args.token_credit_max,
        token_meta_gradient_mode=args.token_meta_gradient_mode,
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

    def build_rollout_engine(
        group_size: int,
        *,
        temperature: float,
        top_p: float,
        top_k: int,
    ):
        return rollout_engine_type(
            policy,
            policy_bundle.tokenizer,
            group_size=group_size,
            max_new_tokens=args.max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            generation_micro_batch_size=args.generation_micro_batch_size,
            logprob_micro_batch_size=args.policy_micro_batch_size,
            logprob_max_tokens_per_micro_batch=(args.policy_max_tokens_per_micro_batch),
            **rollout_kwargs,
        )

    gradient_sampling = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }
    validation_sampling = {
        "temperature": args.validation_temperature,
        "top_p": args.validation_top_p,
        "top_k": args.validation_top_k,
    }
    support_rollouts = build_rollout_engine(
        args.support_group_size,
        **gradient_sampling,
    )
    query_rollouts = build_rollout_engine(
        args.query_group_size,
        **gradient_sampling,
    )
    validation_support_rollouts = build_rollout_engine(
        args.validation_support_group_size,
        **gradient_sampling,
    )
    validation_query_rollouts = build_rollout_engine(
        args.validation_query_group_size,
        **validation_sampling,
    )
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
        f"({local_problem_batch_size} per rank); global rollout problem batch is "
        f"{global_rollout_problem_batch_size} "
        f"({local_rollout_problem_batch_size} per rank, "
        f"{num_rollout_problem_batches} rollout batches); global gradient problem "
        f"microbatch is {global_problem_micro_batch_size} "
        f"({local_problem_micro_batch_size} per rank, "
        f"{num_problem_micro_batches} accumulation microbatches); first-order "
        f"VJP forward batch is {args.first_order_vjp_forward_batch_size}; policy "
        f"forward backend is "
        f"{'sdpa_math' if args.token_meta_coefficient > 0 else args.attn_implementation}; "
        f"token meta-gradient mode is "
        f"{args.token_meta_gradient_mode if args.token_meta_coefficient > 0 else 'disabled'}."
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
        rollout_totals = torch.zeros(9, dtype=torch.float32, device=accelerator.device)
        rollout_problem_batches = _fixed_size_batches(
            problems,
            local_rollout_problem_batch_size,
        )

        step_indices.set_postfix_str(
            f"problems={global_problem_batch_size} stage=support-rollout"
        )
        with timings.measure("support_rollout"):
            support_batches = support_rollouts.generate_batches(
                rollout_problem_batches,
                [
                    [initial_fast] * local_rollout_problem_batch_size
                    for _ in rollout_problem_batches
                ],
                show_progress=accelerator.is_main_process,
                progress_description=f"{batch_prefix} support",
                seed_batches=_generation_seed_batches(
                    rollout_problem_batches,
                    base_seed=args.seed,
                    step=step,
                    phase="train_support",
                    rank=accelerator.process_index,
                ),
            )
        support_groups = tuple(group for batch in support_batches for group in batch)
        if len(support_groups) != local_problem_batch_size:
            raise RuntimeError("Support rollout count does not match problem batch.")
        del support_batches

        # Repartition CPU-backed rollout results into the smaller gradient
        # microbatches without changing problem order.
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
            supports = list(support_groups[start:end])
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
                        support = support.with_reference_logprobs(support.old_logprobs)
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
        del support_groups

        def generate_query_branch(*, token_branch: bool) -> None:
            branch = "token" if token_branch else "sequence"
            phase = "train_token_query" if token_branch else "train_query"
            metric_offset = 6 if token_branch else 3
            timing_prefix = "token_" if token_branch else ""
            generation_fast_parameters = []
            use_base_token_query = (
                token_branch and args.token_meta_gradient_mode == "gradient_alignment"
            )
            if use_base_token_query:
                base_fast_parameters = {
                    name: value.detach().to("cpu")
                    for name, value in initial_fast.items()
                }
                generation_fast_parameters = [
                    base_fast_parameters
                ] * local_problem_batch_size
                timings.values[f"{timing_prefix}generation_adaptation"] = 0.0
            else:
                step_indices.set_postfix_str(
                    f"problems={global_problem_batch_size} "
                    f"stage={branch}-generation-adaptation"
                )
                with timings.measure(f"{timing_prefix}generation_adaptation"):
                    for cached_microbatch in tqdm(
                        rollout_microbatches,
                        total=len(rollout_microbatches),
                        desc=f"{batch_prefix} {branch} generation adaptation",
                        unit="microbatch",
                        leave=True,
                        disable=not accelerator.is_main_process,
                    ):
                        supports = tuple(
                            support.to(accelerator.device)
                            for support in cached_microbatch.supports
                        )
                        adaptations = (
                            algorithm.adapt_token_tasks(
                                supports, initial_fast, differentiable=False
                            )
                            if token_branch
                            else algorithm.adapt_tasks(
                                supports,
                                initial_fast,
                                differentiable=False,
                                supervise_confidence=False,
                                show_progress=False,
                            )
                        )
                        generation_fast_parameters.extend(
                            {
                                name: value.detach().to("cpu")
                                for name, value in adaptation.fast_parameters.items()
                            }
                            for adaptation in adaptations
                        )
                        del supports, adaptations
            if len(generation_fast_parameters) != local_problem_batch_size:
                raise RuntimeError(
                    f"{branch} adapted parameter count does not match problem batch."
                )
            fast_batches = _fixed_size_batches(
                generation_fast_parameters,
                local_rollout_problem_batch_size,
            )
            step_indices.set_postfix_str(
                f"problems={global_problem_batch_size} stage={branch}-query-rollout"
            )
            with timings.measure(f"{timing_prefix}query_rollout"):
                generated_batches = query_rollouts.generate_batches(
                    rollout_problem_batches,
                    fast_batches,
                    show_progress=accelerator.is_main_process,
                    progress_description=f"{batch_prefix} {branch} query",
                    seed_batches=_generation_seed_batches(
                        rollout_problem_batches,
                        base_seed=args.seed,
                        step=step,
                        phase=phase,
                        rank=accelerator.process_index,
                    ),
                )
            query_groups = tuple(
                group for batch in generated_batches for group in batch
            )
            if len(query_groups) != local_problem_batch_size:
                raise RuntimeError(
                    f"{branch} query count does not match problem batch."
                )
            del generation_fast_parameters, fast_batches, generated_batches

            for microbatch_index, cached_microbatch in enumerate(rollout_microbatches):
                start = microbatch_index * local_problem_micro_batch_size
                end = start + local_problem_micro_batch_size
                queries = list(query_groups[start:end])
                with timings.measure(f"{timing_prefix}query_verifier_and_cache"):
                    for problem_index, (problem, query) in enumerate(
                        zip(cached_microbatch.problems, queries, strict=True)
                    ):
                        verification = verifier(
                            query.texts,
                            problem.ground_truth,
                            device=query.device,
                        )
                        query = query.with_verification(
                            verification.rewards,
                            verification.correctness,
                        )
                        if args.outer_kl_coefficient > 0:
                            query = query_rollouts.add_reference_logprobs(
                                query,
                                initial_fast,
                                show_progress=False,
                            )
                        _append_rollout_diagnostics(
                            rollout_log_path,
                            step=step,
                            phase=phase,
                            problem=problem,
                            group=query,
                            verification=verification,
                            max_new_tokens=query_rollouts.max_new_tokens,
                            eos_token_id=query_rollouts.tokenizer.eos_token_id,
                        )
                        rollout_totals[
                            metric_offset
                        ] += query.verifier_rewards.sum().to(accelerator.device)
                        rollout_totals[
                            metric_offset + 1
                        ] += query.correctness_labels.sum().to(accelerator.device)
                        rollout_totals[metric_offset + 2] += query.group_size
                        queries[problem_index] = query
                    cached_queries = tuple(
                        _cache_rollout_group(query) for query in queries
                    )
                rollout_microbatches[microbatch_index] = replace(
                    cached_microbatch,
                    **(
                        {"token_queries": cached_queries}
                        if token_branch
                        else {"queries": cached_queries}
                    ),
                )
                if accelerator.device.type == "cuda":
                    torch.cuda.empty_cache()
            del query_groups

        if args.meta_coefficient > 0:
            generate_query_branch(token_branch=False)
        if args.token_meta_coefficient > 0:
            generate_query_branch(token_branch=True)
        del rollout_problem_batches

        if args.meta_coefficient > 0 and any(
            microbatch.queries is None for microbatch in rollout_microbatches
        ):
            raise RuntimeError("Sequence query rollout cache is incomplete.")
        if args.token_meta_coefficient > 0 and any(
            microbatch.token_queries is None for microbatch in rollout_microbatches
        ):
            raise RuntimeError("Token query rollout cache is incomplete.")
        if accelerator.is_main_process:
            timing_record = {
                "event": "pipeline_timing",
                "step": step,
                "completed_steps": step,
                "timing_seconds": dict(timings.values),
            }
            _append_metric(metrics_path, timing_record)
            print(json.dumps(timing_record, sort_keys=True), flush=True)
        if args.rollout_only:
            rollout_totals = accelerator.reduce(rollout_totals, reduction="sum")
            if accelerator.is_main_process:
                rollout_record = {
                    "event": "rollout_only",
                    "rollout_only": True,
                    "step": step,
                    "completed_steps": step,
                    "support_mean_reward": (
                        rollout_totals[0] / rollout_totals[2]
                    ).item(),
                    "support_accuracy": (rollout_totals[1] / rollout_totals[2]).item(),
                    "query_mean_reward": (
                        (rollout_totals[3] / rollout_totals[5]).item()
                        if args.meta_coefficient > 0
                        else 0.0
                    ),
                    "query_accuracy": (
                        (rollout_totals[4] / rollout_totals[5]).item()
                        if args.meta_coefficient > 0
                        else 0.0
                    ),
                    "token_query_mean_reward": (
                        (rollout_totals[6] / rollout_totals[8]).item()
                        if args.token_meta_coefficient > 0
                        else 0.0
                    ),
                    "token_query_accuracy": (
                        (rollout_totals[7] / rollout_totals[8]).item()
                        if args.token_meta_coefficient > 0
                        else 0.0
                    ),
                    "num_problems": global_problem_batch_size,
                    "problem_micro_batch_size": global_problem_micro_batch_size,
                    "rollout_problem_batch_size": global_rollout_problem_batch_size,
                    "support_responses": int(rollout_totals[2].item()),
                    "query_responses": int(rollout_totals[5].item()),
                    "token_query_responses": int(rollout_totals[8].item()),
                    "timing_seconds": dict(timings.values),
                }
                _append_metric(metrics_path, rollout_record)
                print(json.dumps(rollout_record, sort_keys=True), flush=True)
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
                                f"train step {step + 1} outer " f"{outer_iteration + 1}"
                            ),
                        )
                if accelerator.is_main_process:
                    component_record = {
                        "event": "component_gradient_norms",
                        "step": step,
                        "completed_steps": step,
                        "outer_iteration": outer_iteration,
                        "global_outer_iteration": global_outer_iteration,
                        **component_gradient_norms,
                    }
                    _append_metric(metrics_path, component_record)
                    print(json.dumps(component_record, sort_keys=True), flush=True)

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
                with timings.measure(f"{outer_timing_prefix}/optimizer_moments_to_gpu"):
                    _transfer_confidence_optimizer_moments(
                        confidence_optimizer,
                        target_device=accelerator.device,
                        accelerator=accelerator,
                        outer_iteration=outer_iteration,
                        metrics_path=metrics_path,
                        step=step,
                    )
                optimizer_moments_offloaded = False
            outer_iterations.set_postfix_str("stage=optimizer-step")
            with timings.measure(f"{outer_timing_prefix}/optimizer_step"):
                confidence_optimizer.step()
            if args.offload_confidence_optimizer:
                outer_iterations.set_postfix_str("stage=optimizer-moments-to-cpu")
                with timings.measure(f"{outer_timing_prefix}/optimizer_moments_to_cpu"):
                    _transfer_confidence_optimizer_moments(
                        confidence_optimizer,
                        target_device=torch.device("cpu"),
                        accelerator=accelerator,
                        outer_iteration=outer_iteration,
                        metrics_path=metrics_path,
                        step=step,
                    )
                optimizer_moments_offloaded = True
            outer_iterations.set_postfix_str("stage=metrics")

            metrics = accelerator.reduce(local_metric_sums, reduction="sum")
            metrics = metrics / global_problem_batch_size
            reduced_gradient_norm = accelerator.reduce(
                gradient_norm.detach(), reduction="mean"
            )
            if accelerator.is_main_process:
                train_record = {
                    "event": "train_outer",
                    "step": step,
                    "completed_steps": step,
                    "outer_iteration": outer_iteration,
                    "global_outer_iteration": global_outer_iteration,
                    "problem_batch_size": global_problem_batch_size,
                    "problem_micro_batch_size": global_problem_micro_batch_size,
                    "rollout_problem_batch_size": global_rollout_problem_batch_size,
                    "loss": metrics[0].item(),
                    "meta_loss": metrics[1].item(),
                    "token_meta_loss": metrics[2].item(),
                    "bce": metrics[3].item(),
                    "ranking": metrics[4].item(),
                    "outer_clip_fraction": metrics[5].item(),
                    "token_outer_clip_fraction": metrics[6].item(),
                    "inner_clip_fraction": metrics[7].item(),
                    "token_inner_clip_fraction": metrics[8].item(),
                    "support_reward": metrics[9].item(),
                    "query_reward": metrics[10].item(),
                    "token_query_reward": metrics[11].item(),
                    "support_accuracy": metrics[12].item(),
                    "query_accuracy": metrics[13].item(),
                    "token_query_accuracy": metrics[14].item(),
                    "outer_mean_kl": metrics[15].item(),
                    "token_outer_mean_kl": metrics[16].item(),
                    "inner_mean_kl": metrics[17].item(),
                    "token_inner_mean_kl": metrics[18].item(),
                    "confidence_probability_mean": metrics[19].item(),
                    "confidence_probability_std": torch.sqrt(
                        torch.clamp(metrics[20] - metrics[19].square(), min=0)
                    ).item(),
                    "token_credit_mean": metrics[21].item(),
                    "token_credit_std": torch.sqrt(
                        torch.clamp(metrics[22] - metrics[21].square(), min=0)
                    ).item(),
                    "support_pass_at_group": metrics[23].item(),
                    "query_pass_at_group": metrics[24].item(),
                    "token_query_pass_at_group": metrics[25].item(),
                    "token_credit_abs_mean": metrics[26].item(),
                    "token_credit_saturation_fraction": metrics[27].item(),
                    "confidence_gradient_norm": reduced_gradient_norm.item(),
                    "confidence_learning_rate": confidence_optimizer.param_groups[0][
                        "lr"
                    ],
                    "support_responses": (
                        global_problem_batch_size * args.support_group_size
                    ),
                    "query_responses": (
                        global_problem_batch_size
                        * args.query_group_size
                        * int(args.meta_coefficient > 0)
                    ),
                    "token_query_responses": (
                        global_problem_batch_size
                        * args.query_group_size
                        * int(args.token_meta_coefficient > 0)
                    ),
                    "timing_seconds": dict(timings.values),
                    **component_gradient_norms,
                }
                _append_metric(metrics_path, train_record)
                print(json.dumps(train_record, sort_keys=True), flush=True)

        del problems, rollout_microbatches, rollout_totals

        should_evaluate = (
            args.eval_steps > 0 and (step + 1) % args.eval_steps == 0
        ) or step + 1 == args.max_steps
        if should_evaluate:
            _evaluate(
                step=step + 1,
                problems=validation_problems,
                algorithm=algorithm,
                support_rollouts=validation_support_rollouts,
                query_rollouts=validation_query_rollouts,
                verifier=verifier,
                initial_fast=initial_fast,
                confidence_model=confidence_model,
                accelerator=accelerator,
                rollout_log_path=rollout_log_path,
                metrics_path=metrics_path,
                base_seed=args.seed,
            )

        should_save = (step + 1) % args.save_steps == 0 or step + 1 == args.max_steps
        if should_save:
            _save_committed_checkpoint(
                checkpoint=args.output_dir / f"checkpoint-{step + 1}",
                completed_steps=step + 1,
                expected_optimizer_steps=(step + 1) * args.outer_iterations,
                confidence_optimizer=confidence_optimizer,
                accelerator=accelerator,
                metrics_path=metrics_path,
                rollout_log_path=rollout_log_path,
            )

    if accelerator.is_main_process:
        _atomic_write_json(
            args.output_dir / "final_checkpoint.json",
            {
                "completed_steps": args.max_steps,
                "checkpoint": str(args.output_dir / f"checkpoint-{args.max_steps}"),
            },
        )
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
