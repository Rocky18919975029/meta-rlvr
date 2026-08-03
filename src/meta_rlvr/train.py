from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
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
    parser.add_argument("--generation-micro-batch-size", type=int, default=4)
    parser.add_argument("--policy-micro-batch-size", type=int, default=1)
    parser.add_argument("--confidence-micro-batch-size", type=int, default=2)
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

    algorithm = BilevelGRPO(
        policy=policy,
        confidence_model=confidence_model,
        inner_config=inner_config,
        meta_config=meta_config,
        query_advantage_config=query_advantage_config,
        query_grpo_config=query_grpo_config,
        policy_micro_batch_size=args.policy_micro_batch_size,
        confidence_micro_batch_size=args.confidence_micro_batch_size,
    )
    rollout_kwargs: dict[str, object] = {}
    rollout_engine_type = TransformersRolloutEngine
    if args.rollout_backend == "vllm":
        base_urls = [url.strip() for url in args.vllm_base_urls.split(",")]
        if any(not url for url in base_urls):
            raise ValueError("--vllm-base-urls contains an empty URL.")
        if len(base_urls) != accelerator.num_processes:
            raise ValueError(
                f"Expected {accelerator.num_processes} vLLM URLs, got "
                f"{len(base_urls)}."
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
    accelerator.print(
        f"Loaded {len(all_problems)} unique problems; "
        f"rank {accelerator.process_index} owns {len(local_problems)}; "
        f"validation has {len(validation_problems)} semantically unique problems."
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
        problem = _next_problem(local_problems, step=step, seed=args.seed)
        step_indices.set_postfix_str(f"problem={problem.uid} stage=support-rollout")
        progress_prefix = f"train step {step + 1} problem {problem.uid}"
        support = support_rollouts.generate(
            problem,
            initial_fast,
            show_progress=accelerator.is_main_process,
            progress_description=f"{progress_prefix} support",
        )
        step_indices.set_postfix_str(f"problem={problem.uid} stage=support-verifier")
        support_verification = verifier(
            support.texts,
            problem.ground_truth,
            device=accelerator.device,
        )
        support = support.with_verification(
            support_verification.rewards,
            support_verification.correctness,
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
        if args.inner_kl_coefficient > 0:
            support = support.with_reference_logprobs(support.old_logprobs)

        cached_query = None
        outer_iterations = tqdm(
            range(args.outer_iterations),
            desc=f"train step {step + 1}: outer updates",
            unit="update",
            leave=True,
            disable=not accelerator.is_main_process,
        )
        for outer_iteration in outer_iterations:
            confidence_optimizer.zero_grad(set_to_none=True)

            if cached_query is None:
                outer_iterations.set_postfix_str("stage=generation-adaptation")
                generation_adaptation = algorithm.adapt_task(
                    support,
                    initial_fast,
                    differentiable=False,
                    supervise_confidence=False,
                    show_progress=accelerator.is_main_process,
                    progress_prefix=f"{progress_prefix} generation adaptation",
                )
                outer_iterations.set_postfix_str("stage=query-rollout")
                cached_query = query_rollouts.generate(
                    problem,
                    generation_adaptation.fast_parameters,
                    show_progress=accelerator.is_main_process,
                    progress_description=f"{progress_prefix} query",
                )
                del generation_adaptation
                outer_iterations.set_postfix_str("stage=query-verifier")
                query_verification = verifier(
                    cached_query.texts,
                    problem.ground_truth,
                    device=accelerator.device,
                )
                cached_query = cached_query.with_verification(
                    query_verification.rewards,
                    query_verification.correctness,
                )
                _append_rollout_diagnostics(
                    rollout_log_path,
                    step=step,
                    phase="train_query",
                    problem=problem,
                    group=cached_query,
                    verification=query_verification,
                    max_new_tokens=query_rollouts.max_new_tokens,
                    eos_token_id=query_rollouts.tokenizer.eos_token_id,
                )
                if args.rollout_only:
                    rollout_totals = torch.stack(
                        (
                            support.verifier_rewards.sum(),
                            support.correctness_labels.sum(),
                            torch.tensor(
                                support.group_size,
                                dtype=torch.float32,
                                device=accelerator.device,
                            ),
                            cached_query.verifier_rewards.sum(),
                            cached_query.correctness_labels.sum(),
                            torch.tensor(
                                cached_query.group_size,
                                dtype=torch.float32,
                                device=accelerator.device,
                            ),
                        )
                    )
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
                                    "num_problems": accelerator.num_processes,
                                    "support_responses": int(rollout_totals[2].item()),
                                    "query_responses": int(rollout_totals[5].item()),
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                    accelerator.wait_for_everyone()
                    accelerator.end_training()
                    return
                if args.outer_kl_coefficient > 0:
                    cached_query = query_rollouts.add_reference_logprobs(
                        cached_query,
                        initial_fast,
                        show_progress=accelerator.is_main_process,
                        progress_description=f"{progress_prefix} query reference",
                    )

            outer_iterations.set_postfix_str("stage=meta-forward")
            output = algorithm.outer_loss(
                support,
                cached_query,
                initial_fast,
                show_progress=accelerator.is_main_process,
                progress_prefix=(f"{progress_prefix} outer {outer_iteration + 1}"),
            )
            outer_iterations.set_postfix_str("stage=meta-backward")
            accelerator.backward(output.loss)
            gradient_norm = accelerator.clip_grad_norm_(
                confidence_model.parameters(), args.max_grad_norm
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(
                    "Confidence gradient norm is non-finite; refusing to update "
                    "or save a corrupted checkpoint."
                )
            confidence_optimizer.step()
            outer_iterations.set_postfix_str("stage=metrics")

            metrics = torch.stack(
                (
                    output.loss.detach(),
                    output.meta_grpo.loss.detach(),
                    output.adaptation.confidence_loss.bce.detach(),
                    output.adaptation.confidence_loss.ranking.detach(),
                    output.meta_grpo.clip_fraction.detach(),
                    output.adaptation.inner_losses[-1].clip_fraction.detach(),
                    support.verifier_rewards.mean(),
                    cached_query.verifier_rewards.mean(),
                    support.correctness_labels.mean(),
                    cached_query.correctness_labels.mean(),
                    gradient_norm.detach(),
                )
            )
            metrics = accelerator.reduce(metrics, reduction="mean")
            if accelerator.is_main_process:
                print(
                    json.dumps(
                        {
                            "step": step,
                            "outer_iteration": outer_iteration,
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
                            "confidence_gradient_norm": metrics[10].item(),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            del output

        del support, cached_query, support_verification, query_verification

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
