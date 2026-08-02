from __future__ import annotations

import argparse
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
from .rollout import TransformersRolloutEngine
from .verifier import DAPOMathVerifier


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
    parser.add_argument("--strict-box-verify", action="store_true")

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
    if (
        args.validation_max_problems is not None
        and args.validation_max_problems <= 0
    ):
        raise ValueError("validation_max_problems must be positive.")
    if args.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive.")
    if args.confidence_learning_rate <= 0:
        raise ValueError("confidence_learning_rate must be positive.")
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
        raise ValueError("lora_target_modules must be a non-empty comma-separated list.")


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
                round_index * accelerator.num_processes
                + accelerator.process_index
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
            base_rewards = verifier(
                support.texts,
                problem.ground_truth,
                device=accelerator.device,
            )
            adapted_rewards = verifier(
                query.texts,
                problem.ground_truth,
                device=accelerator.device,
            )
            if valid:
                local_metrics += torch.tensor(
                    (
                        base_rewards.sum().item(),
                        float(base_rewards.numel()),
                        adapted_rewards.sum().item(),
                        float(adapted_rewards.numel()),
                        float(torch.any(base_rewards == 1).item()),
                        float(torch.any(adapted_rewards == 1).item()),
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
                    "validation/base_pass_at_group": (
                        totals[4] / len(problems)
                    ).item(),
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
    inner_config, meta_config, query_advantage_config, query_grpo_config = _configs(args)

    from accelerate import Accelerator
    from accelerate.utils import set_seed

    accelerator = Accelerator()
    if accelerator.mixed_precision != "bf16":
        raise ValueError(
            "The HPC launch must configure Accelerate mixed_precision='bf16'."
        )
    set_seed(args.seed, device_specific=True)
    _write_run_config(args, args.output_dir, accelerator)

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
        zero_init_output=True,
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
        match = re.fullmatch(
            r"checkpoint-(\d+)", args.resume_from_checkpoint.name
        )
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
    support_rollouts = TransformersRolloutEngine(
        policy,
        policy_bundle.tokenizer,
        group_size=args.support_group_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        generation_micro_batch_size=args.generation_micro_batch_size,
        logprob_micro_batch_size=args.policy_micro_batch_size,
    )
    query_rollouts = TransformersRolloutEngine(
        policy,
        policy_bundle.tokenizer,
        group_size=args.query_group_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        generation_micro_batch_size=args.generation_micro_batch_size,
        logprob_micro_batch_size=args.policy_micro_batch_size,
    )
    verifier = DAPOMathVerifier(strict_box_verify=args.strict_box_verify)

    all_problems = load_unique_dapo_problems(args.train_parquet)
    validation_problems = load_semantically_unique_dapo_problems(
        args.validation_parquet
    )
    if args.validation_max_problems is not None:
        validation_problems = validation_problems[
            : args.validation_max_problems
        ]
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
        step_indices.set_postfix_str(
            f"problem={problem.uid} stage=support-rollout"
        )
        progress_prefix = f"train step {step + 1} problem {problem.uid}"
        support = support_rollouts.generate(
            problem,
            initial_fast,
            show_progress=accelerator.is_main_process,
            progress_description=f"{progress_prefix} support",
        )
        support_rewards = verifier(
            support.texts,
            problem.ground_truth,
            device=accelerator.device,
        )
        support = support.with_rewards(support_rewards)
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
                generation_adaptation = algorithm.adapt_task(
                    support,
                    initial_fast,
                    differentiable=False,
                    supervise_confidence=False,
                    show_progress=accelerator.is_main_process,
                    progress_prefix=f"{progress_prefix} generation adaptation",
                )
                cached_query = query_rollouts.generate(
                    problem,
                    generation_adaptation.fast_parameters,
                    show_progress=accelerator.is_main_process,
                    progress_description=f"{progress_prefix} query",
                )
                del generation_adaptation
                query_rewards = verifier(
                    cached_query.texts,
                    problem.ground_truth,
                    device=accelerator.device,
                )
                cached_query = cached_query.with_rewards(query_rewards)
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
                progress_prefix=(
                    f"{progress_prefix} outer {outer_iteration + 1}"
                ),
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
                            "support_accuracy": metrics[6].item(),
                            "query_accuracy": metrics[7].item(),
                            "confidence_gradient_norm": metrics[8].item(),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            del output

        if (step + 1) % args.save_steps == 0:
            accelerator.wait_for_everyone()
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
        )

    accelerator.wait_for_everyone()
    accelerator.save_state(args.output_dir / "final")


if __name__ == "__main__":
    main()
