from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from .bilevel import BilevelGRPO
from .data import load_unique_dapo_problems, rank_shard
from .fidelity import (
    bounded_token_credits,
    finalize_gradient_comparison,
    gradient_comparison,
    parameter_gradients,
)
from .fidelity_preflight import PROBE_SAMPLING, inspect_fidelity_checkpoint
from .functional import chunked_token_logprobs, clone_fast_parameters
from .losses import (
    TOKEN_CREDIT_PARAMETERIZATION_VERSIONS,
    grpo_policy_loss,
    group_advantages,
)
from .models import load_confidence_model, load_policy_with_lora
from .optim import fast_optimizer_step, initial_fast_optimizer_state
from .rollout import VLLMHybridRolloutEngine
from .train import (
    _configs,
    _generation_seed,
    _load_accelerator_state_without_optimizer,
    _problem_batch,
)
from .types import RolloutGroup
from .verifier import DAPOMathVerifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare an exact one-step token-credit meta-gradient with its "
            "Taylor gradient-alignment approximation."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vllm-base-urls", required=True)
    parser.add_argument("--problem-batch-size", type=int, required=True)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=3072)
    parser.add_argument("--token-credit-max", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inner-learning-rate", type=float)
    parser.add_argument("--inner-optimizer", choices=["sgd", "adamw"])
    parser.add_argument("--policy-micro-batch-size", type=int)
    parser.add_argument("--confidence-micro-batch-size", type=int)
    parser.add_argument("--policy-max-tokens-per-micro-batch", type=int)
    parser.add_argument("--confidence-max-tokens-per-micro-batch", type=int)
    parser.add_argument("--token-jvp-response-micro-batch-size", type=int)
    parser.add_argument("--token-jvp-logprob-position-chunk-size", type=int)
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--control-timeout", type=float, default=120.0)
    parser.add_argument("--max-mean-absolute-logprob-delta", type=float, default=0.03)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "problem_batch_size",
        "group_size",
        "max_new_tokens",
        "policy_micro_batch_size",
        "confidence_micro_batch_size",
        "policy_max_tokens_per_micro_batch",
        "confidence_max_tokens_per_micro_batch",
        "token_jvp_response_micro_batch_size",
        "token_jvp_logprob_position_chunk_size",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive.")
    if args.group_size < 2:
        raise ValueError("group_size must be at least two.")
    if args.token_credit_max <= 0:
        raise ValueError("token_credit_max must be positive.")
    if args.inner_learning_rate is not None and args.inner_learning_rate <= 0:
        raise ValueError("inner_learning_rate must be positive.")
    if args.request_timeout <= 0 or args.control_timeout <= 0:
        raise ValueError("vLLM timeouts must be positive.")
    if args.max_mean_absolute_logprob_delta <= 0:
        raise ValueError("max_mean_absolute_logprob_delta must be positive.")


def _verified(group: RolloutGroup, problem, verifier) -> RolloutGroup:
    result = verifier(group.texts, problem.ground_truth, device=group.device)
    return group.with_verification(result.rewards, result.correctness)


def _mean_absolute_logprob_delta(group: RolloutGroup) -> float:
    if group.rollout_logprobs is None:
        raise ValueError("Fidelity rollouts require raw vLLM log-probabilities.")
    selected = group.completion_mask
    return (
        (group.rollout_logprobs[selected] - group.old_logprobs[selected])
        .abs()
        .mean()
        .item()
    )


def _query_loss(
    algorithm: BilevelGRPO,
    query: RolloutGroup,
    fast_parameters,
) -> torch.Tensor:
    if query.verifier_rewards is None:
        raise ValueError("Fidelity query rollouts must be verified.")
    current_logprobs = chunked_token_logprobs(
        algorithm.policy,
        query,
        fast_parameters=fast_parameters,
        micro_batch_size=algorithm.policy_micro_batch_size,
        max_tokens_per_micro_batch=algorithm.policy_max_tokens_per_micro_batch,
        activation_checkpointing=True,
        show_progress=False,
    )
    advantages = group_advantages(
        query.verifier_rewards.detach(),
        algorithm.query_advantage_config,
    )
    return grpo_policy_loss(
        current_logprobs,
        query.old_logprobs,
        query.completion_mask,
        advantages,
        algorithm.query_grpo_config,
        reference_logprobs=query.reference_logprobs,
    ).loss


def _direct_token_adaptation(
    algorithm: BilevelGRPO,
    support: RolloutGroup,
    initial_fast_parameters,
    *,
    token_credit_max: float,
    differentiable: bool,
):
    logits = algorithm._token_confidence_logits_batch(
        (support,),
        differentiable=differentiable,
        show_progress=False,
        progress_description="fidelity token-credit scoring",
    )[0]
    credits = bounded_token_credits(
        logits,
        support.completion_mask,
        maximum=token_credit_max,
        parameterization=algorithm.token_credit_parameterization,
    )
    if not differentiable:
        credits = credits.detach()
    fast_parameters = clone_fast_parameters(initial_fast_parameters)
    optimizer_state = initial_fast_optimizer_state(
        fast_parameters,
        algorithm.inner_config.optimizer,
    )
    gradients, inner_output = algorithm._token_gradient_operator(
        support,
        credits,
        fast_parameters,
    )
    adapted_parameters, optimizer_state = fast_optimizer_step(
        fast_parameters,
        gradients,
        optimizer_state,
        algorithm.inner_config.optimizer,
    )
    return (
        fast_parameters,
        adapted_parameters,
        optimizer_state,
        credits,
        inner_output,
    )


def _distributed_sum(accelerator, values: list[float]) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=accelerator.device)
    return accelerator.reduce(tensor, reduction="sum").cpu().tolist()


def _distributed_mean(accelerator, values: list[float]) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=accelerator.device)
    return accelerator.reduce(tensor, reduction="mean").cpu().tolist()


def _distributed_max(accelerator, values: list[float]) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=accelerator.device)
    return accelerator.reduce(tensor, reduction="max").cpu().tolist()


def main() -> None:
    args = parse_args()
    _validate_args(args)
    checkpoint = args.checkpoint.resolve()
    dataset_parquet = args.dataset_parquet.resolve()
    output_dir = args.output_dir.resolve()
    checkpoint_metadata = inspect_fidelity_checkpoint(
        checkpoint,
        expected_world_size=args.problem_batch_size,
    )
    source_run_dir = checkpoint.parent
    source_config = json.loads(
        (source_run_dir / "run_config.json").read_text(encoding="utf-8")
    )
    trainer_state = json.loads(
        (checkpoint / "trainer_state.json").read_text(encoding="utf-8")
    )
    from accelerate import Accelerator
    from accelerate.utils import set_seed

    accelerator = Accelerator()
    if accelerator.mixed_precision != "bf16":
        raise ValueError("Fidelity probe requires bf16 Accelerate mode.")
    if accelerator.num_processes != int(trainer_state["world_size"]):
        raise ValueError("Probe world size must equal checkpoint world size.")
    if args.problem_batch_size != accelerator.num_processes:
        raise ValueError(
            "The first fidelity probe requires exactly one problem per GPU."
        )
    urls = [item.strip() for item in args.vllm_base_urls.split(",")]
    if len(urls) != accelerator.num_processes or any(not item for item in urls):
        raise ValueError("Expected one non-empty vLLM URL per process rank.")

    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=False)
    accelerator.wait_for_everyone()
    set_seed(args.seed, device_specific=True)

    inverse_parameterizations = {
        version: name
        for name, version in TOKEN_CREDIT_PARAMETERIZATION_VERSIONS.items()
    }
    token_credit_parameterization = inverse_parameterizations.get(
        source_config.get("token_credit_parameterization")
    )
    if token_credit_parameterization is None:
        raise ValueError("Unsupported token-credit checkpoint parameterization.")
    effective = dict(source_config)
    effective.update(
        {
            "inner_iterations": 1,
            "meta_coefficient": 0.0,
            "token_meta_coefficient": 1.0,
            "bce_coefficient": 0.0,
            "ranking_coefficient": 0.0,
        }
    )
    if args.inner_learning_rate is not None:
        effective["inner_learning_rate"] = args.inner_learning_rate
    if args.inner_optimizer is not None:
        effective["inner_optimizer"] = args.inner_optimizer
    inner_config, meta_config, query_advantage, query_grpo = _configs(
        SimpleNamespace(**effective)
    )

    model_kwargs = {"attn_implementation": "sdpa"}
    confidence_model = load_confidence_model(
        str(args.model),
        dtype=source_config["dtype"],
        trust_remote_code=bool(source_config["trust_remote_code"]),
        model_kwargs=model_kwargs,
        enable_sequence_head=False,
        enable_token_head=True,
    )
    confidence_model = accelerator.prepare(confidence_model)
    _load_accelerator_state_without_optimizer(accelerator, checkpoint)
    # Qwen activates gradient checkpointing only in training mode. Its dropout
    # is zero here, so exact and Taylor forwards remain deterministic.
    confidence_model.train()

    target_modules = tuple(
        item.strip() for item in source_config["lora_target_modules"].split(",")
    )
    policy_bundle = load_policy_with_lora(
        str(args.model),
        lora_rank=int(source_config["lora_rank"]),
        lora_alpha=int(source_config["lora_alpha"]),
        target_modules=target_modules,
        dtype=source_config["dtype"],
        force_sdpa_math=True,
        trust_remote_code=bool(source_config["trust_remote_code"]),
        model_kwargs=model_kwargs,
    )
    policy = policy_bundle.model.to(accelerator.device)

    def configured(name: str, fallback: int | None = None):
        explicit = getattr(args, name)
        if explicit is not None:
            return explicit
        value = source_config.get(name, fallback)
        if value is None:
            raise ValueError(f"Missing required configuration field: {name}.")
        return int(value)

    algorithm = BilevelGRPO(
        policy=policy,
        confidence_model=confidence_model,
        inner_config=inner_config,
        meta_config=meta_config,
        query_advantage_config=query_advantage,
        query_grpo_config=query_grpo,
        policy_micro_batch_size=configured("policy_micro_batch_size"),
        first_order_vjp_forward_batch_size=int(
            source_config.get("first_order_vjp_forward_batch_size", 1)
        ),
        confidence_micro_batch_size=configured("confidence_micro_batch_size"),
        policy_max_tokens_per_micro_batch=configured(
            "policy_max_tokens_per_micro_batch", 4096
        ),
        confidence_max_tokens_per_micro_batch=configured(
            "confidence_max_tokens_per_micro_batch", 4096
        ),
        token_jvp_response_micro_batch_size=configured(
            "token_jvp_response_micro_batch_size", 4
        ),
        token_jvp_logprob_position_chunk_size=configured(
            "token_jvp_logprob_position_chunk_size", 256
        ),
        token_credit_max=args.token_credit_max,
        token_credit_parameterization=token_credit_parameterization,
    )
    initial_fast = {
        name: value.to(accelerator.device)
        for name, value in policy_bundle.initial_fast_parameters.items()
    }
    rollout_engine = VLLMHybridRolloutEngine(
        policy,
        policy_bundle.tokenizer,
        group_size=args.group_size,
        max_new_tokens=args.max_new_tokens,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        generation_micro_batch_size=1,
        logprob_micro_batch_size=algorithm.policy_micro_batch_size,
        logprob_max_tokens_per_micro_batch=(
            algorithm.policy_max_tokens_per_micro_batch
        ),
        base_url=urls[accelerator.process_index],
        adapter_root=(
            Path("/dev/shm")
            / f"meta-rlvr-fidelity-{output_dir.name}"
            / f"rank-{accelerator.process_index}"
        ),
        request_timeout=args.request_timeout,
        control_timeout=args.control_timeout,
    )
    verifier = DAPOMathVerifier(strict_box_verify=True)
    problems = rank_shard(
        load_unique_dapo_problems(dataset_parquet),
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    problem = _problem_batch(problems, step=0, batch_size=1, seed=args.seed)[0]
    support_seed = _generation_seed(
        base_seed=args.seed,
        step=0,
        phase="fidelity_support",
        rank=accelerator.process_index,
        problem_uid=problem.uid,
    )
    query_seed = _generation_seed(
        base_seed=args.seed,
        step=0,
        phase="fidelity_query",
        rank=accelerator.process_index,
        problem_uid=problem.uid,
    )

    if accelerator.is_main_process:
        print(
            json.dumps(
                {
                    "event": "meta_gradient_fidelity_started",
                    "checkpoint": str(checkpoint),
                    "checkpoint_step": int(trainer_state["completed_steps"]),
                    "problem_batch_size": args.problem_batch_size,
                    "support_group_size": args.group_size,
                    "query_group_size": args.group_size,
                    "max_new_tokens": args.max_new_tokens,
                    "token_credit_max": args.token_credit_max,
                    "inner_optimizer": effective["inner_optimizer"],
                    "inner_learning_rate": effective["inner_learning_rate"],
                    "source_checkpoint_sampling": checkpoint_metadata[
                        "source_sampling"
                    ],
                    "probe_sampling": PROBE_SAMPLING,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    started = time.perf_counter()
    support, query = rollout_engine.generate_batch(
        [problem, problem],
        [initial_fast, initial_fast],
        show_progress=accelerator.is_main_process,
        progress_description="fidelity independent support/query",
        seeds=[support_seed, query_seed],
    )
    rollout_seconds = time.perf_counter() - started
    support = _verified(support, problem, verifier)
    query = _verified(query, problem, verifier)
    parity = max(
        _mean_absolute_logprob_delta(support),
        _mean_absolute_logprob_delta(query),
    )
    parity_tensor = torch.tensor(parity, dtype=torch.float64, device=accelerator.device)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(parity_tensor, op=torch.distributed.ReduceOp.MAX)
    global_parity = parity_tensor.item()
    if global_parity > args.max_mean_absolute_logprob_delta:
        raise RuntimeError(
            f"vLLM/PyTorch mean log-probability delta {global_parity:.6f} exceeds "
            f"{args.max_mean_absolute_logprob_delta:.6f}."
        )

    accelerator.print("fidelity stage: exact one-step unrolled meta-gradient")
    torch.cuda.reset_peak_memory_stats(accelerator.device)
    confidence_model.zero_grad(set_to_none=True)
    exact_started = time.perf_counter()
    (
        exact_base_fast,
        exact_adapted_fast,
        _,
        exact_credits,
        exact_inner,
    ) = _direct_token_adaptation(
        algorithm,
        support,
        initial_fast,
        token_credit_max=args.token_credit_max,
        differentiable=True,
    )
    exact_loss = _query_loss(algorithm, query, exact_adapted_fast)
    accelerator.backward(exact_loss)
    exact_gradients = parameter_gradients(confidence_model)
    exact_seconds = time.perf_counter() - exact_started
    exact_peak = torch.cuda.max_memory_allocated(accelerator.device) / 1024**3
    exact_loss_value = exact_loss.detach().item()
    exact_credit_values = exact_credits.detach()[support.completion_mask]
    exact_credit_summary = [
        exact_credit_values.sum().item(),
        exact_credit_values.square().sum().item(),
        float(exact_credit_values.numel()),
        exact_credit_values.abs().max().item(),
        (exact_credit_values.abs() >= 0.95 * args.token_credit_max).sum().item(),
    ]
    inner_loss_value = exact_inner.loss.detach().item()
    del exact_loss, exact_base_fast, exact_adapted_fast, exact_credits, exact_inner
    confidence_model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()

    accelerator.print("fidelity stage: Taylor gradient-alignment meta-gradient")
    torch.cuda.reset_peak_memory_stats(accelerator.device)
    alignment_started = time.perf_counter()
    alignment_base_fast = clone_fast_parameters(initial_fast)
    base_query_loss = _query_loss(algorithm, query, alignment_base_fast)
    query_gradient_values = torch.autograd.grad(
        base_query_loss,
        tuple(alignment_base_fast.values()),
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )
    (
        support_base_fast,
        alignment_adapted_fast,
        _,
        alignment_credits,
        _,
    ) = _direct_token_adaptation(
        algorithm,
        support,
        initial_fast,
        token_credit_max=args.token_credit_max,
        differentiable=True,
    )
    if alignment_base_fast.keys() != alignment_adapted_fast.keys():
        raise RuntimeError("Fast parameter names changed during adaptation.")
    alignment_loss = torch.stack(
        [
            (
                query_gradient.detach()
                * (alignment_adapted_fast[name] - support_base_fast[name])
            ).sum()
            for name, query_gradient in zip(
                alignment_base_fast,
                query_gradient_values,
                strict=True,
            )
        ]
    ).sum()
    accelerator.backward(alignment_loss)
    approximation_gradients = parameter_gradients(confidence_model)
    alignment_seconds = time.perf_counter() - alignment_started
    alignment_peak = torch.cuda.max_memory_allocated(accelerator.device) / 1024**3
    base_loss_value = base_query_loss.detach().item()
    predicted_loss_delta = alignment_loss.detach().item()
    del (
        base_query_loss,
        alignment_loss,
        alignment_base_fast,
        support_base_fast,
        alignment_adapted_fast,
        alignment_credits,
        query_gradient_values,
    )

    local_comparison = gradient_comparison(
        exact_gradients,
        approximation_gradients,
    )
    comparison_values = _distributed_sum(
        accelerator,
        [
            local_comparison[name]
            for name in (
                "dot",
                "exact_square",
                "approximation_square",
                "difference_square",
                "same_sign",
                "active",
                "elements",
            )
        ],
    )
    comparison = finalize_gradient_comparison(
        dict(
            zip(
                (
                    "dot",
                    "exact_square",
                    "approximation_square",
                    "difference_square",
                    "same_sign",
                    "active",
                    "elements",
                ),
                comparison_values,
                strict=True,
            )
        )
    )
    del exact_gradients, approximation_gradients
    confidence_model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()

    accelerator.print("fidelity stage: fresh adapted-query rollout")
    (
        _,
        generation_adapted_fast,
        _,
        _,
        _,
    ) = _direct_token_adaptation(
        algorithm,
        support,
        initial_fast,
        token_credit_max=args.token_credit_max,
        differentiable=False,
    )
    generation_adapted_fast = {
        name: value.detach() for name, value in generation_adapted_fast.items()
    }
    adapted_query = rollout_engine.generate(
        problem,
        generation_adapted_fast,
        show_progress=accelerator.is_main_process,
        progress_description="fidelity adapted query",
        seed=query_seed,
    )
    adapted_query = _verified(adapted_query, problem, verifier)

    means = _distributed_mean(
        accelerator,
        [
            exact_loss_value,
            base_loss_value,
            exact_loss_value - base_loss_value,
            predicted_loss_delta,
            inner_loss_value,
            rollout_seconds,
            exact_seconds,
            alignment_seconds,
            exact_peak,
            alignment_peak,
        ],
    )
    maxima = _distributed_max(
        accelerator,
        [rollout_seconds, exact_seconds, alignment_seconds, exact_peak, alignment_peak],
    )
    counts = _distributed_sum(
        accelerator,
        [
            support.correctness_labels.sum().item(),
            query.correctness_labels.sum().item(),
            adapted_query.correctness_labels.sum().item(),
            float(args.group_size),
        ],
    )
    credit_totals = _distributed_sum(
        accelerator,
        [
            exact_credit_summary[0],
            exact_credit_summary[1],
            exact_credit_summary[2],
            exact_credit_summary[4],
        ],
    )
    credit_maximum = torch.tensor(
        exact_credit_summary[3], dtype=torch.float64, device=accelerator.device
    )
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(credit_maximum, op=torch.distributed.ReduceOp.MAX)
    credit_mean = credit_totals[0] / credit_totals[2]
    credit_variance = max(0.0, credit_totals[1] / credit_totals[2] - credit_mean**2)

    problem_record = {
        "problem_uid": problem.uid,
        "support_accuracy": support.correctness_labels.mean().item(),
        "base_query_accuracy": query.correctness_labels.mean().item(),
        "adapted_query_accuracy": adapted_query.correctness_labels.mean().item(),
        "base_query_loss": base_loss_value,
        "adapted_fixed_query_loss": exact_loss_value,
        "actual_fixed_query_loss_delta": exact_loss_value - base_loss_value,
        "taylor_predicted_loss_delta": predicted_loss_delta,
        "support_seed": support_seed,
        "query_seed": query_seed,
    }
    problem_path = output_dir / f"problem-rank-{accelerator.process_index}.json"
    problem_path.write_text(
        json.dumps(problem_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        actual_delta = means[2]
        predicted_delta = means[3]
        result = {
            "event": "meta_gradient_fidelity_completed",
            "checkpoint": str(checkpoint),
            "checkpoint_step": int(trainer_state["completed_steps"]),
            "dataset_parquet": str(dataset_parquet),
            "problem_batch_size": args.problem_batch_size,
            "support_group_size": args.group_size,
            "query_group_size": args.group_size,
            "source_checkpoint_sampling": checkpoint_metadata["source_sampling"],
            "sampling": PROBE_SAMPLING,
            "token_credit": {
                "parameterization": token_credit_parameterization,
                "maximum": args.token_credit_max,
                "cross_trajectory_normalization": False,
                "mean": credit_mean,
                "std": math.sqrt(credit_variance),
                "max_absolute": credit_maximum.item(),
                "saturation_fraction": credit_totals[3] / credit_totals[2],
                "tokens": int(credit_totals[2]),
            },
            "inner_update": {
                "iterations": 1,
                "optimizer": effective["inner_optimizer"],
                "learning_rate": float(effective["inner_learning_rate"]),
                "mean_inner_loss": means[4],
            },
            "gradient": comparison,
            "fixed_query_taylor": {
                "base_loss": means[1],
                "adapted_loss": means[0],
                "actual_loss_delta": actual_delta,
                "predicted_loss_delta": predicted_delta,
                "delta_sign_match": (
                    actual_delta == 0
                    or predicted_delta == 0
                    or (actual_delta > 0) == (predicted_delta > 0)
                ),
                "absolute_prediction_error": abs(actual_delta - predicted_delta),
            },
            "fresh_generation": {
                "common_random_numbers": True,
                "support_accuracy": counts[0] / counts[3],
                "base_query_accuracy": counts[1] / counts[3],
                "adapted_query_accuracy": counts[2] / counts[3],
                "adapted_minus_base_accuracy": (counts[2] - counts[1]) / counts[3],
                "responses_per_phase": int(counts[3]),
            },
            "logprob_parity": {
                "maximum_rank_mean_absolute_delta": global_parity,
                "threshold": args.max_mean_absolute_logprob_delta,
            },
            "timing_seconds_per_rank_mean": {
                "base_support_query_rollout": means[5],
                "exact_gradient": means[6],
                "taylor_gradient": means[7],
            },
            "timing_seconds_per_rank_max": {
                "base_support_query_rollout": maxima[0],
                "exact_gradient": maxima[1],
                "taylor_gradient": maxima[2],
            },
            "peak_allocated_gib_per_rank_mean": {
                "exact_gradient": means[8],
                "taylor_gradient": means[9],
            },
            "peak_allocated_gib_per_rank_max": {
                "exact_gradient": maxima[3],
                "taylor_gradient": maxima[4],
            },
            "seconds": time.perf_counter() - started,
        }
        (output_dir / "fidelity_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, sort_keys=True), flush=True)
    accelerator.wait_for_everyone()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
