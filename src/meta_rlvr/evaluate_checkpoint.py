from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from tqdm.auto import tqdm

from .bilevel import BilevelGRPO
from .data import MathProblem, load_semantically_unique_dapo_problems
from .losses import (
    DEFAULT_TOKEN_CREDIT_PARAMETERIZATION,
    TOKEN_CREDIT_CROSS_TRAJECTORY_NORMALIZATION,
    TOKEN_CREDIT_PARAMETERIZATION_VERSIONS,
)
from .models import load_confidence_model, load_policy_with_lora
from .optim import FastOptimizerState
from .rollout import VLLMHybridRolloutEngine
from .train import (
    _configs,
    _generation_seed,
    _load_accelerator_state_without_optimizer,
)
from .types import RolloutGroup
from .verifier import DAPOMathVerifier, VerificationBatch


DEFAULT_EVALUATION_PARQUET = Path("/data/user/zhongal/data/reschedule/aime24.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Evaluate a Meta-RLVR confidence checkpoint on any DAPO parquet.")
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset-parquet",
        type=Path,
        default=DEFAULT_EVALUATION_PARQUET,
        help=f"Evaluation parquet (default: {DEFAULT_EVALUATION_PARQUET}).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vllm-base-urls", required=True)
    parser.add_argument("--support-group-size", type=int, default=16)
    parser.add_argument("--base-query-group-size", type=int, default=32)
    parser.add_argument("--adapted-query-group-size", type=int, default=32)
    parser.add_argument("--max-problems", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inner-iterations", type=int)
    parser.add_argument("--adaptation-rounds", type=int, default=1)
    parser.add_argument("--inner-learning-rate", type=float)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--local-rollout-batch-size", type=int, default=8)
    parser.add_argument("--local-adaptation-batch-size", type=int, default=2)
    parser.add_argument("--adaptation-temperature", type=float, default=1.0)
    parser.add_argument("--adaptation-top-p", type=float, default=1.0)
    parser.add_argument("--adaptation-top-k", type=int, default=0)
    parser.add_argument("--query-temperature", type=float, default=1.0)
    parser.add_argument("--query-top-p", type=float, default=0.7)
    parser.add_argument("--query-top-k", type=int, default=0)
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--control-timeout", type=float, default=120.0)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "support_group_size": args.support_group_size,
        "base_query_group_size": args.base_query_group_size,
        "adapted_query_group_size": args.adapted_query_group_size,
        "local_rollout_batch_size": args.local_rollout_batch_size,
        "local_adaptation_batch_size": args.local_adaptation_batch_size,
        "adaptation_rounds": args.adaptation_rounds,
        "request_timeout": args.request_timeout,
        "control_timeout": args.control_timeout,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    if (
        min(
            args.support_group_size,
            args.base_query_group_size,
            args.adapted_query_group_size,
        )
        < 2
    ):
        raise ValueError("Every rollout group size must be at least two.")
    if args.max_problems is not None and args.max_problems <= 0:
        raise ValueError("max_problems must be positive.")
    if args.inner_iterations is not None and args.inner_iterations <= 0:
        raise ValueError("inner_iterations must be positive.")
    if args.inner_learning_rate is not None and args.inner_learning_rate <= 0:
        raise ValueError("inner_learning_rate must be positive.")
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")
    for name in ("adaptation_temperature", "query_temperature"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive.")
    for name in ("adaptation_top_p", "query_top_p"):
        value = getattr(args, name)
        if not 0 < value <= 1:
            raise ValueError(f"{name} must be in (0, 1].")
    for name in ("adaptation_top_k", "query_top_k"):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be non-negative.")


def _adaptation_mode(source_config: dict[str, object]) -> str:
    sequence_enabled = float(source_config.get("meta_coefficient", 0.0)) > 0
    token_enabled = float(source_config.get("token_meta_coefficient", 0.0)) > 0
    if sequence_enabled == token_enabled:
        raise ValueError(
            "Checkpoint evaluation requires exactly one enabled meta branch."
        )
    return "sequence" if sequence_enabled else "token"


def _response_confidences(
    adaptation,
    support: RolloutGroup,
    mode: str,
) -> list[float]:
    if mode == "sequence":
        return adaptation.confidence_probabilities.detach().cpu().tolist()
    credits = adaptation.token_credits.detach()
    mask = support.completion_mask.to(credits.dtype)
    return ((credits * mask).sum(dim=1) / mask.sum(dim=1)).cpu().tolist()


def _chunks(items: list, size: int) -> list[list]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(
    group: RolloutGroup,
    problem: MathProblem,
    verifier: DAPOMathVerifier,
) -> tuple[RolloutGroup, VerificationBatch]:
    verification = verifier(
        group.texts,
        problem.ground_truth,
        device=torch.device("cpu"),
    )
    return (
        group.with_verification(
            verification.rewards,
            verification.correctness,
        ),
        verification,
    )


def _write_group(
    stream,
    *,
    checkpoint_step: int,
    phase: str,
    problem: MathProblem,
    group: RolloutGroup,
    verification: VerificationBatch,
    seed: int,
    max_new_tokens: int,
    eos_token_id: int,
    confidence_probabilities: list[float] | None = None,
    adaptation_mode: str | None = None,
    adaptation_round: int | None = None,
) -> None:
    completion_lengths = group.completion_mask.sum(dim=1).tolist()
    for response_index, response in enumerate(group.texts):
        selected_ids = group.input_ids[response_index, 1:][
            group.completion_mask[response_index]
        ]
        ended_with_eos = int(selected_ids[-1].item()) == eos_token_id
        record = {
            "checkpoint_step": checkpoint_step,
            "phase": phase,
            "problem_uid": problem.uid,
            "data_source": problem.data_source,
            "ground_truth": problem.ground_truth,
            "response_index": response_index,
            "seed": seed,
            "completion_tokens": int(completion_lengths[response_index]),
            "ended_with_eos": ended_with_eos,
            "hit_max_new_tokens": (
                int(completion_lengths[response_index]) == max_new_tokens
                and not ended_with_eos
            ),
            "prediction": verification.predictions[response_index],
            "reward": verification.rewards[response_index].item(),
            "correct": verification.correctness[response_index].item(),
            "response": response,
        }
        if confidence_probabilities is not None:
            if adaptation_mode not in ("sequence", "token"):
                raise ValueError("Adaptation mode is required for support scores.")
            score_name = (
                "confidence_probability"
                if adaptation_mode == "sequence"
                else "mean_token_credit"
            )
            record[score_name] = confidence_probabilities[response_index]
        if adaptation_round is not None:
            record["adaptation_round"] = adaptation_round
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    stream.flush()


def _correctness(group: RolloutGroup) -> list[float]:
    if group.correctness_labels is None:
        raise ValueError("Verified rollout group is required.")
    return group.correctness_labels.tolist()


def _optimizer_state_to(
    state: FastOptimizerState,
    device: torch.device | str,
) -> FastOptimizerState:
    return FastOptimizerState(
        step=state.step,
        first_moment={
            name: value.to(device) for name, value in state.first_moment.items()
        },
        second_moment={
            name: value.to(device) for name, value in state.second_moment.items()
        },
    )


def _support_round_summaries(
    totals: torch.Tensor,
    *,
    adaptation_mode: str,
) -> list[dict[str, object]]:
    summaries = []
    for round_index, values_tensor in enumerate(totals, start=1):
        values = values_tensor.tolist()
        problems = int(values[3])
        score_prefix = "confidence" if adaptation_mode == "sequence" else "token_credit"
        summaries.append(
            {
                "round": round_index,
                "accuracy": values[0] / values[1],
                "responses": int(values[1]),
                "correct": int(values[0]),
                "pass_at_group": values[2] / problems,
                "passed_problems": int(values[2]),
                "num_unique_problems": problems,
                f"{score_prefix}_mean": values[4] / values[1],
                f"{score_prefix}_correct_mean": (
                    None if values[6] == 0 else values[5] / values[6]
                ),
                f"{score_prefix}_incorrect_mean": (
                    None if values[8] == 0 else values[7] / values[8]
                ),
            }
        )
    return summaries


def _summary_from_totals(
    totals: torch.Tensor,
    *,
    support_group_size: int,
    base_query_group_size: int,
    adapted_query_group_size: int,
    adaptation_mode: str,
) -> dict[str, object]:
    values = totals.tolist()
    problems = int(values[11])

    def group(correct_index: int, count_index: int, pass_index: int, size: int):
        return {
            "accuracy": values[correct_index] / values[count_index],
            "pass_at_group": values[pass_index] / problems,
            "group_size": size,
            "responses": int(values[count_index]),
            "correct": int(values[correct_index]),
            "passed_problems": int(values[pass_index]),
        }

    support = group(0, 1, 6, support_group_size)
    base_query = group(2, 3, 7, base_query_group_size)
    adapted_query = group(4, 5, 8, adapted_query_group_size)
    base_total_responses = int(values[1] + values[3])
    meta_total_responses = int(values[1] + values[5])
    base_total_correct = int(values[0] + values[2])
    meta_total_correct = int(values[0] + values[4])
    result = {
        "num_unique_problems": problems,
        "support": support,
        "base_query": base_query,
        "adapted_query": adapted_query,
        "query_accuracy_delta": (adapted_query["accuracy"] - base_query["accuracy"]),
        "query_pass_delta": (
            adapted_query["pass_at_group"] - base_query["pass_at_group"]
        ),
        "base_total": {
            "group_size": support_group_size + base_query_group_size,
            "responses": base_total_responses,
            "correct": base_total_correct,
            "accuracy": base_total_correct / base_total_responses,
            "passed_problems": int(values[9]),
            "pass_at_group": values[9] / problems,
        },
        "meta_total": {
            "group_size": support_group_size + adapted_query_group_size,
            "responses": meta_total_responses,
            "correct": meta_total_correct,
            "accuracy": meta_total_correct / meta_total_responses,
            "passed_problems": int(values[10]),
            "pass_at_group": values[10] / problems,
        },
        "adapted_query_better_problems": int(values[12]),
        "adapted_query_worse_problems": int(values[13]),
        "adapted_query_equal_problems": int(values[14]),
        "confidence": {
            "mean": values[15] / values[16],
            "correct_mean": (None if values[18] == 0 else values[17] / values[18]),
            "incorrect_mean": (None if values[20] == 0 else values[19] / values[20]),
            "brier": values[21] / values[16],
            "bce": values[22] / values[16],
            "responses": int(values[16]),
        },
    }
    if adaptation_mode == "token":
        token_credit = result.pop("confidence")
        token_credit.pop("brier")
        token_credit.pop("bce")
        result["token_credit"] = token_credit
    return result


def main() -> None:
    args = parse_args()
    _validate_args(args)
    checkpoint = args.checkpoint.resolve()
    dataset_path = args.dataset_parquet.resolve()
    output_dir = args.output_dir.resolve()
    source_run_dir = checkpoint.parent
    source_config = json.loads(
        (source_run_dir / "run_config.json").read_text(encoding="utf-8")
    )
    adaptation_mode = _adaptation_mode(source_config)
    if (
        adaptation_mode == "token"
        and source_config.get("attn_implementation") != "sdpa"
    ):
        raise ValueError(
            "Corrected token-confidence evaluation requires a checkpoint trained "
            "with attn_implementation='sdpa'. Legacy eager token checkpoints used "
            "a rollout/policy-logprob mismatch and must not be compared as corrected "
            "runs."
        )
    if adaptation_mode == "token" and "token_credit_max" not in source_config:
        raise ValueError(
            "This checkpoint predates the independent tanh token-credit definition. "
            "Evaluate it with the matching legacy code; do not silently reinterpret "
            "its token head."
        )
    token_parameterization = None
    if adaptation_mode == "token":
        version = source_config.get("token_credit_parameterization")
        inverse_versions = {
            saved_version: name
            for name, saved_version in TOKEN_CREDIT_PARAMETERIZATION_VERSIONS.items()
        }
        token_parameterization = inverse_versions.get(version)
    if adaptation_mode == "token" and (
        token_parameterization is None
        or source_config.get("token_credit_cross_trajectory_normalization")
        is not TOKEN_CREDIT_CROSS_TRAJECTORY_NORMALIZATION
    ):
        raise ValueError("Token-credit checkpoint metadata does not match this code.")
    if adaptation_mode == "token" and args.adaptation_rounds != 1:
        raise ValueError("Token checkpoint evaluation currently requires one round.")
    trainer_state = json.loads(
        (checkpoint / "trainer_state.json").read_text(encoding="utf-8")
    )
    checkpoint_step = int(trainer_state["completed_steps"])

    from accelerate import Accelerator
    from accelerate.utils import set_seed

    accelerator = Accelerator()
    if accelerator.mixed_precision != "bf16":
        raise ValueError("Checkpoint evaluation requires bf16 Accelerate mode.")
    if int(trainer_state["world_size"]) != accelerator.num_processes:
        raise ValueError(
            "Evaluation must use the checkpoint world size: "
            f"{trainer_state['world_size']}."
        )
    urls = [url.strip() for url in args.vllm_base_urls.split(",")]
    if len(urls) != accelerator.num_processes:
        raise ValueError("Expected one vLLM URL per distributed rank.")

    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=False)
    accelerator.wait_for_everyone()

    model_seed = int(source_config["seed"])
    set_seed(model_seed, device_specific=True)
    model_kwargs = {
        "attn_implementation": source_config.get("attn_implementation", "sdpa")
    }
    confidence_model = load_confidence_model(
        str(args.model),
        dtype=source_config["dtype"],
        trust_remote_code=bool(source_config["trust_remote_code"]),
        model_kwargs=model_kwargs,
        enable_sequence_head=adaptation_mode == "sequence",
        enable_token_head=adaptation_mode == "token",
    )
    confidence_model = accelerator.prepare(confidence_model)
    target_modules = tuple(
        item.strip() for item in source_config["lora_target_modules"].split(",")
    )
    policy_bundle = load_policy_with_lora(
        str(args.model),
        lora_rank=int(source_config["lora_rank"]),
        lora_alpha=int(source_config["lora_alpha"]),
        target_modules=target_modules,
        dtype=source_config["dtype"],
        force_sdpa_math=adaptation_mode == "token",
        trust_remote_code=bool(source_config["trust_remote_code"]),
        model_kwargs=model_kwargs,
    )
    policy = policy_bundle.model.to(accelerator.device)
    _load_accelerator_state_without_optimizer(accelerator, checkpoint)
    confidence_model.eval()
    accelerator.wait_for_everyone()

    effective_config = dict(source_config)
    effective_config.setdefault("token_meta_coefficient", 0.0)
    if args.inner_iterations is not None:
        effective_config["inner_iterations"] = args.inner_iterations
    if args.inner_learning_rate is not None:
        effective_config["inner_learning_rate"] = args.inner_learning_rate
    inner_config, meta_config, query_advantage, query_grpo = _configs(
        SimpleNamespace(**effective_config)
    )
    algorithm = BilevelGRPO(
        policy=policy,
        confidence_model=confidence_model,
        inner_config=inner_config,
        meta_config=meta_config,
        query_advantage_config=query_advantage,
        query_grpo_config=query_grpo,
        policy_micro_batch_size=int(source_config["policy_micro_batch_size"]),
        first_order_vjp_forward_batch_size=int(
            source_config["first_order_vjp_forward_batch_size"]
        ),
        confidence_micro_batch_size=int(source_config["confidence_micro_batch_size"]),
        policy_max_tokens_per_micro_batch=source_config.get(
            "policy_max_tokens_per_micro_batch"
        ),
        confidence_max_tokens_per_micro_batch=source_config.get(
            "confidence_max_tokens_per_micro_batch"
        ),
        token_jvp_response_micro_batch_size=int(
            source_config.get("token_jvp_response_micro_batch_size", 4)
        ),
        token_jvp_logprob_position_chunk_size=int(
            source_config.get("token_jvp_logprob_position_chunk_size", 256)
        ),
        token_credit_max=float(source_config.get("token_credit_max", 1.0)),
        token_credit_parameterization=(
            token_parameterization or DEFAULT_TOKEN_CREDIT_PARAMETERIZATION
        ),
    )
    max_new_tokens = (
        int(source_config["max_new_tokens"])
        if args.max_new_tokens is None
        else args.max_new_tokens
    )

    def rollout_engine(
        group_size: int,
        *,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> VLLMHybridRolloutEngine:
        return VLLMHybridRolloutEngine(
            policy,
            policy_bundle.tokenizer,
            group_size=group_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            generation_micro_batch_size=int(
                source_config["generation_micro_batch_size"]
            ),
            logprob_micro_batch_size=int(source_config["policy_micro_batch_size"]),
            logprob_max_tokens_per_micro_batch=source_config.get(
                "policy_max_tokens_per_micro_batch"
            ),
            base_url=urls[accelerator.process_index],
            adapter_root=(
                Path("/dev/shm")
                / f"meta-rlvr-eval-{output_dir.name}"
                / f"rank-{accelerator.process_index}"
            ),
            request_timeout=args.request_timeout,
            control_timeout=args.control_timeout,
        )

    adaptation_sampling = {
        "temperature": args.adaptation_temperature,
        "top_p": args.adaptation_top_p,
        "top_k": args.adaptation_top_k,
    }
    query_sampling = {
        "temperature": args.query_temperature,
        "top_p": args.query_top_p,
        "top_k": args.query_top_k,
    }
    support_engine = rollout_engine(
        args.support_group_size,
        **adaptation_sampling,
    )
    base_query_engine = rollout_engine(
        args.base_query_group_size,
        **query_sampling,
    )
    adapted_query_engine = rollout_engine(
        args.adapted_query_group_size,
        **query_sampling,
    )
    verifier = DAPOMathVerifier(strict_box_verify=True)

    problems = load_semantically_unique_dapo_problems(dataset_path)
    if args.max_problems is not None:
        problems = problems[: args.max_problems]
    rounds = math.ceil(len(problems) / accelerator.num_processes)
    local_problems = []
    valid_flags = []
    for round_index in range(rounds):
        problem_index = (
            round_index * accelerator.num_processes + accelerator.process_index
        )
        valid = problem_index < len(problems)
        local_problems.append(problems[problem_index] if valid else problems[0])
        valid_flags.append(valid)
    problem_batches = _chunks(local_problems, args.local_rollout_batch_size)
    initial_fast = {
        name: value.to(accelerator.device)
        for name, value in policy_bundle.initial_fast_parameters.items()
    }

    evaluation_config = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_step,
        "checkpoint_world_size": int(trainer_state["world_size"]),
        "adaptation_mode": adaptation_mode,
        "policy_forward_backend": (
            "sdpa_math"
            if adaptation_mode == "token"
            else model_kwargs["attn_implementation"]
        ),
        "source_run_dir": str(source_run_dir),
        "dataset_parquet": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "num_unique_problems": len(problems),
        "model": str(args.model),
        "support_group_size": args.support_group_size,
        "adaptation_rounds": args.adaptation_rounds,
        "total_support_group_size": (args.support_group_size * args.adaptation_rounds),
        "base_query_group_size": args.base_query_group_size,
        "adapted_query_group_size": args.adapted_query_group_size,
        "seed": args.seed,
        "seed_schedule": "checkpoint-independent",
        "inner_iterations": inner_config.num_iterations,
        "inner_learning_rate": inner_config.optimizer.learning_rate,
        "token_credit_parameterization": (
            source_config["token_credit_parameterization"]
            if adaptation_mode == "token"
            else None
        ),
        "token_credit_max": (
            float(source_config["token_credit_max"])
            if adaptation_mode == "token"
            else None
        ),
        "token_credit_cross_trajectory_normalization": (
            TOKEN_CREDIT_CROSS_TRAJECTORY_NORMALIZATION
            if adaptation_mode == "token"
            else None
        ),
        "max_new_tokens": max_new_tokens,
        "adaptation_temperature": args.adaptation_temperature,
        "adaptation_top_p": args.adaptation_top_p,
        "adaptation_top_k": args.adaptation_top_k,
        "query_temperature": args.query_temperature,
        "query_top_p": args.query_top_p,
        "query_top_k": args.query_top_k,
        "source_training_temperature": float(source_config["temperature"]),
        "source_training_top_p": float(source_config["top_p"]),
        "source_training_top_k": int(source_config["top_k"]),
        "local_rollout_batch_size": args.local_rollout_batch_size,
        "local_adaptation_batch_size": args.local_adaptation_batch_size,
    }
    if accelerator.is_main_process:
        (output_dir / "evaluation_config.json").write_text(
            json.dumps(evaluation_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {"event": "evaluation_started", **evaluation_config},
                sort_keys=True,
            ),
            flush=True,
        )

    rollout_path = output_dir / f"rollouts-rank-{accelerator.process_index}.jsonl"
    problem_path = output_dir / (
        f"problem-metrics-rank-{accelerator.process_index}.jsonl"
    )
    started = time.perf_counter()
    query_seeds = [
        _generation_seed(
            base_seed=args.seed,
            step=0,
            phase="checkpoint_evaluation_query",
            rank=0,
            problem_uid=problem.uid,
        )
        for problem in local_problems
    ]

    with rollout_path.open("w", encoding="utf-8") as rollout_stream:
        current_fast_parameters = [initial_fast] * len(local_problems)
        current_optimizer_states: list[FastOptimizerState] | None = None
        confidence_probabilities = [[] for _ in local_problems]
        support_correctness = [[] for _ in local_problems]
        support_seeds_by_round: list[list[int]] = []
        inner_metrics = [[] for _ in local_problems]
        inner_optimizer_steps = [[] for _ in local_problems]
        support_round_totals = torch.zeros(
            (args.adaptation_rounds, 9),
            dtype=torch.float64,
            device=accelerator.device,
        )

        for round_index in range(args.adaptation_rounds):
            round_number = round_index + 1
            support_phase = (
                "checkpoint_evaluation_support"
                if round_index == 0
                else f"checkpoint_evaluation_support_round_{round_number}"
            )
            support_seeds = [
                _generation_seed(
                    base_seed=args.seed,
                    step=0,
                    phase=support_phase,
                    rank=0,
                    problem_uid=problem.uid,
                )
                for problem in local_problems
            ]
            support_seeds_by_round.append(support_seeds)
            support_batches = support_engine.generate_batches(
                problem_batches,
                _chunks(
                    current_fast_parameters,
                    args.local_rollout_batch_size,
                ),
                show_progress=accelerator.is_main_process,
                progress_description=(
                    "checkpoint evaluation support "
                    f"round {round_number}/{args.adaptation_rounds}"
                ),
                seed_batches=_chunks(
                    support_seeds,
                    args.local_rollout_batch_size,
                ),
            )
            support_groups = [group for batch in support_batches for group in batch]
            verified_supports = []
            support_verifications = []
            for problem, group in zip(local_problems, support_groups, strict=True):
                group, verification = _verify(group, problem, verifier)
                verified_supports.append(group)
                support_verifications.append(verification)
            del support_groups, support_batches

            next_fast_parameters = []
            next_optimizer_states = []
            round_confidence_probabilities = []
            round_inner_metrics = []
            round_optimizer_steps = []
            index_batches = _chunks(
                list(range(len(verified_supports))),
                args.local_adaptation_batch_size,
            )
            adaptation_progress = tqdm(
                index_batches,
                desc=(
                    "checkpoint evaluation adaptation "
                    f"round {round_number}/{args.adaptation_rounds}"
                ),
                unit="batch",
                disable=not accelerator.is_main_process,
            )
            for indices in adaptation_progress:
                device_batch = tuple(
                    verified_supports[index].to(accelerator.device) for index in indices
                )
                if round_index == 0:
                    if adaptation_mode == "sequence":
                        adaptations = algorithm.adapt_tasks(
                            device_batch,
                            initial_fast,
                            differentiable=False,
                            supervise_confidence=False,
                            show_progress=False,
                        )
                    else:
                        adaptations = algorithm.adapt_token_tasks(
                            device_batch,
                            initial_fast,
                            differentiable=False,
                        )
                else:
                    if adaptation_mode != "sequence":
                        raise RuntimeError(
                            "Continued token adaptation is not enabled here."
                        )
                    if current_optimizer_states is None:
                        raise RuntimeError("Missing continued inner optimizer state.")
                    device_fast = tuple(
                        {
                            name: value.to(accelerator.device)
                            for name, value in current_fast_parameters[index].items()
                        }
                        for index in indices
                    )
                    device_optimizer_states = tuple(
                        _optimizer_state_to(
                            current_optimizer_states[index],
                            accelerator.device,
                        )
                        for index in indices
                    )
                    adaptations = algorithm.continue_adapt_tasks(
                        device_batch,
                        device_fast,
                        device_optimizer_states,
                        show_progress=False,
                    )
                    del device_fast, device_optimizer_states
                for local_index, adaptation in enumerate(adaptations):
                    next_fast_parameters.append(
                        {
                            name: value.detach().to("cpu")
                            for name, value in adaptation.fast_parameters.items()
                        }
                    )
                    next_optimizer_states.append(
                        _optimizer_state_to(adaptation.optimizer_state, "cpu")
                    )
                    round_confidence_probabilities.append(
                        _response_confidences(
                            adaptation,
                            device_batch[local_index],
                            adaptation_mode,
                        )
                    )
                    round_inner_metrics.append(
                        [
                            {
                                "loss": output.loss.item(),
                                "clip_fraction": output.clip_fraction.item(),
                                "mean_kl": output.mean_kl.item(),
                            }
                            for output in adaptation.inner_losses
                        ]
                    )
                    round_optimizer_steps.append(adaptation.optimizer_state.step)
                del device_batch, adaptations
            del index_batches, adaptation_progress

            for index, (problem, group, seed, valid) in enumerate(
                zip(
                    local_problems,
                    verified_supports,
                    support_seeds,
                    valid_flags,
                    strict=True,
                )
            ):
                probabilities = round_confidence_probabilities[index]
                correctness = _correctness(group)
                confidence_probabilities[index].extend(probabilities)
                support_correctness[index].extend(correctness)
                inner_metrics[index].append(round_inner_metrics[index])
                inner_optimizer_steps[index].append(round_optimizer_steps[index])
                if not valid:
                    continue
                _write_group(
                    rollout_stream,
                    checkpoint_step=checkpoint_step,
                    phase="support",
                    problem=problem,
                    group=group,
                    verification=support_verifications[index],
                    seed=seed,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=policy_bundle.tokenizer.eos_token_id,
                    confidence_probabilities=probabilities,
                    adaptation_mode=adaptation_mode,
                    adaptation_round=round_number,
                )
                confidence_sum = sum(probabilities)
                correct_confidence_sum = sum(
                    probability
                    for probability, correct in zip(
                        probabilities, correctness, strict=True
                    )
                    if correct == 1
                )
                correct_count = sum(correctness)
                support_round_totals[round_index] += torch.tensor(
                    (
                        correct_count,
                        len(correctness),
                        any(correctness),
                        1,
                        confidence_sum,
                        correct_confidence_sum,
                        correct_count,
                        confidence_sum - correct_confidence_sum,
                        len(correctness) - correct_count,
                    ),
                    dtype=torch.float64,
                    device=accelerator.device,
                )
            current_fast_parameters = next_fast_parameters
            current_optimizer_states = next_optimizer_states
            del verified_supports, support_verifications

        fast_parameters = current_fast_parameters

        base_batches = base_query_engine.generate_batches(
            problem_batches,
            [[initial_fast] * len(batch) for batch in problem_batches],
            show_progress=accelerator.is_main_process,
            progress_description="checkpoint evaluation base query",
            seed_batches=_chunks(query_seeds, args.local_rollout_batch_size),
            compute_old_logprobs=False,
        )
        base_groups = [group for batch in base_batches for group in batch]
        base_correctness = []
        for problem, group, seed, valid in zip(
            local_problems, base_groups, query_seeds, valid_flags, strict=True
        ):
            group, verification = _verify(group, problem, verifier)
            if valid:
                _write_group(
                    rollout_stream,
                    checkpoint_step=checkpoint_step,
                    phase="base_query",
                    problem=problem,
                    group=group,
                    verification=verification,
                    seed=seed,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=policy_bundle.tokenizer.eos_token_id,
                )
            base_correctness.append(_correctness(group))
        del base_groups, base_batches

        adapted_batches = adapted_query_engine.generate_batches(
            problem_batches,
            _chunks(fast_parameters, args.local_rollout_batch_size),
            show_progress=accelerator.is_main_process,
            progress_description="checkpoint evaluation adapted query",
            seed_batches=_chunks(query_seeds, args.local_rollout_batch_size),
            compute_old_logprobs=False,
        )
        del fast_parameters
        adapted_groups = [group for batch in adapted_batches for group in batch]
        totals = torch.zeros(23, dtype=torch.float64, device=accelerator.device)
        with problem_path.open("w", encoding="utf-8") as problem_stream:
            for index, (problem, group, seed, valid) in enumerate(
                zip(
                    local_problems,
                    adapted_groups,
                    query_seeds,
                    valid_flags,
                    strict=True,
                )
            ):
                group, verification = _verify(group, problem, verifier)
                if not valid:
                    continue
                _write_group(
                    rollout_stream,
                    checkpoint_step=checkpoint_step,
                    phase="adapted_query",
                    problem=problem,
                    group=group,
                    verification=verification,
                    seed=seed,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=policy_bundle.tokenizer.eos_token_id,
                    adaptation_round=args.adaptation_rounds,
                )
                support_correct = support_correctness[index]
                base_correct = base_correctness[index]
                adapted_correct = _correctness(group)
                support_pass = any(support_correct)
                base_pass = any(base_correct)
                adapted_pass = any(adapted_correct)
                base_score = sum(base_correct)
                adapted_score = sum(adapted_correct)
                probabilities = confidence_probabilities[index]
                confidence_sum = sum(probabilities)
                correct_confidence_sum = sum(
                    probability
                    for probability, correct in zip(
                        probabilities, support_correct, strict=True
                    )
                    if correct == 1
                )
                incorrect_confidence_sum = confidence_sum - correct_confidence_sum
                correct_count = sum(support_correct)
                incorrect_count = len(support_correct) - correct_count
                if adaptation_mode == "sequence":
                    brier_sum = sum(
                        (probability - correct) ** 2
                        for probability, correct in zip(
                            probabilities, support_correct, strict=True
                        )
                    )
                    bce_sum = sum(
                        -correct * math.log(max(probability, 1e-12))
                        - (1 - correct) * math.log(max(1 - probability, 1e-12))
                        for probability, correct in zip(
                            probabilities, support_correct, strict=True
                        )
                    )
                else:
                    brier_sum = 0.0
                    bce_sum = 0.0
                totals += torch.tensor(
                    (
                        sum(support_correct),
                        len(support_correct),
                        base_score,
                        len(base_correct),
                        adapted_score,
                        len(adapted_correct),
                        support_pass,
                        base_pass,
                        adapted_pass,
                        support_pass or base_pass,
                        support_pass or adapted_pass,
                        1,
                        adapted_score / len(adapted_correct)
                        > base_score / len(base_correct),
                        adapted_score / len(adapted_correct)
                        < base_score / len(base_correct),
                        adapted_score / len(adapted_correct)
                        == base_score / len(base_correct),
                        confidence_sum,
                        len(probabilities),
                        correct_confidence_sum,
                        correct_count,
                        incorrect_confidence_sum,
                        incorrect_count,
                        brier_sum,
                        bce_sum,
                    ),
                    dtype=torch.float64,
                    device=accelerator.device,
                )
                problem_record = {
                    "checkpoint_step": checkpoint_step,
                    "problem_uid": problem.uid,
                    "data_source": problem.data_source,
                    "ground_truth": problem.ground_truth,
                    "prompt": problem.conversation,
                    "support_accuracy": sum(support_correct) / len(support_correct),
                    "base_query_accuracy": base_score / len(base_correct),
                    "adapted_query_accuracy": adapted_score / len(adapted_correct),
                    "support_pass": support_pass,
                    "base_query_pass": base_pass,
                    "adapted_query_pass": adapted_pass,
                    "adaptation_rounds": args.adaptation_rounds,
                    "support_seed": support_seeds_by_round[0][index],
                    "support_seeds": [seeds[index] for seeds in support_seeds_by_round],
                    "query_seed": query_seeds[index],
                    "adaptation_round_metrics": inner_metrics[index],
                    "inner_optimizer_steps": inner_optimizer_steps[index],
                    "inner_iterations": [
                        metric
                        for round_metrics in inner_metrics[index]
                        for metric in round_metrics
                    ],
                }
                if adaptation_mode == "sequence":
                    problem_record["confidence_probabilities"] = (
                        confidence_probabilities[index]
                    )
                    problem_record["mean_confidence"] = sum(
                        confidence_probabilities[index]
                    ) / len(confidence_probabilities[index])
                else:
                    problem_record["response_mean_token_credits"] = (
                        confidence_probabilities[index]
                    )
                    problem_record["mean_token_credit"] = sum(
                        confidence_probabilities[index]
                    ) / len(confidence_probabilities[index])
                problem_stream.write(
                    json.dumps(problem_record, ensure_ascii=False) + "\n"
                )

    totals = accelerator.reduce(totals, reduction="sum")
    support_round_totals = accelerator.reduce(
        support_round_totals,
        reduction="sum",
    )
    if accelerator.is_main_process:
        summary = {
            "event": "checkpoint_evaluation_completed",
            "checkpoint": str(checkpoint),
            "checkpoint_step": checkpoint_step,
            "adaptation_mode": adaptation_mode,
            "dataset_parquet": str(dataset_path),
            "seed": args.seed,
            "adaptation_rounds": args.adaptation_rounds,
            "inner_iterations_per_round": inner_config.num_iterations,
            "total_inner_iterations": (
                args.adaptation_rounds * inner_config.num_iterations
            ),
            "support_rounds": _support_round_summaries(
                support_round_totals,
                adaptation_mode=adaptation_mode,
            ),
            "seconds": time.perf_counter() - started,
            **_summary_from_totals(
                totals,
                support_group_size=(args.support_group_size * args.adaptation_rounds),
                base_query_group_size=args.base_query_group_size,
                adapted_query_group_size=args.adapted_query_group_size,
                adaptation_mode=adaptation_mode,
            ),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, sort_keys=True), flush=True)
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
