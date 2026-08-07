from __future__ import annotations

import argparse
import gc
import json
import math
import time
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from .data import load_semantically_unique_dapo_problems
from .functional import (
    materialized_fast_parameters,
    sequence_microbatches,
    token_logprobs,
)
from .models import load_policy_with_lora
from .rollout import VLLMHybridRolloutEngine
from .types import RolloutGroup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare vLLM raw log-probabilities with PyTorch SDPA and eager "
            "on exactly the same generated token sequences and LoRA weights."
        )
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--vllm-base-url", required=True)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=3072)
    parser.add_argument("--response-micro-batch-size", type=int, default=4)
    parser.add_argument("--jvp-sequence-tokens", type=int, default=3072)
    parser.add_argument("--max-tokens-per-micro-batch", type=int, default=16384)
    parser.add_argument("--logprob-position-chunk-size", type=int, default=256)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--control-timeout", type=float, default=120.0)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "group_size",
        "max_new_tokens",
        "response_micro_batch_size",
        "jvp_sequence_tokens",
        "max_tokens_per_micro_batch",
        "logprob_position_chunk_size",
        "lora_rank",
        "lora_alpha",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive.")
    if args.group_size < 2:
        raise ValueError("group_size must be at least two.")


def _policy_logprobs(
    policy: nn.Module,
    group: RolloutGroup,
    fast_parameters: Mapping[str, Tensor] | None,
    *,
    response_micro_batch_size: int,
    max_tokens_per_micro_batch: int,
    logprob_position_chunk_size: int,
) -> Tensor:
    row_batches = sequence_microbatches(
        group,
        max_sequences=response_micro_batch_size,
        max_tokens=max_tokens_per_micro_batch,
    )
    rows: list[Tensor | None] = [None] * group.group_size
    policy.eval()
    with torch.no_grad():
        for row_indices in row_batches:
            output = token_logprobs(
                policy,
                group,
                fast_parameters=fast_parameters,
                row_indices=row_indices,
                logprob_position_chunk_size=logprob_position_chunk_size,
            )
            for offset, row_index in enumerate(row_indices):
                rows[row_index] = output[offset : offset + 1]
    if any(row is None for row in rows):
        raise RuntimeError("Log-probability probe omitted one or more responses.")
    return torch.cat([row for row in rows if row is not None], dim=0)


def _comparison(
    left: Tensor,
    right: Tensor,
    mask: Tensor,
) -> dict[str, float | int]:
    delta = (left[mask] - right[mask]).float()
    absolute = delta.abs()
    quantiles = torch.quantile(
        absolute,
        torch.tensor((0.5, 0.9, 0.99, 0.999), device=absolute.device),
    )
    return {
        "token_count": delta.numel(),
        "mean_delta": delta.mean().item(),
        "mean_absolute_delta": absolute.mean().item(),
        "max_absolute_delta": absolute.max().item(),
        "p50_absolute_delta": quantiles[0].item(),
        "p90_absolute_delta": quantiles[1].item(),
        "p99_absolute_delta": quantiles[2].item(),
        "p999_absolute_delta": quantiles[3].item(),
        "fraction_absolute_delta_gt_0.1": (absolute > 0.1).float().mean().item(),
        "fraction_absolute_delta_gt_0.5": (absolute > 0.5).float().mean().item(),
        "fraction_absolute_delta_gt_1.0": (absolute > 1.0).float().mean().item(),
    }


def _cuda_memory() -> dict[str, float]:
    gib = 1024**3
    return {
        "allocated_gib": torch.cuda.memory_allocated() / gib,
        "reserved_gib": torch.cuda.memory_reserved() / gib,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / gib,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / gib,
    }


def _fixed_length_jvp_group(
    source: RolloutGroup,
    *,
    sequence_tokens: int,
    device: torch.device,
) -> RolloutGroup:
    first_completion_position = int(
        source.completion_mask[0].nonzero(as_tuple=False)[0].item()
    )
    prompt_tokens = first_completion_position + 1
    if sequence_tokens <= prompt_tokens:
        raise ValueError("JVP sequence length must exceed the tokenized prompt length.")
    source_length = int(source.attention_mask[0].sum().item())
    completion = source.input_ids[0, prompt_tokens:source_length].tolist()
    if not completion:
        raise RuntimeError("The source rollout contains no completion tokens.")
    requested_completion_tokens = sequence_tokens - prompt_tokens
    repeated = (completion * math.ceil(requested_completion_tokens / len(completion)))[
        :requested_completion_tokens
    ]
    sequence = torch.tensor(
        source.input_ids[0, :prompt_tokens].tolist() + repeated,
        dtype=torch.long,
        device=device,
    )
    input_ids = sequence.unsqueeze(0).repeat(source.group_size, 1)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    completion_mask = torch.zeros(
        (source.group_size, sequence_tokens - 1),
        dtype=torch.bool,
        device=device,
    )
    completion_mask[:, prompt_tokens - 1 :] = True
    return RolloutGroup(
        input_ids=input_ids,
        attention_mask=attention_mask,
        completion_mask=completion_mask,
        old_logprobs=torch.zeros_like(completion_mask, dtype=torch.float32),
        texts=tuple("sdpa-math-jvp-probe" for _ in range(source.group_size)),
    )


def _sdpa_math_exact_jvp(
    policy: nn.Module,
    group: RolloutGroup,
    fast_parameters: Mapping[str, Tensor],
    *,
    response_micro_batch_size: int,
    max_tokens_per_micro_batch: int,
    logprob_position_chunk_size: int,
    seed: int,
) -> tuple[Tensor, dict[str, float | int | bool]]:
    names = tuple(fast_parameters)
    primals = tuple(fast_parameters[name].detach() for name in names)
    tangent_scale = math.sqrt(sum(value.numel() for value in primals))
    torch.manual_seed(seed)
    tangents = tuple(torch.randn_like(value) / tangent_scale for value in primals)
    row_batches = sequence_microbatches(
        group,
        max_sequences=response_micro_batch_size,
        max_tokens=max_tokens_per_micro_batch,
    )

    def run_jvp() -> tuple[Tensor, Tensor]:
        primal_rows: list[Tensor | None] = [None] * group.group_size
        tangent_rows: list[Tensor | None] = [None] * group.group_size
        for row_indices in row_batches:

            def selected_logprobs(*values: Tensor) -> Tensor:
                fast = dict(zip(names, values, strict=True))
                return token_logprobs(
                    policy,
                    group,
                    fast_parameters=fast,
                    row_indices=row_indices,
                    activation_checkpointing=False,
                    logprob_position_chunk_size=logprob_position_chunk_size,
                )

            with sdpa_kernel(backends=[SDPBackend.MATH]):
                primal, tangent = torch.func.jvp(
                    selected_logprobs,
                    primals,
                    tangents,
                )
            for offset, row_index in enumerate(row_indices):
                primal_rows[row_index] = primal[offset : offset + 1]
                tangent_rows[row_index] = tangent[offset : offset + 1]
        if any(value is None for value in primal_rows + tangent_rows):
            raise RuntimeError("SDPA math JVP omitted one or more responses.")
        return (
            torch.cat([value for value in primal_rows if value is not None]),
            torch.cat([value for value in tangent_rows if value is not None]),
        )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    primal, tangent = run_jvp()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    if not torch.isfinite(primal).all() or not torch.isfinite(tangent).all():
        raise RuntimeError("SDPA math exact JVP returned non-finite values.")
    active = tangent[group.completion_mask]
    if torch.count_nonzero(active).item() == 0:
        raise RuntimeError("SDPA math exact JVP is zero on completion tokens.")
    completion_tokens = int(group.completion_mask.sum().item())
    metrics: dict[str, float | int | bool] = {
        "supported": True,
        "seconds": seconds,
        "microbatches": len(row_batches),
        "response_micro_batch_size": response_micro_batch_size,
        "responses_per_second": group.group_size / seconds,
        "completion_tokens_per_second": completion_tokens / seconds,
        "active_l2": active.float().norm().item(),
        "sequence_tokens": int(group.input_ids.shape[1]),
        "completion_tokens": completion_tokens,
        **_cuda_memory(),
    }
    return primal.detach().to("cpu"), metrics


def _load_policy(args: argparse.Namespace, attention: str):
    target_modules = tuple(
        item.strip() for item in args.lora_target_modules.split(",") if item.strip()
    )
    return load_policy_with_lora(
        str(args.model),
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        dtype="bfloat16",
        gradient_checkpointing=False,
        model_kwargs={"attn_implementation": attention},
    )


def main() -> None:
    args = parse_args()
    _validate_args(args)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("The parity probe requires exactly one visible CUDA GPU.")
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    problem = load_semantically_unique_dapo_problems(args.parquet)[0]

    started = time.perf_counter()
    sdpa_bundle = _load_policy(args, "sdpa")
    sdpa_policy = sdpa_bundle.model.to(device).eval()
    fast_cpu = {
        name: value.detach().to("cpu").contiguous()
        for name, value in sdpa_bundle.initial_fast_parameters.items()
    }
    engine = VLLMHybridRolloutEngine(
        sdpa_policy,
        sdpa_bundle.tokenizer,
        group_size=args.group_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        generation_micro_batch_size=args.group_size,
        logprob_micro_batch_size=args.response_micro_batch_size,
        logprob_max_tokens_per_micro_batch=args.max_tokens_per_micro_batch,
        base_url=args.vllm_base_url,
        adapter_root=args.adapter_root,
        request_timeout=args.request_timeout,
        control_timeout=args.control_timeout,
    )
    group = engine.generate(
        problem,
        sdpa_bundle.initial_fast_parameters,
        show_progress=True,
        progress_description="vLLM/SDPA parity probe",
        seed=args.seed,
    ).to("cpu")
    sdpa_logprobs = group.old_logprobs.clone()
    rollout_logprobs = group.rollout_logprobs
    if rollout_logprobs is None:
        raise RuntimeError("vLLM did not return rollout log-probabilities.")
    device_group = group.to(device)
    device_fast = {name: value.to(device) for name, value in fast_cpu.items()}
    with sdpa_kernel(backends=[SDPBackend.MATH]):
        sdpa_math_logprobs = _policy_logprobs(
            sdpa_policy,
            device_group,
            device_fast,
            response_micro_batch_size=args.response_micro_batch_size,
            max_tokens_per_micro_batch=args.max_tokens_per_micro_batch,
            logprob_position_chunk_size=args.logprob_position_chunk_size,
        ).to("cpu")
    sdpa_math_parity = _comparison(
        rollout_logprobs,
        sdpa_math_logprobs,
        group.completion_mask,
    )
    print(
        json.dumps(
            {
                "event": "sdpa_math_logprob_parity_completed",
                "vllm_vs_sdpa_math": sdpa_math_parity,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    jvp_group = _fixed_length_jvp_group(
        group,
        sequence_tokens=args.jvp_sequence_tokens,
        device=device,
    )
    try:
        sdpa_math_jvp_primal, sdpa_math_jvp = _sdpa_math_exact_jvp(
            sdpa_policy,
            jvp_group,
            device_fast,
            response_micro_batch_size=args.response_micro_batch_size,
            max_tokens_per_micro_batch=args.max_tokens_per_micro_batch,
            logprob_position_chunk_size=args.logprob_position_chunk_size,
            seed=args.seed + 1,
        )
    except (NotImplementedError, torch.OutOfMemoryError) as error:
        sdpa_math_jvp_primal = None
        sdpa_math_jvp = {
            "supported": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        torch.cuda.empty_cache()
    else:
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            sdpa_math_jvp_direct = _policy_logprobs(
                sdpa_policy,
                jvp_group,
                device_fast,
                response_micro_batch_size=args.response_micro_batch_size,
                max_tokens_per_micro_batch=args.max_tokens_per_micro_batch,
                logprob_position_chunk_size=args.logprob_position_chunk_size,
            ).to("cpu")
        sdpa_math_jvp["primal_vs_direct_mean_absolute_delta"] = _comparison(
            sdpa_math_jvp_primal,
            sdpa_math_jvp_direct,
            jvp_group.completion_mask.to("cpu"),
        )["mean_absolute_delta"]
    print(
        json.dumps(
            {
                "event": "sdpa_math_exact_jvp_completed",
                "sdpa_math_exact_jvp": sdpa_math_jvp,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    del device_group, device_fast, jvp_group, sdpa_math_jvp_primal
    del engine, sdpa_policy, sdpa_bundle
    gc.collect()
    torch.cuda.empty_cache()

    eager_bundle = _load_policy(args, "eager")
    eager_policy = eager_bundle.model.to(device).eval()
    eager_names = set(eager_bundle.initial_fast_parameters)
    if eager_names != set(fast_cpu):
        raise RuntimeError("SDPA and eager LoRA parameter names differ.")
    device_group = group.to(device)
    device_fast = {name: value.to(device) for name, value in fast_cpu.items()}
    eager_logprobs = _policy_logprobs(
        eager_policy,
        device_group,
        device_fast,
        response_micro_batch_size=args.response_micro_batch_size,
        max_tokens_per_micro_batch=args.max_tokens_per_micro_batch,
        logprob_position_chunk_size=args.logprob_position_chunk_size,
    ).to("cpu")
    with materialized_fast_parameters(eager_policy, device_fast):
        eager_materialized_logprobs = _policy_logprobs(
            eager_policy,
            device_group,
            None,
            response_micro_batch_size=args.response_micro_batch_size,
            max_tokens_per_micro_batch=args.max_tokens_per_micro_batch,
            logprob_position_chunk_size=args.logprob_position_chunk_size,
        ).to("cpu")

    lora_b = [value for name, value in fast_cpu.items() if ".lora_B." in name]
    if not lora_b:
        raise RuntimeError("The probe found no LoRA B parameters.")
    summary = {
        "event": "vllm_logprob_parity_probe_completed",
        "problem_uid": problem.uid,
        "group_size": args.group_size,
        "completion_tokens": int(group.completion_mask.sum().item()),
        "max_sequence_tokens": int(group.attention_mask.sum(dim=1).max().item()),
        "lora_b_max_absolute_value": max(value.abs().max().item() for value in lora_b),
        "vllm_vs_sdpa": _comparison(
            rollout_logprobs,
            sdpa_logprobs,
            group.completion_mask,
        ),
        "vllm_vs_sdpa_math": sdpa_math_parity,
        "sdpa_math_exact_jvp": sdpa_math_jvp,
        "vllm_vs_eager": _comparison(
            rollout_logprobs,
            eager_logprobs,
            group.completion_mask,
        ),
        "vllm_vs_eager_materialized": _comparison(
            rollout_logprobs,
            eager_materialized_logprobs,
            group.completion_mask,
        ),
        "eager_functional_vs_materialized": _comparison(
            eager_logprobs,
            eager_materialized_logprobs,
            group.completion_mask,
        ),
        "eager_vs_sdpa": _comparison(
            eager_logprobs,
            sdpa_logprobs,
            group.completion_mask,
        ),
        "seconds": time.perf_counter() - started,
    }
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
