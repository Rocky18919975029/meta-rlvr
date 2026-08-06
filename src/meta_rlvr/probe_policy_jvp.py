from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from .functional import token_logprobs
from .models import load_policy_with_lora
from .types import RolloutGroup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe Qwen/PEFT token-logprob forward-mode JVP support."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "eager"),
        default="sdpa",
    )
    parser.add_argument("--max-sequence-length", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--response-micro-batch-size", type=int, default=2)
    parser.add_argument("--logprob-position-chunk-size", type=int)
    parser.add_argument("--skip-duality-check", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duality-relative-tolerance", type=float, default=0.05)
    return parser.parse_args()


def _memory() -> dict[str, float]:
    gib = 1024**3
    return {
        "allocated_gib": torch.cuda.memory_allocated() / gib,
        "reserved_gib": torch.cuda.memory_reserved() / gib,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / gib,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / gib,
    }


def _timed(stage: str, function):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = function()
    torch.cuda.synchronize()
    metrics = {
        "event": "policy_jvp_stage",
        "stage": stage,
        "seconds": time.perf_counter() - started,
        **_memory(),
    }
    print(json.dumps(metrics, sort_keys=True), flush=True)
    return output, metrics


def _probe_group(
    tokenizer,
    device: torch.device,
    max_length: int,
    group_size: int,
) -> RolloutGroup:
    messages = [
        {
            "role": "user",
            "content": "Compute 2 + 2. Put the final answer in \\boxed{}.",
        }
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    completion = (
        "We compute the sum directly. Two plus two equals four.\n"
        "\\boxed{4}"
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    if len(prompt_ids) >= max_length:
        prompt_ids = prompt_ids[-(max_length - 1) :]
    completion_length = max_length - len(prompt_ids)
    repeats = math.ceil(completion_length / len(completion_ids))
    completion_ids = (completion_ids * repeats)[:completion_length]
    input_ids = torch.tensor(
        [prompt_ids + completion_ids],
        dtype=torch.long,
        device=device,
    ).repeat(group_size, 1)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    completion_mask = torch.zeros(
        (input_ids.shape[0], input_ids.shape[1] - 1),
        dtype=torch.bool,
        device=device,
    )
    completion_mask[:, len(prompt_ids) - 1 :] = True
    return RolloutGroup(
        input_ids=input_ids,
        attention_mask=attention_mask,
        completion_mask=completion_mask,
        old_logprobs=torch.zeros_like(completion_mask, dtype=torch.float32),
        texts=tuple(completion for _ in range(group_size)),
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The policy JVP probe requires CUDA.")
    if args.max_sequence_length < 16:
        raise ValueError("max_sequence_length must be at least 16.")
    if args.group_size < 2:
        raise ValueError("group_size must be at least 2.")
    if not 1 <= args.response_micro_batch_size <= args.group_size:
        raise ValueError(
            "response_micro_batch_size must be between 1 and group_size."
        )
    if (
        args.logprob_position_chunk_size is not None
        and args.logprob_position_chunk_size <= 0
    ):
        raise ValueError("logprob_position_chunk_size must be positive.")
    if args.duality_relative_tolerance <= 0:
        raise ValueError("duality_relative_tolerance must be positive.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda", 0)
    config = json.loads(args.run_config.read_text(encoding="utf-8"))
    target_modules = tuple(
        item.strip() for item in config["lora_target_modules"].split(",")
    )

    bundle, load_metrics = _timed(
        "load_policy",
        lambda: load_policy_with_lora(
            str(args.model),
            lora_rank=int(config["lora_rank"]),
            lora_alpha=int(config["lora_alpha"]),
            target_modules=target_modules,
            dtype=config["dtype"],
            gradient_checkpointing=False,
            trust_remote_code=bool(config["trust_remote_code"]),
            model_kwargs={"attn_implementation": args.attn_implementation},
        ),
    )
    policy, move_metrics = _timed(
        "move_policy_to_gpu",
        lambda: bundle.model.to(device),
    )
    policy.eval()
    group = _probe_group(
        bundle.tokenizer,
        device,
        args.max_sequence_length,
        args.group_size,
    )
    names = tuple(bundle.initial_fast_parameters)
    primals = tuple(
        bundle.initial_fast_parameters[name].to(device).detach()
        for name in names
    )
    tangent_scale = math.sqrt(sum(value.numel() for value in primals))
    tangents = tuple(
        torch.randn_like(value) / tangent_scale for value in primals
    )

    def selected_logprobs(
        row_start: int,
        row_end: int,
        *parameter_values: torch.Tensor,
    ) -> torch.Tensor:
        fast_parameters = dict(zip(names, parameter_values, strict=True))
        return token_logprobs(
            policy,
            group,
            fast_parameters=fast_parameters,
            row_start=row_start,
            row_end=row_end,
            activation_checkpointing=False,
            logprob_position_chunk_size=args.logprob_position_chunk_size,
        )

    row_intervals = tuple(
        (
            start,
            min(start + args.response_micro_batch_size, args.group_size),
        )
        for start in range(0, args.group_size, args.response_micro_batch_size)
    )

    def forward_jvp():
        primal_batches: list[torch.Tensor] = []
        tangent_batches: list[torch.Tensor] = []
        for row_start, row_end in row_intervals:
            primal, tangent = torch.func.jvp(
                lambda *values: selected_logprobs(
                    row_start,
                    row_end,
                    *values,
                ),
                primals,
                tangents,
            )
            primal_batches.append(primal)
            tangent_batches.append(tangent)
        return torch.cat(primal_batches, dim=0), torch.cat(tangent_batches, dim=0)

    (jvp_primal, jvp_tangent), jvp_metrics = _timed(
        "forward_mode_jvp",
        forward_jvp,
    )
    if not torch.isfinite(jvp_primal).all():
        raise RuntimeError("JVP primal output contains non-finite values.")
    if not torch.isfinite(jvp_tangent).all():
        raise RuntimeError("JVP tangent output contains non-finite values.")
    active_jvp = jvp_tangent[group.completion_mask]
    if torch.count_nonzero(active_jvp).item() == 0:
        raise RuntimeError("JVP is identically zero on completion tokens.")

    reverse_metrics = None
    lhs = None
    rhs = None
    relative_error_value = None
    if not args.skip_duality_check:
        reverse_primals = tuple(
            value.detach().requires_grad_(True) for value in primals
        )
        cotangent = jvp_tangent.detach()

        def reverse_vjp():
            output_batches: list[torch.Tensor] = []
            gradient_sums = [torch.zeros_like(value) for value in reverse_primals]
            for row_start, row_end in row_intervals:
                output = selected_logprobs(
                    row_start,
                    row_end,
                    *reverse_primals,
                )
                gradients = torch.autograd.grad(
                    output,
                    reverse_primals,
                    grad_outputs=cotangent[row_start:row_end],
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=False,
                )
                output_batches.append(output.detach())
                gradient_sums = [
                    total + gradient
                    for total, gradient in zip(
                        gradient_sums,
                        gradients,
                        strict=True,
                    )
                ]
            return torch.cat(output_batches, dim=0), tuple(gradient_sums)

        (reverse_output, reverse_gradients), reverse_metrics = _timed(
            "reverse_mode_vjp",
            reverse_vjp,
        )
        torch.testing.assert_close(
            jvp_primal,
            reverse_output,
            rtol=0.02,
            atol=0.02,
        )
        lhs = jvp_tangent.float().square().sum()
        rhs = sum(
            (tangent.float() * gradient.float()).sum()
            for tangent, gradient in zip(
                tangents,
                reverse_gradients,
                strict=True,
            )
        )
        relative_error = (lhs - rhs).abs() / lhs.abs().clamp_min(1e-12)
        relative_error_value = float(relative_error.item())
        if relative_error_value > args.duality_relative_tolerance:
            raise RuntimeError(
                "JVP/VJP duality check failed: "
                f"relative_error={relative_error_value:.6g}."
            )

    result = {
        "event": "policy_jvp_probe_succeeded",
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "attn_implementation": args.attn_implementation,
        "requested_sequence_tokens": args.max_sequence_length,
        "sequence_tokens": int(group.input_ids.shape[1]),
        "group_size": args.group_size,
        "response_micro_batch_size": args.response_micro_batch_size,
        "logprob_position_chunk_size": args.logprob_position_chunk_size,
        "jvp_microbatches": len(row_intervals),
        "completion_tokens": int(group.completion_mask[0].sum().item()),
        "total_completion_tokens": int(group.completion_mask.sum().item()),
        "fast_parameter_tensors": len(primals),
        "fast_parameter_elements": sum(value.numel() for value in primals),
        "jvp_active_l2": float(active_jvp.float().norm().item()),
        "jvp_responses_per_second": args.group_size / jvp_metrics["seconds"],
        "jvp_completion_tokens_per_second": (
            int(group.completion_mask.sum().item()) / jvp_metrics["seconds"]
        ),
        "duality_check_skipped": args.skip_duality_check,
        "duality_lhs": None if lhs is None else float(lhs.item()),
        "duality_rhs": None if rhs is None else float(rhs.item()),
        "duality_relative_error": relative_error_value,
        "duality_relative_tolerance": args.duality_relative_tolerance,
        "load": load_metrics,
        "move_to_gpu": move_metrics,
        "jvp": jvp_metrics,
        "vjp": reverse_metrics,
    }
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
