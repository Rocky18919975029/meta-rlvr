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


def _probe_group(tokenizer, device: torch.device, max_length: int) -> RolloutGroup:
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
    ).repeat(2, 1)
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
        texts=(completion, completion),
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The policy JVP probe requires CUDA.")
    if args.max_sequence_length < 16:
        raise ValueError("max_sequence_length must be at least 16.")
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

    def selected_logprobs(*parameter_values: torch.Tensor) -> torch.Tensor:
        fast_parameters = dict(zip(names, parameter_values, strict=True))
        return token_logprobs(
            policy,
            group,
            fast_parameters=fast_parameters,
            activation_checkpointing=False,
        )[0]

    (jvp_primal, jvp_tangent), jvp_metrics = _timed(
        "forward_mode_jvp",
        lambda: torch.func.jvp(selected_logprobs, primals, tangents),
    )
    if not torch.isfinite(jvp_primal).all():
        raise RuntimeError("JVP primal output contains non-finite values.")
    if not torch.isfinite(jvp_tangent).all():
        raise RuntimeError("JVP tangent output contains non-finite values.")
    active_jvp = jvp_tangent[group.completion_mask[0]]
    if torch.count_nonzero(active_jvp).item() == 0:
        raise RuntimeError("JVP is identically zero on completion tokens.")

    reverse_primals = tuple(value.detach().requires_grad_(True) for value in primals)
    cotangent = jvp_tangent.detach()

    def reverse_vjp():
        reverse_output = selected_logprobs(*reverse_primals)
        gradients = torch.autograd.grad(
            reverse_output,
            reverse_primals,
            grad_outputs=cotangent,
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )
        return reverse_output, gradients

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
        for tangent, gradient in zip(tangents, reverse_gradients, strict=True)
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
        "completion_tokens": int(group.completion_mask[0].sum().item()),
        "fast_parameter_tensors": len(primals),
        "fast_parameter_elements": sum(value.numel() for value in primals),
        "jvp_active_l2": float(active_jvp.float().norm().item()),
        "duality_lhs": float(lhs.item()),
        "duality_rhs": float(rhs.item()),
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
