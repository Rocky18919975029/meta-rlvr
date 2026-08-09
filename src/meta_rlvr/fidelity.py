from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from .losses import bounded_token_credits


def gradient_comparison(
    exact: Mapping[str, Tensor],
    approximation: Mapping[str, Tensor],
) -> dict[str, float]:
    """Compare two gradient mappings without flattening them into one tensor."""
    if not exact or exact.keys() != approximation.keys():
        raise ValueError("Gradient mappings must have identical non-empty keys.")
    dot = 0.0
    exact_square = 0.0
    approximation_square = 0.0
    difference_square = 0.0
    same_sign = 0
    active = 0
    elements = 0
    for name in exact:
        left = exact[name].detach().to(dtype=torch.float64, device="cpu")
        right = approximation[name].detach().to(dtype=torch.float64, device="cpu")
        if left.shape != right.shape:
            raise ValueError(f"Gradient shape mismatch for {name!r}.")
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise ValueError(f"Non-finite gradient found for {name!r}.")
        dot += torch.sum(left * right).item()
        exact_square += torch.sum(left.square()).item()
        approximation_square += torch.sum(right.square()).item()
        difference_square += torch.sum((left - right).square()).item()
        nonzero = (left != 0) | (right != 0)
        same_sign += int(((torch.sign(left) == torch.sign(right)) & nonzero).sum())
        active += int(nonzero.sum())
        elements += left.numel()
    return {
        "dot": dot,
        "exact_square": exact_square,
        "approximation_square": approximation_square,
        "difference_square": difference_square,
        "same_sign": float(same_sign),
        "active": float(active),
        "elements": float(elements),
    }


def parameter_gradients(model: nn.Module) -> dict[str, Tensor]:
    gradients = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            raise RuntimeError(f"Trainable parameter {name!r} received no gradient.")
        gradients[name] = parameter.grad.detach().float().cpu().clone()
    if not gradients:
        raise RuntimeError("Model exposes no trainable parameter gradients.")
    return gradients


def finalize_gradient_comparison(
    totals: Mapping[str, float],
) -> dict[str, float | bool | None]:
    required = {
        "dot",
        "exact_square",
        "approximation_square",
        "difference_square",
        "same_sign",
        "active",
        "elements",
    }
    if totals.keys() != required:
        raise ValueError("Gradient comparison totals have unexpected fields.")
    exact_norm = totals["exact_square"] ** 0.5
    approximation_norm = totals["approximation_square"] ** 0.5
    conclusive = exact_norm > 0 and approximation_norm > 0
    return {
        "conclusive": conclusive,
        "cosine": (
            None
            if not conclusive
            else totals["dot"] / (exact_norm * approximation_norm)
        ),
        "exact_norm": exact_norm,
        "approximation_norm": approximation_norm,
        "norm_ratio": (None if exact_norm == 0 else approximation_norm / exact_norm),
        "relative_l2_error": (
            None if exact_norm == 0 else totals["difference_square"] ** 0.5 / exact_norm
        ),
        "active_sign_agreement": (
            None if totals["active"] == 0 else totals["same_sign"] / totals["active"]
        ),
        "active_elements": int(totals["active"]),
        "parameter_elements": int(totals["elements"]),
    }
