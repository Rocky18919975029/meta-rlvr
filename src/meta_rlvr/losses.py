from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import AdvantageConfig, ConfidenceLossConfig, GRPOLossConfig


@dataclass(frozen=True)
class ConfidenceLossOutput:
    loss: Tensor
    bce: Tensor
    ranking: Tensor
    num_positive: int
    num_negative: int


@dataclass(frozen=True)
class GRPOLossOutput:
    loss: Tensor
    policy_loss: Tensor
    mean_kl: Tensor
    clip_fraction: Tensor


def _validate_vector(name: str, value: Tensor, *, minimum_size: int = 1) -> None:
    if value.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if value.numel() < minimum_size:
        raise ValueError(f"{name} must contain at least {minimum_size} values.")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point.")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values.")


def confidence_losses(
    logits: Tensor,
    labels: Tensor,
    config: ConfidenceLossConfig,
) -> ConfidenceLossOutput:
    _validate_vector("logits", logits)
    _validate_vector("labels", labels)
    if logits.shape != labels.shape:
        raise ValueError("logits and labels must have identical shapes.")
    if torch.any((labels != 0) & (labels != 1)):
        raise ValueError("Confidence labels must contain only 0 and 1.")

    bce = F.binary_cross_entropy_with_logits(logits, labels)
    positive = labels == 1
    negative = labels == 0
    num_positive = int(positive.sum().item())
    num_negative = int(negative.sum().item())

    if num_positive == 0 or num_negative == 0:
        ranking = logits.sum() * 0.0
    else:
        differences = logits[positive][:, None] - logits[negative][None, :]
        ranking = -F.logsigmoid(differences).mean()

    loss = (
        config.bce_coefficient * bce
        + config.ranking_coefficient * ranking
    )
    return ConfidenceLossOutput(
        loss=loss,
        bce=bce,
        ranking=ranking,
        num_positive=num_positive,
        num_negative=num_negative,
    )


def group_advantages(rewards: Tensor, config: AdvantageConfig) -> Tensor:
    _validate_vector("rewards", rewards, minimum_size=2)
    k = rewards.numel()

    stats_rewards = rewards if config.differentiate_group_stats else rewards.detach()
    if config.baseline == "group_mean":
        centered = rewards - stats_rewards.mean()
    elif config.baseline == "leave_one_out":
        baselines = (stats_rewards.sum() - stats_rewards) / (k - 1)
        centered = rewards - baselines
    elif config.baseline == "none":
        centered = rewards
    else:
        raise ValueError(f"Unsupported baseline: {config.baseline}")

    if config.scale == "group_std":
        scale_source = centered if config.differentiate_group_stats else centered.detach()
        # Match GRPO's sample standard deviation (torch.std correction=1),
        # while placing epsilon inside sqrt so equal confidences have a finite
        # meta-gradient.
        variance = scale_source.square().sum() / (k - 1)
        epsilon = torch.as_tensor(
            config.std_epsilon, dtype=variance.dtype, device=variance.device
        )
        std = torch.sqrt(variance + epsilon.square())
        advantages = centered / std
    elif config.scale == "floored_group_std":
        scale_source = centered if config.differentiate_group_stats else centered.detach()
        variance = scale_source.square().sum() / (k - 1)
        floor = torch.as_tensor(
            config.std_floor, dtype=variance.dtype, device=variance.device
        )
        std = torch.sqrt(torch.maximum(variance, floor.square()))
        advantages = centered / std
    elif config.scale in ("center_only", "none"):
        advantages = centered
    else:
        raise ValueError(f"Unsupported scale mode: {config.scale}")

    probabilities = rewards
    if config.group_gate == "max_confidence":
        if torch.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("max_confidence gate requires rewards in [0, 1].")
        advantages = advantages * probabilities.max()
    elif config.group_gate == "probability_any":
        if torch.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("probability_any gate requires rewards in [0, 1].")
        advantages = advantages * (1.0 - torch.prod(1.0 - probabilities))
    elif config.group_gate != "none":
        raise ValueError(f"Unsupported group gate: {config.group_gate}")

    if not torch.isfinite(advantages).all():
        raise ValueError("Advantage computation produced non-finite values.")
    return advantages


def grpo_policy_loss(
    current_logprobs: Tensor,
    old_logprobs: Tensor,
    completion_mask: Tensor,
    advantages: Tensor,
    config: GRPOLossConfig,
    *,
    reference_logprobs: Tensor | None = None,
) -> GRPOLossOutput:
    if current_logprobs.ndim != 2:
        raise ValueError("current_logprobs must have shape [K, T].")
    if old_logprobs.shape != current_logprobs.shape:
        raise ValueError("old_logprobs must match current_logprobs.")
    if completion_mask.shape != current_logprobs.shape:
        raise ValueError("completion_mask must match current_logprobs.")
    if completion_mask.dtype != torch.bool:
        raise TypeError("completion_mask must be torch.bool.")
    if advantages.shape != (current_logprobs.shape[0],):
        raise ValueError("advantages must have shape [K].")
    if torch.any(completion_mask.sum(dim=1) == 0):
        raise ValueError("Every response must contain at least one active token.")
    for name, value in (
        ("current_logprobs", current_logprobs),
        ("old_logprobs", old_logprobs),
        ("advantages", advantages),
    ):
        if not value.is_floating_point():
            raise TypeError(f"{name} must be floating point.")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values.")

    token_advantages = advantages[:, None]
    if config.use_importance_ratio:
        log_ratio = current_logprobs - old_logprobs
        ratios = torch.exp(log_ratio)
        unclipped = ratios * token_advantages
        if config.use_clipping:
            clipped_ratios = torch.clamp(
                ratios,
                min=1.0 - config.clip_epsilon_low,
                max=1.0 + config.clip_epsilon_high,
            )
            clipped = clipped_ratios * token_advantages
            surrogate = torch.minimum(unclipped, clipped)
            clipped_tokens = (unclipped != clipped) & completion_mask
        else:
            surrogate = unclipped
            clipped_tokens = torch.zeros_like(completion_mask)
    else:
        surrogate = current_logprobs * token_advantages
        clipped_tokens = torch.zeros_like(completion_mask)

    if config.kl_coefficient > 0:
        if reference_logprobs is None:
            raise ValueError("Positive KL coefficient requires reference_logprobs.")
        if reference_logprobs.shape != current_logprobs.shape:
            raise ValueError("reference_logprobs must match current_logprobs.")
        if not torch.isfinite(reference_logprobs).all():
            raise ValueError("reference_logprobs contains non-finite values.")
        reference_log_ratio = reference_logprobs - current_logprobs
        kl = torch.exp(reference_log_ratio) - reference_log_ratio - 1.0
    else:
        kl = torch.zeros_like(current_logprobs)

    token_objective = surrogate - config.kl_coefficient * kl
    mask = completion_mask.to(token_objective.dtype)
    token_loss = -token_objective * mask

    if config.token_normalization == "per_response":
        loss = (token_loss.sum(dim=1) / mask.sum(dim=1)).mean()
    elif config.token_normalization == "global_tokens":
        loss = token_loss.sum() / mask.sum()
    elif config.token_normalization == "sequence_sum":
        loss = token_loss.sum(dim=1).mean()
    else:
        raise ValueError(
            f"Unsupported token normalization: {config.token_normalization}"
        )

    policy_token_loss = -surrogate * mask
    if config.token_normalization == "per_response":
        policy_loss = (
            policy_token_loss.sum(dim=1) / mask.sum(dim=1)
        ).mean()
    elif config.token_normalization == "global_tokens":
        policy_loss = policy_token_loss.sum() / mask.sum()
    else:
        policy_loss = policy_token_loss.sum(dim=1).mean()

    mean_kl = (kl * mask).sum() / mask.sum()
    clip_fraction = clipped_tokens.to(current_logprobs.dtype).sum() / mask.sum()
    return GRPOLossOutput(
        loss=loss,
        policy_loss=policy_loss,
        mean_kl=mean_kl,
        clip_fraction=clip_fraction,
    )
