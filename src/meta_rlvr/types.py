from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor


@dataclass(frozen=True)
class RolloutGroup:
    """One problem and its group of sampled completions.

    ``completion_mask`` and token log-probabilities align with next-token
    predictions, so their shape is ``[group_size, sequence_length - 1]``.
    """

    input_ids: Tensor
    attention_mask: Tensor
    completion_mask: Tensor
    old_logprobs: Tensor
    texts: tuple[str, ...]
    verifier_rewards: Tensor | None = None
    correctness_labels: Tensor | None = None
    reference_logprobs: Tensor | None = None

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [K, L].")
        if self.input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must be an integer tensor.")

        k, length = self.input_ids.shape
        if k < 2:
            raise ValueError("A GRPO rollout group requires at least two responses.")
        if length < 2:
            raise ValueError("Each sequence must contain at least two tokens.")
        if self.attention_mask.shape != (k, length):
            raise ValueError("attention_mask must have the same shape as input_ids.")
        if self.completion_mask.shape != (k, length - 1):
            raise ValueError("completion_mask must have shape [K, L - 1].")
        if self.old_logprobs.shape != (k, length - 1):
            raise ValueError("old_logprobs must have shape [K, L - 1].")
        if len(self.texts) != k:
            raise ValueError("texts must contain exactly one completion per row.")

        if self.attention_mask.dtype != torch.bool:
            raise TypeError("attention_mask must be torch.bool.")
        if self.completion_mask.dtype != torch.bool:
            raise TypeError("completion_mask must be torch.bool.")
        if torch.any(self.completion_mask & ~self.attention_mask[:, 1:]):
            raise ValueError("completion_mask cannot select padding tokens.")
        if torch.any(self.completion_mask.sum(dim=1) == 0):
            raise ValueError("Every response must contain at least one completion token.")

        self._validate_floating_tensor("old_logprobs", self.old_logprobs)
        if self.reference_logprobs is not None:
            if self.reference_logprobs.shape != (k, length - 1):
                raise ValueError("reference_logprobs must have shape [K, L - 1].")
            self._validate_floating_tensor(
                "reference_logprobs", self.reference_logprobs
            )
        if self.verifier_rewards is not None:
            if self.verifier_rewards.shape != (k,):
                raise ValueError("verifier_rewards must have shape [K].")
            self._validate_floating_tensor(
                "verifier_rewards", self.verifier_rewards
            )
            if torch.any(
                (self.verifier_rewards != -1) & (self.verifier_rewards != 1)
            ):
                raise ValueError("verifier_rewards must contain only -1 and 1.")
        if self.correctness_labels is not None:
            if self.correctness_labels.shape != (k,):
                raise ValueError("correctness_labels must have shape [K].")
            self._validate_floating_tensor(
                "correctness_labels", self.correctness_labels
            )
            if torch.any(
                (self.correctness_labels != 0) & (self.correctness_labels != 1)
            ):
                raise ValueError("correctness_labels must contain only 0 and 1.")
        if (self.verifier_rewards is None) != (self.correctness_labels is None):
            raise ValueError(
                "verifier_rewards and correctness_labels must be set together."
            )
        if self.verifier_rewards is not None and not torch.equal(
            self.verifier_rewards, 2.0 * self.correctness_labels - 1.0
        ):
            raise ValueError(
                "verifier_rewards must equal 2 * correctness_labels - 1."
            )

        devices = {
            self.input_ids.device,
            self.attention_mask.device,
            self.completion_mask.device,
            self.old_logprobs.device,
        }
        if self.reference_logprobs is not None:
            devices.add(self.reference_logprobs.device)
        if self.verifier_rewards is not None:
            devices.add(self.verifier_rewards.device)
        if self.correctness_labels is not None:
            devices.add(self.correctness_labels.device)
        if len(devices) != 1:
            raise ValueError("All rollout tensors must be on the same device.")

    @staticmethod
    def _validate_floating_tensor(name: str, value: Tensor) -> None:
        if not value.is_floating_point():
            raise TypeError(f"{name} must be floating point.")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values.")

    @property
    def group_size(self) -> int:
        return self.input_ids.shape[0]

    def with_verification(
        self,
        verifier_rewards: Tensor,
        correctness_labels: Tensor,
    ) -> "RolloutGroup":
        return replace(
            self,
            verifier_rewards=verifier_rewards,
            correctness_labels=correctness_labels,
        )

    def with_reference_logprobs(self, reference_logprobs: Tensor) -> "RolloutGroup":
        return replace(self, reference_logprobs=reference_logprobs)
