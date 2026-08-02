from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .third_party.dapo_math import compute_score


@dataclass(frozen=True)
class VerificationBatch:
    """Official policy rewards and binary correctness labels."""

    rewards: Tensor
    correctness: Tensor


class DAPOMathVerifier:
    """Strict wrapper around the official DAPO math verifier."""

    def __init__(self, *, strict_box_verify: bool = False) -> None:
        self.strict_box_verify = strict_box_verify

    def __call__(
        self,
        responses: tuple[str, ...],
        ground_truth: str,
        *,
        device: torch.device,
    ) -> VerificationBatch:
        if not responses:
            raise ValueError("Verifier requires at least one response.")
        if not isinstance(ground_truth, str) or not ground_truth:
            raise ValueError("ground_truth must be a non-empty string.")
        rewards = []
        correctness = []
        for response in responses:
            if not isinstance(response, str):
                raise TypeError("Every response must be a string.")
            result = compute_score(
                solution_str=response,
                ground_truth=ground_truth,
                strict_box_verify=self.strict_box_verify,
            )
            if set(result) != {"score", "acc", "pred"}:
                raise ValueError("Unexpected DAPO verifier result schema.")
            rewards.append(float(result["score"]))
            correctness.append(float(bool(result["acc"])))
        return VerificationBatch(
            rewards=torch.tensor(rewards, dtype=torch.float32, device=device),
            correctness=torch.tensor(
                correctness, dtype=torch.float32, device=device
            ),
        )
