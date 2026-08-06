from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


BaselineMode = Literal["group_mean", "leave_one_out", "none"]
ScaleMode = Literal["group_std", "center_only", "floored_group_std", "none"]
GroupGateMode = Literal["none", "max_confidence", "probability_any"]
TokenNormalization = Literal["per_response", "global_tokens", "sequence_sum"]
FastOptimizerName = Literal["sgd", "adamw"]
MetaGradientMode = Literal["first_order", "second_order"]


@dataclass(frozen=True)
class AdvantageConfig:
    baseline: BaselineMode = "group_mean"
    scale: ScaleMode = "group_std"
    std_epsilon: float = 1e-4
    std_floor: float | None = None
    group_gate: GroupGateMode = "none"
    differentiate_group_stats: bool = True

    def __post_init__(self) -> None:
        if self.std_epsilon <= 0:
            raise ValueError("std_epsilon must be positive.")
        if self.scale == "floored_group_std":
            if self.std_floor is None or self.std_floor <= 0:
                raise ValueError("floored_group_std requires a positive std_floor.")
        elif self.std_floor is not None:
            raise ValueError("std_floor is only valid for floored_group_std.")


@dataclass(frozen=True)
class GRPOLossConfig:
    use_importance_ratio: bool = True
    use_clipping: bool = True
    clip_epsilon_low: float = 0.2
    clip_epsilon_high: float = 0.2
    kl_coefficient: float = 0.0
    token_normalization: TokenNormalization = "per_response"

    def __post_init__(self) -> None:
        if self.use_clipping and not self.use_importance_ratio:
            raise ValueError("PPO clipping requires importance ratios.")
        if self.clip_epsilon_low < 0 or self.clip_epsilon_high < 0:
            raise ValueError("Clipping epsilons must be non-negative.")
        if self.kl_coefficient < 0:
            raise ValueError("kl_coefficient must be non-negative.")


@dataclass(frozen=True)
class FastOptimizerConfig:
    name: FastOptimizerName = "adamw"
    learning_rate: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("Adam betas must be in [0, 1).")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")


@dataclass(frozen=True)
class InnerLoopConfig:
    num_iterations: int = 4
    meta_gradient_mode: MetaGradientMode = "first_order"
    advantage: AdvantageConfig = field(default_factory=AdvantageConfig)
    grpo: GRPOLossConfig = field(default_factory=GRPOLossConfig)
    optimizer: FastOptimizerConfig = field(default_factory=FastOptimizerConfig)

    def __post_init__(self) -> None:
        if self.num_iterations <= 0:
            raise ValueError("num_iterations must be positive.")


@dataclass(frozen=True)
class ConfidenceLossConfig:
    bce_coefficient: float = 1.0
    ranking_coefficient: float = 1.0

    def __post_init__(self) -> None:
        if self.bce_coefficient < 0 or self.ranking_coefficient < 0:
            raise ValueError("Confidence loss coefficients must be non-negative.")


@dataclass(frozen=True)
class MetaLossConfig:
    meta_coefficient: float = 1.0
    confidence: ConfidenceLossConfig = field(default_factory=ConfidenceLossConfig)
    token_meta_coefficient: float = 0.0

    def __post_init__(self) -> None:
        if self.meta_coefficient < 0 or self.token_meta_coefficient < 0:
            raise ValueError("Meta loss coefficients must be non-negative.")
