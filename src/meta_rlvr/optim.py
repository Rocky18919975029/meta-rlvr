from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from .config import FastOptimizerConfig
from .functional import ParameterDict


@dataclass(frozen=True)
class FastOptimizerState:
    step: int
    first_moment: ParameterDict
    second_moment: ParameterDict


def initial_fast_optimizer_state(
    parameters: Mapping[str, Tensor],
    config: FastOptimizerConfig,
) -> FastOptimizerState:
    if not parameters:
        raise ValueError("Fast parameter mapping cannot be empty.")
    if config.name == "sgd":
        return FastOptimizerState(step=0, first_moment={}, second_moment={})
    if config.name != "adamw":
        raise ValueError(f"Unsupported fast optimizer: {config.name}")
    return FastOptimizerState(
        step=0,
        first_moment={name: torch.zeros_like(value) for name, value in parameters.items()},
        second_moment={name: torch.zeros_like(value) for name, value in parameters.items()},
    )


def fast_optimizer_step(
    parameters: Mapping[str, Tensor],
    gradients: Mapping[str, Tensor],
    state: FastOptimizerState,
    config: FastOptimizerConfig,
) -> tuple[ParameterDict, FastOptimizerState]:
    if parameters.keys() != gradients.keys():
        raise ValueError("Fast parameters and gradients must have identical keys.")
    if any(gradient is None for gradient in gradients.values()):
        raise ValueError("Every fast parameter must receive a gradient.")

    if config.name == "sgd":
        updated = {
            name: parameter
            - config.learning_rate
            * (gradient + config.weight_decay * parameter)
            for name, (parameter, gradient) in (
                (key, (parameters[key], gradients[key])) for key in parameters
            )
        }
        return updated, FastOptimizerState(
            step=state.step + 1,
            first_moment={},
            second_moment={},
        )

    if config.name != "adamw":
        raise ValueError(f"Unsupported fast optimizer: {config.name}")
    if state.first_moment.keys() != parameters.keys():
        raise ValueError("Adam first-moment state does not match fast parameters.")
    if state.second_moment.keys() != parameters.keys():
        raise ValueError("Adam second-moment state does not match fast parameters.")

    step = state.step + 1
    first_moment: ParameterDict = {}
    second_moment: ParameterDict = {}
    updated: ParameterDict = {}
    for name, parameter in parameters.items():
        gradient = gradients[name]
        first = (
            config.beta1 * state.first_moment[name]
            + (1.0 - config.beta1) * gradient
        )
        second = (
            config.beta2 * state.second_moment[name]
            + (1.0 - config.beta2) * gradient.square()
        )
        first_moment[name] = first
        second_moment[name] = second

        first_hat = first / (1.0 - config.beta1**step)
        second_hat = second / (1.0 - config.beta2**step)
        decayed = parameter * (1.0 - config.learning_rate * config.weight_decay)
        updated[name] = decayed - config.learning_rate * first_hat / (
            torch.sqrt(second_hat) + config.epsilon
        )

    return updated, FastOptimizerState(
        step=step,
        first_moment=first_moment,
        second_moment=second_moment,
    )

