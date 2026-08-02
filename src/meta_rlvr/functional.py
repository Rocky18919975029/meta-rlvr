from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager

import torch
from torch import Tensor, nn
from torch.func import functional_call
from torch.utils.checkpoint import checkpoint

from .types import RolloutGroup


ParameterDict = dict[str, Tensor]


def trainable_parameter_state(
    model: nn.Module,
    *,
    required_name_substring: str | None = None,
) -> ParameterDict:
    state: ParameterDict = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if required_name_substring is not None and required_name_substring not in name:
            raise ValueError(
                f"Unexpected trainable policy parameter {name!r}; expected every "
                f"trainable parameter name to contain {required_name_substring!r}."
            )
        state[name] = parameter.detach().clone().requires_grad_(True)
    if not state:
        raise ValueError("Policy model has no trainable fast parameters.")
    return state


def clone_fast_parameters(parameters: Mapping[str, Tensor]) -> ParameterDict:
    if not parameters:
        raise ValueError("Fast parameter mapping cannot be empty.")
    return {
        name: value.detach().clone().requires_grad_(True)
        for name, value in parameters.items()
    }


def token_logprobs(
    model: nn.Module,
    group: RolloutGroup,
    *,
    fast_parameters: Mapping[str, Tensor] | None = None,
    row_start: int = 0,
    row_end: int | None = None,
    activation_checkpointing: bool = False,
) -> Tensor:
    if row_end is None:
        row_end = group.group_size
    if row_start < 0 or row_end <= row_start or row_end > group.group_size:
        raise ValueError("Invalid rollout row interval.")
    input_ids = group.input_ids[row_start:row_end]
    attention_mask = group.attention_mask[row_start:row_end]
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": False,
        "return_dict": True,
    }

    def forward_with_fast_parameters(*parameter_values: Tensor) -> Tensor:
        if fast_parameters is None:
            if parameter_values:
                raise RuntimeError("Unexpected functional parameter values.")
            outputs = model(**kwargs)
        else:
            parameter_names = tuple(fast_parameters)
            if len(parameter_values) != len(parameter_names):
                raise RuntimeError("Functional parameter count changed during forward.")
            state = dict(zip(parameter_names, parameter_values, strict=True))
            outputs = functional_call(
                model,
                state,
                args=(),
                kwargs=kwargs,
                strict=False,
            )

        logits = getattr(outputs, "logits", None)
        if logits is None:
            raise TypeError("Policy model must return logits.")
        if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
            raise ValueError(
                f"Unexpected policy logits shape {tuple(logits.shape)}; expected "
                f"[{input_ids.shape[0]}, {input_ids.shape[1]}, vocabulary]."
            )

        next_token_logits = logits[:, :-1, :].float()
        next_tokens = input_ids[:, 1:]
        selected_logits = next_token_logits.gather(
            -1, next_tokens.unsqueeze(-1)
        ).squeeze(-1)
        return selected_logits - torch.logsumexp(next_token_logits, dim=-1)

    if fast_parameters is None:
        if activation_checkpointing:
            raise ValueError(
                "activation_checkpointing requires explicit fast parameters."
            )
        selected = forward_with_fast_parameters()
    else:
        if not fast_parameters:
            raise ValueError("fast_parameters cannot be empty.")
        values = tuple(fast_parameters.values())
        if activation_checkpointing:
            selected = checkpoint(
                forward_with_fast_parameters,
                *values,
                use_reentrant=False,
            )
        else:
            selected = forward_with_fast_parameters(*values)

    expected = group.completion_mask[row_start:row_end]
    if selected.shape != expected.shape:
        raise RuntimeError("Selected token log-probability shape is inconsistent.")
    return selected


def chunked_token_logprobs(
    model: nn.Module,
    group: RolloutGroup,
    *,
    fast_parameters: Mapping[str, Tensor],
    micro_batch_size: int,
    activation_checkpointing: bool,
) -> Tensor:
    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive.")
    chunks = [
        token_logprobs(
            model,
            group,
            fast_parameters=fast_parameters,
            row_start=start,
            row_end=min(start + micro_batch_size, group.group_size),
            activation_checkpointing=activation_checkpointing,
        )
        for start in range(0, group.group_size, micro_batch_size)
    ]
    return torch.cat(chunks, dim=0)


@contextmanager
def materialized_fast_parameters(
    model: nn.Module,
    fast_parameters: Mapping[str, Tensor],
):
    """Temporarily copy fast parameters into a module for non-differentiable generation."""

    if not fast_parameters:
        raise ValueError("fast_parameters cannot be empty.")
    named_parameters = dict(model.named_parameters())
    unknown = set(fast_parameters).difference(named_parameters)
    if unknown:
        raise KeyError(f"Unknown fast parameter names: {sorted(unknown)}")

    backups = {
        name: named_parameters[name].detach().clone()
        for name in fast_parameters
    }
    try:
        with torch.no_grad():
            for name, value in fast_parameters.items():
                target = named_parameters[name]
                if target.shape != value.shape:
                    raise ValueError(
                        f"Fast parameter {name!r} has shape {tuple(value.shape)}, "
                        f"expected {tuple(target.shape)}."
                    )
                target.copy_(value.detach())
        yield
    finally:
        with torch.no_grad():
            for name, backup in backups.items():
                named_parameters[name].copy_(backup)
