from __future__ import annotations

import inspect
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext

import torch
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.func import functional_call
from torch.utils.checkpoint import checkpoint

from .types import RolloutGroup


ParameterDict = dict[str, Tensor]
_FORCE_SDPA_MATH_ATTRIBUTE = "_meta_rlvr_force_sdpa_math"


def enable_sdpa_math_policy_forwards(model: nn.Module) -> None:
    """Make policy forwards use the SDPA math backend for exact JVP parity."""

    setattr(model, _FORCE_SDPA_MATH_ATTRIBUTE, True)


def policy_forward_context(model: nn.Module):
    if getattr(model, _FORCE_SDPA_MATH_ATTRIBUTE, False):
        return sdpa_kernel(backends=[SDPBackend.MATH])
    return nullcontext()


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


def sequence_microbatches(
    group: RolloutGroup,
    *,
    max_sequences: int,
    max_tokens: int | None,
) -> tuple[tuple[int, ...], ...]:
    """Group rows by dense token cost while preserving exact sample semantics."""

    if max_sequences <= 0:
        raise ValueError("max_sequences must be positive.")
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be positive when provided.")
    lengths = [int(value) for value in group.attention_mask.sum(dim=1).tolist()]
    if max_tokens is None:
        return tuple(
            tuple(range(start, min(start + max_sequences, group.group_size)))
            for start in range(0, group.group_size, max_sequences)
        )
    over_budget = [length for length in lengths if length > max_tokens]
    if over_budget:
        raise ValueError(
            f"A sequence length {max(over_budget)} exceeds the token budget "
            f"{max_tokens}."
        )

    # Longest-first packing prevents a short response from inheriting the dense
    # padding cost of an unrelated long response.  Results are restored to the
    # original group order after each model forward.
    ordered = sorted(range(group.group_size), key=lengths.__getitem__, reverse=True)
    batches: list[tuple[int, ...]] = []
    current: list[int] = []
    current_max_length = 0
    for index in ordered:
        candidate_max_length = max(current_max_length, lengths[index])
        candidate_size = len(current) + 1
        if current and (
            candidate_size > max_sequences
            or candidate_max_length * candidate_size > max_tokens
        ):
            batches.append(tuple(current))
            current = []
            current_max_length = 0
        current.append(index)
        current_max_length = max(current_max_length, lengths[index])
    if current:
        batches.append(tuple(current))
    if sorted(index for batch in batches for index in batch) != list(
        range(group.group_size)
    ):
        raise RuntimeError("Token microbatch packing lost or duplicated rows.")
    return tuple(batches)


def _supports_logits_to_keep(model: nn.Module) -> bool:
    parameters = inspect.signature(model.forward).parameters.values()
    return any(
        parameter.name == "logits_to_keep"
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def token_logprobs(
    model: nn.Module,
    group: RolloutGroup,
    *,
    fast_parameters: Mapping[str, Tensor] | None = None,
    row_start: int = 0,
    row_end: int | None = None,
    row_indices: tuple[int, ...] | None = None,
    activation_checkpointing: bool = False,
    logprob_position_chunk_size: int | None = None,
) -> Tensor:
    if logprob_position_chunk_size is not None and logprob_position_chunk_size <= 0:
        raise ValueError("logprob_position_chunk_size must be positive.")
    if row_indices is not None:
        if row_start != 0 or row_end is not None:
            raise ValueError("row_indices cannot be combined with a row interval.")
        if not row_indices or len(set(row_indices)) != len(row_indices):
            raise ValueError("row_indices must contain distinct rows.")
        if min(row_indices) < 0 or max(row_indices) >= group.group_size:
            raise ValueError("row_indices contains an out-of-range row.")
        row_selector = torch.tensor(
            row_indices,
            dtype=torch.long,
            device=group.input_ids.device,
        )
        input_ids = group.input_ids.index_select(0, row_selector)
        attention_mask = group.attention_mask.index_select(0, row_selector)
        expected_full = group.completion_mask.index_select(0, row_selector)
    else:
        if row_end is None:
            row_end = group.group_size
        if row_start < 0 or row_end <= row_start or row_end > group.group_size:
            raise ValueError("Invalid rollout row interval.")
        input_ids = group.input_ids[row_start:row_end]
        attention_mask = group.attention_mask[row_start:row_end]
        expected_full = group.completion_mask[row_start:row_end]

    sequence_length = int(attention_mask.sum(dim=1).max().item())
    if sequence_length < 2:
        raise ValueError("A policy microbatch must contain at least two tokens.")
    input_ids = input_ids[:, :sequence_length]
    attention_mask = attention_mask[:, :sequence_length]
    expected = expected_full[:, : sequence_length - 1]
    prediction_positions = torch.nonzero(expected.any(dim=0), as_tuple=False).flatten()
    if prediction_positions.numel() == 0:
        raise ValueError("A policy microbatch contains no completion positions.")
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": False,
        "return_dict": True,
    }
    uses_selected_logits = _supports_logits_to_keep(model)
    if uses_selected_logits:
        kwargs["logits_to_keep"] = prediction_positions

    def forward_with_fast_parameters(*parameter_values: Tensor) -> Tensor:
        # Keep this context inside the checkpointed function.  Non-reentrant
        # activation checkpointing executes this function again during backward;
        # an outer context would not cover that recomputation.
        with policy_forward_context(model):
            if fast_parameters is None:
                if parameter_values:
                    raise RuntimeError("Unexpected functional parameter values.")
                outputs = model(**kwargs)
            else:
                parameter_names = tuple(fast_parameters)
                if len(parameter_values) != len(parameter_names):
                    raise RuntimeError(
                        "Functional parameter count changed during forward."
                    )
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
        if logits.ndim != 3 or logits.shape[0] != input_ids.shape[0]:
            raise ValueError(f"Unexpected policy logits shape {tuple(logits.shape)}.")
        if logits.shape[1] == input_ids.shape[1]:
            completion_logits = logits.index_select(1, prediction_positions)
        elif uses_selected_logits and logits.shape[1] == prediction_positions.numel():
            completion_logits = logits
        else:
            raise ValueError(
                "Policy logits must cover either the full sequence or exactly "
                "the requested completion positions."
            )

        next_tokens = input_ids.index_select(1, prediction_positions + 1)
        position_chunk_size = (
            completion_logits.shape[1]
            if logprob_position_chunk_size is None
            else logprob_position_chunk_size
        )
        selected_logprob_chunks = []
        for start in range(0, completion_logits.shape[1], position_chunk_size):
            end = min(start + position_chunk_size, completion_logits.shape[1])
            logits_chunk = completion_logits[:, start:end].float()
            token_chunk = next_tokens[:, start:end]
            selected_logits = logits_chunk.gather(
                -1, token_chunk.unsqueeze(-1)
            ).squeeze(-1)
            selected_logprob_chunks.append(
                selected_logits - torch.logsumexp(logits_chunk, dim=-1)
            )
        return torch.cat(selected_logprob_chunks, dim=1)

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

    if selected.shape != (input_ids.shape[0], prediction_positions.numel()):
        raise RuntimeError("Selected token log-probability shape is inconsistent.")
    aligned = selected.new_zeros(expected_full.shape)
    return torch.index_copy(aligned, 1, prediction_positions, selected)


def chunked_token_logprobs(
    model: nn.Module,
    group: RolloutGroup,
    *,
    fast_parameters: Mapping[str, Tensor],
    micro_batch_size: int,
    max_tokens_per_micro_batch: int | None = None,
    activation_checkpointing: bool,
    show_progress: bool = False,
    progress_description: str = "policy log-probabilities",
) -> Tensor:
    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive.")
    row_batches = sequence_microbatches(
        group,
        max_sequences=micro_batch_size,
        max_tokens=max_tokens_per_micro_batch,
    )
    starts = range(len(row_batches))
    if show_progress:
        from tqdm.auto import tqdm

        starts = tqdm(
            starts,
            total=len(row_batches),
            desc=progress_description,
            unit="microbatch",
            leave=True,
        )
    rows: list[Tensor | None] = [None] * group.group_size
    for batch_index in starts:
        row_indices = row_batches[batch_index]
        output = token_logprobs(
            model,
            group,
            fast_parameters=fast_parameters,
            row_indices=row_indices,
            activation_checkpointing=activation_checkpointing,
        )
        for offset, row_index in enumerate(row_indices):
            rows[row_index] = output[offset : offset + 1]
    if any(row is None for row in rows):
        raise RuntimeError("Policy token batching returned an incomplete group.")
    return torch.cat([row for row in rows if row is not None], dim=0)


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
        name: named_parameters[name].detach().clone() for name in fast_parameters
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
