from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import (
    AdvantageConfig,
    GRPOLossConfig,
    InnerLoopConfig,
    MetaLossConfig,
    TokenMetaGradientMode,
)
from .functional import (
    ParameterDict,
    chunked_token_logprobs,
    clone_fast_parameters,
    sequence_microbatches,
    token_logprobs,
)
from .losses import (
    ConfidenceLossOutput,
    GRPOLossOutput,
    bounded_token_credits,
    confidence_losses,
    group_advantages,
    grpo_policy_loss,
    token_grpo_policy_loss,
)
from .optim import (
    FastOptimizerState,
    fast_optimizer_step,
    initial_fast_optimizer_state,
)
from .types import RolloutGroup


@dataclass(frozen=True)
class TaskAdaptation:
    fast_parameters: ParameterDict
    optimizer_state: FastOptimizerState
    confidence_logits: Tensor
    confidence_probabilities: Tensor
    confidence_loss: ConfidenceLossOutput | None
    inner_losses: tuple[GRPOLossOutput, ...]


@dataclass(frozen=True)
class TaskOuterLoss:
    loss: Tensor
    meta_grpo: GRPOLossOutput
    adaptation: TaskAdaptation
    query_advantages: Tensor


@dataclass(frozen=True)
class TokenTaskAdaptation:
    fast_parameters: ParameterDict
    optimizer_state: FastOptimizerState
    token_confidence_logits: Tensor
    token_credits: Tensor
    inner_losses: tuple[GRPOLossOutput, ...]


@dataclass(frozen=True)
class TokenTaskOuterLoss:
    loss: Tensor
    meta_objective: Tensor
    meta_grpo: GRPOLossOutput
    token_confidence_logits: Tensor
    token_credits: Tensor
    inner_grpo: GRPOLossOutput
    query_advantages: Tensor
    adaptation: TokenTaskAdaptation | None


@dataclass(frozen=True)
class TokenGradientAlignmentContext:
    """Policy-only quantities reused by every confidence outer iteration."""

    support_logprobs: Tensor
    support_directional_logprobs: Tensor
    query_grpo: GRPOLossOutput
    query_advantages: Tensor

    def to(self, device: torch.device | str) -> TokenGradientAlignmentContext:
        return TokenGradientAlignmentContext(
            support_logprobs=self.support_logprobs.to(device),
            support_directional_logprobs=(self.support_directional_logprobs.to(device)),
            query_grpo=GRPOLossOutput(
                loss=self.query_grpo.loss.to(device),
                policy_loss=self.query_grpo.policy_loss.to(device),
                mean_kl=self.query_grpo.mean_kl.to(device),
                clip_fraction=self.query_grpo.clip_fraction.to(device),
            ),
            query_advantages=self.query_advantages.to(device),
        )


class BilevelGRPO:
    """Differentiable per-problem adapter adaptation and outer meta objective."""

    def __init__(
        self,
        policy: nn.Module,
        confidence_model: nn.Module,
        inner_config: InnerLoopConfig,
        meta_config: MetaLossConfig,
        query_advantage_config: AdvantageConfig,
        query_grpo_config: GRPOLossConfig,
        *,
        policy_micro_batch_size: int = 4,
        first_order_vjp_forward_batch_size: int = 1,
        confidence_micro_batch_size: int = 4,
        policy_max_tokens_per_micro_batch: int | None = None,
        confidence_max_tokens_per_micro_batch: int | None = None,
        token_jvp_response_micro_batch_size: int = 4,
        token_jvp_logprob_position_chunk_size: int = 256,
        token_credit_max: float = 1.0,
        token_meta_gradient_mode: TokenMetaGradientMode = "gradient_alignment",
    ) -> None:
        if policy_micro_batch_size <= 0 or confidence_micro_batch_size <= 0:
            raise ValueError("Micro-batch sizes must be positive.")
        if first_order_vjp_forward_batch_size <= 0:
            raise ValueError("First-order VJP forward batch size must be positive.")
        if token_jvp_response_micro_batch_size <= 0:
            raise ValueError("Token JVP response micro-batch size must be positive.")
        if token_jvp_logprob_position_chunk_size <= 0:
            raise ValueError("Token JVP position chunk size must be positive.")
        if token_credit_max <= 0:
            raise ValueError("Maximum token-credit magnitude must be positive.")
        if token_meta_gradient_mode not in ("gradient_alignment", "unrolled"):
            raise ValueError("Unsupported token meta-gradient mode.")
        self.policy = policy
        self.confidence_model = confidence_model
        self.inner_config = inner_config
        self.meta_config = meta_config
        self.query_advantage_config = query_advantage_config
        self.query_grpo_config = query_grpo_config
        self.policy_micro_batch_size = policy_micro_batch_size
        self.first_order_vjp_forward_batch_size = first_order_vjp_forward_batch_size
        self.confidence_micro_batch_size = confidence_micro_batch_size
        self.policy_max_tokens_per_micro_batch = policy_max_tokens_per_micro_batch
        self.confidence_max_tokens_per_micro_batch = (
            confidence_max_tokens_per_micro_batch
        )
        self.token_jvp_response_micro_batch_size = token_jvp_response_micro_batch_size
        self.token_jvp_logprob_position_chunk_size = (
            token_jvp_logprob_position_chunk_size
        )
        self.token_credit_max = token_credit_max
        self.token_meta_gradient_mode = token_meta_gradient_mode

    @staticmethod
    def _equalize_distributed_batch_count(
        batches: list[list[tuple[int, int, int]]],
        *,
        device: torch.device,
    ) -> list[list[tuple[int, int, int]]]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            count = torch.tensor(len(batches), dtype=torch.int64, device=device)
            torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.MAX)
            target_count = int(count.item())
        else:
            target_count = len(batches)
        while len(batches) < target_count:
            split_index = max(
                range(len(batches)), key=lambda index: len(batches[index])
            )
            selected = batches[split_index]
            if len(selected) < 2:
                raise RuntimeError(
                    "Distributed confidence batching requires more non-empty "
                    "microbatches than local sequences can provide."
                )
            midpoint = len(selected) // 2
            batches[split_index : split_index + 1] = [
                selected[:midpoint],
                selected[midpoint:],
            ]
        return batches

    def _confidence_row_batches(
        self,
        supports: tuple[RolloutGroup, ...],
    ) -> list[list[tuple[int, int, int]]]:
        entries: list[tuple[int, int, int]] = []
        for problem_index, support in enumerate(supports):
            lengths = support.attention_mask.sum(dim=1).tolist()
            entries.extend(
                (problem_index, response_index, int(length))
                for response_index, length in enumerate(lengths)
            )
        entries.sort(key=lambda item: item[2], reverse=True)
        batches: list[list[tuple[int, int, int]]] = []
        current: list[tuple[int, int, int]] = []
        current_max_length = 0
        for entry in entries:
            candidate_max_length = max(current_max_length, entry[2])
            candidate_size = len(current) + 1
            token_limit_exceeded = (
                self.confidence_max_tokens_per_micro_batch is not None
                and candidate_max_length * candidate_size
                > self.confidence_max_tokens_per_micro_batch
            )
            if current and (
                candidate_size > self.confidence_micro_batch_size
                or token_limit_exceeded
            ):
                batches.append(current)
                current = []
                current_max_length = 0
            if (
                self.confidence_max_tokens_per_micro_batch is not None
                and entry[2] > self.confidence_max_tokens_per_micro_batch
            ):
                raise ValueError(
                    f"Confidence sequence length {entry[2]} exceeds token budget "
                    f"{self.confidence_max_tokens_per_micro_batch}."
                )
            current.append(entry)
            current_max_length = max(current_max_length, entry[2])
        if current:
            batches.append(current)
        return self._equalize_distributed_batch_count(
            batches,
            device=supports[0].input_ids.device,
        )

    def _confidence_logits_batch(
        self,
        supports: tuple[RolloutGroup, ...],
        *,
        differentiable: bool,
        show_progress: bool,
        progress_description: str,
    ) -> tuple[Tensor, ...]:
        if not supports:
            raise ValueError("Confidence scoring requires at least one support group.")
        devices = {support.device for support in supports}
        if len(devices) != 1:
            raise ValueError("Batched confidence groups must share one device.")
        batches = self._confidence_row_batches(supports)
        batch_indices = range(len(batches))
        if show_progress:
            from tqdm.auto import tqdm

            batch_indices = tqdm(
                batch_indices,
                total=len(batches),
                desc=progress_description,
                unit="microbatch",
                leave=True,
            )
        logits_by_problem: list[list[Tensor | None]] = [
            [None] * support.group_size for support in supports
        ]
        for batch_index in batch_indices:
            entries = batches[batch_index]
            max_length = max(entry[2] for entry in entries)
            input_rows = []
            mask_rows = []
            for problem_index, response_index, length in entries:
                support = supports[problem_index]
                input_row = support.input_ids[response_index, :length]
                mask_row = support.attention_mask[response_index, :length]
                if length < max_length:
                    input_row = torch.nn.functional.pad(
                        input_row, (0, max_length - length), value=0
                    )
                    mask_row = torch.nn.functional.pad(
                        mask_row, (0, max_length - length), value=False
                    )
                input_rows.append(input_row)
                mask_rows.append(mask_row)
            input_ids = torch.stack(input_rows)
            attention_mask = torch.stack(mask_rows)
            if differentiable:
                batch_logits = self.confidence_model(input_ids, attention_mask)
            else:
                with torch.no_grad():
                    batch_logits = self.confidence_model(input_ids, attention_mask)
            for offset, (problem_index, response_index, _) in enumerate(entries):
                logits_by_problem[problem_index][response_index] = batch_logits[
                    offset : offset + 1
                ]
        outputs = []
        for rows in logits_by_problem:
            if any(row is None for row in rows):
                raise RuntimeError(
                    "Confidence token batching returned incomplete logits."
                )
            outputs.append(torch.cat([row for row in rows if row is not None]))
        return tuple(outputs)

    def _confidence_logits(
        self,
        support: RolloutGroup,
        *,
        differentiable: bool,
        show_progress: bool,
        progress_description: str,
    ) -> Tensor:
        return self._confidence_logits_batch(
            (support,),
            differentiable=differentiable,
            show_progress=show_progress,
            progress_description=progress_description,
        )[0]

    def _token_confidence_logits_batch(
        self,
        supports: tuple[RolloutGroup, ...],
        *,
        differentiable: bool,
        show_progress: bool,
        progress_description: str,
    ) -> tuple[Tensor, ...]:
        if not supports:
            raise ValueError("Token confidence scoring requires support groups.")
        batches = self._confidence_row_batches(supports)
        batch_indices = range(len(batches))
        if show_progress:
            from tqdm.auto import tqdm

            batch_indices = tqdm(
                batch_indices,
                total=len(batches),
                desc=progress_description,
                unit="microbatch",
                leave=True,
            )
        logits_by_problem: list[list[Tensor | None]] = [
            [None] * support.group_size for support in supports
        ]
        for batch_index in batch_indices:
            entries = batches[batch_index]
            max_length = max(entry[2] for entry in entries)
            input_rows = []
            mask_rows = []
            for problem_index, response_index, length in entries:
                support = supports[problem_index]
                input_row = support.input_ids[response_index, :length]
                mask_row = support.attention_mask[response_index, :length]
                if length < max_length:
                    input_row = torch.nn.functional.pad(
                        input_row, (0, max_length - length), value=0
                    )
                    mask_row = torch.nn.functional.pad(
                        mask_row, (0, max_length - length), value=False
                    )
                input_rows.append(input_row)
                mask_rows.append(mask_row)
            input_ids = torch.stack(input_rows)
            attention_mask = torch.stack(mask_rows)
            if differentiable:
                batch_logits = self.confidence_model(
                    input_ids, attention_mask, output="token"
                )
            else:
                with torch.no_grad():
                    batch_logits = self.confidence_model(
                        input_ids, attention_mask, output="token"
                    )
            for offset, (problem_index, response_index, length) in enumerate(entries):
                width = supports[problem_index].completion_mask.shape[1]
                row = batch_logits[offset, : length - 1]
                logits_by_problem[problem_index][response_index] = (
                    torch.nn.functional.pad(row, (0, width - row.numel()))
                )
        outputs = []
        for rows in logits_by_problem:
            if any(row is None for row in rows):
                raise RuntimeError(
                    "Token confidence batching returned incomplete logits."
                )
            outputs.append(torch.stack([row for row in rows if row is not None]))
        return tuple(outputs)

    def confidence_supervision_loss(
        self,
        support: RolloutGroup,
        *,
        show_progress: bool = False,
        progress_description: str = "confidence supervision",
    ) -> ConfidenceLossOutput:
        if support.correctness_labels is None:
            raise ValueError(
                "Support correctness labels are required for confidence supervision."
            )
        logits = self._confidence_logits(
            support,
            differentiable=True,
            show_progress=show_progress,
            progress_description=progress_description,
        )
        return confidence_losses(
            logits,
            support.correctness_labels,
            self.meta_config.confidence,
        )

    def _group_normalization_weight(
        self,
        support: RolloutGroup,
        response_index: int,
    ) -> Tensor:
        token_count = support.completion_mask[response_index].sum()
        if self.inner_config.grpo.token_normalization == "global_tokens":
            return token_count / support.completion_mask.sum()
        return (
            torch.ones((), dtype=torch.float32, device=support.input_ids.device)
            / support.group_size
        )

    def _aggregate_inner_outputs(
        self,
        support: RolloutGroup,
        outputs: list[GRPOLossOutput],
    ) -> GRPOLossOutput:
        if len(outputs) != support.group_size:
            raise RuntimeError("Expected one inner loss output per response.")
        loss_weights = torch.stack(
            [
                self._group_normalization_weight(support, index)
                for index in range(support.group_size)
            ]
        )
        token_counts = support.completion_mask.sum(dim=1).to(torch.float32)
        token_weights = token_counts / token_counts.sum()
        losses = torch.stack([output.loss.detach() for output in outputs])
        policy_losses = torch.stack([output.policy_loss.detach() for output in outputs])
        mean_kls = torch.stack([output.mean_kl.detach() for output in outputs])
        clip_fractions = torch.stack(
            [output.clip_fraction.detach() for output in outputs]
        )
        loss_weights = loss_weights.to(losses.dtype)
        token_weights = token_weights.to(losses.dtype)
        return GRPOLossOutput(
            loss=(losses * loss_weights).sum(),
            policy_loss=(policy_losses * loss_weights).sum(),
            mean_kl=(mean_kls * token_weights).sum(),
            clip_fraction=(clip_fractions * token_weights).sum(),
        )

    def _first_order_inner_gradients(
        self,
        support: RolloutGroup,
        advantages: Tensor,
        fast_parameters: Mapping[str, Tensor],
        *,
        show_progress: bool,
        progress_description: str,
    ) -> tuple[ParameterDict, GRPOLossOutput]:
        """Differentiate rewards while stopping the policy-gradient Jacobian.

        PPO's policy gradient is piecewise linear in each sequence advantage.
        We compute its unit-advantage gradient for every response, detach that
        gradient from the policy graph, and weight it by the differentiable
        confidence advantage. This retains the confidence-to-adapter path but
        never asks an attention kernel for a double backward.
        """

        names = tuple(fast_parameters)
        parameter_values = tuple(fast_parameters[name] for name in names)
        nonnegative = advantages.detach() >= 0
        signs = torch.where(
            nonnegative,
            torch.ones_like(advantages),
            -torch.ones_like(advantages),
        )
        accumulated = {
            name: torch.zeros_like(value) for name, value in fast_parameters.items()
        }

        response_outputs: list[GRPOLossOutput | None] = [None] * support.group_size
        row_batches = sequence_microbatches(
            support,
            max_sequences=self.first_order_vjp_forward_batch_size,
            max_tokens=self.policy_max_tokens_per_micro_batch,
        )
        response_batches = range(len(row_batches))
        if show_progress:
            from tqdm.auto import tqdm

            response_batches = tqdm(
                response_batches,
                total=len(row_batches),
                desc=progress_description,
                unit="microbatch",
                leave=True,
            )
        for batch_index in response_batches:
            row_indices = row_batches[batch_index]
            current_logprobs = token_logprobs(
                self.policy,
                support,
                fast_parameters=fast_parameters,
                row_indices=row_indices,
                activation_checkpointing=False,
            )
            unit_losses = []
            kl_losses = []
            for offset, response_index in enumerate(row_indices):
                old_logprobs = support.old_logprobs[response_index : response_index + 1]
                completion_mask = support.completion_mask[
                    response_index : response_index + 1
                ]
                reference_logprobs = (
                    None
                    if support.reference_logprobs is None
                    else support.reference_logprobs[response_index : response_index + 1]
                )
                response_advantage = advantages[response_index : response_index + 1]
                response_current = current_logprobs[offset : offset + 1]
                response_output = grpo_policy_loss(
                    response_current,
                    old_logprobs,
                    completion_mask,
                    response_advantage,
                    self.inner_config.grpo,
                    reference_logprobs=reference_logprobs,
                )
                response_outputs[response_index] = GRPOLossOutput(
                    loss=response_output.loss.detach(),
                    policy_loss=response_output.policy_loss.detach(),
                    mean_kl=response_output.mean_kl.detach(),
                    clip_fraction=response_output.clip_fraction.detach(),
                )
                normalization_weight = self._group_normalization_weight(
                    support, response_index
                ).to(response_output.loss.dtype)
                unit_loss = grpo_policy_loss(
                    response_current,
                    old_logprobs,
                    completion_mask,
                    signs[response_index : response_index + 1],
                    self.inner_config.grpo,
                    reference_logprobs=reference_logprobs,
                ).policy_loss
                unit_losses.append(unit_loss * normalization_weight)
                if self.inner_config.grpo.kl_coefficient > 0:
                    zero_advantage = torch.zeros_like(response_advantage)
                    kl_loss = grpo_policy_loss(
                        response_current,
                        old_logprobs,
                        completion_mask,
                        zero_advantage,
                        self.inner_config.grpo,
                        reference_logprobs=reference_logprobs,
                    ).loss
                    kl_losses.append(kl_loss * normalization_weight)

            # A batched VJP materializes a leading response dimension through
            # the complete Qwen backward graph. On a 7B policy this can require
            # tens of GiB even when the corresponding batched forward is small.
            # Reuse that forward graph but consume one unit-advantage VJP at a
            # time. The detached VJP and differentiable advantage preserve the
            # same first-order meta-gradient without materializing a batched
            # policy-gradient Jacobian.
            for unit_index, (response_index, unit_loss) in enumerate(
                zip(row_indices, unit_losses, strict=True)
            ):
                retain_graph = (
                    unit_index + 1 < len(unit_losses)
                    or self.inner_config.grpo.kl_coefficient > 0
                )
                unit_gradient_values = torch.autograd.grad(
                    unit_loss,
                    parameter_values,
                    create_graph=False,
                    retain_graph=retain_graph,
                    allow_unused=False,
                )
                magnitude = advantages[response_index].abs()
                for name, unit_gradient in zip(
                    names, unit_gradient_values, strict=True
                ):
                    accumulated[name] = (
                        accumulated[name] + magnitude * unit_gradient.detach()
                    )

            if self.inner_config.grpo.kl_coefficient > 0:
                kl_gradient_values = torch.autograd.grad(
                    torch.stack(kl_losses).sum(),
                    parameter_values,
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=False,
                )
                for name, kl_gradient in zip(names, kl_gradient_values, strict=True):
                    accumulated[name] = accumulated[name] + kl_gradient.detach()

        if any(output is None for output in response_outputs):
            raise RuntimeError("Inner token batching returned incomplete metrics.")
        return accumulated, self._aggregate_inner_outputs(
            support,
            [output for output in response_outputs if output is not None],
        )

    def _nondifferentiable_inner_gradients(
        self,
        support: RolloutGroup,
        advantages: Tensor,
        fast_parameters: Mapping[str, Tensor],
        *,
        show_progress: bool,
        progress_description: str,
    ) -> tuple[ParameterDict, GRPOLossOutput]:
        names = tuple(fast_parameters)
        parameter_values = tuple(fast_parameters[name] for name in names)
        accumulated = {
            name: torch.zeros_like(value) for name, value in fast_parameters.items()
        }
        response_outputs: list[GRPOLossOutput | None] = [None] * support.group_size
        row_batches = sequence_microbatches(
            support,
            max_sequences=self.policy_micro_batch_size,
            max_tokens=self.policy_max_tokens_per_micro_batch,
        )
        batch_indices = range(len(row_batches))
        if show_progress:
            from tqdm.auto import tqdm

            batch_indices = tqdm(
                batch_indices,
                total=len(row_batches),
                desc=progress_description,
                unit="microbatch",
                leave=True,
            )
        for batch_index in batch_indices:
            row_indices = row_batches[batch_index]
            current_logprobs = token_logprobs(
                self.policy,
                support,
                fast_parameters=fast_parameters,
                row_indices=row_indices,
                activation_checkpointing=False,
            )
            weighted_losses = []
            for offset, response_index in enumerate(row_indices):
                output = grpo_policy_loss(
                    current_logprobs[offset : offset + 1],
                    support.old_logprobs[response_index : response_index + 1],
                    support.completion_mask[response_index : response_index + 1],
                    advantages[response_index : response_index + 1],
                    self.inner_config.grpo,
                    reference_logprobs=(
                        None
                        if support.reference_logprobs is None
                        else support.reference_logprobs[
                            response_index : response_index + 1
                        ]
                    ),
                )
                response_outputs[response_index] = GRPOLossOutput(
                    loss=output.loss.detach(),
                    policy_loss=output.policy_loss.detach(),
                    mean_kl=output.mean_kl.detach(),
                    clip_fraction=output.clip_fraction.detach(),
                )
                weighted_losses.append(
                    output.loss
                    * self._group_normalization_weight(support, response_index).to(
                        output.loss.dtype
                    )
                )
            gradient_values = torch.autograd.grad(
                torch.stack(weighted_losses).sum(),
                parameter_values,
                create_graph=False,
                retain_graph=False,
                allow_unused=False,
            )
            for name, gradient in zip(names, gradient_values, strict=True):
                accumulated[name] = accumulated[name] + gradient.detach()
        if any(output is None for output in response_outputs):
            raise RuntimeError("Inner token batching returned incomplete metrics.")
        return accumulated, self._aggregate_inner_outputs(
            support,
            [output for output in response_outputs if output is not None],
        )

    def _token_batch_loss_weight(
        self,
        support: RolloutGroup,
        row_indices: tuple[int, ...],
    ) -> Tensor:
        if self.inner_config.grpo.token_normalization == "global_tokens":
            selected = torch.tensor(
                row_indices,
                dtype=torch.long,
                device=support.completion_mask.device,
            )
            return (
                support.completion_mask.index_select(0, selected).sum()
                / support.completion_mask.sum()
            )
        return torch.as_tensor(
            len(row_indices) / support.group_size,
            dtype=torch.float32,
            device=support.device,
        )

    def _token_gradient_operator(
        self,
        support: RolloutGroup,
        token_credits: Tensor,
        fast_parameters: Mapping[str, Tensor],
    ) -> tuple[ParameterDict, GRPOLossOutput]:
        """First-order policy gradient with an exact JVP to token credits."""
        algorithm = self
        names = tuple(fast_parameters)
        config = self.inner_config.grpo
        group = support

        class TokenPolicyGradient(torch.autograd.Function):
            @staticmethod
            def forward(ctx, advantages, *parameter_values):
                local_values = tuple(
                    value.detach().requires_grad_(True) for value in parameter_values
                )
                local_parameters = dict(zip(names, local_values, strict=True))
                accumulated = [torch.zeros_like(value) for value in local_values]
                metric_sums = [advantages.new_zeros(()) for _ in range(4)]
                row_batches = sequence_microbatches(
                    group,
                    max_sequences=algorithm.policy_micro_batch_size,
                    max_tokens=algorithm.policy_max_tokens_per_micro_batch,
                )
                with torch.enable_grad():
                    for row_indices in row_batches:
                        selector = torch.tensor(
                            row_indices,
                            dtype=torch.long,
                            device=group.device,
                        )
                        current = token_logprobs(
                            algorithm.policy,
                            group,
                            fast_parameters=local_parameters,
                            row_indices=row_indices,
                            activation_checkpointing=False,
                        )
                        output = token_grpo_policy_loss(
                            current,
                            group.old_logprobs.index_select(0, selector),
                            group.completion_mask.index_select(0, selector),
                            advantages.detach().index_select(0, selector),
                            config,
                            reference_logprobs=(
                                None
                                if group.reference_logprobs is None
                                else group.reference_logprobs.index_select(0, selector)
                            ),
                        )
                        loss_weight = algorithm._token_batch_loss_weight(
                            group, row_indices
                        ).to(output.loss.dtype)
                        gradients = torch.autograd.grad(
                            output.loss * loss_weight,
                            local_values,
                            create_graph=False,
                            retain_graph=False,
                            allow_unused=False,
                        )
                        accumulated = [
                            total + gradient.detach()
                            for total, gradient in zip(
                                accumulated, gradients, strict=True
                            )
                        ]
                        token_weight = (
                            group.completion_mask.index_select(0, selector).sum()
                            / group.completion_mask.sum()
                        ).to(output.loss.dtype)
                        metric_sums[0] = (
                            metric_sums[0] + output.loss.detach() * loss_weight
                        )
                        metric_sums[1] = (
                            metric_sums[1] + output.policy_loss.detach() * loss_weight
                        )
                        metric_sums[2] = (
                            metric_sums[2] + output.mean_kl.detach() * token_weight
                        )
                        metric_sums[3] = (
                            metric_sums[3]
                            + output.clip_fraction.detach() * token_weight
                        )
                ctx.save_for_backward(
                    advantages.detach(),
                    *(value.detach() for value in parameter_values),
                )
                ctx.mark_non_differentiable(*metric_sums)
                return (*accumulated, *metric_sums)

            @staticmethod
            def backward(ctx, *output_cotangents):
                saved = ctx.saved_tensors
                advantages = saved[0]
                parameter_values = saved[1:]
                parameter_cotangents = tuple(
                    torch.zeros_like(value) if tangent is None else tangent.detach()
                    for value, tangent in zip(
                        parameter_values,
                        output_cotangents[: len(names)],
                        strict=True,
                    )
                )
                advantage_gradient = torch.zeros_like(advantages)
                row_batches = sequence_microbatches(
                    group,
                    max_sequences=algorithm.token_jvp_response_micro_batch_size,
                    max_tokens=None,
                )
                was_training = algorithm.policy.training
                input_hook_enabled = hasattr(
                    algorithm.policy, "disable_input_require_grads"
                )
                algorithm.policy.eval()
                if input_hook_enabled:
                    algorithm.policy.disable_input_require_grads()
                try:
                    with torch.enable_grad():
                        for row_indices in row_batches:
                            selector = torch.tensor(
                                row_indices,
                                dtype=torch.long,
                                device=group.device,
                            )

                            def selected_logprobs(*values):
                                parameters = dict(zip(names, values, strict=True))
                                return token_logprobs(
                                    algorithm.policy,
                                    group,
                                    fast_parameters=parameters,
                                    row_indices=row_indices,
                                    activation_checkpointing=False,
                                    logprob_position_chunk_size=(
                                        algorithm.token_jvp_logprob_position_chunk_size
                                    ),
                                )

                            current, directional = torch.func.jvp(
                                selected_logprobs,
                                parameter_values,
                                parameter_cotangents,
                            )
                            old = group.old_logprobs.index_select(0, selector)
                            selected_advantages = advantages.index_select(0, selector)
                            mask = group.completion_mask.index_select(0, selector)
                            if config.use_importance_ratio:
                                ratios = torch.exp(current - old)
                                coefficient = ratios
                                if config.use_clipping:
                                    clipped_ratios = torch.clamp(
                                        ratios,
                                        min=1.0 - config.clip_epsilon_low,
                                        max=1.0 + config.clip_epsilon_high,
                                    )
                                    unclipped_active = torch.where(
                                        selected_advantages >= 0,
                                        ratios <= clipped_ratios,
                                        ratios >= clipped_ratios,
                                    )
                                    coefficient = coefficient * unclipped_active
                            else:
                                coefficient = torch.ones_like(current)
                            if config.token_normalization == "per_response":
                                normalization = 1.0 / (
                                    group.group_size * mask.sum(dim=1, keepdim=True)
                                )
                            elif config.token_normalization == "global_tokens":
                                normalization = torch.ones_like(current) / mask.sum()
                                normalization = normalization * (
                                    mask.sum() / group.completion_mask.sum()
                                )
                            elif config.token_normalization == "sequence_sum":
                                normalization = (
                                    torch.ones_like(current) / group.group_size
                                )
                            else:
                                raise ValueError(
                                    "Unsupported token normalization: "
                                    f"{config.token_normalization}"
                                )
                            row_gradient = (
                                -directional * coefficient * normalization * mask
                            )
                            advantage_gradient.index_copy_(
                                0,
                                selector,
                                row_gradient.to(advantage_gradient.dtype),
                            )
                finally:
                    if input_hook_enabled:
                        algorithm.policy.enable_input_require_grads()
                    algorithm.policy.train(was_training)
                return (advantage_gradient, *(None for _ in names))

        outputs = TokenPolicyGradient.apply(
            token_credits,
            *(fast_parameters[name] for name in names),
        )
        gradient_values = outputs[: len(names)]
        loss, policy_loss, mean_kl, clip_fraction = outputs[len(names) :]
        return (
            dict(zip(names, gradient_values, strict=True)),
            GRPOLossOutput(
                loss=loss,
                policy_loss=policy_loss,
                mean_kl=mean_kl,
                clip_fraction=clip_fraction,
            ),
        )

    def _nondifferentiable_token_inner_gradients(
        self,
        support: RolloutGroup,
        token_credits: Tensor,
        fast_parameters: Mapping[str, Tensor],
    ) -> tuple[ParameterDict, GRPOLossOutput]:
        detached_credits = token_credits.detach().requires_grad_(True)
        gradients, output = self._token_gradient_operator(
            support,
            detached_credits,
            fast_parameters,
        )
        return (
            {name: value.detach() for name, value in gradients.items()},
            output,
        )

    def adapt_task(
        self,
        support: RolloutGroup,
        initial_fast_parameters: Mapping[str, Tensor],
        *,
        differentiable: bool = True,
        supervise_confidence: bool = True,
        show_progress: bool = False,
        progress_prefix: str = "adaptation",
    ) -> TaskAdaptation:
        return self.adapt_tasks(
            (support,),
            initial_fast_parameters,
            differentiable=differentiable,
            supervise_confidence=supervise_confidence,
            show_progress=show_progress,
            progress_prefix=progress_prefix,
        )[0]

    def adapt_tasks(
        self,
        supports: tuple[RolloutGroup, ...],
        initial_fast_parameters: Mapping[str, Tensor],
        *,
        differentiable: bool = True,
        supervise_confidence: bool = True,
        show_progress: bool = False,
        progress_prefix: str = "adaptation",
    ) -> tuple[TaskAdaptation, ...]:
        if not supports:
            raise ValueError("Task adaptation requires at least one support group.")
        confidence_logits_batch = self._confidence_logits_batch(
            supports,
            differentiable=differentiable,
            show_progress=show_progress,
            progress_description=f"{progress_prefix}: confidence scoring",
        )
        outputs = []
        for problem_index, (support, confidence_logits) in enumerate(
            zip(supports, confidence_logits_batch, strict=True)
        ):
            outputs.append(
                self._adapt_task_from_confidence_logits(
                    support,
                    confidence_logits,
                    initial_fast_parameters,
                    differentiable=differentiable,
                    supervise_confidence=supervise_confidence,
                    show_progress=show_progress and len(supports) == 1,
                    progress_prefix=(
                        progress_prefix
                        if len(supports) == 1
                        else f"{progress_prefix} problem {problem_index + 1}"
                    ),
                )
            )
        return tuple(outputs)

    def continue_adapt_tasks(
        self,
        supports: tuple[RolloutGroup, ...],
        initial_fast_parameter_groups: tuple[Mapping[str, Tensor], ...],
        initial_optimizer_states: tuple[FastOptimizerState, ...],
        *,
        show_progress: bool = False,
        progress_prefix: str = "adaptation",
    ) -> tuple[TaskAdaptation, ...]:
        """Continue non-differentiable task adaptation across on-policy rounds."""
        if not supports:
            raise ValueError("Task adaptation requires at least one support group.")
        if not (
            len(supports)
            == len(initial_fast_parameter_groups)
            == len(initial_optimizer_states)
        ):
            raise ValueError(
                "Every support must have one fast-parameter and optimizer state."
            )
        confidence_logits_batch = self._confidence_logits_batch(
            supports,
            differentiable=False,
            show_progress=show_progress,
            progress_description=f"{progress_prefix}: confidence scoring",
        )
        return tuple(
            self._adapt_task_from_confidence_logits(
                support,
                confidence_logits,
                fast_parameters,
                initial_optimizer_state=optimizer_state,
                differentiable=False,
                supervise_confidence=False,
                show_progress=show_progress and len(supports) == 1,
                progress_prefix=(
                    progress_prefix
                    if len(supports) == 1
                    else f"{progress_prefix} problem {problem_index + 1}"
                ),
            )
            for problem_index, (
                support,
                confidence_logits,
                fast_parameters,
                optimizer_state,
            ) in enumerate(
                zip(
                    supports,
                    confidence_logits_batch,
                    initial_fast_parameter_groups,
                    initial_optimizer_states,
                    strict=True,
                )
            )
        )

    def _adapt_task_from_confidence_logits(
        self,
        support: RolloutGroup,
        confidence_logits: Tensor,
        initial_fast_parameters: Mapping[str, Tensor],
        *,
        initial_optimizer_state: FastOptimizerState | None = None,
        differentiable: bool,
        supervise_confidence: bool,
        show_progress: bool,
        progress_prefix: str,
    ) -> TaskAdaptation:
        supervision_enabled = (
            self.meta_config.confidence.bce_coefficient > 0
            or self.meta_config.confidence.ranking_coefficient > 0
        )
        if (
            supervise_confidence
            and supervision_enabled
            and support.correctness_labels is None
        ):
            raise ValueError(
                "Support correctness labels are required for BCE/ranking supervision."
            )
        if confidence_logits.shape != (support.group_size,):
            raise ValueError("Confidence logits must match the support group size.")
        confidence_probabilities = torch.sigmoid(confidence_logits)
        confidence_loss = None
        if supervise_confidence and supervision_enabled:
            confidence_loss = confidence_losses(
                confidence_logits,
                support.correctness_labels,
                self.meta_config.confidence,
            )
        adaptation_rewards = (
            confidence_probabilities
            if differentiable
            else confidence_probabilities.detach()
        )
        advantages = group_advantages(
            adaptation_rewards,
            self.inner_config.advantage,
        )

        fast_parameters = clone_fast_parameters(initial_fast_parameters)
        if initial_optimizer_state is None:
            optimizer_state = initial_fast_optimizer_state(
                fast_parameters,
                self.inner_config.optimizer,
            )
        else:
            optimizer_state = FastOptimizerState(
                step=initial_optimizer_state.step,
                first_moment={
                    name: value.detach().clone()
                    for name, value in initial_optimizer_state.first_moment.items()
                },
                second_moment={
                    name: value.detach().clone()
                    for name, value in initial_optimizer_state.second_moment.items()
                },
            )
        inner_outputs: list[GRPOLossOutput] = []

        for inner_iteration in range(self.inner_config.num_iterations):
            names = tuple(fast_parameters)
            if not differentiable:
                gradients, inner_output = self._nondifferentiable_inner_gradients(
                    support,
                    advantages,
                    fast_parameters,
                    show_progress=show_progress,
                    progress_description=(
                        f"{progress_prefix}: inner "
                        f"{inner_iteration + 1}/{self.inner_config.num_iterations}"
                    ),
                )
            elif self.inner_config.meta_gradient_mode == "first_order":
                gradients, inner_output = self._first_order_inner_gradients(
                    support,
                    advantages,
                    fast_parameters,
                    show_progress=show_progress,
                    progress_description=(
                        f"{progress_prefix}: inner "
                        f"{inner_iteration + 1}/{self.inner_config.num_iterations}"
                    ),
                )
            else:
                current_logprobs = chunked_token_logprobs(
                    self.policy,
                    support,
                    fast_parameters=fast_parameters,
                    micro_batch_size=self.policy_micro_batch_size,
                    max_tokens_per_micro_batch=(self.policy_max_tokens_per_micro_batch),
                    activation_checkpointing=True,
                    show_progress=show_progress,
                    progress_description=(
                        f"{progress_prefix}: exact inner forward "
                        f"{inner_iteration + 1}/{self.inner_config.num_iterations}"
                    ),
                )
                inner_output = grpo_policy_loss(
                    current_logprobs,
                    support.old_logprobs,
                    support.completion_mask,
                    advantages,
                    self.inner_config.grpo,
                    reference_logprobs=support.reference_logprobs,
                )
                gradient_values = torch.autograd.grad(
                    inner_output.loss,
                    tuple(fast_parameters[name] for name in names),
                    create_graph=differentiable,
                    retain_graph=differentiable,
                    allow_unused=False,
                )
                gradients = dict(zip(names, gradient_values, strict=True))
            fast_parameters, optimizer_state = fast_optimizer_step(
                fast_parameters,
                gradients,
                optimizer_state,
                self.inner_config.optimizer,
            )
            if not differentiable:
                fast_parameters = {
                    name: value.detach().requires_grad_(True)
                    for name, value in fast_parameters.items()
                }
                optimizer_state = type(optimizer_state)(
                    step=optimizer_state.step,
                    first_moment={
                        name: value.detach()
                        for name, value in optimizer_state.first_moment.items()
                    },
                    second_moment={
                        name: value.detach()
                        for name, value in optimizer_state.second_moment.items()
                    },
                )
            inner_outputs.append(inner_output)

        return TaskAdaptation(
            fast_parameters=fast_parameters,
            optimizer_state=optimizer_state,
            confidence_logits=confidence_logits,
            confidence_probabilities=confidence_probabilities,
            confidence_loss=confidence_loss,
            inner_losses=tuple(inner_outputs),
        )

    def adapt_token_task(
        self,
        support: RolloutGroup,
        initial_fast_parameters: Mapping[str, Tensor],
        *,
        differentiable: bool = True,
    ) -> TokenTaskAdaptation:
        return self.adapt_token_tasks(
            (support,),
            initial_fast_parameters,
            differentiable=differentiable,
        )[0]

    def adapt_token_tasks(
        self,
        supports: tuple[RolloutGroup, ...],
        initial_fast_parameters: Mapping[str, Tensor],
        *,
        differentiable: bool = True,
    ) -> tuple[TokenTaskAdaptation, ...]:
        if self.meta_config.token_meta_coefficient <= 0:
            raise RuntimeError("Token adaptation is disabled by its zero coefficient.")
        if differentiable and self.inner_config.meta_gradient_mode != "first_order":
            raise ValueError(
                "Token confidence supports first-order meta-gradients only."
            )
        logits_batch = self._token_confidence_logits_batch(
            supports,
            differentiable=differentiable,
            show_progress=False,
            progress_description="token confidence scoring",
        )
        outputs = []
        for support, logits in zip(supports, logits_batch, strict=True):
            if logits.shape != support.completion_mask.shape:
                raise ValueError("Token confidence logits must match completion_mask.")
            credits = bounded_token_credits(
                logits,
                support.completion_mask,
                maximum=self.token_credit_max,
            )
            adaptation_credits = credits if differentiable else credits.detach()
            fast_parameters = clone_fast_parameters(initial_fast_parameters)
            optimizer_state = initial_fast_optimizer_state(
                fast_parameters,
                self.inner_config.optimizer,
            )
            inner_outputs = []
            for _ in range(self.inner_config.num_iterations):
                if differentiable:
                    gradients, inner_output = self._token_gradient_operator(
                        support,
                        adaptation_credits,
                        fast_parameters,
                    )
                else:
                    gradients, inner_output = (
                        self._nondifferentiable_token_inner_gradients(
                            support,
                            adaptation_credits,
                            fast_parameters,
                        )
                    )
                fast_parameters, optimizer_state = fast_optimizer_step(
                    fast_parameters,
                    gradients,
                    optimizer_state,
                    self.inner_config.optimizer,
                )
                if not differentiable:
                    fast_parameters = {
                        name: value.detach().requires_grad_(True)
                        for name, value in fast_parameters.items()
                    }
                    optimizer_state = FastOptimizerState(
                        step=optimizer_state.step,
                        first_moment={
                            name: value.detach()
                            for name, value in optimizer_state.first_moment.items()
                        },
                        second_moment={
                            name: value.detach()
                            for name, value in optimizer_state.second_moment.items()
                        },
                    )
                inner_outputs.append(inner_output)
            outputs.append(
                TokenTaskAdaptation(
                    fast_parameters=fast_parameters,
                    optimizer_state=optimizer_state,
                    token_confidence_logits=logits,
                    token_credits=credits,
                    inner_losses=tuple(inner_outputs),
                )
            )
        return tuple(outputs)

    def _token_outer_loss_from_adaptation(
        self,
        query: RolloutGroup,
        adaptation: TokenTaskAdaptation,
        *,
        show_progress: bool,
        progress_prefix: str,
    ) -> TokenTaskOuterLoss:
        if query.verifier_rewards is None:
            raise ValueError("Query verifier rewards are required for token meta loss.")
        current_query_logprobs = chunked_token_logprobs(
            self.policy,
            query,
            fast_parameters=adaptation.fast_parameters,
            micro_batch_size=self.policy_micro_batch_size,
            max_tokens_per_micro_batch=self.policy_max_tokens_per_micro_batch,
            activation_checkpointing=True,
            show_progress=show_progress,
            progress_description=f"{progress_prefix}: token query forward",
        )
        query_advantages = group_advantages(
            query.verifier_rewards.detach(),
            self.query_advantage_config,
        )
        meta_grpo = grpo_policy_loss(
            current_query_logprobs,
            query.old_logprobs,
            query.completion_mask,
            query_advantages,
            self.query_grpo_config,
            reference_logprobs=query.reference_logprobs,
        )
        return TokenTaskOuterLoss(
            loss=self.meta_config.token_meta_coefficient * meta_grpo.loss,
            meta_objective=meta_grpo.loss,
            meta_grpo=meta_grpo,
            token_confidence_logits=adaptation.token_confidence_logits,
            token_credits=adaptation.token_credits,
            inner_grpo=adaptation.inner_losses[-1],
            query_advantages=query_advantages,
            adaptation=adaptation,
        )

    @staticmethod
    def _detached_grpo(output: GRPOLossOutput) -> GRPOLossOutput:
        return GRPOLossOutput(
            loss=output.loss.detach(),
            policy_loss=output.policy_loss.detach(),
            mean_kl=output.mean_kl.detach(),
            clip_fraction=output.clip_fraction.detach(),
        )

    def _support_logprob_jvp(
        self,
        support: RolloutGroup,
        fast_parameters: Mapping[str, Tensor],
        parameter_tangents: tuple[Tensor, ...],
    ) -> tuple[Tensor, Tensor]:
        names = tuple(fast_parameters)
        parameter_values = tuple(fast_parameters[name] for name in names)
        if len(parameter_values) != len(parameter_tangents):
            raise ValueError("Token JVP tangent count does not match fast parameters.")
        primal = torch.zeros_like(support.old_logprobs)
        directional = torch.zeros_like(support.old_logprobs)
        row_batches = sequence_microbatches(
            support,
            max_sequences=self.token_jvp_response_micro_batch_size,
            max_tokens=None,
        )
        was_training = self.policy.training
        input_hook_enabled = hasattr(self.policy, "disable_input_require_grads")
        self.policy.eval()
        if input_hook_enabled:
            self.policy.disable_input_require_grads()
        try:
            with torch.enable_grad():
                for row_indices in row_batches:
                    selector = torch.tensor(
                        row_indices,
                        dtype=torch.long,
                        device=support.device,
                    )

                    def selected_logprobs(*values: Tensor) -> Tensor:
                        parameters = dict(zip(names, values, strict=True))
                        return token_logprobs(
                            self.policy,
                            support,
                            fast_parameters=parameters,
                            row_indices=row_indices,
                            activation_checkpointing=False,
                            logprob_position_chunk_size=(
                                self.token_jvp_logprob_position_chunk_size
                            ),
                        )

                    selected_primal, selected_directional = torch.func.jvp(
                        selected_logprobs,
                        parameter_values,
                        parameter_tangents,
                    )
                    primal.index_copy_(0, selector, selected_primal.detach())
                    directional.index_copy_(
                        0,
                        selector,
                        selected_directional.detach(),
                    )
        finally:
            if input_hook_enabled:
                self.policy.enable_input_require_grads()
            self.policy.train(was_training)
        return primal, directional

    def token_gradient_alignment_context(
        self,
        support: RolloutGroup,
        query: RolloutGroup,
        initial_fast_parameters: Mapping[str, Tensor],
    ) -> TokenGradientAlignmentContext:
        """Build the fixed policy-side direction for the single-layer surrogate."""
        if query.verifier_rewards is None:
            raise ValueError("Query verifier rewards are required for token meta loss.")
        fast_parameters = clone_fast_parameters(initial_fast_parameters)
        current_query_logprobs = chunked_token_logprobs(
            self.policy,
            query,
            fast_parameters=fast_parameters,
            micro_batch_size=self.policy_micro_batch_size,
            max_tokens_per_micro_batch=self.policy_max_tokens_per_micro_batch,
            activation_checkpointing=True,
            show_progress=False,
            progress_description="token alignment query gradient",
        )
        query_advantages = group_advantages(
            query.verifier_rewards.detach(),
            self.query_advantage_config,
        )
        query_grpo = grpo_policy_loss(
            current_query_logprobs,
            query.old_logprobs,
            query.completion_mask,
            query_advantages,
            self.query_grpo_config,
            reference_logprobs=query.reference_logprobs,
        )
        query_gradients = torch.autograd.grad(
            query_grpo.loss,
            tuple(fast_parameters.values()),
            create_graph=False,
            retain_graph=False,
        )
        support_logprobs, support_directional = self._support_logprob_jvp(
            support,
            fast_parameters,
            tuple(gradient.detach() for gradient in query_gradients),
        )
        return TokenGradientAlignmentContext(
            support_logprobs=support_logprobs,
            support_directional_logprobs=support_directional,
            query_grpo=self._detached_grpo(query_grpo),
            query_advantages=query_advantages.detach(),
        )

    def token_gradient_alignment_contexts_batch(
        self,
        supports: tuple[RolloutGroup, ...],
        queries: tuple[RolloutGroup, ...],
        initial_fast_parameters: Mapping[str, Tensor],
    ) -> tuple[TokenGradientAlignmentContext, ...]:
        if not supports or len(supports) != len(queries):
            raise ValueError("Token alignment requires matching support/query groups.")
        return tuple(
            self.token_gradient_alignment_context(
                support, query, initial_fast_parameters
            )
            for support, query in zip(supports, queries, strict=True)
        )

    def _alignment_reduction(
        self,
        values: Tensor,
        completion_mask: Tensor,
    ) -> Tensor:
        mask = completion_mask.to(values.dtype)
        if self.inner_config.grpo.token_normalization == "per_response":
            return (values.mul(mask).sum(dim=1) / mask.sum(dim=1)).mean()
        if self.inner_config.grpo.token_normalization == "global_tokens":
            return values.mul(mask).sum() / mask.sum()
        if self.inner_config.grpo.token_normalization == "sequence_sum":
            return values.mul(mask).sum(dim=1).mean()
        raise ValueError(
            "Unsupported token normalization: "
            f"{self.inner_config.grpo.token_normalization}"
        )

    def token_gradient_alignment_losses_batch(
        self,
        supports: tuple[RolloutGroup, ...],
        contexts: tuple[TokenGradientAlignmentContext, ...],
    ) -> tuple[TokenTaskOuterLoss, ...]:
        if not supports or len(supports) != len(contexts):
            raise ValueError("Token alignment requires one context per support.")
        logits_batch = self._token_confidence_logits_batch(
            supports,
            differentiable=True,
            show_progress=False,
            progress_description="token confidence scoring",
        )
        outputs = []
        effective_step_size = (
            self.inner_config.optimizer.learning_rate * self.inner_config.num_iterations
        )
        for support, context, logits in zip(
            supports,
            contexts,
            logits_batch,
            strict=True,
        ):
            credits = bounded_token_credits(
                logits,
                support.completion_mask,
                maximum=self.token_credit_max,
            )
            meta_objective = effective_step_size * self._alignment_reduction(
                credits * context.support_directional_logprobs,
                support.completion_mask,
            )
            inner_grpo = token_grpo_policy_loss(
                context.support_logprobs,
                support.old_logprobs,
                support.completion_mask,
                credits.detach(),
                self.inner_config.grpo,
                reference_logprobs=support.reference_logprobs,
            )
            outputs.append(
                TokenTaskOuterLoss(
                    loss=self.meta_config.token_meta_coefficient * meta_objective,
                    meta_objective=meta_objective,
                    meta_grpo=context.query_grpo,
                    token_confidence_logits=logits,
                    token_credits=credits,
                    inner_grpo=inner_grpo,
                    query_advantages=context.query_advantages,
                    adaptation=None,
                )
            )
        return tuple(outputs)

    def token_outer_losses_batch(
        self,
        supports: tuple[RolloutGroup, ...],
        queries: tuple[RolloutGroup, ...],
        initial_fast_parameters: Mapping[str, Tensor],
        *,
        show_progress: bool = False,
        progress_prefix: str = "token outer",
    ) -> tuple[TokenTaskOuterLoss, ...]:
        if not supports or len(supports) != len(queries):
            raise ValueError("Token outer loss requires matching support/query groups.")
        if self.token_meta_gradient_mode == "gradient_alignment":
            contexts = self.token_gradient_alignment_contexts_batch(
                supports,
                queries,
                initial_fast_parameters,
            )
            return self.token_gradient_alignment_losses_batch(supports, contexts)
        adaptations = self.adapt_token_tasks(
            supports,
            initial_fast_parameters,
            differentiable=True,
        )
        return tuple(
            self._token_outer_loss_from_adaptation(
                query,
                adaptation,
                show_progress=show_progress,
                progress_prefix=progress_prefix,
            )
            for query, adaptation in zip(queries, adaptations, strict=True)
        )

    def outer_loss(
        self,
        support: RolloutGroup,
        query: RolloutGroup,
        initial_fast_parameters: Mapping[str, Tensor],
        *,
        adaptation: TaskAdaptation | None = None,
        show_progress: bool = False,
        progress_prefix: str = "outer",
    ) -> TaskOuterLoss:
        if adaptation is None:
            return self.outer_losses_batch(
                (support,),
                (query,),
                initial_fast_parameters,
                show_progress=show_progress,
                progress_prefix=progress_prefix,
            )[0]
        return self._outer_loss_from_adaptation(
            query,
            adaptation,
            show_progress=show_progress,
            progress_prefix=progress_prefix,
        )

    def _outer_loss_from_adaptation(
        self,
        query: RolloutGroup,
        adaptation: TaskAdaptation,
        *,
        show_progress: bool,
        progress_prefix: str,
    ) -> TaskOuterLoss:
        if query.verifier_rewards is None:
            raise ValueError("Query verifier rewards are required for the meta loss.")
        supervision_enabled = (
            self.meta_config.confidence.bce_coefficient > 0
            or self.meta_config.confidence.ranking_coefficient > 0
        )
        if adaptation.confidence_loss is None and supervision_enabled:
            raise ValueError(
                "Outer training requires an adaptation with confidence supervision."
            )

        current_query_logprobs = chunked_token_logprobs(
            self.policy,
            query,
            fast_parameters=adaptation.fast_parameters,
            micro_batch_size=self.policy_micro_batch_size,
            max_tokens_per_micro_batch=self.policy_max_tokens_per_micro_batch,
            activation_checkpointing=True,
            show_progress=show_progress,
            progress_description=f"{progress_prefix}: query forward",
        )
        query_advantages = group_advantages(
            query.verifier_rewards.detach(),
            self.query_advantage_config,
        )
        meta_grpo = grpo_policy_loss(
            current_query_logprobs,
            query.old_logprobs,
            query.completion_mask,
            query_advantages,
            self.query_grpo_config,
            reference_logprobs=query.reference_logprobs,
        )
        loss = self.meta_config.meta_coefficient * meta_grpo.loss
        if adaptation.confidence_loss is not None:
            loss = loss + adaptation.confidence_loss.loss
        return TaskOuterLoss(
            loss=loss,
            meta_grpo=meta_grpo,
            adaptation=adaptation,
            query_advantages=query_advantages,
        )

    def outer_losses_batch(
        self,
        supports: tuple[RolloutGroup, ...],
        queries: tuple[RolloutGroup, ...],
        initial_fast_parameters: Mapping[str, Tensor],
        *,
        show_progress: bool = False,
        progress_prefix: str = "outer",
    ) -> tuple[TaskOuterLoss, ...]:
        if not supports or len(supports) != len(queries):
            raise ValueError(
                "Batched outer loss requires equal non-empty support/query groups."
            )
        adaptations = self.adapt_tasks(
            supports,
            initial_fast_parameters,
            differentiable=True,
            supervise_confidence=True,
            show_progress=show_progress,
            progress_prefix=progress_prefix,
        )
        return tuple(
            self._outer_loss_from_adaptation(
                query,
                adaptation,
                show_progress=show_progress and len(queries) == 1,
                progress_prefix=(
                    progress_prefix
                    if len(queries) == 1
                    else f"{progress_prefix} problem {index + 1}"
                ),
            )
            for index, (query, adaptation) in enumerate(
                zip(queries, adaptations, strict=True)
            )
        )
