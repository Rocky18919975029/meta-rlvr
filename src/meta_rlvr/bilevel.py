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
)
from .functional import (
    ParameterDict,
    chunked_token_logprobs,
    clone_fast_parameters,
    token_logprobs,
)
from .losses import (
    ConfidenceLossOutput,
    GRPOLossOutput,
    confidence_losses,
    grpo_policy_loss,
    group_advantages,
)
from .optim import fast_optimizer_step, initial_fast_optimizer_state
from .types import RolloutGroup


@dataclass(frozen=True)
class TaskAdaptation:
    fast_parameters: ParameterDict
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
        confidence_micro_batch_size: int = 4,
    ) -> None:
        if policy_micro_batch_size <= 0 or confidence_micro_batch_size <= 0:
            raise ValueError("Micro-batch sizes must be positive.")
        self.policy = policy
        self.confidence_model = confidence_model
        self.inner_config = inner_config
        self.meta_config = meta_config
        self.query_advantage_config = query_advantage_config
        self.query_grpo_config = query_grpo_config
        self.policy_micro_batch_size = policy_micro_batch_size
        self.confidence_micro_batch_size = confidence_micro_batch_size

    def _confidence_logits(
        self,
        support: RolloutGroup,
        *,
        differentiable: bool,
    ) -> Tensor:
        chunks: list[Tensor] = []
        for start in range(
            0, support.group_size, self.confidence_micro_batch_size
        ):
            end = min(
                start + self.confidence_micro_batch_size,
                support.group_size,
            )
            if differentiable:
                logits = self.confidence_model(
                    support.input_ids[start:end],
                    support.attention_mask[start:end],
                )
            else:
                with torch.no_grad():
                    logits = self.confidence_model(
                        support.input_ids[start:end],
                        support.attention_mask[start:end],
                    )
            chunks.append(logits)
        return torch.cat(chunks, dim=0)

    def _group_normalization_weight(
        self,
        support: RolloutGroup,
        response_index: int,
    ) -> Tensor:
        token_count = support.completion_mask[response_index].sum()
        if self.inner_config.grpo.token_normalization == "global_tokens":
            return token_count / support.completion_mask.sum()
        return torch.ones(
            (), dtype=torch.float32, device=support.input_ids.device
        ) / support.group_size

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
        policy_losses = torch.stack(
            [output.policy_loss.detach() for output in outputs]
        )
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
            name: torch.zeros_like(value)
            for name, value in fast_parameters.items()
        }

        response_outputs: list[GRPOLossOutput] = []
        response_indices = range(advantages.numel())
        if show_progress:
            from tqdm.auto import tqdm

            response_indices = tqdm(
                response_indices,
                total=advantages.numel(),
                desc=progress_description,
                unit="response",
                leave=True,
            )
        for response_index in response_indices:
            current_logprobs = token_logprobs(
                self.policy,
                support,
                fast_parameters=fast_parameters,
                row_start=response_index,
                row_end=response_index + 1,
                activation_checkpointing=False,
            )
            old_logprobs = support.old_logprobs[
                response_index : response_index + 1
            ]
            completion_mask = support.completion_mask[
                response_index : response_index + 1
            ]
            reference_logprobs = (
                None
                if support.reference_logprobs is None
                else support.reference_logprobs[
                    response_index : response_index + 1
                ]
            )
            response_advantage = advantages[
                response_index : response_index + 1
            ]
            response_output = grpo_policy_loss(
                current_logprobs,
                old_logprobs,
                completion_mask,
                response_advantage,
                self.inner_config.grpo,
                reference_logprobs=reference_logprobs,
            )
            response_outputs.append(
                GRPOLossOutput(
                    loss=response_output.loss.detach(),
                    policy_loss=response_output.policy_loss.detach(),
                    mean_kl=response_output.mean_kl.detach(),
                    clip_fraction=response_output.clip_fraction.detach(),
                )
            )
            unit_advantages = signs[response_index : response_index + 1]
            unit_loss = grpo_policy_loss(
                current_logprobs,
                old_logprobs,
                completion_mask,
                unit_advantages,
                self.inner_config.grpo,
                reference_logprobs=reference_logprobs,
            ).policy_loss
            normalization_weight = self._group_normalization_weight(
                support, response_index
            ).to(unit_loss.dtype)
            unit_gradient_values = torch.autograd.grad(
                unit_loss * normalization_weight,
                parameter_values,
                create_graph=False,
                retain_graph=self.inner_config.grpo.kl_coefficient > 0,
                allow_unused=False,
            )
            magnitude = torch.where(
                nonnegative[response_index],
                advantages[response_index],
                -advantages[response_index],
            )
            for name, unit_gradient in zip(
                names, unit_gradient_values, strict=True
            ):
                accumulated[name] = (
                    accumulated[name] + magnitude * unit_gradient.detach()
                )

            if self.inner_config.grpo.kl_coefficient > 0:
                zero_advantage = torch.zeros_like(response_advantage)
                kl_loss = grpo_policy_loss(
                    current_logprobs,
                    old_logprobs,
                    completion_mask,
                    zero_advantage,
                    self.inner_config.grpo,
                    reference_logprobs=reference_logprobs,
                ).loss
                kl_gradient_values = torch.autograd.grad(
                    kl_loss * normalization_weight,
                    parameter_values,
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=False,
                )
                for name, kl_gradient in zip(
                    names, kl_gradient_values, strict=True
                ):
                    accumulated[name] = (
                        accumulated[name] + kl_gradient.detach()
                    )

        return accumulated, self._aggregate_inner_outputs(
            support, response_outputs
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
        if supervise_confidence and support.verifier_rewards is None:
            raise ValueError(
                "Support verifier rewards are required for BCE/ranking supervision."
            )

        confidence_logits = self._confidence_logits(
            support, differentiable=differentiable
        )
        confidence_probabilities = torch.sigmoid(confidence_logits)
        confidence_loss = None
        if supervise_confidence:
            confidence_loss = confidence_losses(
                confidence_logits,
                support.verifier_rewards,
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
        optimizer_state = initial_fast_optimizer_state(
            fast_parameters,
            self.inner_config.optimizer,
        )
        inner_outputs: list[GRPOLossOutput] = []

        for inner_iteration in range(self.inner_config.num_iterations):
            names = tuple(fast_parameters)
            if (
                not differentiable
                or self.inner_config.meta_gradient_mode == "first_order"
            ):
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
            confidence_logits=confidence_logits,
            confidence_probabilities=confidence_probabilities,
            confidence_loss=confidence_loss,
            inner_losses=tuple(inner_outputs),
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
        if query.verifier_rewards is None:
            raise ValueError("Query verifier rewards are required for the meta loss.")
        if adaptation is None:
            adaptation = self.adapt_task(
                support,
                initial_fast_parameters,
                show_progress=show_progress,
                progress_prefix=progress_prefix,
            )
        if adaptation.confidence_loss is None:
            raise ValueError(
                "Outer training requires an adaptation with confidence supervision."
            )

        current_query_logprobs = chunked_token_logprobs(
            self.policy,
            query,
            fast_parameters=adaptation.fast_parameters,
            micro_batch_size=self.policy_micro_batch_size,
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
        loss = (
            self.meta_config.meta_coefficient * meta_grpo.loss
            + adaptation.confidence_loss.loss
        )
        return TaskOuterLoss(
            loss=loss,
            meta_grpo=meta_grpo,
            adaptation=adaptation,
            query_advantages=query_advantages,
        )
