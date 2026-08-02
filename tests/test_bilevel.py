from types import SimpleNamespace
from dataclasses import replace

import torch
from torch import nn

from meta_rlvr.bilevel import BilevelGRPO
from meta_rlvr.confidence import SequenceConfidenceModel
from meta_rlvr.config import (
    AdvantageConfig,
    ConfidenceLossConfig,
    FastOptimizerConfig,
    GRPOLossConfig,
    InnerLoopConfig,
    MetaLossConfig,
)
from meta_rlvr.functional import token_logprobs, trainable_parameter_state
from meta_rlvr.types import RolloutGroup


class ToyPolicy(nn.Module):
    def __init__(self, vocabulary_size: int = 7, hidden_size: int = 5) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, hidden_size)
        self.base_head = nn.Linear(hidden_size, vocabulary_size, bias=False)
        self.adapter = nn.Parameter(torch.zeros(hidden_size, vocabulary_size))
        self.embedding.weight.requires_grad_(False)
        self.base_head.weight.requires_grad_(False)

    def forward(
        self,
        input_ids,
        attention_mask,
        use_cache,
        return_dict,
    ):
        assert not use_cache
        assert return_dict
        hidden = self.embedding(input_ids)
        logits = self.base_head(hidden) + hidden @ self.adapter
        return SimpleNamespace(logits=logits)


class ToyConfidenceBackbone(nn.Module):
    def __init__(self, vocabulary_size: int = 7, hidden_size: int = 5) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, hidden_size)
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            initializer_range=0.02,
        )

    def forward(self, input_ids, attention_mask, return_dict):
        assert return_dict
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def make_group(policy: nn.Module, rewards: torch.Tensor) -> RolloutGroup:
    correctness = rewards
    verifier_rewards = 2.0 * correctness - 1.0
    input_ids = torch.tensor(
        [
            [1, 2, 3, 4],
            [1, 3, 2, 5],
            [1, 4, 5, 6],
        ]
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    completion_mask = torch.tensor(
        [
            [False, True, True],
            [False, True, True],
            [False, True, True],
        ]
    )
    provisional = RolloutGroup(
        input_ids=input_ids,
        attention_mask=attention_mask,
        completion_mask=completion_mask,
        old_logprobs=torch.zeros((3, 3)),
        texts=("a", "b", "c"),
        verifier_rewards=verifier_rewards,
        correctness_labels=correctness,
    )
    old_logprobs = token_logprobs(policy, provisional).detach()
    return RolloutGroup(
        input_ids=input_ids,
        attention_mask=attention_mask,
        completion_mask=completion_mask,
        old_logprobs=old_logprobs,
        reference_logprobs=old_logprobs,
        texts=("a", "b", "c"),
        verifier_rewards=verifier_rewards,
        correctness_labels=correctness,
    )


def test_meta_loss_backpropagates_through_inner_update_to_confidence() -> None:
    torch.manual_seed(4)
    policy = ToyPolicy()
    confidence = SequenceConfidenceModel(
        ToyConfidenceBackbone(),
        hidden_size=5,
    )
    initial_fast = trainable_parameter_state(policy)
    support = make_group(policy, torch.tensor([1.0, 0.0, 0.0]))
    query = make_group(policy, torch.tensor([1.0, 0.0, 1.0]))

    algorithm = BilevelGRPO(
        policy=policy,
        confidence_model=confidence,
        inner_config=InnerLoopConfig(
            num_iterations=2,
            optimizer=FastOptimizerConfig(name="sgd", learning_rate=0.1),
            grpo=GRPOLossConfig(
                use_importance_ratio=True,
                use_clipping=True,
                kl_coefficient=0.0,
            ),
        ),
        meta_config=MetaLossConfig(
            meta_coefficient=1.0,
            confidence=ConfidenceLossConfig(
                bce_coefficient=0.0,
                ranking_coefficient=0.0,
            ),
        ),
        query_advantage_config=AdvantageConfig(),
        query_grpo_config=GRPOLossConfig(),
    )
    output = algorithm.outer_loss(support, query, initial_fast)
    gradients = torch.autograd.grad(
        output.loss,
        tuple(confidence.parameters()),
        allow_unused=True,
    )
    assert any(
        gradient is not None and torch.any(gradient != 0)
        for gradient in gradients
    )


def test_first_order_inner_update_has_same_value_as_direct_policy_gradient() -> None:
    torch.manual_seed(9)
    policy = ToyPolicy()
    confidence = SequenceConfidenceModel(
        ToyConfidenceBackbone(), hidden_size=5
    )
    initial_fast = trainable_parameter_state(policy)
    support = make_group(policy, torch.tensor([1.0, 0.0, 1.0]))
    support = replace(
        support,
        completion_mask=torch.tensor(
            [
                [False, True, True],
                [False, False, True],
                [False, True, True],
            ]
        ),
    )
    for token_normalization in (
        "per_response",
        "global_tokens",
        "sequence_sum",
    ):
        algorithm = BilevelGRPO(
            policy=policy,
            confidence_model=confidence,
            inner_config=InnerLoopConfig(
                num_iterations=2,
                meta_gradient_mode="first_order",
                optimizer=FastOptimizerConfig(name="sgd", learning_rate=0.1),
                grpo=GRPOLossConfig(
                    token_normalization=token_normalization,
                    kl_coefficient=0.03,
                ),
            ),
            meta_config=MetaLossConfig(),
            query_advantage_config=AdvantageConfig(),
            query_grpo_config=GRPOLossConfig(),
        )
        first_order = algorithm.adapt_task(
            support, initial_fast, differentiable=True
        )
        exact_algorithm = BilevelGRPO(
            policy=policy,
            confidence_model=confidence,
            inner_config=replace(
                algorithm.inner_config,
                meta_gradient_mode="second_order",
            ),
            meta_config=MetaLossConfig(),
            query_advantage_config=AdvantageConfig(),
            query_grpo_config=GRPOLossConfig(),
        )
        direct = exact_algorithm.adapt_task(
            support, initial_fast, differentiable=True
        )
        for name in first_order.fast_parameters:
            torch.testing.assert_close(
                first_order.fast_parameters[name].detach(),
                direct.fast_parameters[name].detach(),
            )


def test_task_adapters_start_from_independent_fast_parameter_copies() -> None:
    policy = ToyPolicy()
    state = trainable_parameter_state(policy)
    first = {name: value.detach().clone().requires_grad_(True) for name, value in state.items()}
    second = {name: value.detach().clone().requires_grad_(True) for name, value in state.items()}
    with torch.no_grad():
        first["adapter"].add_(1.0)
    assert not torch.equal(first["adapter"], second["adapter"])
    torch.testing.assert_close(second["adapter"], state["adapter"])


def test_inference_adaptation_does_not_require_verifier_labels() -> None:
    policy = ToyPolicy()
    confidence = SequenceConfidenceModel(
        ToyConfidenceBackbone(), hidden_size=5
    )
    support = replace(
        make_group(policy, torch.tensor([1.0, 0.0, 0.0])),
        verifier_rewards=None,
        correctness_labels=None,
    )
    algorithm = BilevelGRPO(
        policy=policy,
        confidence_model=confidence,
        inner_config=InnerLoopConfig(
            num_iterations=1,
            optimizer=FastOptimizerConfig(name="sgd", learning_rate=0.1),
        ),
        meta_config=MetaLossConfig(),
        query_advantage_config=AdvantageConfig(),
        query_grpo_config=GRPOLossConfig(),
    )
    adaptation = algorithm.adapt_task(
        support,
        trainable_parameter_state(policy),
        differentiable=False,
        supervise_confidence=False,
    )
    assert adaptation.confidence_loss is None
