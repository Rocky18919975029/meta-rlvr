from contextlib import contextmanager, nullcontext
from dataclasses import replace
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

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
from meta_rlvr.data import ChatMessage, MathProblem
import meta_rlvr.functional as functional_module
from meta_rlvr.functional import (
    chunked_token_logprobs,
    enable_sdpa_math_policy_forwards,
    sdpa_math_checkpoint_contexts,
    sequence_microbatches,
    token_logprobs,
    trainable_parameter_state,
)
from meta_rlvr.losses import bounded_token_credits, token_grpo_policy_loss
from meta_rlvr.optim import fast_optimizer_step, initial_fast_optimizer_state
from meta_rlvr.train import (
    CachedRolloutMicrobatch,
    _accumulate_outer_batch,
    _measure_component_gradient_norms,
)
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


def test_token_policy_sdpa_math_context_wraps_the_actual_forward(monkeypatch) -> None:
    torch.manual_seed(0)
    policy = ToyPolicy()
    group = make_group(policy, torch.tensor([1.0, 0.0, 1.0]))
    events = []

    @contextmanager
    def observed_sdpa_kernel(*, backends):
        events.append(("enter", tuple(backends)))
        try:
            yield
        finally:
            events.append(("exit", tuple(backends)))

    monkeypatch.setattr(functional_module, "sdpa_kernel", observed_sdpa_kernel)
    token_logprobs(policy, group)
    assert events == []

    enable_sdpa_math_policy_forwards(policy)
    token_logprobs(policy, group)

    assert [event[0] for event in events] == ["enter", "exit"]
    assert events[0][1] == (functional_module.SDPBackend.MATH,)


def test_token_policy_checkpoint_recomputation_uses_sdpa_math(monkeypatch) -> None:
    policy = ToyPolicy()
    events = []

    @contextmanager
    def observed_sdpa_kernel(*, backends):
        events.append(("enter", tuple(backends)))
        try:
            yield
        finally:
            events.append(("exit", tuple(backends)))

    monkeypatch.setattr(functional_module, "sdpa_kernel", observed_sdpa_kernel)
    enable_sdpa_math_policy_forwards(policy)
    value = torch.randn(4, requires_grad=True)
    output = checkpoint(
        lambda tensor: torch.sin(tensor).square(),
        value,
        use_reentrant=False,
        context_fn=sdpa_math_checkpoint_contexts,
    )
    output.sum().backward()

    assert [event[0] for event in events] == ["enter", "exit", "enter", "exit"]
    assert all(event[1] == (functional_module.SDPBackend.MATH,) for event in events)


class HookedToyPolicy(ToyPolicy):
    def __init__(self) -> None:
        super().__init__()
        self._input_gradient_hook = None
        self.enable_input_require_grads()

    def enable_input_require_grads(self) -> None:
        if self._input_gradient_hook is not None:
            raise RuntimeError("Input-gradient hook is already enabled.")
        self._input_gradient_hook = self.embedding.register_forward_hook(
            lambda _module, _inputs, output: output.requires_grad_(True)
        )

    def disable_input_require_grads(self) -> None:
        if self._input_gradient_hook is None:
            raise RuntimeError("Input-gradient hook is not enabled.")
        self._input_gradient_hook.remove()
        self._input_gradient_hook = None


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
    assert output.adaptation.confidence_loss is None
    gradients = torch.autograd.grad(
        output.loss,
        tuple(confidence.parameters()),
        allow_unused=True,
    )
    assert any(
        gradient is not None and torch.any(gradient != 0) for gradient in gradients
    )


def test_token_meta_loss_backpropagates_through_exact_jvp() -> None:
    torch.manual_seed(41)
    policy = ToyPolicy()
    confidence = SequenceConfidenceModel(
        ToyConfidenceBackbone(),
        hidden_size=5,
        enable_sequence_head=False,
        enable_token_head=True,
    )
    initial_fast = trainable_parameter_state(policy)
    support = make_group(policy, torch.tensor([1.0, 0.0, 1.0]))
    query = make_group(policy, torch.tensor([0.0, 1.0, 1.0]))
    algorithm = BilevelGRPO(
        policy=policy,
        confidence_model=confidence,
        inner_config=InnerLoopConfig(
            num_iterations=1,
            optimizer=FastOptimizerConfig(name="sgd", learning_rate=0.1),
        ),
        meta_config=MetaLossConfig(
            meta_coefficient=0.0,
            token_meta_coefficient=1.0,
            confidence=ConfidenceLossConfig(
                bce_coefficient=0.0,
                ranking_coefficient=0.0,
            ),
        ),
        query_advantage_config=AdvantageConfig(),
        query_grpo_config=GRPOLossConfig(),
        token_jvp_response_micro_batch_size=2,
    )
    output = algorithm.token_outer_losses_batch((support,), (query,), initial_fast)[0]
    assert algorithm.token_meta_gradient_mode == "gradient_alignment"
    assert output.adaptation is None
    gradients = torch.autograd.grad(
        output.loss,
        tuple(confidence.parameters()),
        allow_unused=True,
    )
    assert confidence.score is None
    assert any(
        gradient is not None and torch.any(gradient != 0) for gradient in gradients
    )


def test_alignment_jvp_does_not_build_reverse_autograd_graph(monkeypatch) -> None:
    torch.manual_seed(411)
    policy = ToyPolicy()
    confidence = SequenceConfidenceModel(
        ToyConfidenceBackbone(),
        hidden_size=5,
        enable_sequence_head=False,
        enable_token_head=True,
    )
    initial_fast = trainable_parameter_state(policy)
    support = make_group(policy, torch.tensor([1.0, 0.0, 1.0]))
    query = make_group(policy, torch.tensor([0.0, 1.0, 1.0]))
    original_jvp = torch.func.jvp
    observations = []

    def checked_jvp(function, primals, tangents, **kwargs):
        observations.append(
            {
                "grad_enabled": torch.is_grad_enabled(),
                "primal_requires_grad": tuple(value.requires_grad for value in primals),
                "tangent_requires_grad": tuple(
                    value.requires_grad for value in tangents
                ),
            }
        )
        return original_jvp(function, primals, tangents, **kwargs)

    monkeypatch.setattr(torch.func, "jvp", checked_jvp)
    algorithm = BilevelGRPO(
        policy=policy,
        confidence_model=confidence,
        inner_config=InnerLoopConfig(
            num_iterations=1,
            optimizer=FastOptimizerConfig(name="sgd", learning_rate=0.1),
        ),
        meta_config=MetaLossConfig(
            meta_coefficient=0.0,
            token_meta_coefficient=1.0,
            confidence=ConfidenceLossConfig(
                bce_coefficient=0.0,
                ranking_coefficient=0.0,
            ),
        ),
        query_advantage_config=AdvantageConfig(),
        query_grpo_config=GRPOLossConfig(),
        token_jvp_response_micro_batch_size=2,
    )
    output = algorithm.token_outer_losses_batch((support,), (query,), initial_fast)[0]
    output.loss.backward()

    assert observations
    assert all(not observation["grad_enabled"] for observation in observations)
    assert all(
        not any(observation["primal_requires_grad"]) for observation in observations
    )
    assert all(
        not any(observation["tangent_requires_grad"]) for observation in observations
    )
    assert any(
        parameter.grad is not None and torch.any(parameter.grad != 0)
        for parameter in confidence.parameters()
    )


def test_token_unrolled_mode_retains_task_adapter_path() -> None:
    torch.manual_seed(42)
    policy = ToyPolicy()
    confidence = SequenceConfidenceModel(
        ToyConfidenceBackbone(),
        hidden_size=5,
        enable_sequence_head=False,
        enable_token_head=True,
    )
    initial_fast = trainable_parameter_state(policy)
    support = make_group(policy, torch.tensor([1.0, 0.0, 1.0]))
    query = make_group(policy, torch.tensor([0.0, 1.0, 1.0]))
    algorithm = BilevelGRPO(
        policy=policy,
        confidence_model=confidence,
        inner_config=InnerLoopConfig(
            num_iterations=1,
            optimizer=FastOptimizerConfig(name="sgd", learning_rate=0.1),
        ),
        meta_config=MetaLossConfig(
            meta_coefficient=0.0,
            token_meta_coefficient=1.0,
        ),
        query_advantage_config=AdvantageConfig(),
        query_grpo_config=GRPOLossConfig(),
        token_meta_gradient_mode="unrolled",
    )
    output = algorithm.token_outer_losses_batch((support,), (query,), initial_fast)[0]
    assert output.adaptation is not None
    torch.testing.assert_close(output.meta_objective, output.meta_grpo.loss)


def test_token_first_order_operator_has_standard_grpo_update_value() -> None:
    torch.manual_seed(43)
    policy = ToyPolicy()
    confidence = SequenceConfidenceModel(
        ToyConfidenceBackbone(),
        hidden_size=5,
        enable_sequence_head=False,
        enable_token_head=True,
    )
    initial_fast = trainable_parameter_state(policy)
    support = make_group(policy, torch.tensor([1.0, 0.0, 1.0]))
    inner = InnerLoopConfig(
        num_iterations=1,
        optimizer=FastOptimizerConfig(name="sgd", learning_rate=0.1),
    )
    algorithm = BilevelGRPO(
        policy=policy,
        confidence_model=confidence,
        inner_config=inner,
        meta_config=MetaLossConfig(
            meta_coefficient=0.0,
            token_meta_coefficient=1.0,
        ),
        query_advantage_config=AdvantageConfig(),
        query_grpo_config=GRPOLossConfig(),
        policy_micro_batch_size=3,
    )
    adaptation = algorithm.adapt_token_task(support, initial_fast, differentiable=True)

    logits = confidence(
        support.input_ids,
        support.attention_mask,
        output="token",
    )
    credits = bounded_token_credits(
        logits,
        support.completion_mask,
        maximum=1.0,
    )
    current = token_logprobs(
        policy,
        support,
        fast_parameters=initial_fast,
    )
    direct_loss = token_grpo_policy_loss(
        current,
        support.old_logprobs,
        support.completion_mask,
        credits,
        inner.grpo,
        reference_logprobs=support.reference_logprobs,
    )
    names = tuple(initial_fast)
    direct_gradients = torch.autograd.grad(
        direct_loss.loss,
        tuple(initial_fast[name] for name in names),
    )
    expected, _ = fast_optimizer_step(
        initial_fast,
        dict(zip(names, direct_gradients, strict=True)),
        initial_fast_optimizer_state(initial_fast, inner.optimizer),
        inner.optimizer,
    )
    for name in names:
        torch.testing.assert_close(
            adaptation.fast_parameters[name].detach(),
            expected[name].detach(),
        )
    torch.testing.assert_close(adaptation.token_credits, credits)


def test_sequence_only_path_never_invokes_token_jvp(monkeypatch) -> None:
    torch.manual_seed(47)
    policy = ToyPolicy()
    confidence = SequenceConfidenceModel(ToyConfidenceBackbone(), hidden_size=5)
    initial_fast = trainable_parameter_state(policy)
    support = make_group(policy, torch.tensor([1.0, 0.0, 1.0]))
    query = make_group(policy, torch.tensor([0.0, 1.0, 1.0]))

    def unexpected(*args, **kwargs):
        raise AssertionError("disabled token JVP was evaluated")

    monkeypatch.setattr(torch.func, "jvp", unexpected)
    algorithm = BilevelGRPO(
        policy=policy,
        confidence_model=confidence,
        inner_config=InnerLoopConfig(
            num_iterations=1,
            optimizer=FastOptimizerConfig(name="sgd", learning_rate=0.1),
        ),
        meta_config=MetaLossConfig(
            meta_coefficient=1.0,
            token_meta_coefficient=0.0,
            confidence=ConfidenceLossConfig(
                bce_coefficient=0.0,
                ranking_coefficient=0.0,
            ),
        ),
        query_advantage_config=AdvantageConfig(),
        query_grpo_config=GRPOLossConfig(),
    )
    output = algorithm.outer_loss(support, query, initial_fast)
    output.loss.backward()
    assert confidence.token_score is None


def test_token_jvp_temporarily_disables_transformers_input_gradient_hook() -> None:
    torch.manual_seed(49)
    policy = HookedToyPolicy()
    confidence = SequenceConfidenceModel(
        ToyConfidenceBackbone(),
        hidden_size=5,
        enable_sequence_head=False,
        enable_token_head=True,
    )
    initial_fast = trainable_parameter_state(policy)
    support = make_group(policy, torch.tensor([1.0, 0.0, 1.0]))
    query = make_group(policy, torch.tensor([0.0, 1.0, 1.0]))
    algorithm = BilevelGRPO(
        policy=policy,
        confidence_model=confidence,
        inner_config=InnerLoopConfig(
            num_iterations=1,
            optimizer=FastOptimizerConfig(name="sgd", learning_rate=0.1),
        ),
        meta_config=MetaLossConfig(
            meta_coefficient=0.0,
            token_meta_coefficient=1.0,
            confidence=ConfidenceLossConfig(
                bce_coefficient=0.0,
                ranking_coefficient=0.0,
            ),
        ),
        query_advantage_config=AdvantageConfig(),
        query_grpo_config=GRPOLossConfig(),
        token_jvp_response_micro_batch_size=2,
    )
    output = algorithm.token_outer_losses_batch((support,), (query,), initial_fast)[0]
    output.loss.backward()
    assert policy.training
    assert policy._input_gradient_hook is not None


def test_first_order_inner_update_has_same_value_as_direct_policy_gradient() -> None:
    torch.manual_seed(9)
    policy = ToyPolicy()
    confidence = SequenceConfidenceModel(ToyConfidenceBackbone(), hidden_size=5)
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
        first_order = algorithm.adapt_task(support, initial_fast, differentiable=True)
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
        direct = exact_algorithm.adapt_task(support, initial_fast, differentiable=True)
        for name in first_order.fast_parameters:
            torch.testing.assert_close(
                first_order.fast_parameters[name].detach(),
                direct.fast_parameters[name].detach(),
            )


def test_first_order_vjp_forward_batch_preserves_update_and_meta_gradient() -> None:
    torch.manual_seed(13)
    policy = ToyPolicy()
    confidence = SequenceConfidenceModel(ToyConfidenceBackbone(), hidden_size=5)
    initial_fast = trainable_parameter_state(policy)
    support = make_group(policy, torch.tensor([1.0, 0.0, 1.0]))
    query = make_group(policy, torch.tensor([0.0, 1.0, 1.0]))

    def build_algorithm(vjp_forward_batch_size: int) -> BilevelGRPO:
        return BilevelGRPO(
            policy=policy,
            confidence_model=confidence,
            inner_config=InnerLoopConfig(
                num_iterations=2,
                meta_gradient_mode="first_order",
                optimizer=FastOptimizerConfig(name="sgd", learning_rate=0.1),
                grpo=GRPOLossConfig(kl_coefficient=0.03),
            ),
            meta_config=MetaLossConfig(),
            query_advantage_config=AdvantageConfig(),
            query_grpo_config=GRPOLossConfig(),
            policy_micro_batch_size=3,
            first_order_vjp_forward_batch_size=vjp_forward_batch_size,
        )

    serial_output = build_algorithm(1).outer_loss(support, query, initial_fast)
    serial_gradients = torch.autograd.grad(
        serial_output.loss,
        tuple(confidence.parameters()),
    )
    packed_output = build_algorithm(3).outer_loss(support, query, initial_fast)
    packed_gradients = torch.autograd.grad(
        packed_output.loss,
        tuple(confidence.parameters()),
    )

    torch.testing.assert_close(serial_output.loss, packed_output.loss)
    for name in serial_output.adaptation.fast_parameters:
        torch.testing.assert_close(
            serial_output.adaptation.fast_parameters[name],
            packed_output.adaptation.fast_parameters[name],
        )
    for serial_gradient, packed_gradient in zip(
        serial_gradients, packed_gradients, strict=True
    ):
        torch.testing.assert_close(serial_gradient, packed_gradient)


def test_task_adapters_start_from_independent_fast_parameter_copies() -> None:
    policy = ToyPolicy()
    state = trainable_parameter_state(policy)
    first = {
        name: value.detach().clone().requires_grad_(True)
        for name, value in state.items()
    }
    second = {
        name: value.detach().clone().requires_grad_(True)
        for name, value in state.items()
    }
    with torch.no_grad():
        first["adapter"].add_(1.0)
    assert not torch.equal(first["adapter"], second["adapter"])
    torch.testing.assert_close(second["adapter"], state["adapter"])


def test_inference_adaptation_does_not_require_verifier_labels() -> None:
    policy = ToyPolicy()
    confidence = SequenceConfidenceModel(ToyConfidenceBackbone(), hidden_size=5)
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


def test_batched_generation_adaptation_matches_single_response_updates() -> None:
    torch.manual_seed(21)
    policy = ToyPolicy()
    confidence = SequenceConfidenceModel(ToyConfidenceBackbone(), hidden_size=5)
    support = make_group(policy, torch.tensor([1.0, 0.0, 1.0]))
    initial_fast = trainable_parameter_state(policy)

    def build_algorithm(policy_micro_batch_size: int) -> BilevelGRPO:
        return BilevelGRPO(
            policy=policy,
            confidence_model=confidence,
            inner_config=InnerLoopConfig(
                num_iterations=2,
                optimizer=FastOptimizerConfig(name="sgd", learning_rate=0.1),
                grpo=GRPOLossConfig(kl_coefficient=0.03),
            ),
            meta_config=MetaLossConfig(),
            query_advantage_config=AdvantageConfig(),
            query_grpo_config=GRPOLossConfig(),
            policy_micro_batch_size=policy_micro_batch_size,
        )

    serial = build_algorithm(1).adapt_task(
        support,
        initial_fast,
        differentiable=False,
        supervise_confidence=False,
    )
    batched = build_algorithm(3).adapt_task(
        support,
        initial_fast,
        differentiable=False,
        supervise_confidence=False,
    )
    for name in serial.fast_parameters:
        torch.testing.assert_close(
            serial.fast_parameters[name], batched.fast_parameters[name]
        )


def test_continued_adaptation_preserves_fast_adam_state() -> None:
    torch.manual_seed(23)
    policy = ToyPolicy()
    confidence = SequenceConfidenceModel(ToyConfidenceBackbone(), hidden_size=5)
    support = make_group(policy, torch.tensor([1.0, 0.0, 1.0]))
    initial_fast = trainable_parameter_state(policy)

    def build_algorithm(iterations: int) -> BilevelGRPO:
        return BilevelGRPO(
            policy=policy,
            confidence_model=confidence,
            inner_config=InnerLoopConfig(
                num_iterations=iterations,
                optimizer=FastOptimizerConfig(name="adamw", learning_rate=0.01),
            ),
            meta_config=MetaLossConfig(),
            query_advantage_config=AdvantageConfig(),
            query_grpo_config=GRPOLossConfig(),
        )

    first_round = build_algorithm(2).adapt_task(
        support,
        initial_fast,
        differentiable=False,
        supervise_confidence=False,
    )
    second_round = build_algorithm(2).continue_adapt_tasks(
        (support,),
        (first_round.fast_parameters,),
        (first_round.optimizer_state,),
    )[0]
    direct = build_algorithm(4).adapt_task(
        support,
        initial_fast,
        differentiable=False,
        supervise_confidence=False,
    )

    assert first_round.optimizer_state.step == 2
    assert second_round.optimizer_state.step == 4
    for name in direct.fast_parameters:
        torch.testing.assert_close(
            second_round.fast_parameters[name], direct.fast_parameters[name]
        )
        torch.testing.assert_close(
            second_round.optimizer_state.first_moment[name],
            direct.optimizer_state.first_moment[name],
        )
        torch.testing.assert_close(
            second_round.optimizer_state.second_moment[name],
            direct.optimizer_state.second_moment[name],
        )


def test_token_aware_microbatches_trim_padding_without_changing_logprobs() -> None:
    policy = ToyPolicy()
    input_ids = torch.tensor(
        (
            (1, 2, 3, 4, 5),
            (1, 2, 3, 0, 0),
            (1, 2, 3, 4, 0),
        )
    )
    attention_mask = torch.tensor(
        (
            (True, True, True, True, True),
            (True, True, True, False, False),
            (True, True, True, True, False),
        )
    )
    completion_mask = torch.tensor(
        (
            (False, True, True, True),
            (False, True, False, False),
            (False, True, True, False),
        )
    )
    group = RolloutGroup(
        input_ids=input_ids,
        attention_mask=attention_mask,
        completion_mask=completion_mask,
        old_logprobs=torch.zeros((3, 4)),
        texts=("a", "b", "c"),
    )
    fast = trainable_parameter_state(policy)
    expected = token_logprobs(policy, group, fast_parameters=fast)
    actual = chunked_token_logprobs(
        policy,
        group,
        fast_parameters=fast,
        micro_batch_size=3,
        max_tokens_per_micro_batch=8,
        activation_checkpointing=False,
    )
    assert sequence_microbatches(group, max_sequences=3, max_tokens=8) == ((0,), (2, 1))
    torch.testing.assert_close(actual, expected)


def test_selected_policy_logits_match_full_vocabulary_projection() -> None:
    class SelectedLogitsToyPolicy(ToyPolicy):
        def forward(
            self,
            input_ids,
            attention_mask,
            use_cache,
            return_dict,
            logits_to_keep=None,
        ):
            outputs = super().forward(input_ids, attention_mask, use_cache, return_dict)
            if logits_to_keep is not None:
                outputs.logits = outputs.logits.index_select(1, logits_to_keep)
            return outputs

    torch.manual_seed(22)
    full_policy = ToyPolicy()
    selected_policy = SelectedLogitsToyPolicy()
    selected_policy.load_state_dict(full_policy.state_dict())
    group = make_group(full_policy, torch.tensor([1.0, 0.0, 1.0]))
    torch.testing.assert_close(
        token_logprobs(selected_policy, group),
        token_logprobs(full_policy, group),
    )


def test_position_chunked_logprobs_preserve_forward_jvp() -> None:
    torch.manual_seed(23)
    policy = ToyPolicy()
    group = make_group(policy, torch.tensor([1.0, 0.0, 1.0]))
    fast = trainable_parameter_state(policy)
    names = tuple(fast)
    primals = tuple(fast[name].detach() for name in names)
    tangents = tuple(torch.randn_like(value) for value in primals)

    def logprobs(position_chunk_size, *parameter_values):
        return token_logprobs(
            policy,
            group,
            fast_parameters=dict(zip(names, parameter_values, strict=True)),
            logprob_position_chunk_size=position_chunk_size,
        )

    full_primal, full_tangent = torch.func.jvp(
        lambda *values: logprobs(None, *values),
        primals,
        tangents,
    )
    chunked_primal, chunked_tangent = torch.func.jvp(
        lambda *values: logprobs(1, *values),
        primals,
        tangents,
    )
    torch.testing.assert_close(chunked_primal, full_primal)
    torch.testing.assert_close(chunked_tangent, full_tangent)


def test_confidence_scoring_batches_responses_across_problems() -> None:
    policy = ToyPolicy()
    confidence = SequenceConfidenceModel(ToyConfidenceBackbone(), hidden_size=5)
    first = make_group(policy, torch.tensor([1.0, 0.0, 1.0]))
    second = make_group(policy, torch.tensor([0.0, 1.0, 0.0]))
    algorithm = BilevelGRPO(
        policy=policy,
        confidence_model=confidence,
        inner_config=InnerLoopConfig(),
        meta_config=MetaLossConfig(),
        query_advantage_config=AdvantageConfig(),
        query_grpo_config=GRPOLossConfig(),
        confidence_micro_batch_size=6,
    )
    batched = algorithm._confidence_logits_batch(
        (first, second),
        differentiable=True,
        show_progress=False,
        progress_description="test",
    )
    torch.testing.assert_close(
        batched[0],
        confidence(first.input_ids, first.attention_mask),
    )
    torch.testing.assert_close(
        batched[1],
        confidence(second.input_ids, second.attention_mask),
    )


class CPUAccelerator:
    device = torch.device("cpu")
    is_main_process = False

    @staticmethod
    def backward(loss: torch.Tensor) -> None:
        loss.backward()

    @staticmethod
    def clip_grad_norm_(parameters, max_norm: float) -> torch.Tensor:
        return torch.nn.utils.clip_grad_norm_(tuple(parameters), max_norm)

    @staticmethod
    def reduce(value: torch.Tensor, *, reduction: str) -> torch.Tensor:
        assert reduction == "mean"
        return value

    @staticmethod
    def wait_for_everyone() -> None:
        return None

    @staticmethod
    def no_sync(model):
        return nullcontext()


def _problem(uid: str) -> MathProblem:
    return MathProblem(
        uid=uid,
        messages=(ChatMessage(role="user", content=f"problem {uid}"),),
        ground_truth="0",
        data_source="unit_test",
    )


def _parameter_gradient_norm(
    loss: torch.Tensor,
    parameters: tuple[nn.Parameter, ...],
) -> torch.Tensor:
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    squared_norms = [
        gradient.float().square().sum()
        for gradient in gradients
        if gradient is not None
    ]
    if not squared_norms:
        raise AssertionError("Expected at least one confidence parameter gradient.")
    return torch.stack(squared_norms).sum().sqrt()


def test_component_gradient_norms_are_exact_and_leave_no_gradients() -> None:
    torch.manual_seed(12)
    policy = ToyPolicy()
    confidence = SequenceConfidenceModel(ToyConfidenceBackbone(), hidden_size=5)
    initial_fast = trainable_parameter_state(policy)
    support = make_group(policy, torch.tensor([1.0, 0.0, 1.0]))
    query = make_group(policy, torch.tensor([0.0, 1.0, 0.0]))
    coefficients = {"meta": 2.0, "bce": 3.0, "ranking": 4.0}
    algorithm = BilevelGRPO(
        policy=policy,
        confidence_model=confidence,
        inner_config=InnerLoopConfig(
            num_iterations=1,
            optimizer=FastOptimizerConfig(name="sgd", learning_rate=0.1),
        ),
        meta_config=MetaLossConfig(
            meta_coefficient=coefficients["meta"],
            confidence=ConfidenceLossConfig(
                bce_coefficient=coefficients["bce"],
                ranking_coefficient=coefficients["ranking"],
            ),
        ),
        query_advantage_config=AdvantageConfig(),
        query_grpo_config=GRPOLossConfig(),
    )
    parameters = tuple(confidence.parameters())
    initial_parameters = tuple(parameter.detach().clone() for parameter in parameters)
    expected = {
        "meta": _parameter_gradient_norm(
            algorithm.outer_loss(support, query, initial_fast).meta_grpo.loss,
            parameters,
        ),
        "bce": _parameter_gradient_norm(
            algorithm.confidence_supervision_loss(support).bce,
            parameters,
        ),
        "ranking": _parameter_gradient_norm(
            algorithm.confidence_supervision_loss(support).ranking,
            parameters,
        ),
    }
    optimizer = torch.optim.AdamW(confidence.parameters(), lr=1e-3)
    rollout_microbatches = [
        CachedRolloutMicrobatch(
            problems=(_problem("0"),),
            supports=(support,),
            queries=(query,),
        )
    ]

    measured = _measure_component_gradient_norms(
        algorithm=algorithm,
        rollout_microbatches=rollout_microbatches,
        local_problem_batch_size=1,
        initial_fast=initial_fast,
        confidence_model=confidence,
        confidence_optimizer=optimizer,
        accelerator=CPUAccelerator(),
        progress_prefix="test",
    )

    for component, expected_norm in expected.items():
        torch.testing.assert_close(
            torch.tensor(measured[f"gradient_norm/{component}_raw"]),
            expected_norm,
        )
        torch.testing.assert_close(
            torch.tensor(measured[f"gradient_norm/{component}_weighted"]),
            coefficients[component] * expected_norm,
        )
    assert measured["gradient_norm/measurement_seconds"] >= 0.0
    for parameter, initial_parameter in zip(
        confidence.parameters(), initial_parameters, strict=True
    ):
        torch.testing.assert_close(parameter, initial_parameter)
    assert not optimizer.state
    assert all(parameter.grad is None for parameter in confidence.parameters())


def test_component_gradient_norms_skip_zero_coefficient_losses() -> None:
    torch.manual_seed(13)
    policy = ToyPolicy()
    confidence = SequenceConfidenceModel(ToyConfidenceBackbone(), hidden_size=5)
    initial_fast = trainable_parameter_state(policy)
    support = make_group(policy, torch.tensor([1.0, 0.0, 1.0]))
    query = make_group(policy, torch.tensor([0.0, 1.0, 0.0]))
    algorithm = BilevelGRPO(
        policy=policy,
        confidence_model=confidence,
        inner_config=InnerLoopConfig(
            num_iterations=1,
            optimizer=FastOptimizerConfig(name="sgd", learning_rate=0.1),
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
    optimizer = torch.optim.AdamW(confidence.parameters(), lr=1e-3)
    measured = _measure_component_gradient_norms(
        algorithm=algorithm,
        rollout_microbatches=[
            CachedRolloutMicrobatch(
                problems=(_problem("0"),),
                supports=(support,),
                queries=(query,),
            )
        ],
        local_problem_batch_size=1,
        initial_fast=initial_fast,
        confidence_model=confidence,
        confidence_optimizer=optimizer,
        accelerator=CPUAccelerator(),
        progress_prefix="test",
    )

    assert "gradient_norm/meta_raw" in measured
    assert "gradient_norm/bce_raw" not in measured
    assert "gradient_norm/ranking_raw" not in measured


def test_problem_microbatch_accumulation_matches_full_batch_gradient() -> None:
    torch.manual_seed(16)
    policy = ToyPolicy()
    confidence = SequenceConfidenceModel(ToyConfidenceBackbone(), hidden_size=5)
    initial_fast = trainable_parameter_state(policy)
    supports = (
        make_group(policy, torch.tensor([1.0, 0.0, 1.0])),
        make_group(policy, torch.tensor([0.0, 1.0, 0.0])),
        make_group(policy, torch.tensor([1.0, 1.0, 0.0])),
        make_group(policy, torch.tensor([0.0, 0.0, 1.0])),
    )
    queries = (
        make_group(policy, torch.tensor([0.0, 1.0, 0.0])),
        make_group(policy, torch.tensor([1.0, 0.0, 1.0])),
        make_group(policy, torch.tensor([0.0, 1.0, 1.0])),
        make_group(policy, torch.tensor([1.0, 0.0, 0.0])),
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
    parameters = tuple(confidence.parameters())
    expected_metrics = torch.zeros(26)
    for support, query in zip(supports, queries, strict=True):
        output = algorithm.outer_loss(support, query, initial_fast)
        (output.loss / len(supports)).backward()
        expected_metrics += torch.stack(
            (
                output.loss.detach(),
                output.meta_grpo.loss.detach(),
                torch.tensor(0.0),
                output.adaptation.confidence_loss.bce.detach(),
                output.adaptation.confidence_loss.ranking.detach(),
                output.meta_grpo.clip_fraction.detach(),
                torch.tensor(0.0),
                output.adaptation.inner_losses[-1].clip_fraction.detach(),
                torch.tensor(0.0),
                support.verifier_rewards.mean(),
                query.verifier_rewards.mean(),
                torch.tensor(0.0),
                support.correctness_labels.mean(),
                query.correctness_labels.mean(),
                torch.tensor(0.0),
                output.meta_grpo.mean_kl.detach(),
                torch.tensor(0.0),
                output.adaptation.inner_losses[-1].mean_kl.detach(),
                torch.tensor(0.0),
                output.adaptation.confidence_probabilities.detach().mean(),
                output.adaptation.confidence_probabilities.detach().square().mean(),
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.any(support.correctness_labels == 1).float(),
                torch.any(query.correctness_labels == 1).float(),
                torch.tensor(0.0),
            )
        )
    expected_gradients = tuple(
        parameter.grad.detach().clone() for parameter in parameters
    )
    confidence.zero_grad(set_to_none=True)
    rollout_microbatches = [
        CachedRolloutMicrobatch(
            problems=(_problem("0"), _problem("1")),
            supports=supports[:2],
            queries=queries[:2],
        ),
        CachedRolloutMicrobatch(
            problems=(_problem("2"), _problem("3")),
            supports=supports[2:],
            queries=queries[2:],
        ),
    ]

    actual_metrics = _accumulate_outer_batch(
        algorithm=algorithm,
        rollout_microbatches=rollout_microbatches,
        local_problem_batch_size=4,
        initial_fast=initial_fast,
        accelerator=CPUAccelerator(),
        confidence_model=confidence,
        defer_gradient_sync=True,
        progress_description="test",
    )

    torch.testing.assert_close(actual_metrics, expected_metrics)
    for parameter, expected_gradient in zip(
        parameters, expected_gradients, strict=True
    ):
        torch.testing.assert_close(parameter.grad, expected_gradient)


def test_combined_sequence_and_token_meta_losses_accumulate_independently() -> None:
    torch.manual_seed(53)
    policy = ToyPolicy()
    confidence = SequenceConfidenceModel(
        ToyConfidenceBackbone(),
        hidden_size=5,
        enable_sequence_head=True,
        enable_token_head=True,
    )
    initial_fast = trainable_parameter_state(policy)
    support = make_group(policy, torch.tensor([1.0, 0.0, 1.0]))
    sequence_query = make_group(policy, torch.tensor([0.0, 1.0, 1.0]))
    token_query = make_group(policy, torch.tensor([1.0, 1.0, 0.0]))
    algorithm = BilevelGRPO(
        policy=policy,
        confidence_model=confidence,
        inner_config=InnerLoopConfig(
            num_iterations=1,
            optimizer=FastOptimizerConfig(name="sgd", learning_rate=0.1),
        ),
        meta_config=MetaLossConfig(
            meta_coefficient=0.3,
            token_meta_coefficient=0.7,
            confidence=ConfidenceLossConfig(
                bce_coefficient=0.0,
                ranking_coefficient=0.0,
            ),
        ),
        query_advantage_config=AdvantageConfig(),
        query_grpo_config=GRPOLossConfig(),
        token_jvp_response_micro_batch_size=2,
    )
    sequence_output = algorithm.outer_loss(support, sequence_query, initial_fast)
    token_output = algorithm.token_outer_losses_batch(
        (support,), (token_query,), initial_fast
    )[0]
    expected_loss = sequence_output.loss + token_output.loss
    expected_loss.backward()
    expected_gradients = tuple(
        parameter.grad.detach().clone() for parameter in confidence.parameters()
    )
    confidence.zero_grad(set_to_none=True)

    rollout_microbatches = [
        CachedRolloutMicrobatch(
            problems=(_problem("combined"),),
            supports=(support,),
            queries=(sequence_query,),
            token_queries=(token_query,),
        )
    ]
    metrics = _accumulate_outer_batch(
        algorithm=algorithm,
        rollout_microbatches=rollout_microbatches,
        local_problem_batch_size=1,
        initial_fast=initial_fast,
        accelerator=CPUAccelerator(),
        confidence_model=confidence,
        progress_description="combined",
    )
    torch.testing.assert_close(metrics[0], expected_loss.detach())
    torch.testing.assert_close(metrics[1], sequence_output.meta_grpo.loss.detach())
    torch.testing.assert_close(metrics[2], token_output.meta_objective.detach())
    assert rollout_microbatches[0].token_alignment_contexts is not None

    def fail_if_alignment_context_is_recomputed(*args, **kwargs):
        raise AssertionError("cached token alignment context was recomputed")

    algorithm.token_gradient_alignment_contexts_batch = (
        fail_if_alignment_context_is_recomputed
    )
    confidence.zero_grad(set_to_none=True)
    cached_metrics = _accumulate_outer_batch(
        algorithm=algorithm,
        rollout_microbatches=rollout_microbatches,
        local_problem_batch_size=1,
        initial_fast=initial_fast,
        accelerator=CPUAccelerator(),
        confidence_model=confidence,
        progress_description="combined cached",
    )
    torch.testing.assert_close(cached_metrics, metrics)
    for parameter, expected_gradient in zip(
        confidence.parameters(), expected_gradients, strict=True
    ):
        torch.testing.assert_close(parameter.grad, expected_gradient)
