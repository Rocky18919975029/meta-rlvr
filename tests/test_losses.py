import pytest
import torch

from meta_rlvr.config import (
    AdvantageConfig,
    ConfidenceLossConfig,
    GRPOLossConfig,
)
from meta_rlvr.losses import (
    bounded_token_credits,
    confidence_losses,
    grpo_policy_loss,
    group_advantages,
    token_credit_derivatives,
    token_grpo_policy_loss,
)


def test_disabled_confidence_losses_skip_bce_and_ranking(monkeypatch) -> None:
    def unexpected(*args, **kwargs):
        raise AssertionError("disabled confidence loss was evaluated")

    monkeypatch.setattr(
        torch.nn.functional, "binary_cross_entropy_with_logits", unexpected
    )
    monkeypatch.setattr(torch.nn.functional, "logsigmoid", unexpected)
    output = confidence_losses(
        torch.tensor([0.2, -0.1], requires_grad=True),
        torch.tensor([1.0, 0.0]),
        ConfidenceLossConfig(bce_coefficient=0.0, ranking_coefficient=0.0),
    )

    assert output.loss.item() == 0.0
    assert output.bce.item() == 0.0
    assert output.ranking.item() == 0.0


def test_ranking_only_skips_bce(monkeypatch) -> None:
    def unexpected(*args, **kwargs):
        raise AssertionError("disabled BCE was evaluated")

    monkeypatch.setattr(
        torch.nn.functional, "binary_cross_entropy_with_logits", unexpected
    )
    output = confidence_losses(
        torch.tensor([0.2, -0.1], requires_grad=True),
        torch.tensor([1.0, 0.0]),
        ConfidenceLossConfig(bce_coefficient=0.0, ranking_coefficient=1.0),
    )
    assert output.ranking.item() > 0.0


def test_bce_only_skips_ranking(monkeypatch) -> None:
    def unexpected(*args, **kwargs):
        raise AssertionError("disabled ranking loss was evaluated")

    monkeypatch.setattr(torch.nn.functional, "logsigmoid", unexpected)
    output = confidence_losses(
        torch.tensor([0.2, -0.1], requires_grad=True),
        torch.tensor([1.0, 0.0]),
        ConfidenceLossConfig(bce_coefficient=1.0, ranking_coefficient=0.0),
    )
    assert output.bce.item() > 0.0
    assert output.ranking.item() == 0.0


def test_qwen_ranking_loss_uses_all_positive_negative_pairs() -> None:
    logits = torch.tensor([2.0, 1.0, -1.0, -2.0], requires_grad=True)
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    output = confidence_losses(
        logits,
        labels,
        ConfidenceLossConfig(bce_coefficient=0.0, ranking_coefficient=1.0),
    )

    differences = logits[:2, None] - logits[2:][None, :]
    expected = -torch.nn.functional.logsigmoid(differences).mean()
    torch.testing.assert_close(output.loss, expected)
    assert output.num_positive == 2
    assert output.num_negative == 2


def test_homogeneous_group_has_zero_ranking_loss_but_nonzero_bce() -> None:
    logits = torch.tensor([0.2, -0.1, 0.3], requires_grad=True)
    labels = torch.zeros(3)
    output = confidence_losses(logits, labels, ConfidenceLossConfig())

    torch.testing.assert_close(output.ranking, torch.tensor(0.0))
    assert output.bce.item() > 0
    output.loss.backward()
    assert logits.grad is not None


def test_group_std_is_affine_invariant() -> None:
    rewards = torch.tensor([0.1, 0.2, 0.7, 0.9])
    config = AdvantageConfig(std_epsilon=1e-8)
    first = group_advantages(rewards, config)
    second = group_advantages(3.0 * rewards + 2.0, config)
    torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(first.std(), torch.tensor(1.0))


def test_group_std_has_finite_gradient_at_equal_confidences() -> None:
    rewards = torch.full((4,), 0.5, requires_grad=True)
    advantages = group_advantages(rewards, AdvantageConfig())
    gradient = torch.autograd.grad(advantages[0], rewards)[0]
    assert torch.isfinite(gradient).all()


def test_center_only_preserves_reward_spread() -> None:
    config = AdvantageConfig(scale="center_only")
    wide = group_advantages(torch.tensor([0.1, 0.9]), config)
    narrow = group_advantages(torch.tensor([0.49, 0.51]), config)
    assert wide.abs().max() > 30 * narrow.abs().max()


def test_floored_std_does_not_amplify_tiny_differences() -> None:
    rewards = torch.tensor([0.499, 0.500, 0.501])
    config = AdvantageConfig(
        scale="floored_group_std",
        std_floor=0.05,
    )
    advantages = group_advantages(rewards, config)
    assert advantages.abs().max().item() < 0.03


def test_grpo_clips_positive_advantage_when_ratio_is_too_large() -> None:
    current = torch.tensor([[torch.log(torch.tensor(2.0))]], requires_grad=True)
    old = torch.zeros_like(current)
    mask = torch.ones_like(current, dtype=torch.bool)
    advantages = torch.tensor([1.0])
    output = grpo_policy_loss(
        current,
        old,
        mask,
        advantages,
        GRPOLossConfig(
            use_importance_ratio=True,
            use_clipping=True,
            clip_epsilon_low=0.2,
            clip_epsilon_high=0.2,
        ),
    )
    torch.testing.assert_close(output.policy_loss, torch.tensor(-1.2))
    torch.testing.assert_close(output.clip_fraction, torch.tensor(1.0))


def test_positive_kl_requires_reference_logprobs() -> None:
    values = torch.zeros((2, 3))
    with pytest.raises(ValueError, match="reference_logprobs"):
        grpo_policy_loss(
            values,
            values,
            torch.ones_like(values, dtype=torch.bool),
            torch.tensor([1.0, -1.0]),
            GRPOLossConfig(kl_coefficient=0.1),
        )


def test_token_credits_do_not_mix_trajectories_or_positions() -> None:
    logits = torch.tensor(
        [[0.1, 0.2, 0.9], [0.5, 0.6, 0.0], [0.9, 0.0, 0.0]],
        requires_grad=True,
    )
    mask = torch.tensor([[True, True, True], [True, True, False], [True, False, False]])
    credits = bounded_token_credits(
        logits,
        mask,
        maximum=1.0,
        parameterization="scaled_tanh",
    )

    torch.testing.assert_close(credits, torch.tanh(logits) * mask)
    derivative = torch.autograd.grad(credits[0, 0], logits)[0]
    assert torch.count_nonzero(derivative) == 1
    assert derivative[0, 0] > 0


def test_scaled_arctan_token_credit_and_derivative_match_definition() -> None:
    logits = torch.tensor([[-2.0, 0.0, 2.0]])
    mask = torch.tensor([[True, True, False]])

    credits = bounded_token_credits(
        logits,
        mask,
        maximum=1.0,
        parameterization="scaled_arctan",
    )
    derivatives = token_credit_derivatives(
        logits,
        mask,
        maximum=1.0,
        parameterization="scaled_arctan",
    )

    expected_credits = (2.0 / torch.pi) * torch.atan(logits) * mask
    expected_derivatives = (2.0 / torch.pi) / (1.0 + logits.square()) * mask
    torch.testing.assert_close(credits, expected_credits)
    torch.testing.assert_close(derivatives, expected_derivatives)


def test_token_grpo_matches_sequence_grpo_for_constant_row_advantages() -> None:
    torch.manual_seed(5)
    current = torch.randn(3, 4) * 0.03
    old = torch.zeros_like(current)
    mask = torch.tensor(
        [
            [True, True, True, True],
            [True, True, False, False],
            [True, True, True, False],
        ]
    )
    advantages = torch.tensor([0.7, -0.2, 0.1])
    config = GRPOLossConfig(token_normalization="per_response")
    sequence = grpo_policy_loss(current, old, mask, advantages, config)
    token = token_grpo_policy_loss(
        current,
        old,
        mask,
        advantages[:, None].expand_as(current),
        config,
    )
    torch.testing.assert_close(token.loss, sequence.loss)
    torch.testing.assert_close(token.policy_loss, sequence.policy_loss)
    torch.testing.assert_close(token.clip_fraction, sequence.clip_fraction)
