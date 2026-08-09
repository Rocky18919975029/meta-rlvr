from __future__ import annotations

import math

import pytest
import torch

from meta_rlvr.fidelity import (
    bounded_token_credits,
    finalize_gradient_comparison,
    gradient_comparison,
    parameter_gradients,
)


def test_bounded_token_credits_are_signed_bounded_and_row_independent() -> None:
    logits = torch.tensor(
        [[-2.0, 0.0, 2.0], [0.5, -0.5, 1.0]],
        requires_grad=True,
    )
    mask = torch.tensor([[True, True, False], [True, True, True]])
    credits = bounded_token_credits(logits, mask, maximum=1.5)

    assert torch.all(credits.abs() <= 1.5)
    assert credits[0, 0] < 0
    assert credits[0, 1] == 0
    assert credits[0, 2] == 0
    assert credits[1, 0] > 0

    derivative = torch.autograd.grad(credits[0, 0], logits)[0]
    assert derivative[0, 0] > 0
    assert torch.count_nonzero(derivative) == 1


def test_gradient_comparison_reports_exact_cosine_and_norm_ratio() -> None:
    exact = {
        "a": torch.tensor([1.0, 2.0]),
        "b": torch.tensor([-2.0]),
    }
    approximation = {
        "a": torch.tensor([2.0, 4.0]),
        "b": torch.tensor([-4.0]),
    }
    result = finalize_gradient_comparison(gradient_comparison(exact, approximation))

    assert result["conclusive"] is True
    assert math.isclose(result["cosine"], 1.0, abs_tol=1e-12)
    assert math.isclose(result["norm_ratio"], 2.0, abs_tol=1e-12)
    assert math.isclose(result["relative_l2_error"], 1.0, abs_tol=1e-12)
    assert result["active_sign_agreement"] == 1.0


def test_gradient_comparison_marks_zero_signal_as_inconclusive() -> None:
    zeros = {"parameter": torch.zeros(4)}
    result = finalize_gradient_comparison(gradient_comparison(zeros, zeros))

    assert result["conclusive"] is False
    assert result["cosine"] is None
    assert result["norm_ratio"] is None
    assert result["active_sign_agreement"] is None


def test_parameter_gradients_skips_empty_fsdp_original_parameter_shards() -> None:
    model = torch.nn.Module()
    model.register_parameter("empty_shard", torch.nn.Parameter(torch.empty(0)))
    model.register_parameter("owned_shard", torch.nn.Parameter(torch.ones(3)))
    model.owned_shard.grad = torch.tensor([1.0, 2.0, 3.0])

    gradients = parameter_gradients(model)

    assert gradients.keys() == {"owned_shard"}
    torch.testing.assert_close(gradients["owned_shard"], model.owned_shard.grad)


def test_parameter_gradients_rejects_disconnected_nonempty_parameter() -> None:
    model = torch.nn.Module()
    model.register_parameter("disconnected", torch.nn.Parameter(torch.ones(3)))

    with pytest.raises(RuntimeError, match="Non-empty trainable parameter"):
        parameter_gradients(model)


def test_taylor_meta_gradient_matches_small_exact_one_step_update() -> None:
    phi = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    initial = torch.tensor([0.2, -0.4], dtype=torch.float64)
    support_basis = torch.tensor([0.7, -0.3], dtype=torch.float64)
    target = torch.tensor([-0.1, 0.5], dtype=torch.float64)
    learning_rate = 1e-3

    credit = torch.tanh(phi)
    adapted = initial + learning_rate * credit * support_basis
    exact_loss = 0.5 * (adapted - target).square().sum()
    exact_gradient = torch.autograd.grad(exact_loss, phi, retain_graph=True)[0]

    base_query_gradient = initial - target
    parameter_delta = adapted - initial
    alignment_loss = (base_query_gradient.detach() * parameter_delta).sum()
    alignment_gradient = torch.autograd.grad(alignment_loss, phi)[0]

    relative_error = (alignment_gradient - exact_gradient).abs() / exact_gradient.abs()
    assert torch.sign(alignment_gradient) == torch.sign(exact_gradient)
    assert relative_error < 1e-2
