import torch

from meta_rlvr.config import FastOptimizerConfig
from meta_rlvr.optim import fast_optimizer_step, initial_fast_optimizer_state


def test_differentiable_adam_has_finite_gradient_at_zero_update() -> None:
    signal = torch.tensor(0.0, requires_grad=True)
    parameters = {"adapter": torch.tensor([1.0, -1.0], requires_grad=True)}
    gradients = {"adapter": signal.expand_as(parameters["adapter"])}
    config = FastOptimizerConfig(
        name="adamw",
        learning_rate=1e-3,
        epsilon=1e-8,
    )
    state = initial_fast_optimizer_state(parameters, config)

    updated, _ = fast_optimizer_step(
        parameters,
        gradients,
        state,
        config,
    )
    meta_gradient = torch.autograd.grad(updated["adapter"].sum(), signal)[0]

    assert torch.isfinite(meta_gradient)
    torch.testing.assert_close(meta_gradient, torch.tensor(-2e5))
