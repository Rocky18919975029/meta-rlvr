from types import SimpleNamespace

import pytest
import torch

from meta_rlvr.train import _adamw_optimizer, _move_adamw_moments


def _initialized_optimizer(*, amsgrad: bool = False):
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0, 3.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.01, amsgrad=amsgrad)
    parameter.square().sum().backward()
    optimizer.step()
    return parameter, optimizer


def test_move_adamw_moments_preserves_values_and_step_state() -> None:
    parameter, optimizer = _initialized_optimizer(amsgrad=True)
    state = optimizer.state[parameter]
    original_step = state["step"]
    expected = {
        key: state[key].clone() for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")
    }

    moved_bytes = _move_adamw_moments(
        SimpleNamespace(optimizer=optimizer),
        torch.device("cpu"),
    )

    assert _adamw_optimizer(SimpleNamespace(optimizer=optimizer)) is optimizer
    assert state["step"] is original_step
    assert moved_bytes == sum(value.nbytes for value in expected.values())
    for key, value in expected.items():
        assert state[key].device.type == "cpu"
        torch.testing.assert_close(state[key], value)


def test_empty_adamw_state_has_no_moments_to_move() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter])

    assert _move_adamw_moments(optimizer, torch.device("cpu")) == 0


def test_optimizer_offload_rejects_unknown_state_layout() -> None:
    parameter, optimizer = _initialized_optimizer()
    optimizer.state[parameter]["unexpected"] = torch.tensor(1.0)

    with pytest.raises(ValueError, match="Unexpected AdamW optimizer state keys"):
        _move_adamw_moments(optimizer, torch.device("cpu"))


def test_optimizer_offload_requires_adamw() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))

    with pytest.raises(TypeError, match="requires torch.optim.AdamW"):
        _adamw_optimizer(torch.optim.SGD([parameter], lr=0.1))
