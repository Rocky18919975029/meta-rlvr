import inspect
import json
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

from meta_rlvr.bilevel import BilevelGRPO
from meta_rlvr.train import (
    _generation_seed,
    _initialize_adamw_state_for_fsdp_load,
    _initialize_or_validate_run,
    _optimizer_step_values,
    _resolve_resume_checkpoint,
    _restore_committed_log_boundaries,
    _validate_optimizer_steps,
)


def _initialized_optimizer(steps: int = 2):
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        parameter.square().sum().backward()
        optimizer.step()
    return optimizer


def test_first_order_vjp_forward_batch_default_is_one() -> None:
    parameter = inspect.signature(BilevelGRPO.__init__).parameters[
        "first_order_vjp_forward_batch_size"
    ]
    assert parameter.default == 1


def test_generation_seed_is_stable_and_phase_specific() -> None:
    arguments = {
        "base_seed": 42,
        "step": 7,
        "rank": 3,
        "problem_uid": "problem-1",
    }
    first = _generation_seed(phase="train_support", **arguments)
    assert first == _generation_seed(phase="train_support", **arguments)
    assert first != _generation_seed(phase="train_query", **arguments)
    assert 0 <= first < 2**32


def test_optimizer_step_validation_detects_missing_or_stale_state() -> None:
    optimizer = _initialized_optimizer(steps=2)
    assert _optimizer_step_values(optimizer) == {2}
    _validate_optimizer_steps(optimizer, expected_steps=2)
    with pytest.raises(RuntimeError, match="not restored exactly"):
        _validate_optimizer_steps(optimizer, expected_steps=1)


def test_fsdp_optimizer_restore_materializes_adamw_state_first() -> None:
    restored = _initialized_optimizer(steps=2).state_dict()
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)

    _initialize_adamw_state_for_fsdp_load(
        optimizer,
        moment_device=torch.device("cpu"),
    )
    assert _optimizer_step_values(optimizer) == {0}

    optimizer.load_state_dict(restored)
    _validate_optimizer_steps(optimizer, expected_steps=2)


def test_partial_dcp_load_restores_state_and_preserves_param_groups(tmp_path) -> None:
    checkpoint = tmp_path / "optimizer_0"
    saved = {
        "optimizer": {
            "state": {"weight": {"step": torch.tensor(1.0)}},
            "param_groups": [{"lr": 0.01, "params": ["weight"]}],
        }
    }
    dist_cp.save(saved, checkpoint_id=checkpoint)
    current = {
        "optimizer": {
            "state": {"weight": {"step": torch.tensor(0.0)}},
            "param_groups": b"current optimizer groups",
        }
    }

    dist_cp.load(
        current,
        checkpoint_id=checkpoint,
        planner=DefaultLoadPlanner(allow_partial_load=True),
    )

    assert current["optimizer"]["state"]["weight"]["step"].item() == 1
    assert current["optimizer"]["param_groups"] == b"current optimizer groups"


def test_resume_restores_exact_append_only_log_boundaries(tmp_path) -> None:
    output = tmp_path / "run"
    checkpoint = output / "checkpoint-3"
    checkpoint.mkdir(parents=True)
    metrics = output / "metrics.jsonl"
    rollout = output / "rollouts-rank-0.jsonl"
    committed_metrics = b'{"event":"checkpoint_committed"}\n'
    committed_rollout = b'{"step":2}\n'
    metrics.write_bytes(committed_metrics + b'{"partial":true}\n')
    rollout.write_bytes(committed_rollout + b'{"step":3}\n')
    (checkpoint / "trainer_state.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "completed_steps": 3,
                "optimizer_steps": 6,
                "world_size": 1,
                "metrics_log_bytes": len(committed_metrics),
            }
        )
    )
    (checkpoint / "rank-0.json").write_text(
        json.dumps(
            {
                "rank": 0,
                "rollout_log_bytes": len(committed_rollout),
            }
        )
    )
    accelerator = SimpleNamespace(
        num_processes=1, process_index=0, is_main_process=True
    )

    state = _restore_committed_log_boundaries(
        checkpoint,
        output_dir=output,
        accelerator=accelerator,
    )

    assert state["completed_steps"] == 3
    assert metrics.read_bytes() == committed_metrics
    assert rollout.read_bytes() == committed_rollout


def test_resume_requires_latest_committed_checkpoint(tmp_path) -> None:
    output = tmp_path / "run"
    old = output / "checkpoint-1"
    latest = output / "checkpoint-2"
    old.mkdir(parents=True)
    latest.mkdir()
    (old / "trainer_state.json").write_text("{}")
    (latest / "trainer_state.json").write_text("{}")
    args = Namespace(output_dir=output, resume_from_checkpoint=old)

    with pytest.raises(ValueError, match="latest committed"):
        _resolve_resume_checkpoint(args)


def test_resume_configuration_is_strict_except_operational_fields(tmp_path) -> None:
    output = tmp_path / "run"
    accelerator = SimpleNamespace(num_processes=4, is_main_process=True)
    initial = Namespace(
        output_dir=output,
        resume_from_checkpoint=None,
        max_steps=10,
        save_steps=1,
        eval_steps=2,
        seed=42,
        inner_iterations=2,
        vllm_base_urls="http://127.0.0.1:10001",
    )
    _initialize_or_validate_run(initial, accelerator=accelerator)

    resumed = Namespace(
        output_dir=output,
        resume_from_checkpoint=output / "checkpoint-4",
        max_steps=20,
        save_steps=2,
        eval_steps=4,
        seed=42,
        inner_iterations=2,
        vllm_base_urls="http://127.0.0.1:20001",
    )
    _initialize_or_validate_run(resumed, accelerator=accelerator)

    resumed.inner_iterations = 3
    with pytest.raises(ValueError, match="inner_iterations"):
        _initialize_or_validate_run(resumed, accelerator=accelerator)


def test_token_credit_definition_is_strict_on_resume(tmp_path) -> None:
    output = tmp_path / "token-run"
    accelerator = SimpleNamespace(num_processes=4, is_main_process=True)
    legacy = Namespace(
        output_dir=output,
        resume_from_checkpoint=None,
        max_steps=1,
        save_steps=1,
        eval_steps=0,
        token_meta_coefficient=1.0,
    )
    _initialize_or_validate_run(legacy, accelerator=accelerator)
    saved = json.loads((output / "run_config.json").read_text())
    assert saved["token_credit_parameterization"] == "scaled_tanh_v1"
    assert saved["token_credit_cross_trajectory_normalization"] is False

    changed = Namespace(
        output_dir=output,
        resume_from_checkpoint=output / "checkpoint-1",
        max_steps=2,
        save_steps=1,
        eval_steps=0,
        token_meta_coefficient=1.0,
        token_credit_max=1.0,
    )
    with pytest.raises(ValueError, match="token_credit_max"):
        _initialize_or_validate_run(changed, accelerator=accelerator)
