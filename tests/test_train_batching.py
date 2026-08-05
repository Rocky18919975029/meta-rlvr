import pytest

from meta_rlvr.data import ChatMessage, MathProblem, rank_shard
from meta_rlvr.train import (
    _fixed_size_batches,
    _local_problem_batch_size,
    _problem_batch,
    _problem_batch_layout,
)


def _problems(count: int) -> list[MathProblem]:
    return [
        MathProblem(
            uid=str(index),
            messages=(ChatMessage(role="user", content=f"problem {index}"),),
            ground_truth=str(index),
            data_source="unit_test",
        )
        for index in range(count)
    ]


def test_problem_batch_size_is_global_and_divided_across_ranks() -> None:
    assert _local_problem_batch_size(None, world_size=4) == (4, 1)
    assert _local_problem_batch_size(32, world_size=4) == (32, 8)
    assert _local_problem_batch_size(32, world_size=8) == (32, 4)
    assert _local_problem_batch_size(64, world_size=8) == (64, 8)

    with pytest.raises(ValueError, match="divisible"):
        _local_problem_batch_size(32, world_size=6)
    with pytest.raises(ValueError, match="at least"):
        _local_problem_batch_size(2, world_size=4)


def test_problem_microbatch_layout_preserves_global_optimizer_batch() -> None:
    assert _problem_batch_layout(512, 32, world_size=8) == (
        512,
        64,
        32,
        4,
        16,
    )
    assert _problem_batch_layout(512, 32, world_size=4) == (
        512,
        128,
        32,
        8,
        16,
    )
    assert _problem_batch_layout(32, None, world_size=8) == (
        32,
        4,
        32,
        4,
        1,
    )

    with pytest.raises(ValueError, match="cannot exceed"):
        _problem_batch_layout(32, 64, world_size=8)
    with pytest.raises(ValueError, match="world size"):
        _problem_batch_layout(32, 12, world_size=8)
    with pytest.raises(ValueError, match="divisible by problem_micro"):
        _problem_batch_layout(96, 64, world_size=8)


def test_rollout_and_gradient_problem_batches_are_independent_and_ordered() -> None:
    assert _problem_batch_layout(
        32,
        32,
        world_size=4,
        sub_batch_name="rollout_problem_batch_size",
    ) == (32, 8, 32, 8, 1)
    assert _problem_batch_layout(32, 4, world_size=4) == (32, 8, 4, 1, 8)

    items = list(range(8))
    rollout_batches = _fixed_size_batches(items, 4)
    flattened = tuple(item for batch in rollout_batches for item in batch)
    gradient_microbatches = _fixed_size_batches(flattened, 1)
    assert rollout_batches == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert gradient_microbatches == [[index] for index in range(8)]

    with pytest.raises(ValueError, match="rollout_problem_batch_size"):
        _problem_batch_layout(
            32,
            12,
            world_size=8,
            sub_batch_name="rollout_problem_batch_size",
        )


def test_problem_batches_are_deterministic_and_advance_by_local_batch() -> None:
    shard = _problems(16)
    first = _problem_batch(shard, step=0, batch_size=4, seed=42)
    repeated = _problem_batch(shard, step=0, batch_size=4, seed=42)
    second = _problem_batch(shard, step=1, batch_size=4, seed=42)

    assert first == repeated
    assert len({problem.uid for problem in first}) == 4
    assert {problem.uid for problem in first}.isdisjoint(
        problem.uid for problem in second
    )


def test_four_rank_problem_batches_form_32_unique_tasks() -> None:
    problems = _problems(128)
    batch = []
    for rank in range(4):
        shard = rank_shard(problems, rank=rank, world_size=4)
        batch.extend(_problem_batch(shard, step=0, batch_size=8, seed=42))

    assert len(batch) == 32
    assert len({problem.uid for problem in batch}) == 32
