import json

import pytest

from meta_rlvr.watch_inner_step_curve import _collect_records, _summarize


def _record(phase, problem, correct, *, adaptation_round=None):
    record = {
        "phase": phase,
        "problem_uid": problem,
        "correct": correct,
    }
    if adaptation_round is not None:
        record["adaptation_round"] = adaptation_round
    return record


def test_live_inner_curve_maps_rollout_phases_to_policy_steps(tmp_path) -> None:
    records = [
        _record("support", "a", 1, adaptation_round=1),
        _record("support", "a", 0, adaptation_round=1),
        _record("support", "b", 0, adaptation_round=1),
        _record("support", "b", 0, adaptation_round=1),
        _record("support", "a", 1, adaptation_round=2),
        _record("support", "a", 1, adaptation_round=2),
        _record("support", "b", 0, adaptation_round=2),
        _record("support", "b", 1, adaptation_round=2),
        _record("base_query", "a", 0),
        _record("base_query", "a", 1),
        _record("base_query", "b", 0),
        _record("base_query", "b", 0),
        _record("adapted_query", "a", 1, adaptation_round=3),
        _record("adapted_query", "a", 1, adaptation_round=3),
        _record("adapted_query", "b", 1, adaptation_round=3),
        _record("adapted_query", "b", 0, adaptation_round=3),
    ]
    path = tmp_path / "rollouts-rank-0.jsonl"
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n{partial",
        encoding="utf-8",
    )

    summaries = _summarize(_collect_records(tmp_path), expected_problems=2)

    support = [r for r in summaries if r["series"] == "on_policy_support"]
    query = [r for r in summaries if r["series"] == "paired_query"]
    assert [r["inner_step"] for r in support] == [0, 1]
    assert [r["accuracy"] for r in support] == pytest.approx([0.25, 0.75])
    assert [r["inner_step"] for r in query] == [0, 3]
    assert [r["accuracy"] for r in query] == pytest.approx([0.25, 0.75])
    assert query[1]["pass_at_group"] == 1.0


def test_on_policy_curve_joins_support_rounds_and_final_rollout(tmp_path) -> None:
    records = [
        _record("support", "a", 0, adaptation_round=1),
        _record("support", "b", 1, adaptation_round=1),
        _record("support", "a", 1, adaptation_round=2),
        _record("support", "b", 1, adaptation_round=2),
        _record("base_query", "a", 0),
        _record("base_query", "b", 0),
        _record("adapted_query", "a", 1, adaptation_round=2),
        _record("adapted_query", "b", 0, adaptation_round=2),
    ]
    path = tmp_path / "rollouts-rank-0.jsonl"
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    summaries = _summarize(
        _collect_records(tmp_path),
        expected_problems=2,
        on_policy_inner_curve=True,
    )

    assert [record["series"] for record in summaries] == [
        "on_policy_rollout",
        "on_policy_rollout",
        "on_policy_rollout",
    ]
    assert [record["inner_step"] for record in summaries] == [0, 1, 2]
    assert [record["accuracy"] for record in summaries] == pytest.approx(
        [0.5, 1.0, 0.5]
    )
