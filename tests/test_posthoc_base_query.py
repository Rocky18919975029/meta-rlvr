import json

from meta_rlvr.posthoc_base_query import _load_adapted_records, _summarize


def test_load_and_summarize_adapted_records(tmp_path) -> None:
    path = tmp_path / "rollouts-rank-0.jsonl"
    records = [
        {"step": 3, "phase": "validation_adapted", "problem_uid": "a", "correct": 1},
        {"step": 3, "phase": "validation_adapted", "problem_uid": "a", "correct": 0},
        {"step": 3, "phase": "validation_adapted", "problem_uid": "b", "correct": 0},
        {"step": 3, "phase": "validation_adapted", "problem_uid": "b", "correct": 0},
        {"step": 2, "phase": "validation_adapted", "problem_uid": "x", "correct": 1},
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    summary = _summarize(_load_adapted_records(tmp_path, step=3))

    assert summary == {
        "accuracy": 0.25,
        "pass_at_group": 0.5,
        "num_unique_problems": 2,
        "responses": 4,
        "group_size": 2,
    }
