import json
from pathlib import Path

from meta_rlvr.plot_checkpoint_curve import _load_records


def _summary(step: int, method: str) -> dict:
    method_offset = 0.002 if method == "token" else 0.0
    base_accuracy = 0.20 + 0.001 * step + method_offset
    base_pass = 0.40 + 0.002 * step + method_offset
    adapted_accuracy = base_accuracy + 0.01 * step
    adapted_pass = base_pass + 0.02 * step
    return {
        "event": "checkpoint_evaluation_completed",
        "checkpoint_step": step,
        "adaptation_mode": method,
        "dataset_parquet": "/data/aime24.parquet",
        "seed": 42,
        "seconds": 10.0,
        "support": {
            "group_size": 16,
            "accuracy": 0.2 + 0.001 * step + method_offset,
        },
        "base_query": {
            "group_size": 32,
            "accuracy": base_accuracy,
            "pass_at_group": base_pass,
        },
        "adapted_query": {
            "group_size": 32,
            "accuracy": adapted_accuracy,
            "pass_at_group": adapted_pass,
        },
        "query_accuracy_delta": adapted_accuracy - base_accuracy,
        "query_pass_delta": adapted_pass - base_pass,
        "base_total": {
            "accuracy": base_accuracy,
            "pass_at_group": base_pass,
        },
        "meta_total": {"accuracy": 0.3, "pass_at_group": 0.5},
    }


def test_load_records_collects_sequence_and_token_curves(tmp_path: Path) -> None:
    evaluations = []
    for method, steps in (("sequence", range(1, 7)), ("token", range(1, 4))):
        for step in steps:
            result_dir = tmp_path / f"{method}-{step}"
            result_dir.mkdir()
            (result_dir / "summary.json").write_text(
                json.dumps(_summary(step, method)),
                encoding="utf-8",
            )
            evaluations.append(
                {
                    "method": method,
                    "step": step,
                    "result_dir": str(result_dir),
                }
            )
    manifest_path = tmp_path / "submission.json"
    manifest_path.write_text(
        json.dumps({"evaluations": evaluations}),
        encoding="utf-8",
    )

    records, _ = _load_records(manifest_path)

    assert len(records) == 9
    assert [
        record["step"] for record in records if record["method"] == "sequence"
    ] == list(range(1, 7))
    assert [
        record["step"] for record in records if record["method"] == "token"
    ] == list(range(1, 4))


def test_load_records_accepts_token_only_curve(tmp_path: Path) -> None:
    evaluations = []
    for step in range(1, 4):
        result_dir = tmp_path / f"token-{step}"
        result_dir.mkdir()
        (result_dir / "summary.json").write_text(
            json.dumps(_summary(step, "token")),
            encoding="utf-8",
        )
        evaluations.append(
            {"method": "token", "step": step, "result_dir": str(result_dir)}
        )
    manifest_path = tmp_path / "submission.json"
    manifest_path.write_text(
        json.dumps({"dataset_label": "MATH-500", "evaluations": evaluations}),
        encoding="utf-8",
    )

    records, manifest = _load_records(manifest_path)

    assert manifest["dataset_label"] == "MATH-500"
    assert [(record["method"], record["step"]) for record in records] == [
        ("token", 1),
        ("token", 2),
        ("token", 3),
    ]
