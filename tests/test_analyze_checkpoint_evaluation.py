from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_posthoc_checkpoint_evaluation_statistics(tmp_path: Path) -> None:
    config = {
        "num_unique_problems": 4,
        "support_group_size": 2,
        "base_query_group_size": 2,
        "adapted_query_group_size": 2,
    }
    (tmp_path / "evaluation_config.json").write_text(json.dumps(config))
    metrics = [
        ("p0", 0.5, 0.0, 0.5, True, False, True),
        ("p1", 0.5, 0.5, 0.0, True, True, False),
        ("p2", 1.0, 0.5, 0.5, True, True, True),
        ("p3", 0.5, 0.0, 0.0, True, False, False),
    ]
    with (tmp_path / "problem-metrics-rank-0.jsonl").open("w") as stream:
        for uid, support, base, adapted, support_pass, base_pass, adapted_pass in metrics:
            stream.write(
                json.dumps(
                    {
                        "problem_uid": uid,
                        "support_accuracy": support,
                        "base_query_accuracy": base,
                        "adapted_query_accuracy": adapted,
                        "support_pass": support_pass,
                        "base_query_pass": base_pass,
                        "adapted_query_pass": adapted_pass,
                    }
                )
                + "\n"
            )
    confidence = {
        "p0": [(0.9, 1), (0.1, 0)],
        "p1": [(0.2, 1), (0.8, 0)],
        "p2": [(0.7, 1), (0.6, 1)],
        "p3": [(0.7, 1), (0.3, 0)],
    }
    with (tmp_path / "rollouts-rank-0.jsonl").open("w") as stream:
        for uid, rows in confidence.items():
            for index, (probability, correct) in enumerate(rows):
                stream.write(
                    json.dumps(
                        {
                            "phase": "support",
                            "problem_uid": uid,
                            "response_index": index,
                            "confidence_probability": probability,
                            "correct": correct,
                        }
                    )
                    + "\n"
                )

    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_checkpoint_evaluation.py",
            "--evaluation-dir",
            str(tmp_path),
            "--bootstrap-samples",
            "1000",
            "--permutation-samples",
            "1000",
            "--seed",
            "7",
        ],
        check=True,
    )
    result = json.loads((tmp_path / "posthoc-statistics.json").read_text())
    assert result["num_unique_problems"] == 4
    assert result["query_accuracy_delta"]["mean"] == 0.0
    assert result["query_accuracy_delta"]["positive_problems"] == 1
    assert result["query_accuracy_delta"]["negative_problems"] == 1
    assert result["confidence_within_problem"]["mixed_problems"] == 3
    assert result["confidence_within_problem"]["macro_pairwise_auc"] == 2 / 3
