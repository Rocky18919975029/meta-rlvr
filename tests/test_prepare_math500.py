from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq


def test_prepare_math500_produces_dapo_parquet(tmp_path: Path) -> None:
    input_path = tmp_path / "test.jsonl"
    output_path = tmp_path / "MATH-500.parquet"
    rows = [
        {
            "problem": f"What is {index}+1?",
            "solution": f"It is \\boxed{{{index + 1}}}.",
            "answer": str(index + 1),
            "subject": "Prealgebra",
            "level": 1,
            "unique_id": f"test/prealgebra/{index}.json",
        }
        for index in range(2)
    ]
    input_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_math500.py",
            "--input-jsonl",
            str(input_path),
            "--output-parquet",
            str(output_path),
        ],
        check=True,
    )

    converted = pq.read_table(output_path).to_pylist()
    assert len(converted) == 2
    assert converted[0]["data_source"] == "HuggingFaceH4/MATH-500"
    assert converted[0]["prompt"] == [
        {
            "role": "user",
            "content": (
                "What is 0+1? Let's think step by step and output the final "
                "answer within \\boxed{}."
            ),
        }
    ]
    assert converted[0]["reward_model"] == {
        "style": "rule",
        "ground_truth": "1",
    }
    assert converted[0]["extra_info"]["index"] == (
        "HuggingFaceH4/MATH-500:test/prealgebra/0.json"
    )
