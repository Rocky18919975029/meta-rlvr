#!/usr/bin/env python3
"""Convert canonical HuggingFaceH4/MATH-500 JSONL to DAPO parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


DATA_SOURCE = "HuggingFaceH4/MATH-500"
ANSWER_INSTRUCTION = (
    " Let's think step by step and output the final answer within \\boxed{}."
)
EXPECTED_KEYS = {
    "problem",
    "solution",
    "answer",
    "subject",
    "level",
    "unique_id",
}

SCHEMA = pa.schema(
    [
        pa.field("data_source", pa.string(), nullable=False),
        pa.field(
            "prompt",
            pa.list_(
                pa.struct(
                    [
                        pa.field("role", pa.string(), nullable=False),
                        pa.field("content", pa.string(), nullable=False),
                    ]
                )
            ),
            nullable=False,
        ),
        pa.field("ability", pa.string(), nullable=False),
        pa.field(
            "reward_model",
            pa.struct(
                [
                    pa.field("style", pa.string(), nullable=False),
                    pa.field("ground_truth", pa.string(), nullable=False),
                ]
            ),
            nullable=False,
        ),
        pa.field(
            "extra_info",
            pa.struct(
                [
                    pa.field("split", pa.string(), nullable=False),
                    pa.field("index", pa.string(), nullable=False),
                    pa.field("source_unique_id", pa.string(), nullable=False),
                    pa.field("subject", pa.string(), nullable=False),
                    pa.field("level", pa.int64(), nullable=False),
                    pa.field("solution", pa.string(), nullable=False),
                ]
            ),
            nullable=False,
        ),
    ]
)


def convert_row(row: dict[str, object]) -> dict[str, object]:
    if set(row) != EXPECTED_KEYS:
        raise ValueError(
            f"MATH-500 row keys differ from the canonical schema: {sorted(row)}"
        )

    problem = row["problem"]
    answer = row["answer"]
    unique_id = row["unique_id"]
    if not isinstance(problem, str) or not problem:
        raise TypeError("problem must be a non-empty string")
    if not isinstance(answer, str) or not answer:
        raise TypeError("answer must be a non-empty string")
    if not isinstance(unique_id, str) or not unique_id:
        raise TypeError("unique_id must be a non-empty string")

    return {
        "data_source": DATA_SOURCE,
        "prompt": [
            {
                "role": "user",
                "content": problem + ANSWER_INSTRUCTION,
            }
        ],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": answer,
        },
        "extra_info": {
            "split": "test",
            "index": f"{DATA_SOURCE}:{unique_id}",
            "source_unique_id": unique_id,
            "subject": row["subject"],
            "level": row["level"],
            "solution": row["solution"],
        },
    }


def prepare(input_jsonl: Path, output_parquet: Path) -> None:
    with input_jsonl.open(encoding="utf-8") as handle:
        source_rows = [json.loads(line) for line in handle]

    converted = [convert_row(row) for row in source_rows]
    identifiers = [row["extra_info"]["index"] for row in converted]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("MATH-500 contains duplicate unique_id values")

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(converted, schema=SCHEMA)
    pq.write_table(table, output_parquet, compression="zstd")
    print(
        json.dumps(
            {
                "input": str(input_jsonl),
                "output": str(output_parquet),
                "rows": table.num_rows,
                "unique_ids": len(set(identifiers)),
            },
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare(args.input_jsonl, args.output_parquet)
