from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported chat role: {self.role!r}")
        if not self.content:
            raise ValueError("Chat message content cannot be empty.")

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class MathProblem:
    uid: str
    messages: tuple[ChatMessage, ...]
    ground_truth: str
    data_source: str

    def __post_init__(self) -> None:
        if not self.uid:
            raise ValueError("Problem uid cannot be empty.")
        if not self.messages:
            raise ValueError("Problem messages cannot be empty.")
        if self.messages[-1].role != "user":
            raise ValueError("The final prompt message must have role='user'.")
        if not self.ground_truth:
            raise ValueError("Problem ground_truth cannot be empty.")
        if not self.data_source:
            raise ValueError("Problem data_source cannot be empty.")

    @property
    def conversation(self) -> list[dict[str, str]]:
        return [message.as_dict() for message in self.messages]


def parse_dapo_row(row: dict[str, Any]) -> MathProblem:
    required = {"prompt", "reward_model", "extra_info", "data_source"}
    missing = required.difference(row)
    if missing:
        raise KeyError(f"DAPO row is missing fields: {sorted(missing)}")

    raw_messages = row["prompt"]
    if not isinstance(raw_messages, list) or not raw_messages:
        raise TypeError("DAPO prompt must be a non-empty list of chat messages.")
    messages: list[ChatMessage] = []
    for index, message in enumerate(raw_messages):
        if not isinstance(message, dict):
            raise TypeError(f"DAPO prompt message {index} must be a mapping.")
        if set(message) != {"role", "content"}:
            raise ValueError(
                f"DAPO prompt message {index} must contain exactly role/content."
            )
        messages.append(
            ChatMessage(role=message["role"], content=message["content"])
        )

    reward_model = row["reward_model"]
    if not isinstance(reward_model, dict) or "ground_truth" not in reward_model:
        raise TypeError("DAPO reward_model must contain ground_truth.")
    ground_truth = reward_model["ground_truth"]
    if not isinstance(ground_truth, str):
        raise TypeError("DAPO ground_truth must be a string.")

    extra_info = row["extra_info"]
    if not isinstance(extra_info, dict) or "index" not in extra_info:
        raise TypeError("DAPO extra_info must contain index.")
    uid = str(extra_info["index"])
    data_source = row["data_source"]
    if not isinstance(data_source, str):
        raise TypeError("DAPO data_source must be a string.")

    return MathProblem(
        uid=uid,
        messages=tuple(messages),
        ground_truth=ground_truth,
        data_source=data_source,
    )


def load_unique_dapo_problems(path: str | Path) -> list[MathProblem]:
    parquet_path = Path(path)
    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)

    from datasets import load_dataset

    dataset = load_dataset(
        "parquet",
        data_files=str(parquet_path),
        split="train",
    )
    unique: dict[str, MathProblem] = {}
    for row_index, row in enumerate(dataset):
        problem = parse_dapo_row(row)
        existing = unique.get(problem.uid)
        if existing is None:
            unique[problem.uid] = problem
        elif existing != problem:
            raise ValueError(
                f"Rows sharing uid={problem.uid!r} disagree at row {row_index}."
            )
    if not unique:
        raise ValueError("DAPO parquet contains no problems.")
    return list(unique.values())


def load_semantically_unique_dapo_problems(
    path: str | Path,
) -> list[MathProblem]:
    """Load evaluation problems, collapsing replicated rollout prompts."""

    parquet_path = Path(path)
    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)

    from datasets import load_dataset

    dataset = load_dataset(
        "parquet",
        data_files=str(parquet_path),
        split="train",
    )
    unique: dict[
        tuple[tuple[tuple[str, str], ...], str, str], MathProblem
    ] = {}
    for row in dataset:
        problem = parse_dapo_row(row)
        key = (
            tuple(
                (message.role, message.content)
                for message in problem.messages
            ),
            problem.ground_truth,
            problem.data_source,
        )
        unique.setdefault(key, problem)
    if not unique:
        raise ValueError("Evaluation parquet contains no problems.")
    return list(unique.values())


def rank_shard(
    problems: list[MathProblem],
    *,
    rank: int,
    world_size: int,
) -> list[MathProblem]:
    if world_size <= 0:
        raise ValueError("world_size must be positive.")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must be in [0, world_size).")
    shard = problems[rank::world_size]
    if not shard:
        raise ValueError(
            f"Rank {rank} received no problems from a dataset of size {len(problems)}."
        )
    return shard
