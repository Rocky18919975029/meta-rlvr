#!/usr/bin/env python3
"""Paired post-hoc statistics for a checkpoint evaluation directory."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--permutation-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _jsonl(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            records.extend(json.loads(line) for line in stream)
    return records


def _bootstrap_ci(
    values: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
) -> list[float]:
    means = np.empty(samples, dtype=np.float64)
    chunk_size = 2_000
    for start in range(0, samples, chunk_size):
        end = min(start + chunk_size, samples)
        indices = rng.integers(0, len(values), size=(end - start, len(values)))
        means[start:end] = values[indices].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def _sign_flip_pvalue(
    values: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
) -> float:
    observed = abs(values.mean())
    exceed = 0
    chunk_size = 2_000
    for start in range(0, samples, chunk_size):
        count = min(chunk_size, samples - start)
        signs = rng.integers(0, 2, size=(count, len(values)), dtype=np.int8)
        signs = signs * 2 - 1
        permuted = np.abs((signs * values).mean(axis=1))
        exceed += int(np.count_nonzero(permuted >= observed - 1e-15))
    return (exceed + 1) / (samples + 1)


def _exact_two_sided_binomial(first: int, second: int) -> float:
    trials = first + second
    if trials == 0:
        return 1.0
    tail = min(first, second)
    probability = 2.0 * sum(
        math.comb(trials, index) for index in range(tail + 1)
    ) / (2**trials)
    return min(1.0, probability)


def _continuous_paired(
    values: np.ndarray,
    *,
    bootstrap_samples: int,
    permutation_samples: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    positive = int(np.count_nonzero(values > 0))
    negative = int(np.count_nonzero(values < 0))
    zero = int(np.count_nonzero(values == 0))
    return {
        "mean": float(values.mean()),
        "percentage_points": float(values.mean() * 100),
        "standard_error": float(values.std(ddof=1) / math.sqrt(len(values))),
        "bootstrap_95_ci": _bootstrap_ci(
            values,
            samples=bootstrap_samples,
            rng=rng,
        ),
        "sign_flip_permutation_pvalue": _sign_flip_pvalue(
            values,
            samples=permutation_samples,
            rng=rng,
        ),
        "positive_problems": positive,
        "negative_problems": negative,
        "equal_problems": zero,
        "exact_sign_test_pvalue": _exact_two_sided_binomial(positive, negative),
    }


def _binary_paired(
    values: np.ndarray,
    *,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    improved = int(np.count_nonzero(values == 1))
    worsened = int(np.count_nonzero(values == -1))
    return {
        "mean": float(values.mean()),
        "percentage_points": float(values.mean() * 100),
        "bootstrap_95_ci": _bootstrap_ci(
            values,
            samples=bootstrap_samples,
            rng=rng,
        ),
        "improved_problems": improved,
        "worsened_problems": worsened,
        "equal_problems": int(np.count_nonzero(values == 0)),
        "exact_mcnemar_pvalue": _exact_two_sided_binomial(improved, worsened),
    }


def analyze(
    evaluation_dir: Path,
    *,
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
) -> dict[str, object]:
    config = json.loads(
        (evaluation_dir / "evaluation_config.json").read_text(encoding="utf-8")
    )
    problem_records = _jsonl(
        sorted(evaluation_dir.glob("problem-metrics-rank-*.jsonl"))
    )
    if len(problem_records) != config["num_unique_problems"]:
        raise ValueError("Problem metric count does not match evaluation_config.json.")
    by_uid = {record["problem_uid"]: record for record in problem_records}
    if len(by_uid) != len(problem_records):
        raise ValueError("Problem metrics contain duplicate problem_uid values.")
    ordered = [by_uid[uid] for uid in sorted(by_uid)]

    support_k = int(config["support_group_size"])
    base_k = int(config["base_query_group_size"])
    adapted_k = int(config["adapted_query_group_size"])
    support_accuracy = np.array(
        [record["support_accuracy"] for record in ordered], dtype=np.float64
    )
    base_accuracy = np.array(
        [record["base_query_accuracy"] for record in ordered], dtype=np.float64
    )
    adapted_accuracy = np.array(
        [record["adapted_query_accuracy"] for record in ordered], dtype=np.float64
    )
    base_total_accuracy = (
        support_k * support_accuracy + base_k * base_accuracy
    ) / (support_k + base_k)
    meta_total_accuracy = (
        support_k * support_accuracy + adapted_k * adapted_accuracy
    ) / (support_k + adapted_k)

    base_pass = np.array(
        [record["base_query_pass"] for record in ordered], dtype=np.int8
    )
    adapted_pass = np.array(
        [record["adapted_query_pass"] for record in ordered], dtype=np.int8
    )
    support_pass = np.array(
        [record["support_pass"] for record in ordered], dtype=np.int8
    )
    base_total_pass = np.maximum(support_pass, base_pass)
    meta_total_pass = np.maximum(support_pass, adapted_pass)

    support_rollouts = defaultdict(list)
    for record in _jsonl(sorted(evaluation_dir.glob("rollouts-rank-*.jsonl"))):
        if record["phase"] == "support":
            support_rollouts[record["problem_uid"]].append(record)
    if set(support_rollouts) != set(by_uid):
        raise ValueError("Support rollout problems do not match problem metrics.")

    separation = []
    macro_auc = []
    micro_wins = 0.0
    micro_pairs = 0
    mixed_uids = []
    pooled_confidence = []
    pooled_correctness = []
    for uid in sorted(support_rollouts):
        records = sorted(
            support_rollouts[uid], key=lambda record: record["response_index"]
        )
        if len(records) != support_k:
            raise ValueError(f"Support group {uid!r} does not have K={support_k}.")
        confidence = np.array(
            [record["confidence_probability"] for record in records],
            dtype=np.float64,
        )
        correctness = np.array(
            [record["correct"] for record in records], dtype=np.int8
        )
        pooled_confidence.extend(confidence.tolist())
        pooled_correctness.extend(correctness.tolist())
        correct_confidence = confidence[correctness == 1]
        incorrect_confidence = confidence[correctness == 0]
        if len(correct_confidence) == 0 or len(incorrect_confidence) == 0:
            continue
        comparisons = correct_confidence[:, None] - incorrect_confidence[None, :]
        wins = np.count_nonzero(comparisons > 0) + 0.5 * np.count_nonzero(
            comparisons == 0
        )
        pairs = comparisons.size
        separation.append(correct_confidence.mean() - incorrect_confidence.mean())
        macro_auc.append(wins / pairs)
        micro_wins += wins
        micro_pairs += pairs
        mixed_uids.append(uid)

    separation_array = np.asarray(separation, dtype=np.float64)
    auc_centered = np.asarray(macro_auc, dtype=np.float64) - 0.5
    query_delta_by_uid = {
        record["problem_uid"]: (
            record["adapted_query_accuracy"] - record["base_query_accuracy"]
        )
        for record in problem_records
    }
    mixed_query_delta = np.array(
        [query_delta_by_uid[uid] for uid in mixed_uids], dtype=np.float64
    )
    pooled_confidence_array = np.asarray(pooled_confidence, dtype=np.float64)
    pooled_correctness_array = np.asarray(pooled_correctness, dtype=np.float64)
    clipped = np.clip(pooled_confidence_array, 1e-12, 1 - 1e-12)

    rng = np.random.default_rng(seed)
    return {
        "event": "checkpoint_evaluation_posthoc_statistics",
        "evaluation_dir": str(evaluation_dir),
        "num_unique_problems": len(ordered),
        "bootstrap_samples": bootstrap_samples,
        "permutation_samples": permutation_samples,
        "analysis_seed": seed,
        "query_accuracy_delta": _continuous_paired(
            adapted_accuracy - base_accuracy,
            bootstrap_samples=bootstrap_samples,
            permutation_samples=permutation_samples,
            rng=rng,
        ),
        "equal_budget_total_accuracy_delta": _continuous_paired(
            meta_total_accuracy - base_total_accuracy,
            bootstrap_samples=bootstrap_samples,
            permutation_samples=permutation_samples,
            rng=rng,
        ),
        "query_pass_delta": _binary_paired(
            adapted_pass - base_pass,
            bootstrap_samples=bootstrap_samples,
            rng=rng,
        ),
        "equal_budget_total_pass_delta": _binary_paired(
            meta_total_pass - base_total_pass,
            bootstrap_samples=bootstrap_samples,
            rng=rng,
        ),
        "confidence_within_problem": {
            "mixed_problems": len(mixed_uids),
            "mean_correct_minus_incorrect": _continuous_paired(
                separation_array,
                bootstrap_samples=bootstrap_samples,
                permutation_samples=permutation_samples,
                rng=rng,
            ),
            "macro_pairwise_auc": float(np.mean(macro_auc)),
            "macro_pairwise_auc_minus_half": _continuous_paired(
                auc_centered,
                bootstrap_samples=bootstrap_samples,
                permutation_samples=permutation_samples,
                rng=rng,
            ),
            "micro_pairwise_auc": float(micro_wins / micro_pairs),
            "separation_query_gain_correlation": float(
                np.corrcoef(separation_array, mixed_query_delta)[0, 1]
            ),
        },
        "confidence_pooled_calibration": {
            "responses": len(pooled_confidence),
            "mean": float(pooled_confidence_array.mean()),
            "accuracy": float(pooled_correctness_array.mean()),
            "brier": float(
                np.mean((pooled_confidence_array - pooled_correctness_array) ** 2)
            ),
            "bce": float(
                np.mean(
                    -pooled_correctness_array * np.log(clipped)
                    - (1 - pooled_correctness_array) * np.log(1 - clipped)
                )
            ),
        },
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0 or args.permutation_samples <= 0:
        raise ValueError("Resampling counts must be positive.")
    evaluation_dir = args.evaluation_dir.resolve()
    result = analyze(
        evaluation_dir,
        bootstrap_samples=args.bootstrap_samples,
        permutation_samples=args.permutation_samples,
        seed=args.seed,
    )
    output = (
        evaluation_dir / "posthoc-statistics.json"
        if args.output is None
        else args.output.resolve()
    )
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    print(f"output={output}")


if __name__ == "__main__":
    main()
