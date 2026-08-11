from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuously plot an on-policy evaluation by inner step."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Render once instead of watching until summary.json appears.",
    )
    return parser.parse_args()


def _read_complete_records(path: Path) -> list[dict]:
    records = []
    if not path.is_file():
        return records
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # The evaluator may still be writing the final line.
                continue
    return records


def _collect_records(run_dir: Path) -> list[dict]:
    records = []
    for path in sorted(run_dir.glob("rollouts-rank-*.jsonl")):
        records.extend(_read_complete_records(path))
    return records


def _summarize(
    records: list[dict],
    *,
    expected_problems: int | None = None,
) -> list[dict]:
    groups: dict[tuple[str, int, int], dict[str, object]] = defaultdict(
        lambda: {
            "correct": 0.0,
            "responses": 0,
            "passed_problems": set(),
            "problem_responses": defaultdict(int),
        }
    )
    for record in records:
        phase = record.get("phase")
        if phase == "support":
            round_number = int(record.get("adaptation_round", 1))
            series = "on_policy_support"
            inner_step = round_number - 1
        elif phase == "base_query":
            series = "paired_query"
            inner_step = 0
        elif phase == "adapted_query":
            series = "paired_query"
            inner_step = int(record.get("adaptation_round", 1))
        else:
            continue
        problem_uid = str(record["problem_uid"])
        key = (series, inner_step, 0)
        group = groups[key]
        correct = float(record["correct"])
        group["correct"] += correct
        group["responses"] += 1
        group["problem_responses"][problem_uid] += 1
        if correct == 1.0:
            group["passed_problems"].add(problem_uid)

    summaries = []
    for (series, inner_step, _), group in groups.items():
        problem_responses = group["problem_responses"]
        group_sizes = set(problem_responses.values())
        if len(group_sizes) != 1:
            # Do not publish a point while one or more problems are incomplete.
            continue
        group_size = group_sizes.pop()
        num_problems = len(problem_responses)
        if expected_problems is not None and num_problems != expected_problems:
            continue
        responses = int(group["responses"])
        summaries.append(
            {
                "series": series,
                "inner_step": inner_step,
                "group_size": group_size,
                "num_problems": num_problems,
                "responses": responses,
                "correct": int(group["correct"]),
                "accuracy": float(group["correct"]) / responses,
                "passed_problems": len(group["passed_problems"]),
                "pass_at_group": len(group["passed_problems"]) / num_problems,
            }
        )
    summaries.sort(key=lambda item: (item["series"], item["inner_step"]))
    return summaries


def _write_outputs(output_dir: Path, records: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "inner_step_curve.json"
    csv_path = output_dir / "inner_step_curve.csv"
    json_path.write_text(
        json.dumps({"records": records}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if records:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "on_policy_support": {
            "label": "On-policy support",
            "marker": "o",
            "color": "#d62728",
        },
        "paired_query": {
            "label": "Paired query",
            "marker": "s",
            "color": "#1f77b4",
        },
    }
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for series, style in styles.items():
        selected = [record for record in records if record["series"] == series]
        if not selected:
            continue
        group_sizes = sorted({record["group_size"] for record in selected})
        k_label = "/".join(str(size) for size in group_sizes)
        label = f"{style['label']} (K={k_label})"
        steps = [record["inner_step"] for record in selected]
        axes[0].plot(
            steps,
            [100 * record["accuracy"] for record in selected],
            marker=style["marker"],
            linewidth=2,
            label=label,
            color=style["color"],
        )
        axes[1].plot(
            steps,
            [100 * record["pass_at_group"] for record in selected],
            marker=style["marker"],
            linewidth=2,
            label=label,
            color=style["color"],
        )
    axes[0].set_title("Response accuracy")
    axes[0].set_ylabel("Accuracy (%)")
    axes[1].set_title("Pass rate (compare only within the same K)")
    axes[1].set_ylabel("Pass@K (%)")
    steps = sorted({record["inner_step"] for record in records})
    for axis in axes:
        axis.set_xlabel("Inner optimizer step")
        axis.set_xticks(steps)
        axis.grid(axis="y", alpha=0.25)
        if records:
            axis.legend(frameon=False)
    figure.suptitle("On-policy inner-loop evaluation (live)")
    temporary = output_dir / "inner_step_curve.tmp.png"
    figure.savefig(temporary, dpi=180, format="png")
    plt.close(figure)
    temporary.replace(output_dir / "inner_step_curve.png")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_dir / "live-inner-step-curve"
    )
    if args.interval <= 0:
        raise ValueError("interval must be positive.")
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    config_path = run_dir / "evaluation_config.json"
    expected_problems = None
    if config_path.is_file():
        expected_problems = int(
            json.loads(config_path.read_text(encoding="utf-8"))[
                "num_unique_problems"
            ]
        )

    last_signature = None
    while True:
        records = _summarize(
            _collect_records(run_dir),
            expected_problems=expected_problems,
        )
        signature = json.dumps(records, sort_keys=True)
        if signature != last_signature:
            _write_outputs(output_dir, records)
            print(
                json.dumps(
                    {
                        "event": "inner_step_curve_updated",
                        "points": len(records),
                        "curve": str(output_dir / "inner_step_curve.png"),
                        "records": records,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            last_signature = signature
        if args.once or (run_dir / "summary.json").is_file():
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
