from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect checkpoint evaluations and plot learning curves."
    )
    parser.add_argument("--submission-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load_records(manifest_path: Path) -> tuple[list[dict], dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for entry in manifest["evaluations"]:
        summary_path = Path(entry["result_dir"]) / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["event"] != "checkpoint_evaluation_completed":
            raise ValueError(f"Incomplete evaluation: {summary_path}")
        if summary["checkpoint_step"] != entry["step"]:
            raise ValueError(f"Checkpoint step mismatch: {summary_path}")
        if summary["adaptation_mode"] != entry["method"]:
            raise ValueError(f"Adaptation mode mismatch: {summary_path}")
        records.append(
            {
                "method": entry["method"],
                "step": entry["step"],
                "result_dir": entry["result_dir"],
                "dataset_parquet": summary["dataset_parquet"],
                "seed": summary["seed"],
                "support_group_size": summary["support"]["group_size"],
                "query_group_size": summary["adapted_query"]["group_size"],
                "support_accuracy": summary["support"]["accuracy"],
                "base_query_accuracy": summary["base_query"]["accuracy"],
                "adapted_query_accuracy": summary["adapted_query"]["accuracy"],
                "query_accuracy_delta": summary["query_accuracy_delta"],
                "base_query_pass_at_group": summary["base_query"][
                    "pass_at_group"
                ],
                "adapted_query_pass_at_group": summary["adapted_query"][
                    "pass_at_group"
                ],
                "query_pass_delta": summary["query_pass_delta"],
                "base_total_accuracy": summary["base_total"]["accuracy"],
                "meta_total_accuracy": summary["meta_total"]["accuracy"],
                "base_total_pass_at_group": summary["base_total"][
                    "pass_at_group"
                ],
                "meta_total_pass_at_group": summary["meta_total"][
                    "pass_at_group"
                ],
                "evaluation_seconds": summary["seconds"],
            }
        )
    records.sort(key=lambda record: (record["method"], record["step"]))
    expected = {"sequence": list(range(1, 7)), "token": list(range(1, 4))}
    for method, steps in expected.items():
        actual = [record["step"] for record in records if record["method"] == method]
        if actual != steps:
            raise ValueError(f"Expected {method} steps {steps}, got {actual}.")
    comparable = {
        (
            record["dataset_parquet"],
            record["seed"],
            record["support_group_size"],
            record["query_group_size"],
        )
        for record in records
    }
    if len(comparable) != 1:
        raise ValueError("Evaluation configurations are not directly comparable.")
    return records, manifest


def _write_csv(path: Path, records: list[dict]) -> None:
    fields = list(records[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _plot(path_png: Path, path_pdf: Path, records: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = {
        "sequence": {
            "label": "Sequence confidence",
            "marker": "o",
            "color": "#1f77b4",
        },
        "token": {
            "label": "Token confidence",
            "marker": "s",
            "color": "#d62728",
        },
    }
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    panels = (
        ("adapted_query_accuracy", "Query accuracy", "Accuracy"),
        ("adapted_query_pass_at_group", "Query pass@32", "Pass@32"),
        ("meta_total_accuracy", "Total accuracy (support + query)", "Accuracy"),
        ("query_accuracy_delta", "Accuracy improvement over base", "Delta"),
        ("query_pass_delta", "Pass@32 improvement over base", "Delta"),
        (
            "meta_total_pass_at_group",
            "Total pass@48 (support + query)",
            "Pass@48",
        ),
    )
    base_fields = {
        "adapted_query_accuracy": "base_query_accuracy",
        "adapted_query_pass_at_group": "base_query_pass_at_group",
        "meta_total_accuracy": "base_total_accuracy",
        "meta_total_pass_at_group": "base_total_pass_at_group",
    }
    for axis, (field, title, ylabel) in zip(axes.flat, panels, strict=True):
        for method, style in methods.items():
            selected = [record for record in records if record["method"] == method]
            axis.plot(
                [record["step"] for record in selected],
                [100 * record[field] for record in selected],
                marker=style["marker"],
                linewidth=2,
                markersize=6,
                label=style["label"],
                color=style["color"],
            )
            if field in base_fields:
                axis.plot(
                    [record["step"] for record in selected],
                    [100 * record[base_fields[field]] for record in selected],
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.65,
                    label=f"{style['label']} base",
                    color=style["color"],
                )
        if field not in base_fields:
            axis.axhline(0, color="0.45", linestyle="--", linewidth=1.2)
        axis.set_title(title)
        axis.set_xlabel("Global training step")
        axis.set_ylabel(f"{ylabel} (%)")
        axis.set_xticks(range(1, 7))
        axis.grid(axis="y", alpha=0.25)
    for axis in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 2]):
        axis.legend(frameon=False)
    figure.suptitle("AIME24 checkpoint evaluation · seed 42 · support K=16 → query K=32")
    figure.savefig(path_png, dpi=200)
    figure.savefig(path_pdf)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    manifest_path = args.submission_manifest.resolve()
    output_dir = args.output_dir.resolve()
    records, manifest = _load_records(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(output_dir / "checkpoint_curve.csv", records)
    (output_dir / "checkpoint_curve.json").write_text(
        json.dumps(
            {"submission": manifest, "records": records},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _plot(
        output_dir / "checkpoint_curve.png",
        output_dir / "checkpoint_curve.pdf",
        records,
    )
    print(f"curve_csv={output_dir / 'checkpoint_curve.csv'}")
    print(f"curve_png={output_dir / 'checkpoint_curve.png'}")
    print(f"curve_pdf={output_dir / 'checkpoint_curve.pdf'}")


if __name__ == "__main__":
    main()
