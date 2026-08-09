import sys

from meta_rlvr.evaluate_checkpoint import parse_args as parse_evaluation_args
from meta_rlvr.train import parse_args as parse_training_args


def test_training_defaults_use_untruncated_on_policy_sampling(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "meta-rlvr-train",
            "--train-parquet",
            "train.parquet",
            "--validation-parquet",
            "validation.parquet",
            "--output-dir",
            "output",
            "--max-new-tokens",
            "128",
        ],
    )

    args = parse_training_args()

    assert args.temperature == 1.0
    assert args.top_p == 1.0
    assert args.top_k == 0
    assert args.validation_temperature == 1.0
    assert args.validation_top_p == 0.7
    assert args.validation_top_k == 0


def test_checkpoint_evaluation_separates_adaptation_and_query_sampling(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "meta-rlvr-evaluate",
            "--checkpoint",
            "checkpoint-1",
            "--output-dir",
            "output",
            "--model",
            "model",
            "--vllm-base-urls",
            "http://127.0.0.1:8000",
        ],
    )

    args = parse_evaluation_args()

    assert args.adaptation_temperature == 1.0
    assert args.adaptation_top_p == 1.0
    assert args.adaptation_top_k == 0
    assert args.query_temperature == 1.0
    assert args.query_top_p == 0.7
    assert args.query_top_k == 0
