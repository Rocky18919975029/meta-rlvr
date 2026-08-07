from types import SimpleNamespace

import pytest
import torch

from meta_rlvr.evaluate_checkpoint import (
    DEFAULT_EVALUATION_PARQUET,
    _adaptation_mode,
    _response_confidences,
    _summary_from_totals,
    _support_round_summaries,
)


def test_default_evaluation_dataset_is_aime24() -> None:
    assert DEFAULT_EVALUATION_PARQUET.name == "aime24.parquet"


def test_checkpoint_evaluation_summary_reports_query_and_total_budgets() -> None:
    summary = _summary_from_totals(
        torch.tensor(
            [
                6,
                8,
                4,
                8,
                5,
                8,
                2,
                1,
                2,
                2,
                3,
                4,
                2,
                1,
                1,
                4,
                8,
                3,
                4,
                1,
                4,
                2,
                4,
            ],
            dtype=torch.float64,
        ),
        support_group_size=2,
        base_query_group_size=2,
        adapted_query_group_size=2,
    )

    assert summary["num_unique_problems"] == 4
    assert summary["base_query"]["accuracy"] == 0.5
    assert summary["adapted_query"]["accuracy"] == 0.625
    assert summary["base_total"]["accuracy"] == 10 / 16
    assert summary["meta_total"]["accuracy"] == 11 / 16
    assert summary["base_total"]["pass_at_group"] == 0.5
    assert summary["meta_total"]["pass_at_group"] == 0.75
    assert summary["adapted_query_better_problems"] == 2
    assert summary["adapted_query_worse_problems"] == 1
    assert summary["adapted_query_equal_problems"] == 1
    assert summary["confidence"] == {
        "mean": 0.5,
        "correct_mean": 0.75,
        "incorrect_mean": 0.25,
        "brier": 0.25,
        "bce": 0.5,
        "responses": 8,
    }


def test_support_round_summaries_keep_rounds_separate() -> None:
    summaries = _support_round_summaries(
        torch.tensor(
            [
                [2, 4, 1, 2, 1.8, 1.2, 2, 0.6, 2],
                [3, 4, 2, 2, 2.4, 2.0, 3, 0.4, 1],
            ],
            dtype=torch.float64,
        )
    )

    assert summaries[0]["round"] == 1
    assert summaries[0]["accuracy"] == 0.5
    assert summaries[0]["pass_at_group"] == 0.5
    assert summaries[1]["round"] == 2
    assert summaries[1]["accuracy"] == 0.75
    assert summaries[1]["pass_at_group"] == 1.0


def test_checkpoint_adaptation_mode_requires_one_meta_branch() -> None:
    assert _adaptation_mode({"meta_coefficient": 1.0}) == "sequence"
    assert _adaptation_mode(
        {"meta_coefficient": 0.0, "token_meta_coefficient": 1.0}
    ) == "token"
    with pytest.raises(ValueError, match="exactly one"):
        _adaptation_mode({"meta_coefficient": 0.0})
    with pytest.raises(ValueError, match="exactly one"):
        _adaptation_mode(
            {"meta_coefficient": 1.0, "token_meta_coefficient": 1.0}
        )


def test_token_response_confidence_averages_only_completion_tokens() -> None:
    adaptation = SimpleNamespace(
        token_confidence_probabilities=torch.tensor(
            [[0.2, 0.4, 0.9], [0.1, 0.3, 0.5]]
        )
    )
    support = SimpleNamespace(
        completion_mask=torch.tensor(
            [[True, True, False], [True, True, True]]
        )
    )

    values = _response_confidences(adaptation, support, "token")

    assert values == pytest.approx([0.3, 0.3])
