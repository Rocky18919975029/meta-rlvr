import torch

from meta_rlvr.evaluate_checkpoint import _summary_from_totals


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
