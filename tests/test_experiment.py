"""Comparison reporting, without training anything."""

import pytest

from newsgroups import metrics
from newsgroups.experiment import RunResult, compare

NAMES = ["a", "b"]


def result(name, acc, params, seconds=10.0):
    truth = [0, 1, 0, 1]
    correct = round(acc * 4)
    predicted = [0, 1, 0, 1][:correct] + [1 - v for v in [0, 1, 0, 1][correct:]]
    return RunResult(
        name=name,
        parameters=params,
        train_seconds=seconds,
        report=metrics.report(truth, predicted, NAMES),
        baseline_accuracy=0.5,
    )


def test_no_runs_is_not_a_crash():
    assert compare([]) == "no runs"


def test_a_single_run_reports_itself_without_a_comparison_line():
    text = compare([result("dense", 1.0, 1000)])
    assert "dense" in text
    assert "leads" not in text


def test_the_comparison_states_the_cost_of_the_winner():
    text = compare([result("dense", 0.5, 1_000), result("transformer", 1.0, 10_000)])
    assert "transformer leads dense" in text
    assert "10.0x the parameters" in text


def test_lift_over_the_baseline_is_reported_not_raw_accuracy():
    run = result("dense", 1.0, 100)
    assert run.lift_over_baseline == pytest.approx(0.5)


def test_summary_carries_accuracy_macro_f1_params_and_time():
    text = result("dense", 1.0, 1234, seconds=42.0).summary()
    for fragment in ("acc", "macro-F1", "1,234", "42s"):
        assert fragment in text
