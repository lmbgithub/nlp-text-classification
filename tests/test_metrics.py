"""Metrics checked against values computed by hand."""

import math

import pytest

from newsgroups.metrics import (
    ClassScore,
    accuracy,
    confusion_matrix,
    macro_f1,
    majority_baseline,
    report,
)

NAMES = ["a", "b", "c"]


def test_perfect_prediction():
    assert accuracy([0, 1, 2], [0, 1, 2]) == 1.0
    assert macro_f1([0, 1, 2], [0, 1, 2], NAMES) == 1.0


def test_everything_wrong():
    assert accuracy([0, 1], [1, 0]) == 0.0


def test_accuracy_is_hand_checkable():
    assert accuracy([0, 0, 1, 1], [0, 1, 1, 1]) == 0.75


def test_f1_of_a_known_confusion():
    # Class "a": 1 true positive, 1 false positive, 1 false negative.
    # precision = recall = 0.5, so F1 = 0.5.
    scores = {s.name: s for s in report([0, 0, 1], [0, 1, 0], NAMES).per_class}
    assert scores["a"].precision == 0.5
    assert scores["a"].recall == 0.5
    assert scores["a"].f1 == 0.5


def test_a_class_never_predicted_scores_zero_not_nan():
    scores = {s.name: s for s in report([0, 0, 1], [0, 0, 0], NAMES).per_class}
    assert scores["b"].precision == 0.0
    assert scores["b"].recall == 0.0
    assert scores["b"].f1 == 0.0


def test_absent_classes_still_appear_in_the_report():
    # Deriving the class list from observed labels would drop "c" entirely and
    # inflate macro-F1 by averaging over fewer, easier classes.
    result = report([0, 1], [0, 1], NAMES)
    assert [s.name for s in result.per_class] == NAMES
    assert result.macro_f1 == pytest.approx(2 / 3)


def test_macro_f1_weights_small_classes_equally():
    # 99 easy documents of class a, 1 of class b that is missed entirely.
    truth = [0] * 99 + [1]
    predicted = [0] * 100
    result = report(truth, predicted, ["a", "b"])
    assert result.accuracy == 0.99
    assert result.macro_f1 < 0.51  # accuracy says excellent, macro-F1 does not


def test_worst_classes_are_the_lowest_f1():
    truth = [0, 1, 2]
    predicted = [0, 0, 2]
    assert report(truth, predicted, NAMES).worst(1)[0].name == "b"


def test_f1_of_an_all_zero_class_score_is_zero():
    assert ClassScore(0, "a", 0, 0.0, 0.0).f1 == 0.0


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        accuracy([0, 1], [0])


def test_empty_evaluation_set_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        accuracy([], [])


def test_majority_baseline_on_balanced_classes_is_near_chance():
    truth = [i % 15 for i in range(150)]
    assert majority_baseline(truth) == pytest.approx(1 / 15)


def test_majority_baseline_on_a_skewed_split():
    assert majority_baseline([0] * 9 + [1]) == 0.9


def test_confusion_matrix_counts_rows_as_truth():
    matrix = confusion_matrix([0, 0, 1], [0, 1, 1], n_classes=2)
    assert matrix == [[1, 1], [0, 1]]


def test_confusion_matrix_rejects_out_of_range_labels():
    with pytest.raises(ValueError, match="out of range"):
        confusion_matrix([0, 5], [0, 0], n_classes=2)


def test_macro_f1_of_an_empty_class_list_is_undefined():
    assert math.isnan(report([0], [0], []).macro_f1)


def test_summary_reports_both_headline_numbers():
    text = report([0, 1], [0, 1], NAMES).summary()
    assert "accuracy" in text and "macro-F1" in text
