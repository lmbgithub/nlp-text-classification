"""Classification metrics, written out so the averaging rule is explicit.

Accuracy on its own is not enough here even though the classes are close to
balanced: the two models being compared can reach the same accuracy while
differing sharply on the small, semantically overlapping categories
(`talk.religion.misc` against `alt.atheism`). Macro-F1 weights every class
equally and makes that visible; micro-averaged F1 would collapse back to
accuracy and hide it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClassScore:
    label: int
    name: str
    support: int
    precision: float
    recall: float

    @property
    def f1(self) -> float:
        # Both zero means the class was never predicted and never present in a
        # correct prediction; F1 is 0, not a division by zero.
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)


@dataclass(frozen=True, slots=True)
class ClassificationReport:
    accuracy: float
    per_class: tuple[ClassScore, ...]

    @property
    def macro_f1(self) -> float:
        if not self.per_class:
            return float("nan")
        return sum(c.f1 for c in self.per_class) / len(self.per_class)

    def worst(self, n: int = 3) -> list[ClassScore]:
        return sorted(self.per_class, key=lambda c: c.f1)[:n]

    def summary(self) -> str:
        lines = [
            f"accuracy {self.accuracy:.4f}   macro-F1 {self.macro_f1:.4f}",
            f"{'class':<26}{'support':>8}{'prec':>8}{'recall':>8}{'f1':>8}",
        ]
        for score in self.per_class:
            lines.append(
                f"{score.name:<26}{score.support:>8}{score.precision:>8.3f}"
                f"{score.recall:>8.3f}{score.f1:>8.3f}"
            )
        return "\n".join(lines)


def _check(truth: Sequence[int], predicted: Sequence[int]) -> None:
    if len(truth) != len(predicted):
        raise ValueError(
            "truth and predictions must be the same length; "
            f"got {len(truth)} and {len(predicted)}"
        )
    if not truth:
        raise ValueError("cannot score an empty evaluation set")


def accuracy(truth: Sequence[int], predicted: Sequence[int]) -> float:
    _check(truth, predicted)
    return sum(1 for t, p in zip(truth, predicted, strict=True) if t == p) / len(truth)


def report(
    truth: Sequence[int], predicted: Sequence[int], names: Sequence[str]
) -> ClassificationReport:
    """Per-class precision/recall/F1 over every class in `names`.

    Classes are taken from `names`, not from the labels observed: a class the
    model never predicts and that never appears in this split still belongs in
    the report, with support 0 and F1 0. Deriving the class list from the data
    makes a model that ignores a whole category look better, not worse.
    """

    _check(truth, predicted)
    scores = []
    for label, name in enumerate(names):
        true_positive = sum(
            1 for t, p in zip(truth, predicted, strict=True) if t == label and p == label
        )
        predicted_positive = sum(1 for p in predicted if p == label)
        actual_positive = sum(1 for t in truth if t == label)
        scores.append(
            ClassScore(
                label=label,
                name=name,
                support=actual_positive,
                precision=true_positive / predicted_positive
                if predicted_positive
                else 0.0,
                recall=true_positive / actual_positive if actual_positive else 0.0,
            )
        )
    return ClassificationReport(
        accuracy=accuracy(truth, predicted), per_class=tuple(scores)
    )


def macro_f1(
    truth: Sequence[int], predicted: Sequence[int], names: Sequence[str]
) -> float:
    return report(truth, predicted, names).macro_f1


def confusion_matrix(
    truth: Sequence[int], predicted: Sequence[int], n_classes: int
) -> list[list[int]]:
    """Rows are true classes, columns predicted."""
    _check(truth, predicted)
    matrix = [[0] * n_classes for _ in range(n_classes)]
    for t, p in zip(truth, predicted, strict=True):
        if not (0 <= t < n_classes and 0 <= p < n_classes):
            raise ValueError(f"label out of range for {n_classes} classes: ({t}, {p})")
        matrix[t][p] += 1
    return matrix


def majority_baseline(truth: Sequence[int]) -> float:
    """The accuracy of always predicting the most common class.

    Every reported accuracy is compared against this. On fifteen roughly
    balanced classes it sits near 0.07, which is exactly why it is worth
    printing: it puts the model's number on a scale.
    """
    _check(truth, truth)
    counts: dict[int, int] = {}
    for label in truth:
        counts[label] = counts.get(label, 0) + 1
    return max(counts.values()) / len(truth)
