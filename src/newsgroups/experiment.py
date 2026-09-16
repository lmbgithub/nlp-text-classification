"""Run both architectures under identical conditions and compare them."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from newsgroups import metrics
from newsgroups.corpus import CATEGORIES, Document, load_corpus
from newsgroups.models import ModelConfig, build_dense, build_transformer, compile_model
from newsgroups.split import Split, train_validation_split


@dataclass(frozen=True, slots=True)
class RunResult:
    name: str
    parameters: int
    train_seconds: float
    report: metrics.ClassificationReport
    baseline_accuracy: float

    @property
    def accuracy(self) -> float:
        return self.report.accuracy

    @property
    def lift_over_baseline(self) -> float:
        return self.accuracy - self.baseline_accuracy

    def summary(self) -> str:
        return (
            f"{self.name:<14}"
            f"acc {self.accuracy:.4f}  "
            f"macro-F1 {self.report.macro_f1:.4f}  "
            f"params {self.parameters:,}  "
            f"{self.train_seconds:.0f}s"
        )


def compare(results: Sequence[RunResult]) -> str:
    """Rank runs and state the cost of the winner.

    Accuracy alone is not the deliverable. A model that wins by half a point
    while costing six times the parameters and four times the training time is
    a different engineering decision from one that wins outright, and the
    comparison should make that visible rather than printing one number.
    """
    if not results:
        return "no runs"

    ordered = sorted(results, key=lambda r: r.report.macro_f1, reverse=True)
    lines = [r.summary() for r in ordered]
    best, *rest = ordered
    if rest:
        runner = rest[0]
        gap = best.report.macro_f1 - runner.report.macro_f1
        cost = best.parameters / runner.parameters if runner.parameters else float("inf")
        lines.append(
            f"\n{best.name} leads {runner.name} by {gap:+.4f} macro-F1 "
            f"at {cost:.1f}x the parameters"
        )
    return "\n".join(lines)


def load_and_split(
    root, *, categories: Sequence[str] = CATEGORIES, seed: int = 1337
) -> tuple[list[Document], Split]:
    documents = load_corpus(root, categories)
    return documents, train_validation_split(documents, seed=seed)


def run(
    split: Split,
    config: ModelConfig,
    *,
    architecture: str,
    epochs: int = 20,
    batch_size: int = 128,
    verbose: int = 2,
) -> RunResult:
    """Vectorize, train and score one architecture.

    The vectorizer is adapted on the **training split only**. Adapting it on the
    full corpus before splitting leaks validation vocabulary into training — a
    small effect on this dataset, and the kind of thing that is impossible to
    find later once it is spread across notebook cells.
    """

    import time

    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.layers import TextVectorization

    keras.utils.set_random_seed(config.seed)

    vectorizer = TextVectorization(
        max_tokens=config.vocab_size, output_sequence_length=config.sequence_length
    )
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(split.texts()).batch(128))

    x_train = vectorizer(np.array([[t] for t in split.texts()])).numpy()
    x_val = vectorizer(np.array([[t] for t in split.texts(validation=True)])).numpy()
    y_train = np.array(split.labels())
    y_val = np.array(split.labels(validation=True))

    builders = {"transformer": build_transformer, "dense": build_dense}
    if architecture not in builders:
        raise ValueError(
            f"unknown architecture {architecture!r}; expected {sorted(builders)}"
        )

    model = compile_model(builders[architecture](config))
    started = time.perf_counter()
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        verbose=verbose,
    )
    elapsed = time.perf_counter() - started

    predicted = model.predict(x_val, verbose=0).argmax(axis=1).tolist()
    categories = _category_names(split)
    return RunResult(
        name=architecture,
        parameters=int(model.count_params()),
        train_seconds=elapsed,
        report=metrics.report(y_val.tolist(), predicted, categories),
        baseline_accuracy=metrics.majority_baseline(y_val.tolist()),
    )


def _category_names(split: Split) -> list[str]:
    """Recover the label -> category mapping from the split itself."""
    mapping: dict[int, str] = {}
    for document in (*split.train, *split.validation):
        mapping.setdefault(document.label, document.category)
    return [mapping.get(label, f"class-{label}") for label in range(max(mapping) + 1)]
