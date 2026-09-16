"""How much of each document survives a fixed sequence length.

`output_sequence_length=200` is the kind of constant that gets copied from a
tutorial and never questioned. On this corpus the median message is longer than
that, so the number quietly decides how much of the data the model is allowed to
see — and it belongs in the report, not in a layer constructor.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TruncationReport:
    sequence_length: int
    documents: int
    median_words: float
    p90_words: float
    p99_words: float
    fully_covered: float
    words_kept: float

    def summary(self) -> str:
        return (
            f"sequence length {self.sequence_length}\n"
            f"  documents            {self.documents:,}\n"
            f"  median words         {self.median_words:.0f}\n"
            f"  90th / 99th pct      {self.p90_words:.0f} / {self.p99_words:.0f}\n"
            f"  documents kept whole {self.fully_covered:.1%}\n"
            f"  of all words kept    {self.words_kept:.1%}"
        )


def word_counts(texts: Iterable[str]) -> list[int]:
    """Whitespace word counts — the same unit the vectorizer truncates in."""
    return [len(text.split()) for text in texts]


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile, `q` in [0, 100].

    Written out rather than pulled from numpy so the statistics half of this
    package keeps no third-party dependency, and so the interpolation rule is
    visible instead of assumed.
    """
    if not values:
        raise ValueError("percentile of an empty sequence is undefined")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must be in [0, 100]; got {q}")

    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])

    position = (len(ordered) - 1) * q / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def analyse(texts: Iterable[str], sequence_length: int) -> TruncationReport:
    """Report what a given sequence length costs on this corpus."""
    counts = word_counts(texts)
    if not counts:
        raise ValueError("cannot analyse truncation on an empty corpus")
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive; got {sequence_length}")

    total = sum(counts)
    kept = sum(min(c, sequence_length) for c in counts)
    return TruncationReport(
        sequence_length=sequence_length,
        documents=len(counts),
        median_words=percentile(counts, 50),
        p90_words=percentile(counts, 90),
        p99_words=percentile(counts, 99),
        fully_covered=sum(1 for c in counts if c <= sequence_length) / len(counts),
        # A corpus of empty documents keeps nothing of nothing; report 1.0
        # rather than dividing by zero.
        words_kept=kept / total if total else 1.0,
    )
