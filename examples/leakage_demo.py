"""Show what header stripping is actually protecting against — offline.

A synthetic corpus is built twice from the same bodies: once with the message
headers left in, once stripped. A trivial classifier that only looks for the
category name in the text then scores near-perfectly on the first and at chance
on the second.

That gap is the whole argument for `strip_headers`. A model trained on the
leaking version reports an excellent validation accuracy and has learnt nothing
about the text.

    python examples/leakage_demo.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from newsgroups.corpus import Document, strip_headers
from newsgroups.metrics import majority_baseline, report
from newsgroups.split import train_validation_split
from newsgroups.truncation import analyse

CATEGORIES = ("sci.space", "rec.autos", "sci.med")
BODIES = {
    "sci.space": "orbit launch telescope mission payload",
    "rec.autos": "engine tyres gearbox mileage dealer",
    "sci.med": "patient dosage symptoms diagnosis trial",
}
RNG = random.Random(1337)


def raw_message(category: str) -> str:
    words = BODIES[category].split()
    RNG.shuffle(words)
    return (
        f"Newsgroups: {category}\n"
        f"From: someone@example.com\n"
        f"Subject: a subject line\n"
        f"\n"
        f"{' '.join(words)}\n"
    )


def build(*, strip: bool) -> list[Document]:
    documents = []
    for label, category in enumerate(CATEGORIES):
        for _ in range(200):
            raw = raw_message(category)
            text = strip_headers(raw) if strip else raw
            documents.append(
                Document(text=text, label=label, category=category, path=None)
            )
    return documents


def cheating_classifier(text: str) -> int:
    """Predict by looking for a category name in the text; guess otherwise."""
    for label, category in enumerate(CATEGORIES):
        if category in text:
            return label
    return RNG.randrange(len(CATEGORIES))


def evaluate(*, strip: bool) -> None:
    split = train_validation_split(build(strip=strip), seed=1337)
    truth = split.labels(validation=True)
    predicted = [cheating_classifier(t) for t in split.texts(validation=True)]
    result = report(truth, predicted, list(CATEGORIES))
    label = "headers stripped" if strip else "headers left in"
    print(
        f"{label:<20} accuracy {result.accuracy:.3f}   "
        f"macro-F1 {result.macro_f1:.3f}   baseline {majority_baseline(truth):.3f}"
    )


def main() -> None:
    evaluate(strip=False)
    evaluate(strip=True)
    print()
    print(analyse((d.text for d in build(strip=True)), sequence_length=3).summary())


if __name__ == "__main__":
    main()
