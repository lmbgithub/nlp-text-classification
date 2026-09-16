"""Train/validation split that cannot silently decouple texts from labels."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from newsgroups.corpus import Document


@dataclass(frozen=True, slots=True)
class Split:
    train: tuple[Document, ...]
    validation: tuple[Document, ...]

    @property
    def sizes(self) -> tuple[int, int]:
        return len(self.train), len(self.validation)

    def texts(self, *, validation: bool = False) -> list[str]:
        return [d.text for d in (self.validation if validation else self.train)]

    def labels(self, *, validation: bool = False) -> list[int]:
        return [d.label for d in (self.validation if validation else self.train)]


def train_validation_split(
    documents: Sequence[Document], *, validation_fraction: float = 0.2, seed: int = 1337
) -> Split:
    """Shuffle and split, keeping each text bound to its own label.

    The version this replaces shuffled the text list and the label list as two
    separate calls, re-seeding a fresh generator in between so the two
    permutations happened to match. It works, and it is one edit away from not
    working: change the seed on one line, add a filter to one list, switch to a
    global RNG, and every label silently belongs to a different document. The
    model still trains, the accuracy lands near chance, and nothing anywhere
    says why.

    Shuffling `Document` objects makes that class of bug unrepresentable.
    """

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "validation_fraction must be strictly between 0 and 1; "
            f"got {validation_fraction}"
        )

    shuffled = list(documents)
    random.Random(seed).shuffle(shuffled)

    n_validation = int(validation_fraction * len(shuffled))
    if n_validation == 0 or n_validation == len(shuffled):
        raise ValueError(
            f"validation_fraction {validation_fraction} leaves an empty split "
            f"for {len(shuffled)} documents"
        )

    return Split(
        train=tuple(shuffled[:-n_validation]),
        validation=tuple(shuffled[-n_validation:]),
    )
