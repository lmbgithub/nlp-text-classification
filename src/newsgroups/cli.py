"""Train both architectures on 20 Newsgroups and print the comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from newsgroups import truncation
from newsgroups.corpus import CATEGORIES, CorpusError, counts_by_category, iter_texts
from newsgroups.experiment import compare, load_and_split, run
from newsgroups.models import ModelConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="newsgroups",
        description="Transformer against a dense baseline on 20 Newsgroups.",
    )
    parser.add_argument(
        "data_dir", type=Path, help="extracted 20_newsgroup directory (see README)"
    )
    parser.add_argument(
        "--architecture",
        choices=["transformer", "dense", "both"],
        default="both",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=200)
    parser.add_argument("--vocab-size", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--categories",
        default=",".join(CATEGORIES),
        help="comma-separated category directories to load",
    )
    parser.add_argument(
        "--corpus-only",
        action="store_true",
        help="report corpus and truncation statistics, then stop (needs no TensorFlow)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    categories = tuple(c.strip() for c in args.categories.split(",") if c.strip())
    if len(categories) < 2:
        print("at least two categories are needed to classify", file=sys.stderr)
        return 2

    try:
        documents, split = load_and_split(
            args.data_dir, categories=categories, seed=args.seed
        )
    except CorpusError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"documents {len(documents):,}   categories {len(categories)}")
    for category, count in sorted(counts_by_category(documents).items()):
        print(f"  {category:<26}{count:>6}")
    print(f"\ntrain / validation  {split.sizes[0]:,} / {split.sizes[1]:,}\n")
    print(truncation.analyse(iter_texts(documents), args.sequence_length).summary())

    if args.corpus_only:
        return 0

    config = ModelConfig(
        vocab_size=args.vocab_size,
        sequence_length=args.sequence_length,
        n_classes=len(categories),
        seed=args.seed,
    )
    architectures = (
        ["transformer", "dense"] if args.architecture == "both" else [args.architecture]
    )

    results = []
    for architecture in architectures:
        print(f"\n--- {architecture}")
        result = run(
            split,
            config,
            architecture=architecture,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        print(result.report.summary())
        results.append(result)

    print("\n=== comparison")
    print(compare(results))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
