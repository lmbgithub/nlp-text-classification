"""Reading 20 Newsgroups from an extracted directory tree.

Each category is a directory; each message is a file. The messages carry RFC-822
style headers, and what happens to those headers decides whether the whole
experiment measures anything at all.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

# The fifteen categories used in this experiment. The full set has twenty; the
# five omitted ones are the near-duplicate `comp.*` and `talk.*` pairs that make
# the confusion matrix hard to read without changing the conclusion. The list is
# explicit rather than "whatever directories exist" so a partial download fails
# loudly instead of silently training on twelve classes.
CATEGORIES: tuple[str, ...] = (
    "alt.atheism",
    "comp.graphics",
    "comp.sys.mac.hardware",
    "comp.windows.x",
    "misc.forsale",
    "rec.autos",
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.crypt",
    "sci.med",
    "sci.space",
    "soc.religion.christian",
    "talk.politics.guns",
    "talk.politics.misc",
    "talk.religion.misc",
)

# Headers whose value is the label, or a trivial paraphrase of it. Leaving any
# of these in the body hands the classifier the answer.
LEAKING_HEADERS = ("newsgroups:", "followup-to:", "xref:", "path:")


class CorpusError(FileNotFoundError):
    """The dataset directory is missing or incomplete."""


@dataclass(frozen=True, slots=True)
class Document:
    text: str
    label: int
    category: str
    path: Path


def strip_headers(raw: str) -> str:
    """Remove the message headers, keeping the body.

    The headers end at the first blank line — that is the format, and this
    splits on it. The obvious shortcut is `lines[10:]`: drop a fixed ten lines
    and hope. It is wrong in both directions, and both failures are silent.

    * A message with fewer than ten header lines loses the start of its body.
    * A message with more than ten keeps the rest of its headers, including
      `Newsgroups: comp.graphics` — which *is* the label. The model then scores
      beautifully by reading the answer off the input, and the number means
      nothing.

    Any leaking header that survives an unconventional message layout is
    dropped explicitly as a second line of defence.
    """

    _head, separator, body = raw.partition("\n\n")
    if not separator:
        # No blank line anywhere: the file is a header block or a body, and
        # there is no way to tell. Treat it as a body with leaking lines removed
        # rather than discarding the document.
        body = raw
    return "\n".join(
        line for line in body.splitlines() if not _is_leaking_header(line)
    ).strip()


def _is_leaking_header(line: str) -> bool:
    return line.lower().startswith(LEAKING_HEADERS)


def load_corpus(
    root: str | Path,
    categories: Sequence[str] = CATEGORIES,
    *,
    encoding: str = "latin-1",
) -> list[Document]:
    """Load every message under `root`, one Document per file.

    `latin-1` decodes any byte sequence, which is what this corpus needs: the
    messages are from 1993 and carry a mix of encodings. Using UTF-8 with
    `errors="strict"` would fail on a handful of files; using it with
    `errors="ignore"` would silently delete characters.
    """

    root = Path(root)
    if not root.is_dir():
        raise CorpusError(
            f"dataset directory not found: {root}. See the README for how to download it."
        )

    documents: list[Document] = []
    for label, category in enumerate(categories):
        directory = root / category
        if not directory.is_dir():
            raise CorpusError(
                f"category directory missing: {directory}. The download is incomplete."
            )
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            text = strip_headers(path.read_text(encoding=encoding))
            if not text:
                continue  # header-only message; it carries no signal
            documents.append(
                Document(text=text, label=label, category=category, path=path)
            )

    if not documents:
        raise CorpusError(f"no usable messages under {root}")
    return documents


def counts_by_category(documents: Sequence[Document]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for doc in documents:
        counts[doc.category] = counts.get(doc.category, 0) + 1
    return counts


def iter_texts(documents: Sequence[Document]) -> Iterator[str]:
    for doc in documents:
        yield doc.text
