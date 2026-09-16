"""20 Newsgroups text classification: a transformer against a dense baseline.

The package separates the parts that can be tested exactly — header stripping,
the train/validation split, truncation coverage, the metrics — from the parts
that need TensorFlow. Only `models` imports Keras, so the whole test suite runs
in under a second with no deep-learning framework installed.
"""

from newsgroups.corpus import Document, load_corpus, strip_headers
from newsgroups.metrics import ClassificationReport, accuracy, macro_f1
from newsgroups.split import Split, train_validation_split

__all__ = [
    "ClassificationReport",
    "Document",
    "Split",
    "accuracy",
    "load_corpus",
    "macro_f1",
    "strip_headers",
    "train_validation_split",
]
