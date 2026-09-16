# nlp-text-classification — a transformer against a dense baseline

Text classification on 20 Newsgroups: how much does a transformer actually buy
over a well-tuned dense network on 20k documents — and what does it cost?

**Core package: standard library only.** TensorFlow is needed to train; nothing
else in the package imports it, so the suite runs in a tenth of a second with no
deep-learning framework installed. **85 tests.**

## Skills demonstrated

**NLP** — corpus ingestion and header stripping on raw Usenet messages, label
leakage detection and defence, vocabulary adaptation confined to the training
split, sequence-truncation analysis, text vectorisation

**Deep learning** — a transformer encoder block (multi-head self-attention,
token and positional embeddings, residual connections, layer normalisation)
against a dense baseline under an identical training configuration; correct
propagation of the `training` flag so dropout is active during training

**ML evaluation** — macro-F1 against accuracy on near-balanced classes,
per-class precision/recall/F1, confusion matrices, majority-class baselines,
classes absent from a split reported rather than dropped

**Software engineering** — standard-library core with the framework isolated to
one module, 85 tests that run in under a second with no TensorFlow installed,
immutable configuration objects, argparse CLI, ruff lint and format, pre-commit
gate, CI across Python 3.10-3.12

**Architecture** — the dependency boundary is the point: `models.py` is the only
module that imports TensorFlow, so corpus handling, splitting, truncation
analysis and metrics are all testable and fast without it

## What the run looks like

Training needs the dataset and TensorFlow. The transcript below is from the
offline example, which demonstrates the single decision that determines whether
any accuracy in this repository means anything:

```
$ python examples/leakage_demo.py
headers left in      accuracy 1.000   macro-F1 1.000   baseline 0.350
headers stripped     accuracy 0.325   macro-F1 0.325   baseline 0.350

sequence length 3
  documents            600
  median words         5
  90th / 99th pct      5 / 5
  documents kept whole 0.0%
  of all words kept    60.0%
```

A classifier that does nothing but search the text for a category name scores
**100%** while the headers are present, and drops to chance once they are
stripped. Every 20 Newsgroups message carries `Newsgroups: comp.graphics` — the
label — in its header block.

Full run:

```bash
python -m newsgroups ~/.keras/datasets/news20_extracted/20_newsgroup
python -m newsgroups <data_dir> --corpus-only    # statistics, no TensorFlow
```

## The six decisions worth discussing

**1. Headers end at the blank line; they are not ten lines long.** The common
version of this notebook drops `lines[10:]` and moves on. That is wrong in both
directions and both failures are silent: a message with fewer than ten header
lines loses the start of its body, and a message with more keeps
`Newsgroups: <category>` — the answer — inside the training text. The transcript
above is what the second failure buys you. `strip_headers` splits on the blank
line that actually delimits the header block, and drops any surviving leaking
header (`Newsgroups:`, `Followup-To:`, `Xref:`, `Path:`) as a second defence.

**2. Shuffle documents, not two parallel lists.** The original shuffles the text
list and the label list in two separate calls, re-seeding a fresh generator in
between so the permutations happen to match. It works, and it is one edit away
from not working — a changed seed, an added filter, a switch to a global RNG,
and every label belongs to a different document. Nothing raises; accuracy just
sits near chance forever. Shuffling `Document` objects makes the bug
unrepresentable.

**3. `training` is forwarded, never hardcoded.** The transformer block in the
original is called as `transformer_block(x, training=False)` at build time,
which bakes inference mode into the graph and disables both dropout layers for
the whole training run. The model trains, reports a validation accuracy, and is
not the model anyone believes they are training.

**4. The output layer is sized from the configuration.** The baseline ended in
`Dense(20)` while training on fifteen categories: five units that can never be
correct, quietly absorbing softmax probability on every prediction.
`ModelConfig.n_classes` is the single source of that number, and it is validated.

**5. The sequence length is reported, not assumed.** `output_sequence_length=200`
is copied from a tutorial and then decides how much of the corpus the model is
allowed to see. `truncation.analyse` reports two different numbers — the share of
documents kept whole and the share of *all words* kept — because they can point
in opposite directions: nine short messages and one long one give 90% of
documents intact and under 5% of the words.

**6. Macro-F1, not accuracy, decides the comparison.** The classes are close to
balanced, so accuracy is not meaningless — but the two architectures can tie on
accuracy while differing sharply on the small overlapping categories
(`talk.religion.misc` against `alt.atheism`). Macro-F1 weights every class
equally; classes absent from a split still appear in the report with F1 0, since
deriving the class list from observed labels inflates the average by quietly
dropping the hardest classes. Every accuracy is printed next to the
majority-class baseline, which on fifteen classes is about 0.067.

## Design

```
src/newsgroups/
  corpus.py       directory walk, header stripping, the leak defence
  split.py        shuffle-and-split over Documents
  truncation.py   what a fixed sequence length costs, with a stdlib percentile
  metrics.py      accuracy, per-class P/R/F1, macro-F1, confusion, baseline
  models.py       the only module that imports TensorFlow
  experiment.py   vectorize, train, score, compare
  cli.py          argument parsing and the report
```

Both architectures run through the same `run()` with the same `ModelConfig`
value object: same vocabulary, sequence length, split, seed, loss, optimizer and
epoch count. Keeping the configuration immutable is what stops the comparison
drifting — it is impossible to give one model a longer sequence by editing a
literal in one cell and not the other.

The vectorizer is adapted on the **training split only**. Adapting on the full
corpus leaks validation vocabulary into training; a small effect here, and
impossible to find later once it is spread across notebook cells.

## Usage

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[dev]"            # tests and the offline example
pip install -r requirements.txt    # TensorFlow, to train
```

```bash
pytest -q
ruff check . && ruff format --check .
pre-commit install                 # format and lint gate on every commit
python examples/leakage_demo.py
python -m newsgroups <data_dir> --corpus-only
python -m newsgroups <data_dir> --architecture transformer --epochs 20
```

As a library:

```python
from newsgroups import load_corpus, train_validation_split
from newsgroups.truncation import analyse

documents = load_corpus("~/.keras/datasets/news20_extracted/20_newsgroup")
split = train_validation_split(documents, seed=1337)
print(analyse((d.text for d in documents), 200).summary())
```

`main.ipynb` runs the same experiment as a narrative, importing the package
rather than redefining it.

## Dataset

**20 Newsgroups** — roughly 20,000 Usenet messages across 20 categories. Not
included; about 17 MB compressed.

```bash
curl -O http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.tar.gz
tar -xzf news20.tar.gz
```

That produces a `20_newsgroup/` directory with one subdirectory per category;
pass its path to the CLI. Keras can also fetch it:

```python
import keras

path = keras.utils.get_file(
    "news20.tar.gz",
    "http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.tar.gz",
    untar=True,
)
# data dir: <path>/../news20_extracted/20_newsgroup
```

This experiment uses fifteen of the twenty categories (see
`newsgroups.corpus.CATEGORIES`); the five omitted ones are near-duplicate
`comp.*` and `talk.*` pairs that crowd the confusion matrix without changing the
conclusion. The list is explicit, so an incomplete download raises instead of
quietly training on fewer classes.

## Scope

- **No pre-trained embeddings.** GloVe or a pre-trained encoder would beat both
  models here, which is exactly why they are excluded: the question is what the
  architecture buys on this much data, and a pre-trained vocabulary answers a
  different question.
- **No hyperparameter search.** Both models use the same defaults. Tuning one
  and not the other is the usual way a transformer comparison gets its result.
- **No repeated runs or confidence intervals.** A single seed shows the
  cost/benefit shape. It does not support a claim about a half-point accuracy
  gap, so no such claim is made here.
- **No fine-tuned BERT.** The comparison is a small transformer trained from
  scratch against a dense baseline, on identical inputs. Fine-tuning a
  pre-trained model is a different experiment with a foregone conclusion.
- **No inference server or export.** Nothing here is meant to be deployed.

## License

MIT — see [LICENSE](LICENSE).
