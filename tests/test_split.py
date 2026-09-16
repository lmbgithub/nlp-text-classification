"""The split must keep every text bound to its own label."""

import pytest

from newsgroups.corpus import Document
from newsgroups.split import train_validation_split


def documents(n=100):
    return [
        Document(text=f"text {i}", label=i % 5, category=f"cat{i % 5}", path=None)
        for i in range(n)
    ]


def test_split_sizes_follow_the_fraction():
    split = train_validation_split(documents(100), validation_fraction=0.2)
    assert split.sizes == (80, 20)


def test_every_document_appears_exactly_once():
    docs = documents(50)
    split = train_validation_split(docs)
    assert sorted(d.text for d in (*split.train, *split.validation)) == sorted(
        d.text for d in docs
    )


def test_texts_and_labels_stay_paired_after_shuffling():
    # The whole point of shuffling Documents rather than two parallel lists.
    split = train_validation_split(documents(60))
    for document, text, label in zip(
        split.train, split.texts(), split.labels(), strict=True
    ):
        assert document.text == text
        assert document.label == label
        assert int(text.split()[-1]) % 5 == label


def test_the_same_seed_gives_the_same_split():
    a = train_validation_split(documents(40), seed=7)
    b = train_validation_split(documents(40), seed=7)
    assert [d.text for d in a.validation] == [d.text for d in b.validation]


def test_a_different_seed_gives_a_different_split():
    a = train_validation_split(documents(200), seed=1)
    b = train_validation_split(documents(200), seed=2)
    assert [d.text for d in a.validation] != [d.text for d in b.validation]


def test_the_split_actually_shuffles():
    docs = documents(200)
    split = train_validation_split(docs)
    assert [d.text for d in split.train] != [d.text for d in docs[:160]]


def test_splitting_does_not_mutate_the_caller_list():
    docs = documents(30)
    before = [d.text for d in docs]
    train_validation_split(docs)
    assert [d.text for d in docs] == before


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_out_of_range_fractions_are_rejected(fraction):
    with pytest.raises(ValueError, match="strictly between"):
        train_validation_split(documents(10), validation_fraction=fraction)


def test_a_fraction_that_rounds_to_an_empty_split_is_rejected():
    with pytest.raises(ValueError, match="empty split"):
        train_validation_split(documents(4), validation_fraction=0.2)


def test_validation_accessors_read_the_validation_half():
    split = train_validation_split(documents(50))
    assert len(split.texts(validation=True)) == len(split.validation)
    assert split.labels(validation=True) == [d.label for d in split.validation]
