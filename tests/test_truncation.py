"""What a fixed sequence length costs, checked against hand-computed values."""

import pytest

from newsgroups.truncation import analyse, percentile, word_counts


def test_word_counts_split_on_whitespace():
    assert word_counts(["one two three", "", "  spaced   out  "]) == [3, 0, 2]


def test_percentile_of_a_single_value():
    assert percentile([5.0], 99) == 5.0


def test_percentile_endpoints_are_the_extremes():
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 0) == 1.0
    assert percentile(values, 100) == 4.0


def test_percentile_interpolates_linearly():
    # position = (4-1) * 0.5 = 1.5 -> halfway between 2 and 3.
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5


def test_percentile_does_not_need_sorted_input():
    assert percentile([4.0, 1.0, 3.0, 2.0], 50) == 2.5


def test_percentile_of_nothing_is_an_error():
    with pytest.raises(ValueError, match="empty"):
        percentile([], 50)


@pytest.mark.parametrize("q", [-1, 101])
def test_percentile_rejects_out_of_range_q(q):
    with pytest.raises(ValueError, match=r"\[0, 100\]"):
        percentile([1.0, 2.0], q)


def test_coverage_when_nothing_is_truncated():
    report = analyse(["a b", "c d e"], sequence_length=10)
    assert report.fully_covered == 1.0
    assert report.words_kept == 1.0


def test_coverage_when_everything_is_truncated():
    # Two documents of 10 words, length 5: half of every document survives.
    report = analyse([" ".join("w" * 1 for _ in range(10))] * 2, sequence_length=5)
    assert report.fully_covered == 0.0
    assert report.words_kept == pytest.approx(0.5)


def test_words_kept_is_not_the_same_as_documents_kept_whole():
    # Nine short documents and one very long one: 90% of documents survive
    # intact while a minority of the words do. Reporting only one of these
    # numbers gives the opposite impression of the other.
    texts = ["a b"] * 9 + [" ".join(["w"] * 1000)]
    report = analyse(texts, sequence_length=5)
    assert report.fully_covered == 0.9
    assert report.words_kept < 0.05


def test_an_all_empty_corpus_does_not_divide_by_zero():
    report = analyse(["", ""], sequence_length=200)
    assert report.words_kept == 1.0
    assert report.median_words == 0


def test_empty_corpus_is_an_error():
    with pytest.raises(ValueError, match="empty corpus"):
        analyse([], sequence_length=200)


@pytest.mark.parametrize("length", [0, -5])
def test_non_positive_sequence_length_is_rejected(length):
    with pytest.raises(ValueError, match="positive"):
        analyse(["a b c"], sequence_length=length)


def test_summary_mentions_both_coverage_numbers():
    text = analyse(["a b c"], sequence_length=2).summary()
    assert "documents kept whole" in text
    assert "of all words kept" in text
