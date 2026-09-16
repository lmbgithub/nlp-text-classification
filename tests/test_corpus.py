"""Header stripping is where this experiment is won or lost."""

import pytest

from newsgroups.corpus import (
    CATEGORIES,
    CorpusError,
    counts_by_category,
    load_corpus,
    strip_headers,
)

MESSAGE = (
    "Newsgroups: comp.graphics\n"
    "From: someone@example.com\n"
    "Subject: rendering\n"
    "\n"
    "How do I ray trace a sphere?\n"
)


def test_headers_are_removed_at_the_blank_line():
    assert strip_headers(MESSAGE) == "How do I ray trace a sphere?"


def test_the_label_never_survives_into_the_body():
    # This is the leak that makes a meaningless model look excellent.
    assert "comp.graphics" not in strip_headers(MESSAGE)


def test_a_short_header_block_keeps_the_whole_body():
    # The `lines[10:]` shortcut would eat the first line of this body.
    raw = "From: a@b\n\nline one\nline two\n"
    assert strip_headers(raw) == "line one\nline two"


def test_a_long_header_block_leaves_no_headers_behind():
    raw = "\n".join(f"X-Header-{i}: value" for i in range(30)) + "\n\nthe body\n"
    assert strip_headers(raw) == "the body"


def test_leaking_headers_are_dropped_even_without_a_blank_line():
    raw = "Newsgroups: sci.med\nreal content\n"
    assert strip_headers(raw) == "real content"


@pytest.mark.parametrize("header", ["Newsgroups:", "Followup-To:", "Xref:", "Path:"])
def test_every_leaking_header_is_recognised(header):
    raw = f"From: a@b\n\n{header} sci.crypt\nkept\n"
    assert strip_headers(raw) == "kept"


def test_leaking_header_matching_is_case_insensitive():
    raw = "From: a@b\n\nNEWSGROUPS: sci.crypt\nkept\n"
    assert strip_headers(raw) == "kept"


def test_a_body_quoting_the_category_in_prose_is_kept():
    raw = "From: a@b\n\nI also read comp.graphics sometimes\n"
    assert "comp.graphics" in strip_headers(raw)


def test_a_header_only_message_becomes_empty():
    assert strip_headers("From: a@b\nSubject: x\n\n\n") == ""


def test_empty_input_is_empty_output():
    assert strip_headers("") == ""


def test_quoted_reply_bodies_survive():
    raw = "From: a@b\n\n> previous message\nmy reply\n"
    assert strip_headers(raw) == "> previous message\nmy reply"


def test_load_corpus_labels_by_category_order(corpus_dir):
    documents = load_corpus(corpus_dir(), ("sci.space", "rec.autos"))
    assert {d.category: d.label for d in documents} == {"sci.space": 0, "rec.autos": 1}


def test_load_corpus_reads_every_file(corpus_dir):
    documents = load_corpus(corpus_dir(per_category=4), ("sci.space", "rec.autos"))
    assert len(documents) == 8
    assert counts_by_category(documents) == {"sci.space": 4, "rec.autos": 4}


def test_load_corpus_strips_headers_from_every_document(corpus_dir):
    documents = load_corpus(corpus_dir(), ("sci.space", "rec.autos"))
    assert all(not d.text.startswith("Newsgroups:") for d in documents)


def test_missing_root_says_where_to_get_the_data(tmp_path):
    with pytest.raises(CorpusError, match="README"):
        load_corpus(tmp_path / "absent")


def test_partial_download_fails_loudly(corpus_dir):
    # Training on twelve of fifteen categories because three directories are
    # missing must be an error, not a quieter experiment.
    root = corpus_dir(("sci.space",))
    with pytest.raises(CorpusError, match="incomplete"):
        load_corpus(root, ("sci.space", "rec.autos"))


def test_empty_categories_produce_an_error(tmp_path):
    root = tmp_path / "20_newsgroup"
    (root / "sci.space").mkdir(parents=True)
    with pytest.raises(CorpusError, match="no usable messages"):
        load_corpus(root, ("sci.space",))


def test_category_list_is_explicit_and_deduplicated():
    assert len(set(CATEGORIES)) == len(CATEGORIES) == 15
