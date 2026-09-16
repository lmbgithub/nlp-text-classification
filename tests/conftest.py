import pytest

from newsgroups.corpus import Document


@pytest.fixture
def make_document(tmp_path):
    def _make(text="body", label=0, category="sci.space"):
        return Document(text=text, label=label, category=category, path=tmp_path / "m")

    return _make


@pytest.fixture
def corpus_dir(tmp_path):
    """A miniature corpus on disk: two categories, three messages each."""

    def _build(categories=("sci.space", "rec.autos"), per_category=3):
        root = tmp_path / "20_newsgroup"
        for c_index, category in enumerate(categories):
            directory = root / category
            directory.mkdir(parents=True)
            for index in range(per_category):
                (directory / f"{index}").write_text(
                    f"Newsgroups: {category}\nFrom: someone@example.com\n"
                    f"Subject: test\n\nbody text {c_index} {index}\n",
                    encoding="latin-1",
                )
        return root

    return _build
