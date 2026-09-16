"""CLI paths that need neither TensorFlow nor the dataset."""

import pytest

from newsgroups.cli import build_parser, main


def test_defaults():
    args = build_parser().parse_args(["data"])
    assert args.architecture == "both"
    assert args.epochs == 20
    assert args.sequence_length == 200


def test_unknown_architecture_is_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["data", "--architecture", "lstm"])


def test_missing_dataset_exits_with_a_usage_code(tmp_path, capsys):
    assert main([str(tmp_path / "absent")]) == 2
    assert "README" in capsys.readouterr().err


def test_a_single_category_cannot_be_classified(tmp_path, capsys):
    assert main([str(tmp_path), "--categories", "sci.space"]) == 2
    assert "at least two categories" in capsys.readouterr().err


def test_corpus_only_runs_without_tensorflow(corpus_dir, capsys):
    root = corpus_dir(("sci.space", "rec.autos"), per_category=20)
    assert main([str(root), "--corpus-only", "--categories", "sci.space,rec.autos"]) == 0
    out = capsys.readouterr().out
    assert "train / validation" in out
    assert "documents kept whole" in out
