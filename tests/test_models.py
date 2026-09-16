"""Configuration validation. Building the graphs needs TensorFlow; these do not."""

import pytest

from newsgroups.models import ModelConfig


def test_defaults_are_the_documented_ones():
    config = ModelConfig()
    assert config.vocab_size == 20_000
    assert config.sequence_length == 200


def test_the_class_count_is_configuration_not_a_literal():
    # The bug this guards: a Dense(20) output layer while training on fifteen
    # categories, so five units can never be correct and silently absorb
    # probability mass on every prediction.
    assert ModelConfig(n_classes=15).n_classes == 15


def test_a_single_class_problem_is_rejected():
    with pytest.raises(ValueError, match="at least 2"):
        ModelConfig(n_classes=1)


@pytest.mark.parametrize("dropout", [-0.1, 1.0, 1.5])
def test_out_of_range_dropout_is_rejected(dropout):
    with pytest.raises(ValueError, match="dropout"):
        ModelConfig(dropout=dropout)


def test_heads_must_divide_the_embedding_dimension():
    with pytest.raises(ValueError, match="divide evenly"):
        ModelConfig(embed_dim=32, num_heads=5)


def test_config_is_immutable_so_the_two_runs_cannot_diverge():
    config = ModelConfig()
    with pytest.raises(AttributeError):
        config.sequence_length = 400
