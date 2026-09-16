"""Keras model factories. The only module that imports TensorFlow.

Both models are built on the same vectorized input so the comparison is about
architecture and nothing else: same vocabulary, same sequence length, same
split, same seed, same loss, same optimizer, same number of epochs.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_VOCAB_SIZE = 20_000
DEFAULT_SEQUENCE_LENGTH = 200


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Everything the two architectures share, in one place.

    Keeping this as a value object is what makes the comparison honest: it is
    impossible to give one model a larger vocabulary or a longer sequence by
    editing a literal in one cell and not the other.
    """

    vocab_size: int = DEFAULT_VOCAB_SIZE
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH
    n_classes: int = 15
    embed_dim: int = 32
    num_heads: int = 2
    ff_dim: int = 32
    dropout: float = 0.1
    seed: int = 1337

    def __post_init__(self) -> None:
        if self.n_classes < 2:
            raise ValueError(f"n_classes must be at least 2; got {self.n_classes}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {self.dropout}")
        if self.embed_dim % self.num_heads:
            raise ValueError(
                f"embed_dim {self.embed_dim} must divide evenly "
                f"by num_heads {self.num_heads}"
            )


def _keras():
    try:
        from tensorflow import keras
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "TensorFlow is not installed; run: pip install -r requirements.txt"
        ) from exc
    return keras


def build_transformer_block(config: ModelConfig):
    keras = _keras()
    layers = keras.layers

    class TransformerBlock(layers.Layer):
        """One pre-norm-free encoder block: attention, residual, feed-forward."""

        def __init__(self, embed_dim, num_heads, ff_dim, rate):
            super().__init__()
            self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
            self.ffn = keras.Sequential(
                [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
            )
            self.norm1 = layers.LayerNormalization(epsilon=1e-6)
            self.norm2 = layers.LayerNormalization(epsilon=1e-6)
            self.drop1 = layers.Dropout(rate)
            self.drop2 = layers.Dropout(rate)

        def call(self, inputs, training=None):
            # `training` is forwarded, never hardcoded. Calling this block with
            # `training=False` at build time — which is what the notebook this
            # replaces did — bakes inference mode into the graph and disables
            # both dropout layers for the entire training run. The model still
            # trains and still reports a validation accuracy; it is simply not
            # the model anyone thinks they are training.
            attended = self.drop1(self.att(inputs, inputs), training=training)
            out = self.norm1(inputs + attended)
            forwarded = self.drop2(self.ffn(out), training=training)
            return self.norm2(out + forwarded)

    return TransformerBlock(
        config.embed_dim, config.num_heads, config.ff_dim, config.dropout
    )


def build_token_and_position_embedding(config: ModelConfig):
    keras = _keras()
    layers = keras.layers

    class TokenAndPositionEmbedding(layers.Layer):
        """Token embeddings plus learned absolute position embeddings.

        Attention is permutation-invariant, so without this the transformer is
        a bag of words with extra steps — and would be beaten by the dense
        baseline for reasons that have nothing to do with attention.
        """

        def __init__(self, maxlen, vocab_size, embed_dim):
            super().__init__()
            self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
            self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
            self.maxlen = maxlen

        def call(self, x):
            import tensorflow as tf

            positions = tf.range(start=0, limit=tf.shape(x)[-1], delta=1)
            return self.token_emb(x) + self.pos_emb(positions)

    return TokenAndPositionEmbedding(
        config.sequence_length, config.vocab_size, config.embed_dim
    )


def build_transformer(config: ModelConfig):
    keras = _keras()
    layers = keras.layers

    inputs = layers.Input(shape=(config.sequence_length,))
    x = build_token_and_position_embedding(config)(inputs)
    x = build_transformer_block(config)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(config.dropout)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(config.dropout)(x)
    outputs = layers.Dense(config.n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="transformer")


def build_dense(config: ModelConfig):
    """The baseline: embed, flatten, one wide hidden layer.

    The output layer is sized from `config.n_classes`, never from a literal.
    The version this replaces ended in `Dense(20)` while training on fifteen
    categories: five output units that could never be correct, absorbing
    probability mass on every prediction, with no error anywhere.
    """
    keras = _keras()
    layers = keras.layers

    return keras.Sequential(
        [
            layers.Input(shape=(config.sequence_length,)),
            layers.Embedding(config.vocab_size, 10),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(config.n_classes, activation="softmax"),
        ],
        name="dense",
    )


def compile_model(model):
    """One loss, one optimizer, for both architectures.

    `sparse_categorical_crossentropy` because the labels are integers, not
    one-hot rows. `binary_crossentropy` on integer labels — which the notebook
    this replaces compiled with first, before recompiling — does not raise; it
    trains, converges to something, and reports a number that means nothing.
    """
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="rmsprop",
        metrics=["accuracy"],
    )
    return model
