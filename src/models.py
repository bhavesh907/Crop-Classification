"""Model definitions: MLP, 1-D CNN, and Bidirectional LSTM."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def build_mlp(
    input_dim: int,
    n_classes: int,
    dropout_rate: float = 0.3,
    l2_reg: float = 1e-5,
) -> keras.Model:
    """Fully-connected baseline operating on the flat feature vector."""
    return keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(l2_reg)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2_reg)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(n_classes, activation="softmax"),
        ],
        name="mlp",
    )


def build_cnn(
    n_timestamps: int,
    features_per_step: int,
    n_classes: int,
    dropout_rate: float = 0.3,
) -> keras.Model:
    """1-D CNN over the temporal axis with global average pooling."""
    return keras.Sequential(
        [
            layers.Input(shape=(n_timestamps, features_per_step)),
            layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv1D(128, kernel_size=3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(dropout_rate),
            layers.Dense(128, activation="relu"),
            layers.Dropout(dropout_rate),
            layers.Dense(n_classes, activation="softmax"),
        ],
        name="cnn",
    )


def build_lstm(
    n_timestamps: int,
    features_per_step: int,
    n_classes: int,
    dropout_rate: float = 0.3,
) -> keras.Model:
    """
    Bidirectional LSTM — the recommended architecture for temporal spectral data.

    Processing the sequence in both directions captures phenological patterns
    that grow toward and away from peak greenness.
    """
    return keras.Sequential(
        [
            layers.Input(shape=(n_timestamps, features_per_step)),
            layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
            layers.Dropout(dropout_rate),
            layers.Bidirectional(layers.LSTM(64)),
            layers.Dropout(dropout_rate),
            layers.Dense(128, activation="relu"),
            layers.Dropout(dropout_rate),
            layers.Dense(n_classes, activation="softmax"),
        ],
        name="bilstm",
    )
