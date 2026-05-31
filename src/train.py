"""Training pipeline with proper splits, scaling, class weights, and callbacks."""
import logging
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from .config import MODELS_DIR, N_TIMESTAMPS, RANDOM_SEED, RESULTS_DIR
from .features import compute_ndvi, get_feature_array
from .models import build_cnn, build_lstm, build_mlp

logger = logging.getLogger(__name__)


def train(
    df,
    model_type: str = "lstm",
    epochs: int = 50,
    batch_size: int = 256,
    test_size: float = 0.15,
    val_size: float = 0.15,
    dropout_rate: float = 0.3,
):
    """
    Train a crop-classification model and return evaluation artefacts.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``preprocess.load_dataset()``.
    model_type : {"mlp", "cnn", "lstm"}
    epochs : int
        Maximum training epochs (EarlyStopping will usually stop earlier).
    batch_size : int
    test_size : float
        Fraction of data held out as the final test set.
    val_size : float
        Fraction of data used for validation during training.
    dropout_rate : float

    Returns
    -------
    model, scaler, encoder, history, (X_test, y_test_cat)
    """
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = compute_ndvi(df)
    X_flat, y_raw = get_feature_array(df, include_ndvi=True)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_raw)
    joblib.dump(encoder, MODELS_DIR / "label_encoder.joblib")
    n_classes = len(encoder.classes_)
    y_cat = keras.utils.to_categorical(y_encoded, num_classes=n_classes)

    # ── Stratified train / val / test split ────────────────────────────────
    X_tv, X_test, y_tv_cat, y_test_cat, y_tv_enc, _ = train_test_split(
        X_flat, y_cat, y_encoded,
        test_size=test_size, random_state=RANDOM_SEED, stratify=y_encoded,
    )
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train_cat, y_val_cat, y_train_enc, _ = train_test_split(
        X_tv, y_tv_cat, y_tv_enc,
        test_size=val_ratio, random_state=RANDOM_SEED, stratify=y_tv_enc,
    )

    logger.info(
        "Split sizes — train: %d  val: %d  test: %d",
        len(X_train), len(X_val), len(X_test),
    )

    # ── Scaling (fit only on train) ────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

    # ── Class weights for imbalanced classes ───────────────────────────────
    class_weights_arr = compute_class_weight(
        "balanced", classes=np.unique(y_encoded), y=y_encoded
    )
    class_weights = dict(enumerate(class_weights_arr))
    logger.info("Class weights: %s", class_weights)

    # ── Build model ────────────────────────────────────────────────────────
    n_features = X_train_s.shape[1]
    features_per_step = n_features // N_TIMESTAMPS

    if model_type == "mlp":
        model = build_mlp(n_features, n_classes, dropout_rate=dropout_rate)
        X_tr, X_v, X_te = X_train_s, X_val_s, X_test_s
    else:
        X_tr = X_train_s.reshape(-1, N_TIMESTAMPS, features_per_step)
        X_v = X_val_s.reshape(-1, N_TIMESTAMPS, features_per_step)
        X_te = X_test_s.reshape(-1, N_TIMESTAMPS, features_per_step)
        if model_type == "cnn":
            model = build_cnn(N_TIMESTAMPS, features_per_step, n_classes, dropout_rate=dropout_rate)
        else:
            model = build_lstm(N_TIMESTAMPS, features_per_step, n_classes, dropout_rate=dropout_rate)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # ── Callbacks ──────────────────────────────────────────────────────────
    ckpt_path = str(MODELS_DIR / f"{model_type}_best.keras")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=7, restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_loss", save_best_only=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1,
        ),
        keras.callbacks.CSVLogger(str(RESULTS_DIR / f"{model_type}_history.csv")),
    ]

    history = model.fit(
        X_tr, y_train_cat,
        validation_data=(X_v, y_val_cat),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    model.save(str(MODELS_DIR / f"{model_type}_final.keras"))
    logger.info("Model saved → %s", MODELS_DIR / f"{model_type}_final.keras")

    return model, scaler, encoder, history, (X_te, y_test_cat)
