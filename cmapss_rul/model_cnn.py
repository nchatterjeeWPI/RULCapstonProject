# cmapss_rul/model_cnn.py
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv1D,
    BatchNormalization,
    Activation,
    Dropout,
    GlobalAveragePooling1D,
    Dense,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback


def build(input_shape: Tuple[int, int],
          filters: int = 64,
          kernel_size: int = 5,
          dropout: float = 0.2,
          dense_units: int = 64,
          lr: float = 1e-3) -> Model:
    """
    Build a simple 1D CNN for RUL regression.

    Args:
        input_shape: (sequence_length, num_features)
        filters: base number of conv filters
        kernel_size: kernel size for conv layers
        dropout: dropout rate
        dense_units: units in the penultimate Dense layer
        lr: learning rate for Adam

    Returns:
        Compiled Keras Model.
    """
    inp = Input(shape=input_shape)

    x = Conv1D(filters=filters, kernel_size=kernel_size, padding="causal")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv1D(filters=filters * 2, kernel_size=kernel_size, padding="causal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout)(x)

    x = Conv1D(filters=filters * 2, kernel_size=kernel_size, padding="causal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout)(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(dense_units, activation="relu")(x)
    out = Dense(1, activation="linear")(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse", metrics=["mae"])
    return model


def train_default(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int = 60,
    batch_size: int = 64,
    lr: float = 1e-3,
    filters: int = 64,
    kernel_size: int = 5,
    dropout: float = 0.2,
    dense_units: int = 64,
    callbacks: Optional[list[Callback]] = None,
) -> Tuple[Model, any]:
    """
    Train the CNN with fixed hyperparameters (no tuner).

    Args:
        X_tr, y_tr: training windows and labels
        X_val, y_val: validation windows and labels
        epochs: max epochs
        batch_size: batch size
        lr: learning rate
        filters, kernel_size, dropout, dense_units: model hyperparams
        callbacks: extra callbacks to append

    Returns:
        (trained_model, history)
    """
    input_shape = X_tr.shape[1:]
    model = build(
        input_shape=input_shape,
        filters=filters,
        kernel_size=kernel_size,
        dropout=dropout,
        dense_units=dense_units,
        lr=lr,
    )

    cb = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
    ]
    if callbacks:
        cb.extend(callbacks)

    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb,
        verbose=1,
    )
    return model, history


# Optional: KerasTuner support to match the unified interface
def tune(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_epochs: int = 60,
    directory: str = "cnn_tuning",
    project_name: str = "cmapss_cnn",
):
    """
    Run simple Hyperband tuning for the CNN. Returns (best_model, best_hp, tuner).
    """
    try:
        import keras_tuner as kt
    except Exception as e:
        raise ImportError("keras-tuner is required for tune(); pip install keras-tuner") from e

    def build_from_hp(hp):
        filters = hp.Choice("filters", [32, 48, 64, 96])
        kernel_size = hp.Choice("kernel_size", [3, 5, 7])
        dropout = hp.Float("dropout", 0.1, 0.5, step=0.1)
        dense_units = hp.Choice("dense_units", [32, 64, 96, 128])
        lr = hp.Float("lr", 1e-4, 3e-3, sampling="log")
        return build(
            input_shape=X_tr.shape[1:],
            filters=filters,
            kernel_size=kernel_size,
            dropout=dropout,
            dense_units=dense_units,
            lr=lr,
        )

    tuner = kt.Hyperband(
        hypermodel=build_from_hp,
        objective="val_loss",
        max_epochs=max_epochs,
        factor=3,
        directory=directory,
        project_name=project_name,
    )

    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    tuner.search(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        callbacks=[es],
        verbose=1,
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_model = build_from_hp(best_hp)
    history = best_model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=best_hp.get("batch_size", 64) if hasattr(best_hp, "get") else 64,
        callbacks=[es],
        verbose=1,
    )
    return best_model, best_hp, tuner, history
