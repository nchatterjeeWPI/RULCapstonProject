# cmapss_rul/model_tcn.py
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv1D,
    Dropout,
    Add,
    GlobalAveragePooling1D,
    Dense,
    BatchNormalization,
    Activation,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback


def _residual_block(x, filters: int, kernel_size: int, dilation_rate: int, dropout: float):
    """
    A causal Temporal Convolutional (TCN) residual block:
      - Conv1D (causal) -> BN -> ReLU -> Dropout
      - Conv1D (causal) -> BN -> ReLU -> Dropout
      - 1x1 Conv skip if channels differ
      - Add skip connection
    """
    h = Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate,
               padding="causal")(x)
    h = BatchNormalization()(h)
    h = Activation("relu")(h)
    h = Dropout(dropout)(h)

    h = Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate,
               padding="causal")(h)
    h = BatchNormalization()(h)
    h = Activation("relu")(h)
    h = Dropout(dropout)(h)

    # Match channels for the residual connection if needed
    if x.shape[-1] != filters:
        x = Conv1D(filters=filters, kernel_size=1, padding="same")(x)

    return Add()([x, h])


def build(input_shape: Tuple[int, int],
          filters: int = 48,
          blocks: int = 4,
          kernel_size: int = 5,
          dropout: float = 0.2,
          lr: float = 1e-3) -> Model:
    """
    Build a TCN-style 1D model for RUL regression.

    Args:
        input_shape: (sequence_length, num_features)
        filters: base number of filters in residual blocks
        blocks: number of residual blocks; dilations = 1,2,4,... (2**i)
        kernel_size: convolution kernel size
        dropout: dropout rate inside blocks
        lr: Adam learning rate

    Returns:
        Compiled Keras Model.
    """
    inp = Input(shape=input_shape)
    x = inp
    for i in range(blocks):
        x = _residual_block(
            x,
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=2 ** i,
            dropout=dropout,
        )

    x = GlobalAveragePooling1D()(x)
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
    filters: int = 48,
    blocks: int = 4,
    kernel_size: int = 5,
    dropout: float = 0.2,
    callbacks: Optional[list[Callback]] = None,
):
    """
    Train the TCN with fixed hyperparameters.

    Returns:
        (trained_model, history)
    """
    model = build(
        input_shape=X_tr.shape[1:],
        filters=filters,
        blocks=blocks,
        kernel_size=kernel_size,
        dropout=dropout,
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


def tune(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_epochs: int = 60,
    directory: str = "tcn_tuning",
    project_name: str = "cmapss_tcn",
):
    """
    Optional Hyperband tuning for TCN. Returns (best_model, best_hp, tuner, history)
    """
    try:
        import keras_tuner as kt
    except Exception as e:
        raise ImportError("keras-tuner is required for tune(); pip install keras-tuner") from e

    def build_from_hp(hp):
        filters = hp.Choice("filters", [32, 48, 64, 96])
        blocks = hp.Int("blocks", min_value=3, max_value=6, step=1)
        kernel_size = hp.Choice("kernel_size", [3, 5, 7])
        dropout = hp.Float("dropout", 0.1, 0.5, step=0.1)
        lr = hp.Float("lr", 1e-4, 3e-3, sampling="log")
        return build(
            input_shape=X_tr.shape[1:],
            filters=filters,
            blocks=blocks,
            kernel_size=kernel_size,
            dropout=dropout,
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

    # Batch size from HP if present, else default
    bs = best_hp.get("batch_size", 64) if hasattr(best_hp, "get") else 64
    history = best_model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=bs,
        callbacks=[es],
        verbose=1,
    )
    return best_model, best_hp, tuner, history
