# ===============================================================
# cmapss_rul/model_tcn.py
# ===============================================================
# This module defines a TCN (Temporal Convolutional Network) model for
# time-series regression of Remaining Useful Life (RUL).
#
# What’s inside:
#   1) _residual_block(): a causal 1D conv residual block (TCN building unit)
#   2) build(): assembles the full TCN model and compiles it
#   3) train_default(): trains the model with fixed hyperparameters
#   4) tune(): optional KerasTuner Hyperband search for TCN hyperparameters
#
# Why TCN?
# TCNs use causal, dilated convolutions to “look back” over time without leaking
# future information. Residual connections help gradients flow and stabilize training.
# ===============================================================

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


# ===============================================================
# 1) TCN RESIDUAL BLOCK (causal convolutions + residual skip)
# ===============================================================
# A residual block stacks two causal Conv1D layers, each followed by batch norm,
# ReLU activation, and dropout. If the input and output channel counts differ,
# a 1x1 convolution matches dimensions so we can add (skip connection).
# Dilations (1, 2, 4, …) let the network see further back in time efficiently.
# ---------------------------------------------------------------
def _residual_block(x, filters: int, kernel_size: int, dilation_rate: int, dropout: float):
    """
    A causal Temporal Convolutional (TCN) residual block:
      - Conv1D (causal) -> BN -> ReLU -> Dropout
      - Conv1D (causal) -> BN -> ReLU -> Dropout
      - 1x1 Conv skip if channels differ
      - Add skip connection
    """
    # First causal conv stack
    h = Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate,
               padding="causal")(x)
    h = BatchNormalization()(h)
    h = Activation("relu")(h)
    h = Dropout(dropout)(h)

    # Second causal conv stack
    h = Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate,
               padding="causal")(h)
    h = BatchNormalization()(h)
    h = Activation("relu")(h)
    h = Dropout(dropout)(h)

    # If input channels != output channels, align with 1x1 conv so shapes match
    if x.shape[-1] != filters:
        x = Conv1D(filters=filters, kernel_size=1, padding="same")(x)

    # Residual add: output = transformed(x) + (possibly projected) x
    return Add()([x, h])


# ===============================================================
# 2) BUILD THE TCN MODEL
# ===============================================================
# Stacks multiple residual blocks with increasing dilation (1, 2, 4, …),
# then pools over time and finishes with a linear Dense(1) for RUL regression.
# ---------------------------------------------------------------
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

    # Stack residual blocks with exponentially increasing dilation
    for i in range(blocks):
        x = _residual_block(
            x,
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=2 ** i,  # 1, 2, 4, ...
            dropout=dropout,
        )

    # Pool feature maps over time to a single vector
    x = GlobalAveragePooling1D()(x)

    # Final regression head: predict a single continuous RUL value
    out = Dense(1, activation="linear")(x)

    # Build and compile the model
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse", metrics=["mae"])
    return model


# ===============================================================
# 3) TRAIN WITH FIXED HYPERPARAMETERS
# ===============================================================
# Trains the TCN using early stopping (to avoid overfitting) and
# a ReduceLROnPlateau scheduler (to lower LR when validation stalls).
# Returns both the trained model and the Keras history object.
# ---------------------------------------------------------------
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
    # Build a fresh model using the supplied input shape and hyperparameters
    model = build(
        input_shape=X_tr.shape[1:],
        filters=filters,
        blocks=blocks,
        kernel_size=kernel_size,
        dropout=dropout,
        lr=lr,
    )

    # Default callbacks: early stopping + LR scheduler on validation loss
    cb = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
    ]
    if callbacks:
        cb.extend(callbacks)

    # Fit the model
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


# ===============================================================
# 4) OPTIONAL HYPERPARAMETER TUNING (KerasTuner Hyperband)
# ===============================================================
# Searches over key TCN hyperparameters (filters, blocks, kernel_size, dropout, lr).
# Uses Hyperband to allocate training budget efficiently. Returns:
#   best_model: a model built from the best hyperparameters
#   best_hp:    the HyperParameters object chosen by the tuner
#   tuner:      the tuner instance (for inspection / dashboards)
#   history:    training history of best_model
# ---------------------------------------------------------------
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
        # Friendly error if keras-tuner isn't installed
        raise ImportError("keras-tuner is required for tune(); pip install keras-tuner") from e

    # Define how to build a model from a set of hyperparameters
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

    # Set up a Hyperband tuner to minimize validation loss
    tuner = kt.Hyperband(
        hypermodel=build_from_hp,
        objective="val_loss",
        max_epochs=max_epochs,
        factor=3,
        directory=directory,
        project_name=project_name,
    )

    # Still keep early stopping in searches to avoid wasting time
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    # Run the search over hyperparameters
    tuner.search(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        callbacks=[es],
        verbose=1,
    )

    # Retrieve the best hyperparameters and build the corresponding model
    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_model = build_from_hp(best_hp)

    # Optional: allow batch_size to be part of HPs (fallback to 64 if not set)
    bs = best_hp.get("batch_size", 64) if hasattr(best_hp, "get") else 64

    # Train the best model configuration end-to-end
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
