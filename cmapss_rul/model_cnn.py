# ===============================================================
# cmapss_rul/model_cnn.py
# ===============================================================
# This module defines a compact 1D CNN (convolutional neural network)
# for time-series regression of Remaining Useful Life (RUL).
#
# What’s inside:
#   1) build(): assemble and compile the CNN
#   2) train_default(): train with fixed hyperparameters (early stop + LR scheduler)
#   3) tune(): optional KerasTuner Hyperband search over CNN hyperparameters
#
# Why a 1D CNN?
# 1D convolutions can detect short-to-medium temporal patterns (motifs)
# in sensor sequences. They’re fast, parallelizable, and often work well
# when patterns are local in time (e.g., a few to tens of steps).
# ===============================================================

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


# ===============================================================
# 1) BUILD THE CNN
# ===============================================================
# We stack several causal Conv1D blocks (so we don’t peek into the future),
# normalize with BatchNorm, apply ReLU nonlinearity, and use Dropout to reduce
# overfitting. GlobalAveragePooling1D collapses the time dimension into a
# single vector before a small Dense head predicts the RUL.
# ---------------------------------------------------------------
def build(
    input_shape: Tuple[int, int],
    filters: int = 64,
    kernel_size: int = 5,
    dropout: float = 0.2,
    dense_units: int = 64,
    lr: float = 1e-3
) -> Model:
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
    # Input tensor: a window of (sequence_length, num_features)
    inp = Input(shape=input_shape)

    # Conv block #1 (causal so time t depends only on ≤ t)
    x = Conv1D(filters=filters, kernel_size=kernel_size, padding="causal")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Conv block #2 (more filters, dropout for regularization)
    x = Conv1D(filters=filters * 2, kernel_size=kernel_size, padding="causal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout)(x)

    # Conv block #3 (same width as #2, more depth)
    x = Conv1D(filters=filters * 2, kernel_size=kernel_size, padding="causal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout)(x)

    # Pool across time to a single vector (no learnable params here)
    x = GlobalAveragePooling1D()(x)

    # Small dense head before the final linear regression output
    x = Dense(dense_units, activation="relu")(x)
    out = Dense(1, activation="linear")(x)  # predict a single continuous RUL value

    # Build + compile
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse", metrics=["mae"])
    return model


# ===============================================================
# 2) TRAIN WITH FIXED HYPERPARAMETERS
# ===============================================================
# Trains the CNN using:
#   - EarlyStopping (stop when val_loss plateaus; restore best weights)
#   - ReduceLROnPlateau (reduce LR when improvements stall)
# Returns both the trained model and the Keras History (learning curves).
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
    # Build a fresh model for the given input shape and hyperparams
    input_shape = X_tr.shape[1:]
    model = build(
        input_shape=input_shape,
        filters=filters,
        kernel_size=kernel_size,
        dropout=dropout,
        dense_units=dense_units,
        lr=lr,
    )

    # Default callbacks: early stopping + LR scheduler
    cb = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
    ]
    if callbacks:
        cb.extend(callbacks)

    # Train the model
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
# 3) OPTIONAL HYPERPARAMETER TUNING (KerasTuner Hyperband)
# ===============================================================
# Searches over key CNN hyperparameters (filters, kernel_size, dropout,
# dense_units, learning rate, optionally batch size).
# Returns:
#   best_model: model built from the best hyperparameters
#   best_hp:    HyperParameters object chosen by the tuner
#   tuner:      the tuner instance (for inspection / dashboards)
#   history:    training history of best_model
# ---------------------------------------------------------------
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
        # Friendly error if keras-tuner isn't installed
        raise ImportError("keras-tuner is required for tune(); pip install keras-tuner") from e

    # Define how to build a model from a set of hyperparameters
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

    # Hyperband tuner: explores many configs efficiently
    tuner = kt.Hyperband(
        hypermodel=build_from_hp,
        objective="val_loss",
        max_epochs=max_epochs,
        factor=3,
        directory=directory,
        project_name=project_name,
    )

    # Early stop inside the search to avoid wasting time on weak trials
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    # Run the search
    tuner.search(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        callbacks=[es],
        verbose=1,
    )

    # Retrieve best hyperparameters and build/train the corresponding model
    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_model = build_from_hp(best_hp)

    # If batch_size is part of HPs, use it; otherwise fall back to 64
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
