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
from collections.abc import Iterable
import shutil
from pathlib import Path

import numpy as np
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv1D,
    Dropout,
    Add,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    Dense,
    BatchNormalization,
    Activation,
    SpatialDropout1D,
    Concatenate,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
# Keras Tuner import (only needed if tune() is called)
try:
    import keras_tuner as kt
except ImportError:
    kt = None  # fallback when keras-tuner is not installed

KERAS_TUNER_AVAILABLE = kt is not None
CallbackType = EarlyStopping | ReduceLROnPlateau


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
    h = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="causal",
        kernel_regularizer=l2(1e-4),
    )(x)
    h = BatchNormalization()(h)
    h = Activation("relu")(h)
    h = Dropout(dropout)(h)

    # Second causal conv stack
    h = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="causal",
        kernel_regularizer=l2(1e-4),
    )(h)
    h = BatchNormalization()(h)
    h = Activation("relu")(h)
    h = Dropout(dropout)(h)

    # If input channels != output channels, align with 1x1 conv so shapes match
    if x.shape[-1] != filters:
        x = Conv1D(
            filters=filters,
            kernel_size=1,
            padding="same",
            kernel_regularizer=l2(1e-4),
        )(x)

    # Residual add: output = transformed(x) + (possibly projected) x
    return Add()([x, h])


# ===============================================================
# 2) BUILD THE TCN MODEL
# ===============================================================
# Stacks multiple residual blocks with increasing dilation (1, 2, 4, …),
# then pools over time and finishes with a linear Dense(1) for RUL regression.
# ---------------------------------------------------------------
def build(input_shape: Tuple[int, ...],
          filters: int = 48,
          blocks: int = 4,
          kernel_size: int = 5,
          dropout: float = 0.2,
          lr: float = 1e-3,
          dense_units: int = 64,
          ) -> Model:

    """
    Build a TCN-style 1D model for RUL regression.

    Args:
        input_shape: (sequence_length, num_features)
        filters: base number of filters in residual blocks
        blocks: number of residual blocks; dilations = 1,2,4,... (2**i)
        kernel_size: convolution kernel size
        dropout: dropout rate inside blocks
        lr: Adam learning rate
        dense_units: number of units in the dense layer before the final output
    Returns:
        Compiled Keras Model.
    """
    inp = Input(shape=input_shape)
    x = inp
    x = SpatialDropout1D(0.1)(x)
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
    avg = GlobalAveragePooling1D()(x)
    mx = GlobalMaxPooling1D()(x)
    x = Concatenate()([avg, mx])

    x = Dense(dense_units, activation="relu")(x)
    x = Dropout(dropout)(x)

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
    callbacks: Optional[Iterable[CallbackType]] = None,
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
    cb: list[CallbackType] = [
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
# 4) HYPERPARAMETER TUNING HELPER - Build model from hp object
# ===============================================================
def build_tcn_model(hp):
    """
    Build a TCN model with hyperparameters from Keras Tuner.
    
    Args:
        hp: HyperParameters object from Keras Tuner
        
    Returns:
        Compiled Keras Model
    """
    # Define hyperparameter search space
    filters = hp.Int("filters", min_value=32, max_value=64, step=16)
    blocks = hp.Int("blocks", min_value=3, max_value=5)
    kernel_size = hp.Choice("kernel_size", values=[3, 5, 7])
    dropout = hp.Float("dropout", min_value=0.1, max_value=0.4, step=0.1)
    lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    
    # Get input shape from hp (set during tuner.search)
    input_shape = hp.get("input_shape")
    
    # Build model using the build() function
    return build(
        input_shape=input_shape,
        filters=filters,
        blocks=blocks,
        kernel_size=kernel_size,
        dropout=dropout,
        lr=lr
    )


# ===============================================================
# 5) OPTIONAL HYPERPARAMETER TUNING (KerasTuner Hyperband)
# ===============================================================
# Searches over key TCN hyperparameters (filters, blocks, kernel_size, dropout, lr).
# Uses Hyperband to allocate training budget efficiently. Returns:
#   best_model: a model built from the best hyperparameters
#   best_hp:    the HyperParameters object chosen by the tuner
#   tuner:      the tuner instance (for inspection / dashboards)
#   history:    training history of best_model
# ---------------------------------------------------------------
def tune(X_tr, y_tr, X_val, y_val, max_epochs=50, directory="tcn_tuning", project_name="cmapss_tcn"):
    """
    Hyperparameter tuning using Keras Tuner (Hyperband).
    Returns: (best_model, best_hyperparameters, tuner, history)
    """
    if not KERAS_TUNER_AVAILABLE:
        raise ImportError("keras-tuner is required for hyperparameter tuning. Install it with: pip install keras-tuner")
    
    # Clear any existing checkpoints that might cause shape conflicts
    tuning_dir = Path(directory) / project_name
    if tuning_dir.exists():
        print(f"[INFO] Cleaning up previous tuning directory: {tuning_dir}")
        shutil.rmtree(tuning_dir)

    # Store input_shape outside the hyperparameter space
    input_shape = X_tr.shape[1:]
    
    # Create a wrapper function that uses the captured input_shape
    def build_model_wrapper(hp):
        # Define hyperparameter search space
        filters = hp.Int("filters", min_value=48, max_value=128, step=16)
        blocks = hp.Int("blocks", min_value=3, max_value=6)
        kernel_size = hp.Choice("kernel_size", values=[3, 5, 7])
        dropout = hp.Float("dropout", min_value=0.1, max_value=0.5, step=0.1)
        lr = hp.Float("lr", min_value=1e-4, max_value=3e-3, sampling="log")

        # dense layer width for the head
        dense_units = hp.Choice("dense_units", values=[32, 64, 96])

        # optional: tune batch size too
        hp.Choice("batch_size", values=[32, 64, 96])

        return build(
            input_shape=input_shape,
            filters=filters,
            blocks=blocks,
            kernel_size=kernel_size,
            dropout=dropout,
            lr=lr,
            dense_units= dense_units,
        )
    '''
    tuner = kt.Hyperband(
        build_model_wrapper,
        objective="val_loss",
        max_epochs=max_epochs,
        factor=3,
        directory=directory,
        project_name=project_name,
        overwrite=True,  # Start fresh to avoid shape conflicts
        hyperband_iterations=1 # Reduced from 2 for saving memory
    )
    '''
    tuner = kt.RandomSearch(
        build_model_wrapper,
        objective="val_loss",
        max_trials=20,
        executions_per_trial=1,
        directory=directory,
        project_name=project_name,
        overwrite=True
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )
    
    print(f"[INFO] Starting hyperparameter search...")
    tuner.search(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        callbacks=[early_stop],
        verbose=1
    )
    
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    # best_model = tuner.get_best_models(num_models=1)[0]
    hp_values = best_hp.values
    bs = hp_values.get("batch_size", 64)
    # Retrain best model from scratch to get full history
    print("\n[INFO] Retraining best model with full epochs...")
    # Build model directly using best hyperparameters
    final_model = build(
        input_shape=input_shape,
        filters=best_hp.get("filters"),
        blocks=best_hp.get("blocks"),
        kernel_size=best_hp.get("kernel_size"),
        dropout=best_hp.get("dropout"),
        lr=best_hp.get("lr"),
        dense_units=best_hp.get("dense_units"),
    )
    history = final_model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=bs,
        callbacks=[early_stop],
        verbose=1
    )
    
    return final_model, best_hp, tuner, history
