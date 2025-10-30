# ===============================================================
# cmapss_rul/model_lstm.py
# ===============================================================
# This module defines an LSTM-based model for time-series regression
# of Remaining Useful Life (RUL).
#
# What’s inside:
#   1) build_lstm(): constructs and compiles a 2-layer LSTM network
#   2) train_default(): trains with fixed hyperparameters (early stopping + LR scheduler)
#   3) tune(): optional KerasTuner Hyperband search for LSTM hyperparameters
#
# Why LSTM?
# LSTMs are recurrent neural networks designed to capture long-term temporal
# dependencies (memory) in sequences (e.g., sensor readings over time).
# ===============================================================

from typing import Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ===============================================================
# 1) BUILD THE LSTM MODEL
# ===============================================================
# Stack two LSTM layers (first returns sequences, second returns last timestep),
# then a Dense head for regression. We use dropout (and recurrent_dropout) to
# reduce overfitting, and gradient clipping to stabilize training.
# ---------------------------------------------------------------
def build_lstm(
    input_shape: Tuple[int, int],
    lstm1_units: int = 64,
    lstm2_units: int = 32,
    dense_units: int = 64,
    dropout: float = 0.2,
    recurrent_dropout: float = 0.1,
    lr: float = 1e-3,
) -> Model:
    # Clear any old TF graph/state (good hygiene between runs in notebooks/scripts)
    tf.keras.backend.clear_session()

    # Input shape: (sequence_length, num_features)
    inp = Input(shape=input_shape, dtype="float32")

    # LSTM #1: returns full sequence so LSTM #2 can process temporal outputs
    x = LSTM(
        lstm1_units,
        return_sequences=True,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout
    )(inp)

    # LSTM #2: returns only the final hidden state (summary of the sequence)
    x = LSTM(
        lstm2_units,
        return_sequences=False,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout
    )(x)

    # Dense head: non-linear layer before final regression
    x = Dense(dense_units, activation="relu")(x)
    x = Dropout(dropout)(x)

    # Final regression output: predict a single continuous RUL value
    out = Dense(1, activation="linear")(x)

    # Build and compile the model
    model = Model(inp, out)
    # Clip gradients (clipnorm) to improve stability on noisy sequences
    opt = Adam(learning_rate=lr, clipnorm=1.0)
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return model


# ===============================================================
# 2) TRAIN WITH FIXED HYPERPARAMETERS
# ===============================================================
# Trains the LSTM using:
#   - EarlyStopping (stops when val_loss stops improving, restores best weights)
#   - ReduceLROnPlateau (reduces LR when validation improvements plateau)
# Handles basic data sanitation (dtype casting, remove non-finite rows).
# ---------------------------------------------------------------
def train_default(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 60,
    batch_size: int = 64,
    lr: float = 1e-3,
    verbose: int = 1,
    **kwargs,  # absorbs unexpected keywords like max_epochs
):
    # Allow alias: some callers pass max_epochs; respect it if present
    if "max_epochs" in kwargs and kwargs["max_epochs"] is not None:
        epochs = kwargs["max_epochs"]

    # Safety: cast arrays to float32 and ensure correct shapes
    X_tr = np.asarray(X_tr, dtype="float32")
    y_tr = np.asarray(y_tr, dtype="float32").reshape(-1, 1)
    X_val = np.asarray(X_val, dtype="float32")
    y_val = np.asarray(y_val, dtype="float32").reshape(-1, 1)

    # Drop any rows containing NaNs/Infs to avoid training errors
    tr_mask = np.isfinite(X_tr).all(axis=(1, 2)) & np.isfinite(y_tr).ravel()
    va_mask = np.isfinite(X_val).all(axis=(1, 2)) & np.isfinite(y_val).ravel()
    X_tr, y_tr = X_tr[tr_mask], y_tr[tr_mask]
    X_val, y_val = X_val[va_mask], y_val[va_mask]

    # Build a fresh LSTM model for these shapes/hyperparams
    model = build_lstm(
        input_shape=X_tr.shape[1:],
        lstm1_units=64,
        lstm2_units=32,
        dense_units=64,
        dropout=0.2,
        recurrent_dropout=0.1,
        lr=lr,
    )

    # Callbacks: early stopping + LR scheduler
    es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    rlrop = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5)

    # Train the model
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es, rlrop],
        verbose=verbose,
    )
    # Return both the trained model and its learning curves
    return model, history  # <<< return tuple


# ===============================================================
# 3) OPTIONAL HYPERPARAMETER TUNING (KerasTuner Hyperband)
# ===============================================================
# Searches over key LSTM hyperparameters (layer sizes, dropouts, LR, batch size).
# Keeps the search space modest so CPU runs remain practical; you can widen it on GPU.
# Returns:
#   best_model: model built from best hyperparameters
#   best_hp:    HyperParameters object chosen by the tuner
#   tuner:      tuner instance (for inspection)
#   history:    training history of best_model
# ---------------------------------------------------------------
def tune(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_epochs: int = 60,
    directory: str = "lstm_tuning",
    project_name: str = "cmapss_lstm",
):
    """
    Hyperband tuning for LSTM. Returns (best_model, best_hp, tuner, history).
    Search space is modest to keep CPU runs practical; expand on GPU.
    """
    try:
        import keras_tuner as kt
    except Exception as e:
        # Friendly error if keras-tuner isn't installed
        raise ImportError("keras-tuner is required for LSTM tune(); pip install keras-tuner") from e

    # Ensure proper dtypes/shapes
    X_tr = np.asarray(X_tr, dtype="float32")
    y_tr = np.asarray(y_tr, dtype="float32").reshape(-1, 1)
    X_val = np.asarray(X_val, dtype="float32")
    y_val = np.asarray(y_val, dtype="float32").reshape(-1, 1)

    # Remove any non-finite rows to avoid tuner failures
    tr_mask = np.isfinite(X_tr).all(axis=(1, 2)) & np.isfinite(y_tr).ravel()
    va_mask = np.isfinite(X_val).all(axis=(1, 2)) & np.isfinite(y_val).ravel()
    X_tr, y_tr = X_tr[tr_mask], y_tr[tr_mask]
    X_val, y_val = X_val[va_mask], y_val[va_mask]

    # Define how to build a model from a set of hyperparameters
    def build_from_hp(hp):
        lstm1_units = hp.Choice("lstm1_units", [32, 64, 96, 128])
        lstm2_units = hp.Choice("lstm2_units", [16, 32, 48, 64])
        dense_units = hp.Choice("dense_units", [32, 64, 96, 128])
        dropout = hp.Float("dropout", 0.1, 0.4, step=0.1)
        recurrent_dropout = hp.Float("recurrent_dropout", 0.0, 0.3, step=0.1)
        lr = hp.Float("lr", 1e-4, 3e-3, sampling="log")
        # Let Hyperband pick batch size too
        _ = hp.Choice("batch_size", [32, 64, 96])

        # Build a candidate LSTM with the sampled hyperparameters
        tf.keras.backend.clear_session()
        inp = Input(shape=X_tr.shape[1:], dtype="float32")
        x = LSTM(lstm1_units, return_sequences=True, dropout=dropout,
                 recurrent_dropout=recurrent_dropout)(inp)
        x = LSTM(lstm2_units, return_sequences=False, dropout=dropout,
                 recurrent_dropout=recurrent_dropout)(x)
        x = Dense(dense_units, activation="relu")(x)
        x = Dropout(dropout)(x)
        out = Dense(1, activation="linear")(x)

        model = Model(inp, out)
        opt = Adam(learning_rate=lr, clipnorm=1.0)
        model.compile(optimizer=opt, loss="mse", metrics=["mae"])
        return model

    # Set up Hyperband tuner to minimize validation loss
    tuner = kt.Hyperband(
        hypermodel=build_from_hp,
        objective="val_loss",
        max_epochs=max_epochs,
        factor=3,
        directory=directory,
        project_name=project_name,
    )

    # Still keep early stopping to save time on bad trials
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    # Run the hyperparameter search
    tuner.search(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        callbacks=[es],
        verbose=1,
    )

    # Retrieve best HPs and build/train the best model configuration
    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_model = build_from_hp(best_hp)

    # If batch_size was part of HPs, use it; otherwise fall back to 64
    bs = best_hp.get("batch_size", 64) if hasattr(best_hp, "get") else 64

    history = best_model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=bs,
        callbacks=[es],
        verbose=1,
    )
    return best_model, best_hp, tuner, history
