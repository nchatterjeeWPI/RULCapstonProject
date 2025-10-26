# cmapss_rul/model_lstm.py
from typing import Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def build_lstm(
    input_shape: Tuple[int, int],
    lstm1_units: int = 64,
    lstm2_units: int = 32,
    dense_units: int = 64,
    dropout: float = 0.2,
    recurrent_dropout: float = 0.1,
    lr: float = 1e-3,
) -> Model:
    tf.keras.backend.clear_session()
    inp = Input(shape=input_shape, dtype="float32")
    x = LSTM(lstm1_units, return_sequences=True, dropout=dropout,
             recurrent_dropout=recurrent_dropout)(inp)
    x = LSTM(lstm2_units, return_sequences=False, dropout=dropout,
             recurrent_dropout=recurrent_dropout)(x)
    x = Dense(dense_units, activation="relu")(x)
    x = Dropout(dropout)(x)
    out = Dense(1, activation="linear")(x)
    model = Model(inp, out)
    opt = Adam(learning_rate=lr, clipnorm=1.0)  # gradient clipping for stability
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return model

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
    # allow alias: max_epochs
    if "max_epochs" in kwargs and kwargs["max_epochs"] is not None:
        epochs = kwargs["max_epochs"]

    # Safety: cast and drop any non-finite rows
    X_tr = np.asarray(X_tr, dtype="float32")
    y_tr = np.asarray(y_tr, dtype="float32").reshape(-1, 1)
    X_val = np.asarray(X_val, dtype="float32")
    y_val = np.asarray(y_val, dtype="float32").reshape(-1, 1)

    tr_mask = np.isfinite(X_tr).all(axis=(1, 2)) & np.isfinite(y_tr).ravel()
    va_mask = np.isfinite(X_val).all(axis=(1, 2)) & np.isfinite(y_val).ravel()
    X_tr, y_tr = X_tr[tr_mask], y_tr[tr_mask]
    X_val, y_val = X_val[va_mask], y_val[va_mask]

    model = build_lstm(
        input_shape=X_tr.shape[1:],
        lstm1_units=64,
        lstm2_units=32,
        dense_units=64,
        dropout=0.2,
        recurrent_dropout=0.1,
        lr=lr,
    )

    es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    rlrop = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5)

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es, rlrop],
        verbose=verbose,
    )
    return model, history  # <<< return tuple
