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

# --- Add this block to model_lstm.py ---
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
        raise ImportError("keras-tuner is required for LSTM tune(); pip install keras-tuner") from e

    # Ensure proper dtypes/shapes
    X_tr = np.asarray(X_tr, dtype="float32")
    y_tr = np.asarray(y_tr, dtype="float32").reshape(-1, 1)
    X_val = np.asarray(X_val, dtype="float32")
    y_val = np.asarray(y_val, dtype="float32").reshape(-1, 1)

    # Remove any non-finite rows
    tr_mask = np.isfinite(X_tr).all(axis=(1, 2)) & np.isfinite(y_tr).ravel()
    va_mask = np.isfinite(X_val).all(axis=(1, 2)) & np.isfinite(y_val).ravel()
    X_tr, y_tr = X_tr[tr_mask], y_tr[tr_mask]
    X_val, y_val = X_val[va_mask], y_val[va_mask]

    def build_from_hp(hp):
        lstm1_units = hp.Choice("lstm1_units", [32, 64, 96, 128])
        lstm2_units = hp.Choice("lstm2_units", [16, 32, 48, 64])
        dense_units = hp.Choice("dense_units", [32, 64, 96, 128])
        dropout = hp.Float("dropout", 0.1, 0.4, step=0.1)
        recurrent_dropout = hp.Float("recurrent_dropout", 0.0, 0.3, step=0.1)
        lr = hp.Float("lr", 1e-4, 3e-3, sampling="log")
        # let Hyperband choose batch size too
        _ = hp.Choice("batch_size", [32, 64, 96])

        import tensorflow as tf
        from tensorflow.keras import Model
        from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam

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

    tuner = kt.Hyperband(
        hypermodel=build_from_hp,
        objective="val_loss",
        max_epochs=max_epochs,
        factor=3,
        directory=directory,
        project_name=project_name,
    )

    from tensorflow.keras.callbacks import EarlyStopping
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    tuner.search(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        callbacks=[es],
        verbose=1,
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_model = build_from_hp(best_hp)

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
# --- end add ---
