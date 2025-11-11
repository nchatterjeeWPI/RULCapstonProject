import numpy as np
from cmapss_rul import model_lstm

def test_lstm_build_compiles():
    model = model_lstm.build_lstm(input_shape=(30, 10))
    assert model is not None
    assert model.count_params() > 0

def test_lstm_train_default_runs():
    X_tr = np.random.rand(8, 30, 10)
    y_tr = np.random.rand(8)
    X_val = np.random.rand(4, 30, 10)
    y_val = np.random.rand(4)
    model, history = model_lstm.train_default(X_tr, y_tr, X_val, y_val, epochs=1)
    assert "loss" in history.history
