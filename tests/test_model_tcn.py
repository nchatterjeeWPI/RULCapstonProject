import numpy as np
from cmapss_rul import model_tcn

def test_tcn_build_compiles():
    model = model_tcn.build(input_shape=(30, 5))
    assert model is not None
    assert model.count_params() > 0

def test_tcn_train_default_runs_forward_pass():
    X_tr = np.random.rand(12, 30, 5).astype("float32")
    y_tr = np.random.rand(12).astype("float32")
    X_val = np.random.rand(4, 30, 5).astype("float32")
    y_val = np.random.rand(4).astype("float32")

    model, history = model_tcn.train_default(
        X_tr, y_tr, X_val, y_val, epochs=1, batch_size=2
    )
    assert "loss" in history.history
