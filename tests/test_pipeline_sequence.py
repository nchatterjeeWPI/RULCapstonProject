import numpy as np
import pandas as pd

from cmapss_rul import pipeline


def test_sequence_generation_creates_expected_keys(monkeypatch):
    """Validate sequence_generation() wires through to sequences helpers and
    returns a dictionary with the expected top-level keys.
    """

    # Minimal train/val/test dataframes — details ignored by the stubs
    base_df = pd.DataFrame(
        {
            "unit": [1, 1, 1],
            "cycle": [1, 2, 3],
            "sensor_1": [0.1, 0.2, 0.3],
            "RUL": [100.0, 99.0, 98.0],
        }
    )
    train_df = base_df.copy()
    val_df = base_df.copy()
    test_data = {"FD001": base_df.copy()}

    # --- Stub sequences helpers used inside pipeline.sequence_generation ---

    def fake_add_regime_onehot(df, K):
        # just passing the same df to keep code moving
        return df

    def fake_create_sequences(df, feature_cols, sequence_length):
        # Return small dummy arrays with shapes derived from params
        X = np.zeros((5, sequence_length, len(feature_cols)))
        y = np.zeros((5, 1))
        meta = {"dummy": True}
        return X, y, meta

    def fake_build_test_sequences_per_dataset(test_data_, sequence_length,
                                              feature_cols):
        datasets = list(test_data_.keys())
        X_test_dict = {
            fd: np.zeros((3, sequence_length, len(feature_cols))) for fd in
            datasets
        }
        y_test_dict = {fd: np.zeros((3, 1)) for fd in datasets}
        engine_ids_test_dict = {fd: np.array([1, 2, 3]) for fd in datasets}
        last_idx_map = {fd: np.array([2]) for fd in datasets}
        return X_test_dict, y_test_dict, engine_ids_test_dict, last_idx_map

    monkeypatch.setattr(pipeline.sequences, "add_regime_onehot", fake_add_regime_onehot)
    monkeypatch.setattr(pipeline.sequences, "create_sequences", fake_create_sequences)
    monkeypatch.setattr(
        pipeline.sequences,
        "build_test_sequences_per_dataset",
        fake_build_test_sequences_per_dataset,
    )

    datasets = ["FD001"]
    sensor_cols = ["sensor_1"]
    setting_cols = []
    sequence_length = 30
    K = 3

    sequences_data = pipeline.sequence_generation(
        train_df,
        val_df,
        test_data,
        datasets,
        sensor_cols,
        setting_cols,
        sequence_length,
        K,
    )

    # Top-level keys that downstream training/eval code expects
    for key in [
        "X_train",
        "y_train",
        "X_val",
        "y_val",
        "X_test_dict",
        "y_test_dict",
        "engine_ids_test_dict",
        "last_idx_map",
    ]:
        assert key in sequences_data

    # Sanity check on one of the shapes from the stub
    X_train = sequences_data["X_train"]
    assert X_train.shape[1] == sequence_length

    # sensor_cols + K regime one-hot columns
    expected_feature_dim = len(sensor_cols) + K + len(setting_cols)
    assert X_train.shape[2] == expected_feature_dim
