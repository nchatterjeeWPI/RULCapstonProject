import pandas as pd

import main


def test_main_runs_without_errors(monkeypatch):
    """Smoke test that main() runs end-to-end when core steps are stubbed.

    This does NOT exercise the heavy pipeline logic – it just verifies that
    the control flow in main() is wired correctly to the high-level steps.
    """

    # --- Stub parse_arguments to return a minimal, consistent config ---
    dummy_args = object()
    config = {
        "datasets": ["FD001"],
        "cap_val": 125,
        "val_size": 0.2,
        "sequence_length": 30,
        "K": 3,
        "architectures": ["lstm"],
        "epochs": 1,
        "use_tuning": False,
        "run_sensor_analysis": False,
    }

    def fake_parse_arguments():
        return dummy_args, config

    monkeypatch.setattr(main, "parse_arguments", fake_parse_arguments)

    # --- Stub setup_and_download ---
    def fake_setup_and_download(args, cfg):
        assert args is dummy_args
        assert cfg is config
        return "paths_obj", "out_dir"

    monkeypatch.setattr(main, "setup_and_download", fake_setup_and_download)

    # --- Build tiny dummy dataframes for downstream steps ---
    train_df = pd.DataFrame(
        {
            "unit": [1, 1, 1],
            "cycle": [1, 2, 3],
            "sensor_1": [0.1, 0.2, 0.3],
        }
    )
    test_df = train_df.copy()
    rul_df = pd.DataFrame({"unit": [1], "RUL": [50]})

    def fake_load_datasets(paths, datasets):
        assert datasets == ["FD001"]
        return {"FD001": train_df}, {"FD001": test_df}, {"FD001": rul_df}

    monkeypatch.setattr(main, "load_datasets", fake_load_datasets)

    # --- Exploration just returns which sensors to keep ---
    def fake_explore_datasets(train_data, test_data):
        assert "FD001" in train_data
        assert "FD001" in test_data
        return ["sensor_1"]

    monkeypatch.setattr(main, "explore_datasets", fake_explore_datasets)

    # --- Preprocess just passes data through (or could tweak it lightly) ---
    def fake_preprocess_data(train_data, test_data, rul_data, datasets, cap_val, sensors_to_keep):
        assert datasets == ["FD001"]
        assert cap_val == 125
        return train_data, test_data

    monkeypatch.setattr(main, "preprocess_data", fake_preprocess_data)

    # --- Train/val split returns two DataFrames with sensor_* columns ---
    def fake_train_val_split(train_data, datasets, val_size):
        assert datasets == ["FD001"]
        assert val_size == 0.2
        # Use the same df for both; main() only cares about the columns
        return train_df, train_df

    monkeypatch.setattr(main, "train_val_split", fake_train_val_split)

    # --- Sensor analysis / selection / regimes / sequences / training / eval ---
    def fake_sensor_analysis_step(train_df_, val_df_, sensor_cols, output_dir, run_analysis):
        # Return whatever structure apply_sensor_selection expects
        return {"selected_sensors": sensor_cols}

    monkeypatch.setattr(main, "sensor_analysis_step", fake_sensor_analysis_step)

    def fake_apply_sensor_selection(train_df_, val_df_, test_data_, datasets, sensor_results, sensor_cols):
        # No-op selection: just pass things through
        return train_df_, val_df_, test_data_, sensor_cols

    monkeypatch.setattr(main, "apply_sensor_selection", fake_apply_sensor_selection)

    def fake_regime_clustering(train_df_, val_df_, test_data_, datasets, K):
        # No regimes or extra setting columns in this smoke test
        setting_cols = []
        sensor_cols = [c for c in train_df_.columns if c.startswith("sensor_")]
        return train_df_, val_df_, test_data_, setting_cols, sensor_cols

    monkeypatch.setattr(main, "regime_clustering", fake_regime_clustering)

    def fake_sequence_generation(train_df_, val_df_, test_data_, datasets, sensor_cols, setting_cols, sequence_length, K):
        # Return a minimal sequences_data dict with expected keys
        return {
            "X_train": object(),
            "y_train": object(),
            "X_val": object(),
            "y_val": object(),
            "X_test_dict": {},
            "y_test_dict": {},
            "engine_ids_test_dict": {},
            "last_idx_map": {},
        }

    monkeypatch.setattr(main, "sequence_generation", fake_sequence_generation)

    def fake_train_models(sequences_data, architectures, epochs, use_tuning):
        assert architectures == ["lstm"]
        assert epochs == 1
        return {"lstm": "trained_model"}

    monkeypatch.setattr(main, "train_models", fake_train_models)

    def fake_test_and_evaluate(trained_models, sequences_data, datasets, output_dir):
        assert "lstm" in trained_models
        return {"lstm": {"rmse": 42.0}}

    monkeypatch.setattr(main, "test_and_evaluate", fake_test_and_evaluate)

    captured = {}

    def fake_report_results(all_results, output_dir, architectures):
        captured["called"] = True
        captured["results"] = all_results
        captured["output_dir"] = output_dir
        captured["architectures"] = architectures

    monkeypatch.setattr(main, "report_results", fake_report_results)

    # --- Run main() and ensure it reaches the reporting step without error ---
    main.main()

    assert captured.get("called", False), "report_results was not called"
    assert captured["results"]["lstm"]["rmse"] == 42.0
    assert captured["output_dir"] == "out_dir"
    assert captured["architectures"] == ["lstm"]
