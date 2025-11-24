import pytest
from pathlib import Path
from cmapss_rul.pipeline import (
    parse_arguments,
    setup_and_download,
    load_datasets,
    explore_datasets,
    preprocess_data,
    train_val_split,
    regime_clustering,
    sensor_analysis_step,
    apply_sensor_selection,
    sequence_generation,
    train_models,
    run_evaluation,
    report_results
)


@pytest.mark.system
def test_fd001_full_pipeline_end_to_end():
    """
    System Test: Runs FD001 end-to-end through the updated main.py workflow 
    using real CMAPSS data (no mocking). Ensures the full pipeline executes,
    produces models, metrics, and final report outputs.
    """


    # STEP 1: Simulate parsed arguments + config loading

    # NOTE: We call parse_arguments() directly so config is real
    args, config = parse_arguments()

    # Override config for faster system testing
    config["datasets"] = ["FD001"]
    config["epochs"] = 1            # keeps runtime small
    config["use_tuning"] = False    # disables keras-tuner
    config["run_sensor_analysis"] = False  # disables SHAP / heavy ops
    config["architectures"] = ["LSTM"]     # keep minimal for test speed


    # STEP 2: Setup directories + dataset paths

    paths, output_dir = setup_and_download(args, config)

    assert output_dir.exists(), "Output directory failed to initialize."


    # STEP 3: Load real CMAPSS datasets
   
    train_data, test_data, rul_data = load_datasets(paths, config["datasets"])

    assert not train_data.empty
    assert not test_data.empty
    assert len(rul_data) > 0


    # STEP 4: Basic exploration

    sensors_to_keep = explore_datasets(train_data, test_data)
    assert len(sensors_to_keep) > 0


    # STEP 5: Preprocess

    train_data, test_data = preprocess_data(
        train_data, test_data, rul_data,
        config["datasets"], config["cap_val"], sensors_to_keep
    )

    assert "RUL" in train_data.columns


    # STEP 6: Train/Validation Split

    train_df, val_df = train_val_split(
        train_data, config["datasets"], config["val_size"]
    )

    assert len(train_df) > 0
    assert len(val_df) > 0


    # STEP 7: Identify Sensor Columns

    sensor_cols = [c for c in train_df.columns if c.startswith("sensor_")]
    assert len(sensor_cols) > 0


    # STEP 8: Sensor analysis (disabled for speed)

    sensor_results = sensor_analysis_step(
        train_df, val_df, sensor_cols, output_dir,
        run_analysis=config["run_sensor_analysis"]
    )


    # STEP 9: Apply sensor selection results

    train_df, val_df, test_data, sensor_cols = apply_sensor_selection(
        train_df, val_df, test_data,
        config["datasets"], sensor_results, sensor_cols
    )

    assert len(sensor_cols) > 0


    # STEP 10: Regime Clustering

    train_df, val_df, test_data, setting_cols, sensor_cols = regime_clustering(
        train_df, val_df, test_data,
        config["datasets"], config["K"]
    )

    assert isinstance(setting_cols, list)


    # STEP 11: Sequence Generation

    sequences_data = sequence_generation(
        train_df, val_df, test_data,
        config["datasets"], sensor_cols, setting_cols,
        config["sequence_length"], config["K"]
    )

    assert "train" in sequences_data
    assert "test" in sequences_data

    # STEP 12: Model Training + Saving
  
    trained_models = train_models(
        sequences_data, config["architectures"],
        config["epochs"], config["use_tuning"]
    )

    assert "LSTM" in trained_models

    model_dir = output_dir / "final_model"
    model_dir.mkdir(exist_ok=True)

    for arch, model in trained_models.items():
        path = model_dir / f"{arch}_final.keras"
        model.save(path)
        assert path.exists()

    # STEP 13: Test & Evaluate

    results = run_evaluation(
        trained_models, sequences_data,
        config["datasets"], output_dir, config
    )

    assert isinstance(results, dict)
    assert "LSTM" in results

    # STEP 14: Report generation

    report_results(results, output_dir, config["architectures"])

    # If we reach this point, the system test passed
    assert True
