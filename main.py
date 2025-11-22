"""
main.py — Entry point for CMAPSS RUL prediction pipeline.

This script orchestrates the high-level workflow:
1. Parse command-line arguments
2. Setup & Download (optional)
3. Load data
4. Basic exploration
5. Preprocess
6. Train/val split
7. Identify sensor columns
8. Sensor analysis (feature selection)
9. Regime clustering
10. Sequence generation
11. Train models
12. Test & evaluate
13. Report results
"""
import os
import time
import warnings
import sys
import json
import platform
from datetime import datetime

# --- Silence most TensorFlow C++ logs ---
# 0 = all logs, 1 = filter INFO, 2 = filter INFO+WARNING, 3 = only ERROR+FATAL
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# --- Silence Python-level warnings (from warnings.warn) ---
warnings.filterwarnings("ignore")

import tensorflow as tf

# Configure TensorFlow memory growth to prevent allocation warnings
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[INFO] Enabled memory growth for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"[WARNING] Could not enable memory growth: {e}")


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
    test_and_evaluate,
    report_results,
    save_models_and_metadata,
)


def main():
    """Main entry point for the RUL prediction pipeline."""
    start = time.perf_counter()
    # 1. Parse command-line arguments
    args, config = parse_arguments()
    
    # 2. Setup & Download (optional)
    paths, output_dir = setup_and_download(args, config)

    # 2b. Persist run configuration for reproducibility
    run_meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "args": vars(args),
        "config": config,
    }

    run_config_path = output_dir / "run_config.json"
    run_config_path.parent.mkdir(parents=True, exist_ok=True)
    with run_config_path.open("w") as f:
        json.dump(run_meta, f, indent=2, default=str)

    print(f"[INFO] Saved run configuration to: {run_config_path}")

    # 3. Load data
    train_data, test_data, rul_data = load_datasets(paths, config['datasets'])
    
    # 4. Basic exploration
    sensors_to_keep = explore_datasets(train_data, test_data)
    # print(sensors_to_keep)
    # 5. Preprocess
    train_data, test_data = preprocess_data(
        train_data, test_data, rul_data, 
        config['datasets'], config['cap_val'], sensors_to_keep
    )
    
    # 6. Train/val split
    train_df, val_df = train_val_split(
        train_data, config['datasets'], config['val_size']
    )

    # 7. Identify sensor columns early
    sensor_cols = [c for c in train_df.columns if c.startswith("sensor_")]
    if not sensor_cols:
        raise RuntimeError("No sensor* columns found.")
    print(f"\n[INFO] Identified {len(sensor_cols)} sensor columns")

    # 8. Sensor analysis (optional - controlled by --use-common-sensors flag)
    sensor_results = sensor_analysis_step(
        train_df, val_df, sensor_cols, output_dir,
        run_analysis=config['run_sensor_analysis']
    )
    
    # Apply sensor selection results
    train_df, val_df, test_data, sensor_cols = apply_sensor_selection(
        train_df, val_df, test_data, config['datasets'], 
        sensor_results, sensor_cols
    )

    # 9. Regime clustering (now uses optimized sensor set)
    train_df, val_df, test_data, setting_cols, sensor_cols = regime_clustering(
        train_df, val_df, test_data, config['datasets'], config['K']
    )

    # 10. Sequence generation
    sequences_data = sequence_generation(
        train_df,
        val_df,
        test_data,
        config["datasets"],
        sensor_cols,
        setting_cols,
        config["sequence_length"],
        config["K"],
    )

    # 11a. Train models
    trained_models = train_models(
        sequences_data,
        config["architectures"],
        config["epochs"],
        config["use_tuning"],
    )

    # 11b. Save models + metadata
    feature_cols = sequences_data["feature_cols"]
    save_models_and_metadata(
        trained_models=trained_models,
        feature_cols=feature_cols,
        config=config,
        output_dir=output_dir,
    )

    # 12. Test & evaluate
    all_results = test_and_evaluate(
        trained_models, sequences_data, 
        config['datasets'], output_dir, config
    )

    # 13. Report results
    report_results(all_results, output_dir, config['architectures'])

    end = time.perf_counter()
    elapsed = (end - start)/3600
    print(f"\n ----> TOTAL RUNTIME:{elapsed:.4f} Hours")

if __name__ == "__main__":
    main()
