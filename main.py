"""
main.py — Entry point for CMAPSS RUL prediction pipeline.

This script orchestrates the high-level workflow:
1. Parse command-line arguments
2. Setup & Download (optional)
3. Load data
4. Basic exploration
5. Preprocess
6. Train/val split
7. Regime clustering
8. Sensor analysis (optional)
9. Sequence generation
10. Train models
11. Test & evaluate
12. Report results
"""

import tensorflow as tf

# Configure TensorFlow memory growth to prevent allocation warnings
gpus = tf.config.list_physical_devices('GPU')
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
    sequence_generation,
    train_models,
    test_and_evaluate,
    report_results
)


def main():
    """Main entry point for the RUL prediction pipeline."""
    
    # 1. Parse command-line arguments
    args, config = parse_arguments()
    
    # 2. Setup & Download (optional)
    paths, output_dir = setup_and_download(args, config)
    
    # 3. Load data
    train_data, test_data, rul_data = load_datasets(paths, config['datasets'])
    
    # 4. Basic exploration
    sensors_to_keep = explore_datasets(train_data, test_data)
    print(sensors_to_keep)
    # 5. Preprocess
    train_data, test_data = preprocess_data(
        train_data, test_data, rul_data, 
        config['datasets'], config['cap_val'], sensors_to_keep
    )
    
    # 6. Train/val split
    train_df, val_df = train_val_split(
        train_data, config['datasets'], config['val_size']
    )
    
    # 7. Regime clustering
    train_df, val_df, test_data, setting_cols, sensor_cols = regime_clustering(
        train_df, val_df, test_data, config['datasets'], config['K']
    )
    
    # 8. Sensor analysis
    sensor_results = sensor_analysis_step(train_df, val_df, sensor_cols, output_dir)

    if sensor_results:
        recommended = sensor_results['recommended_sensors']
        print(f"Using {len(recommended)} recommended sensors: {recommended}")
    
    # 9. Sequence generation
    sequences_data = sequence_generation(
        train_df, val_df, test_data, config['datasets'],
        sensor_cols, setting_cols, config['sequence_length'], config['K']
    )
    
    # 10. Train models
    trained_models = train_models(
        sequences_data, config['architectures'], 
        config['epochs'], config['use_tuning']
    )
    
    # 11. Test & evaluate
    all_results = test_and_evaluate(
        trained_models, sequences_data, 
        config['datasets'], output_dir
    )
    
    # 12. Report results
    report_results(all_results, output_dir, config['architectures'])


if __name__ == "__main__":
    main()
