"""
pipeline.py — High-level pipeline orchestration for RUL prediction.

This module encapsulates the end-to-end pipeline steps:
- Data preprocessing and splitting
- Sensor analysis
- Sequence generation
- Model training and evaluation
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from . import preprocess, regimes, sequences, eval as eval_module, \
    sensor_analysis
from . import model_tcn, model_lstm, model_cnn


def preprocess_data(
        train_data: Dict[str, pd.DataFrame],
        test_data: Dict[str, pd.DataFrame],
        rul_data: Dict[str, pd.DataFrame],
        datasets: List[str],
        cap_val: int,
        sensors_to_keep: List[str] = None
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Preprocess training and test data.

    Args:
        train_data: Dictionary of training DataFrames by dataset
        test_data: Dictionary of test DataFrames by dataset
        rul_data: Dictionary of RUL DataFrames by dataset
        datasets: List of dataset names to process
        cap_val: RUL cap value
        sensors_to_keep: Optional list of sensors to keep

    Returns:
        Tuple of (train_data, test_data) dictionaries
    """
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)

    # Drop unwanted sensors if specified
    if sensors_to_keep:
        print(f"Keeping {len(sensors_to_keep)} sensors")
        preprocess.drop_unwanted_sensors(train_data, sensors_to_keep)

    # Compute RUL for all datasets
    print("Computing RUL values...")
    for fd in datasets:
        train_data[fd] = preprocess.compute_rul_train(train_data[fd])
        test_data[fd] = preprocess.compute_rul_test(test_data[fd],
                                                    rul_data[fd])

    # Cap RUL
    print(f"Capping RUL at {cap_val}")
    for fd in datasets:
        train_data[fd] = preprocess.cap_rul(train_data[fd], cap_val)
        test_data[fd] = preprocess.cap_rul(test_data[fd], cap_val)

    return train_data, test_data


def prepare_train_val_split(
        train_data: Dict[str, pd.DataFrame],
        datasets: List[str],
        val_size: float = 0.2,
        random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combine training data and split into train/val sets.

    Args:
        train_data: Dictionary of training DataFrames
        datasets: List of dataset names
        val_size: Validation split fraction
        random_state: Random seed for splitting

    Returns:
        Tuple of (train_df, val_df)
    """
    print("\n" + "=" * 60)
    print("TRAIN/VAL SPLIT")
    print("=" * 60)

    # Add dataset tags and combine
    regimes.add_dataset_tags(train_data, {}, datasets)
    combined_train = regimes.concat_train(train_data, datasets)

    # Group by dataset + engine to avoid data leakage
    combined_train["group_id"] = (
            combined_train["dataset"].astype(str) + "_" +
            combined_train["engine_id"].astype(str)
    )

    gss = GroupShuffleSplit(n_splits=1, test_size=val_size,
                            random_state=random_state)
    train_idx, val_idx = next(gss.split(
        combined_train,
        groups=combined_train["group_id"].values
    ))

    train_df = combined_train.iloc[train_idx].copy()
    val_df = combined_train.iloc[val_idx].copy()

    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    return train_df, val_df


def apply_regime_clustering(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_data: Dict[str, pd.DataFrame],
        datasets: List[str],
        K: int
) -> Tuple[
    pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame], List[str], List[str]]:
    """
    Apply regime clustering and normalization.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_data: Dictionary of test DataFrames
        datasets: List of dataset names
        K: Number of regimes (clusters)

    Returns:
        Tuple of (train_df, val_df, test_data, setting_cols, sensor_cols)
    """
    print("\n" + "=" * 60)
    print(f"REGIME CLUSTERING (K={K})")
    print("=" * 60)

    # Identify columns
    setting_cols = [c for c in train_df.columns if
                    c.startswith(("op_setting_", "setting_"))]
    sensor_cols = [c for c in train_df.columns if c.startswith("sensor_")]

    if not setting_cols:
        raise RuntimeError("No op_setting* columns found.")
    if not sensor_cols:
        raise RuntimeError("No sensor* columns found.")

    print(f"Settings: {len(setting_cols)} | Sensors: {len(sensor_cols)}")

    # Fit K-means on settings
    km = regimes.fit_kmeans_settings(train_df, setting_cols, k=K)

    # Assign regimes
    train_df = regimes.assign_regime(train_df, km, setting_cols)
    val_df = regimes.assign_regime(val_df, km, setting_cols)
    for fd in datasets:
        test_data[fd] = regimes.assign_regime(test_data[fd], km, setting_cols)

    # Fit and transform sensor scalers per regime
    scalers = regimes.fit_per_regime_sensor_scalers(train_df, sensor_cols,
                                                    "regime_id", K)
    train_df = regimes.transform_sensors_per_regime(train_df, scalers,
                                                    sensor_cols, "regime_id")
    val_df = regimes.transform_sensors_per_regime(val_df, scalers, sensor_cols,
                                                  "regime_id")
    for fd in datasets:
        test_data[fd] = regimes.transform_sensors_per_regime(
            test_data[fd], scalers, sensor_cols, "regime_id"
        )

    # Scale settings
    regimes.scale_settings(train_df,
                           [val_df] + [test_data[fd] for fd in datasets],
                           setting_cols)

    return train_df, val_df, test_data, setting_cols, sensor_cols


def run_sensor_analysis(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        sensor_cols: List[str],
        output_dir: Path
) -> Dict[str, pd.DataFrame]:
    """
    Run comprehensive sensor importance analysis.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        sensor_cols: List of sensor column names
        output_dir: Directory to save results

    Returns:
        Dictionary of analysis results including common sensors
    """
    return sensor_analysis.run_full_analysis(
        train_df=train_df,
        val_df=val_df,
        sensor_cols=sensor_cols,
        output_dir=output_dir,
        rul_col="RUL",
        save_results=True,
        top_n_for_common=5,
        min_methods_for_common=3
    )


def prepare_sequences(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_data: Dict[str, pd.DataFrame],
        datasets: List[str],
        sensor_cols: List[str],
        setting_cols: List[str],
        sequence_length: int,
        K: int
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict, Dict, Dict, Dict]:
    """
    Create sequences for time-series models.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_data: Dictionary of test DataFrames
        datasets: List of dataset names
        sensor_cols: List of sensor column names
        setting_cols: List of setting column names
        sequence_length: Length of sequences
        K: Number of regimes

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test_dict, y_test_dict,
                  engine_ids_test_dict, last_idx_map)
    """
    print("\n" + "=" * 60)
    print(f"SEQUENCE GENERATION (length={sequence_length})")
    print("=" * 60)

    # Add regime one-hot encoding
    sequences.add_regime_onehot(train_df, K)
    sequences.add_regime_onehot(val_df, K)
    for fd in datasets:
        sequences.add_regime_onehot(test_data[fd], K)

    # Define feature columns
    regime_onehot_cols = [f"regime_{r}" for r in range(K)]
    feature_cols = sensor_cols + setting_cols + regime_onehot_cols
    print(f"Total features: {len(feature_cols)}")

    # Create sequences
    X_tr, y_tr, _ = sequences.create_sequences(train_df, feature_cols,
                                               sequence_length)
    X_val, y_val, _ = sequences.create_sequences(val_df, feature_cols,
                                                 sequence_length)

    print(f"Train windows: {X_tr.shape}")
    print(f"Val windows: {X_val.shape}")

    # Test sequences
    X_te_dict, y_te_dict, engine_ids_te_dict, last_idx_map = \
        sequences.build_test_sequences_per_dataset(test_data, sequence_length,
                                                   feature_cols)

    return X_tr, y_tr, X_val, y_val, X_te_dict, y_te_dict, engine_ids_te_dict, last_idx_map


def train_model(
        arch: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int,
        use_tuning: bool
) -> Any:
    """
    Train a model with the specified architecture.

    Args:
        arch: Architecture name ('tcn', 'lstm', or 'cnn')
        X_train: Training sequences
        y_train: Training labels
        X_val: Validation sequences
        y_val: Validation labels
        epochs: Number of training epochs
        use_tuning: Whether to perform hyperparameter tuning

    Returns:
        Trained model
    """
    print("\n" + "=" * 70)
    print(f"TRAINING: {arch.upper()}")
    print("=" * 70)

    # Select model module
    if arch == "tcn":
        mod = model_tcn
        proj = "cmapss_tcn"
    elif arch == "lstm":
        mod = model_lstm
        proj = "cmapss_lstm"
    elif arch == "cnn":
        mod = model_cnn
        proj = "cmapss_cnn"
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    # Train
    if not use_tuning:
        print(f"Training {arch.upper()} with fixed hyperparameters...")
        model, _ = mod.train_default(X_train, y_train, X_val, y_val,
                                     epochs=epochs)
    else:
        if hasattr(mod, "tune"):
            print(f"Performing hyperparameter tuning for {arch.upper()}...")
            best_model, best_hp, tuner, history = mod.tune(
                X_train, y_train, X_val, y_val,
                max_epochs=epochs,
                directory=f"{arch}_tuning",
                project_name=proj
            )
            model = best_model
            try:
                print("Best hyperparameters:", best_hp.values)
            except Exception:
                pass
        else:
            print(
                f"[WARN] Tuning not implemented for '{arch}'. Using fixed hyperparameters.")
            model, _ = mod.train_default(X_train, y_train, X_val, y_val,
                                         epochs=epochs)

    return model


def evaluate_model(
        model: Any,
        arch: str,
        X_test_dict: Dict,
        y_test_dict: Dict,
        engine_ids_test_dict: Dict,
        last_idx_map: Dict,
        datasets: List[str],
        output_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate model and save results.

    Args:
        model: Trained model
        arch: Architecture name
        X_test_dict: Dictionary of test sequences
        y_test_dict: Dictionary of test labels
        engine_ids_test_dict: Dictionary of engine IDs
        last_idx_map: Dictionary mapping engines to their last window index
        datasets: List of dataset names
        output_dir: Directory to save results

    Returns:
        Tuple of (metrics_df, final_predictions_df)
    """
    print("\n" + "=" * 60)
    print(f"EVALUATION: {arch.upper()}")
    print("=" * 60)

    # Per-dataset metrics
    metrics_df = eval_module.per_dataset_metrics(model, X_test_dict,
                                                 y_test_dict, datasets)
    print(f"\n[{arch.upper()}] Per-dataset test metrics:")
    print(metrics_df.to_string(index=False))

    # Final engine predictions
    final_df = eval_module.build_final_engine_table(
        model, X_test_dict, y_test_dict, engine_ids_test_dict, last_idx_map,
        clip_pred=True
    )

    if not final_df.empty:
        print(f"\n[{arch.upper()}] Top 20 engines by |prediction error|:")
        print(final_df.head(20).to_string(index=False))

        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        out_csv = output_dir / f"final_engine_rul_predictions_{arch}.csv"
        final_df.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv.resolve()}")
    else:
        print(f"[{arch.upper()}] No final-window predictions found.")

    return metrics_df, final_df


def run_full_pipeline(
        train_data: Dict[str, pd.DataFrame],
        test_data: Dict[str, pd.DataFrame],
        rul_data: Dict[str, pd.DataFrame],
        architectures: List[str],
        datasets: List[str],
        epochs: int,
        sequence_length: int,
        K: int,
        cap_val: int,
        val_size: float,
        use_tuning: bool,
        sensors_to_keep: List[str] = None,
        run_sensor_analysis_flag: bool = True,
        output_dir: Path = None
) -> Dict[str, Dict[str, Any]]:
    """
    Run the complete RUL prediction pipeline.

    Args:
        train_data: Dictionary of training DataFrames
        test_data: Dictionary of test DataFrames
        rul_data: Dictionary of RUL DataFrames
        architectures: List of architectures to train
        datasets: List of dataset names
        epochs: Number of training epochs
        sequence_length: Length of sequences
        K: Number of regimes
        cap_val: RUL cap value
        val_size: Validation split fraction
        use_tuning: Whether to perform hyperparameter tuning
        sensors_to_keep: Optional list of sensors to keep
        run_sensor_analysis_flag: Whether to run sensor analysis
        output_dir: Output directory for results

    Returns:
        Dictionary of results by architecture
    """
    if output_dir is None:
        output_dir = Path("./_outputs/results")

    # 1. Preprocess data
    train_data, test_data = preprocess_data(
        train_data, test_data, rul_data, datasets, cap_val, sensors_to_keep
    )

    # 2. Train/val split
    train_df, val_df = prepare_train_val_split(train_data, datasets, val_size)

    # 3. Regime clustering and normalization
    train_df, val_df, test_data, setting_cols, sensor_cols = apply_regime_clustering(
        train_df, val_df, test_data, datasets, K
    )

    # 4. Sensor analysis (optional)
    sensor_results = None
    if run_sensor_analysis_flag:
        sensor_results = run_sensor_analysis(
            train_df, val_df, sensor_cols, output_dir / "sensor_analysis"
        )

    # 5. Prepare sequences
    X_tr, y_tr, X_val, y_val, X_te_dict, y_te_dict, engine_ids_te_dict, last_idx_map = \
        prepare_sequences(
            train_df, val_df, test_data, datasets,
            sensor_cols, setting_cols, sequence_length, K
        )

    # 6. Train and evaluate each architecture
    all_results = {}
    for arch in architectures:
        model = train_model(arch, X_tr, y_tr, X_val, y_val, epochs, use_tuning)

        metrics_df, final_df = evaluate_model(
            model, arch, X_te_dict, y_te_dict,
            engine_ids_te_dict, last_idx_map, datasets,
            output_dir / "model"
        )

        all_results[arch] = {
            'model': model,
            'metrics': metrics_df,
            'final_predictions': final_df
        }

    # 7. Summary comparison if multiple architectures
    if len(architectures) > 1:
        print("\n" + "=" * 70)
        print("ARCHITECTURE COMPARISON SUMMARY")
        print("=" * 70)
        for arch_name, results in all_results.items():
            print(f"\n{arch_name.upper()}:")
            print(results['metrics'].to_string(index=False))

    return all_results