"""
pipeline.py — High-level pipeline orchestration for RUL prediction.

This module contains the main pipeline steps that can be called sequentially.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import json
from . import preprocess, regimes, sequences, eval as eval_module, \
    sensor_analysis
from . import model_tcn, model_lstm, model_cnn
from .config import make_paths, ensure_dirs, DEFAULT
from . import download, load, explore
from .cli import parse_args
from . import uncertainty


# ============================================================================
# Parse Command-Line Arguments
# ============================================================================

def parse_arguments() -> Tuple[Any, Dict[str, Any]]:
    """
    Parse command-line arguments and create configuration.

    Returns:
        Tuple of (args, config_dict)
    """
    print("\n" + "=" * 70)
    print("STEP 1: PARSE COMMAND-LINE ARGUMENTS")
    print("=" * 70)

    args = parse_args()

    # Resolve configuration (CLI args override defaults)

    # --- Architectures: allow one or many values from CLI ---
    # args.arch is now either:
    #   - None (not provided)
    #   - ["tcn"], ["lstm"], ["cnn"], ["all"], or combinations like ["tcn", "lstm"]
    if args.arch is None:
        arch_list = [
            DEFAULT.arch]  # fallback to single default arch (e.g., "cnn")
    else:
        arch_list = list(args.arch)

    # If user includes "all" anywhere, run all three
    if "all" in arch_list:
        architectures = ["tcn", "lstm", "cnn"]
    else:
        architectures = arch_list

    use_tuning = (
        DEFAULT.use_tuning if args.tuning is None else (args.tuning == "on")
    )
    epochs = args.epochs if args.epochs is not None else DEFAULT.epochs
    sequence_length = (
        args.sequence_length
        if args.sequence_length is not None
        else DEFAULT.sequence_length
    )
    K = args.regimes_k if args.regimes_k is not None else DEFAULT.k
    cap_val = args.cap if args.cap is not None else DEFAULT.cap
    val_size = args.val_size if args.val_size is not None else DEFAULT.val_size
    datasets = args.datasets or list(DEFAULT.datasets)
    use_common_sensors = args.use_common_sensors
    unc_method = getattr(args, "uncertainty",
                         None) or DEFAULT.uncertainty_method
    alpha = getattr(args, "alpha", None)
    if alpha is None:
        alpha = DEFAULT.alpha
    mc_samples = getattr(args, "mc_samples", None)
    if mc_samples is None:
        mc_samples = DEFAULT.mc_samples
    clip_pred = getattr(DEFAULT, "clip_pred", True)

    config = {
        'architectures': architectures,
        'datasets': datasets,
        'epochs': epochs,
        'sequence_length': sequence_length,
        'K': K,
        'cap_val': cap_val,
        'val_size': val_size,
        'use_tuning': use_tuning,
        'use_common_sensors': use_common_sensors,
        'run_sensor_analysis': use_common_sensors,
        'uncertainty_method': unc_method,
        'alpha': float(alpha),
        'mc_samples': int(mc_samples),
        'clip_pred': bool(clip_pred),
    }

    print(f"Architectures: {architectures}")
    print(f"Datasets: {datasets}")
    print(f"Epochs: {epochs}")
    print(f"Sequence Length: {sequence_length}")
    print(f"Regimes (K): {K}")
    print(f"RUL Cap: {cap_val}")
    print(f"Val Size: {val_size}")
    print(f"Tuning: {'ON' if use_tuning else 'OFF'}")
    print(f"Use Common Sensors: {'YES' if use_common_sensors else 'NO'}")
    print(f"Run Sensor Analysis: {'YES' if use_common_sensors else 'NO'}")
    print(f"Uncertainty: method={unc_method} | alpha={alpha} | mc_samples={mc_samples} | clip_pred={clip_pred}")

    return args, config


# ============================================================================
# Setup & Download
# ============================================================================

def setup_and_download(args: Any, config: Dict[str, Any]) -> Tuple[Any, Path]:
    """
    Setup paths and directories, optionally download data.

    Args:
        args: Parsed command-line arguments
        config: Configuration dictionary

    Returns:
        Tuple of (paths, output_dir)
    """
    print("\n" + "=" * 70)
    print("STEP 2: SETUP & DOWNLOAD")
    print("=" * 70)

    paths = make_paths()
    ensure_dirs(paths)
    output_dir = Path(args.out) if args.out else Path("./_outputs/results")

    if args.download:
        print("[INFO] Downloading datasets...")
        download.fetch_cmaps(paths.raw_data_dir,
                             github_token=args.github_token)
    else:
        print("[INFO] Skipping download")

    print(f"Data directory: {paths.user_data_dir}")
    print(f"Output directory: {output_dir}")

    return paths, output_dir


# ============================================================================
# Load Data
# ============================================================================

def load_datasets(paths: Any, datasets: List[str]) -> Tuple[Dict, Dict, Dict]:
    """
    Load training, test, and RUL data.

    Args:
        paths: Paths object containing data directories
        datasets: List of dataset names to load

    Returns:
        Tuple of (train_data, test_data, rul_data)
    """
    print("\n" + "=" * 70)
    print("STEP 3: LOAD DATA")
    print("=" * 70)

    train_data, test_data, rul_data = load.load_all(paths.user_data_dir,
                                                    datasets)

    print(f"Loaded {len(train_data)} training datasets")
    print(f"Loaded {len(test_data)} test datasets")
    print(f"Loaded {len(rul_data)} RUL datasets")

    return train_data, test_data, rul_data


# ============================================================================
# Basic Exploration
# ============================================================================

def explore_datasets(train_data: Dict, test_data: Dict) -> List[str]:
    """
    Inspect data and identify sensors to keep.

    Args:
        train_data: Dictionary of training DataFrames
        test_data: Dictionary of test DataFrames

    Returns:
        List of sensor names to keep (or None to keep all)
    """
    print("\n" + "=" * 70)
    print("STEP 4: BASIC EXPLORATION")
    print("=" * 70)

    explore.inspect(train_data)
    missing_dupes = explore.missing_and_dupes_report(train_data, test_data)
    print(f"Missing/Duplicate report: {missing_dupes}")

    sensors_to_keep = explore.non_constant_sensors(train_data)
    if sensors_to_keep:
        print(f"Non-constant sensors identified: {len(sensors_to_keep)}")
    else:
        print("All sensors will be kept")
        sensors_to_keep = None

    return sensors_to_keep


# ============================================================================
# Preprocess Data
# ============================================================================

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
    print("\n" + "=" * 70)
    print("STEP 5: PREPROCESS")
    print("=" * 70)

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


# ============================================================================
# Train/Validation Split
# ============================================================================

def train_val_split(
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
    print("\n" + "=" * 70)
    print("STEP 6: TRAIN/VAL SPLIT")
    print("=" * 70)

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


# ============================================================================
# Regime Clustering
# ============================================================================

def regime_clustering(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_data: Dict[str, pd.DataFrame],
        datasets: List[str],
        K: int
) -> Tuple[
    pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame], List[str], List[str]]:
    """
    Apply regime clustering and normalization and applies sensor selection.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_data: Dictionary of test DataFrames
        datasets: List of dataset names
        K: Number of regimes (clusters)

    Returns:
        Tuple of (train_df, val_df, test_data, setting_cols, sensor_cols)
    """
    print("\n" + "=" * 70)
    print(f"STEP 9: REGIME CLUSTERING (K={K})")
    print("=" * 70)

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


# ============================================================================
# Apply Sensor Selection
# ============================================================================

def apply_sensor_selection(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_data: Dict[str, pd.DataFrame],
        datasets: List[str],
        sensor_results: Optional[Dict],
        sensor_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame], List[str]]:
    """
    Apply sensor selection results to filter dataframes.
    
    If sensor_results contains recommended sensors, filters all dataframes
    to only include those sensors. Otherwise, keeps all sensors.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_data: Dictionary of test DataFrames
        datasets: List of dataset names
        sensor_results: Results from sensor analysis (or None)
        sensor_cols: Original list of sensor columns
        
    Returns:
        Tuple of (train_df, val_df, test_data, filtered_sensor_cols)
    """
    if sensor_results and 'recommended_sensors' in sensor_results:
        recommended = sensor_results['recommended_sensors']
        print(f"[INFO] Reducing from {len(sensor_cols)} to {len(recommended)} recommended sensors")
        
        # Filter dataframes to only include recommended sensors
        other_cols = [c for c in train_df.columns if not c.startswith("sensor_")]
        train_df = train_df[other_cols + recommended]
        val_df = val_df[other_cols + recommended]

        for fd in datasets:
            test_other_cols = [c for c in test_data[fd].columns if
                               not c.startswith("sensor_")]
            test_data[fd] = test_data[fd][test_other_cols + recommended]
        
        return train_df, val_df, test_data, recommended
    else:
        print(f"[INFO] Using all {len(sensor_cols)} sensors")
        return train_df, val_df, test_data, sensor_cols


# ============================================================================
# Sensor Analysis
# ============================================================================

def sensor_analysis_step(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        sensor_cols: List[str],
        output_dir: Path,
        run_analysis: bool = True,
        recommended_count: int = 4
) -> Dict[str, pd.DataFrame]:
    """
    Run comprehensive sensor importance analysis (optional).

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        sensor_cols: List of sensor column names
        output_dir: Directory to save results
        run_analysis: Whether to run the analysis
        recommended_count: Number of sensors to recommend

    Returns:
        Dictionary of analysis results (or None if skipped)
    """
    print("\n" + "=" * 70)
    print("STEP 8: SENSOR ANALYSIS (FEATURE SELECTION)")
    print("=" * 70)

    if not run_analysis:
        print("[INFO] Sensor analysis skipped")
        return None

    return sensor_analysis.run_full_analysis(
        train_df=train_df,
        val_df=val_df,
        sensor_cols=sensor_cols,
        output_dir=output_dir / "sensor_analysis",
        rul_col="RUL",
        save_results=True,
        top_n_for_common=5,
        min_methods_for_common=3,
        recommended_sensor_count=recommended_count
    )


# ============================================================================
# Sequence Generation
# ============================================================================

def sequence_generation(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_data: Dict[str, pd.DataFrame],
        datasets: List[str],
        sensor_cols: List[str],
        setting_cols: List[str],
        sequence_length: int,
        K: int
) -> Dict[str, Any]:
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
        Dictionary containing all sequence data
    """
    print("\n" + "=" * 70)
    print(f"STEP 10: SEQUENCE GENERATION (length={sequence_length})")
    print("=" * 70)

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

    return {
        'X_train': X_tr,
        'y_train': y_tr,
        'X_val': X_val,
        'y_val': y_val,
        'X_test_dict': X_te_dict,
        'y_test_dict': y_te_dict,
        'engine_ids_test_dict': engine_ids_te_dict,
        'last_idx_map': last_idx_map,
        "feature_cols": feature_cols,
    }

# ============================================================================
# Save History
# ============================================================================

def _save_history(history, path: Path) -> None:
    """Save a Keras History object as CSV with an epoch column."""
    if history is None:
        return
    hist = history.history or {}
    if not hist:
        return

    df = pd.DataFrame(hist)
    df.insert(0, "epoch", np.arange(1, len(df) + 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[INFO] Saved training history to: {path}")

# ============================================================================
# Train Models
# ============================================================================

def train_models(
        sequences_data: Dict[str, Any],
        architectures: List[str],
        epochs: int,
        use_tuning: bool
) -> Dict[str, Any]:
    """
    Train models for all specified architectures.

    Args:
        sequences_data: Dictionary containing sequence data
        architectures: List of architecture names
        epochs: Number of training epochs
        use_tuning: Whether to perform hyperparameter tuning

    Returns:
        Dictionary of trained models by architecture
    """
    logs_root = Path("_outputs") / "training_logs"
    logs_root.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("STEP 11: TRAIN MODELS")
    print("=" * 70)

    X_train = sequences_data['X_train']
    y_train = sequences_data['y_train']
    X_val = sequences_data['X_val']
    y_val = sequences_data['y_val']

    trained_models = {}

    for arch in architectures:
        print(f"\n--- Training {arch.upper()} ---")

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
            model, history = mod.train_default(X_train, y_train, X_val, y_val,
                                               epochs=epochs)
            # Save training history
            _save_history(history, logs_root / f"{arch}_train_history.csv")

        else:
            if hasattr(mod, "tune"):
                print(
                    f"Performing hyperparameter tuning for {arch.upper()}...")
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
                # Save tuning/selected-model history if available
                _save_history(history, logs_root / f"{arch}_tuning_history.csv")
            else:
                print(
                    f"[WARN] Tuning not implemented for '{arch}'. Using fixed hyperparameters.")
                model, history = mod.train_default(X_train, y_train, X_val,
                                                   y_val,
                                                   epochs=epochs)
                _save_history(history, logs_root / f"{arch}_train_history.csv")

        trained_models[arch] = model

    return trained_models


# ============================================================================
# Test & Evaluate
# ============================================================================

def test_and_evaluate(
        trained_models: Dict[str, Any],
        sequences_data: Dict[str, Any],
        datasets: List[str],
        output_dir: Path,
        config: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate all trained models and save results.

    Args:
        trained_models: Dictionary of trained models
        sequences_data: Dictionary containing sequence data
        datasets: List of dataset names
        output_dir: Directory to save results

    Returns:
        Dictionary of results by architecture
    """
    print("\n" + "=" * 70)
    print("STEP 12: TEST & EVALUATE")
    print("=" * 70)

    X_test_dict = sequences_data['X_test_dict']
    y_test_dict = sequences_data['y_test_dict']
    engine_ids_test_dict = sequences_data['engine_ids_test_dict']
    last_idx_map = sequences_data['last_idx_map']

    # For conformal calibration
    X_val = sequences_data['X_val']
    y_val = sequences_data['y_val']

    unc_method = config.get('uncertainty_method', 'none')
    alpha = float(config.get('alpha', 0.10))
    mc_T = int(config.get('mc_samples', 50))
    clip_pred = bool(config.get('clip_pred', True))

    figs_dir = output_dir / "figures"
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for arch, model in trained_models.items():
        print(f"\n--- Evaluating {arch.upper()} ---")

        # 1) Per-dataset metrics
        metrics_df = eval_module.per_dataset_metrics(model, X_test_dict,
                                                     y_test_dict, datasets)
        print(f"\n[{arch.upper()}] Per-dataset test metrics:")
        print(metrics_df.to_string(index=False))

        metrics_csv = model_dir / f"per_dataset_window_metrics_{arch}.csv"
        metrics_df.to_csv(metrics_csv, index=False)
        print(f"[INFO] Saved per-dataset window metrics to: {metrics_csv}")

        # 2) Calibrate conformal q-hat
        qhat = None
        if unc_method == "conformal":
            print(f"[{arch.upper()}][UNCERTAINTY] Calibrating conformal intervals (alpha={alpha})...")
            qhat = uncertainty.conformal_calibrate(model, X_val, y_val, alpha=alpha, clip_pred=clip_pred)
            print(f"[{arch.upper()}][UNCERTAINTY] qhat = {qhat:.4f}")

        # 3) Final engine predictions
        final_rows = []
        for fd, Xte in X_test_dict.items():
            if Xte is None or Xte.shape[0] == 0 or fd not in last_idx_map:
                continue

            final_idxs = sorted(last_idx_map[fd].values())

            if unc_method == "mc":
                # Memory-safe: MC only on final windows
                print(f"[{arch.upper()}][UNCERTAINTY] MC-dropout on final windows ({mc_T} samples) for {fd}...")
                Xte_final = Xte[final_idxs]
                samples = uncertainty.mc_predict(model, Xte_final, T=mc_T)  # (T, Nfinal)
                mean, lo_fin, hi_fin = uncertainty.mc_interval_from_samples(samples, alpha=alpha, clip_pred=clip_pred)
                yhat_fin = np.clip(mean, 0, None) if clip_pred else mean
            else:
                # Deterministic predictions on all windows; slice final
                yhat_all = model.predict(Xte, verbose=0).reshape(-1)
                if clip_pred:
                    yhat_all = np.clip(yhat_all, 0, None)
                yhat_fin = yhat_all[final_idxs]
                if unc_method == "conformal" and qhat is not None:
                    lo_all, hi_all = uncertainty.conformal_interval(yhat_all, qhat, clip_pred=clip_pred)
                    lo_fin, hi_fin = lo_all[final_idxs], hi_all[final_idxs]
                else:
                    lo_fin = hi_fin = None

            ytrue_all = y_test_dict[fd]
            eids_all = engine_ids_test_dict[fd]
            for j, idx in enumerate(final_idxs):
                y_true = float(ytrue_all[idx])
                y_pred = float(yhat_fin[j])
                row = {
                    "dataset": fd,
                    "engine_id": int(eids_all[idx]),
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "y_pred_lo": float(lo_fin[j]) if lo_fin is not None else None,
                    "y_pred_hi": float(hi_fin[j]) if hi_fin is not None else None,
                }
                final_rows.append(row)

        final_df = pd.DataFrame(final_rows)

        # 4) Save per-arch final-engine CSV
        if not final_df.empty:
            out_csv = model_dir / f"final_engine_rul_predictions_{arch}.csv"
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            final_df.to_csv(out_csv, index=False)
            print(f"Saved: {out_csv.resolve()}")
        else:
            print(f"[{arch.upper()}] No final-window predictions found.")

        # 5) Plots (overall + per-dataset) — SVG for zoomability
        if not final_df.empty:
            all_svg = figs_dir / f"rul_bands_final_all_{arch}.svg"
            eval_module.plot_final_engine_bands(final_df, str(all_svg))
            for fd in sorted(final_df["dataset"].unique()):
                fd_svg = figs_dir / f"rul_bands_final_{fd}_{arch}.svg"
                eval_module.plot_final_engine_bands(final_df, str(fd_svg), dataset=fd)

        # 6) Coverage + interval-width summaries (if bands exist)
        has_lo = "y_pred_lo" in final_df.columns
        has_hi = "y_pred_hi" in final_df.columns
        some_lo = has_lo and pd.to_numeric(final_df["y_pred_lo"], errors="coerce").notna().any()
        some_hi = has_hi and pd.to_numeric(final_df["y_pred_hi"], errors="coerce").notna().any()
        if has_lo and has_hi and some_lo and some_hi:
            cov_df = eval_module.interval_accuracy_summary(final_df)
            cov_csv = model_dir / f"interval_coverage_summary_{arch}.csv"
            cov_df.to_csv(cov_csv, index=False)
            print(f"\n[{arch.upper()}] Interval coverage summary:")
            print(cov_df.to_string(index=False))

            final_df["interval_width"] = (
                    pd.to_numeric(final_df["y_pred_hi"], errors="coerce")
                    - pd.to_numeric(final_df["y_pred_lo"], errors="coerce")
            )
            by_ds = final_df.groupby("dataset")["interval_width"].agg(["count", "mean", "median"]).reset_index()
            print(f"\n[{arch.upper()}] Interval width summary:")
            print(by_ds.to_string(index=False))
        else:
            print(f"[{arch.upper()}] No usable interval bounds found; plots saved (bands may be absent).")

        # Keep the uncertainty-aware final_df for saving + results
        final_df_unc = final_df.copy()

        # Build a separate table (if you still want the extra stats)
        final_df_no_unc = eval_module.build_final_engine_table(
            model, X_test_dict, y_test_dict, engine_ids_test_dict,
            last_idx_map,
            clip_pred=clip_pred,
        )

        # 6b) Extra debug view: top engines by absolute error
        if not final_df_no_unc.empty:
            tmp = final_df_no_unc.copy()
            # abs_delta already exists from build_final_engine_table
            print(f"\n[{arch.upper()}] Top 20 engines by worst case |by prediction error|:")
            print(
                tmp.sort_values("abs_delta", ascending=False)
                .head(20)
                .to_string(index=False)
            )

        # 7) Final-engine metrics: RMSE/CMAPSS + accuracy within ±K cycles
        final_metrics_df = eval_module.final_engine_metrics(
            final_df_no_unc,
            ks=[10, 20]  # adjust thresholds as you like
        )

        if not final_metrics_df.empty:
            metrics_path = model_dir / f"final_engine_metrics_{arch}.csv"
            final_metrics_df.to_csv(metrics_path, index=False)

            # Pretty-print accuracy columns as percentages
            pct_cols = [c for c in final_metrics_df.columns if
                        c.startswith("acc_within_")]
            display_df = final_metrics_df.copy()
            for col in pct_cols:
                display_df[col] = (display_df[col] * 100).round(1).astype(
                    str) + " %"

            print(
                f"\n[{arch.upper()}] Final-engine metrics (RMSE/CMAPSS/Accuracy):")
            print(display_df.to_string(index=False))

        # 8) Store results for later reporting/comparison
        all_results[arch] = {
            "model": model,
            "metrics": metrics_df,  # per-dataset window metrics
            "final_metrics": final_metrics_df,
            # per-dataset + overall engine metrics
            "final_predictions": final_df_unc,  # with intervals if any
            "final_predictions_plain": final_df_no_unc,
        }

    return all_results


# ============================================================================
# Report Results
# ============================================================================

def report_results(
        all_results: Dict[str, Dict[str, Any]],
        output_dir: Path,
        architectures: List[str]
):
    """
    Generate final summary report.

    Args:
        all_results: Dictionary of results by architecture
        output_dir: Output directory
        architectures: List of architectures trained
    """
    print("\n" + "=" * 70)
    print("STEP 13: REPORT RESULTS")
    print("=" * 70)

    # 1) Per-dataset window metrics (existing behavior)
    if len(architectures) > 1:
        print(
            "\nARCHITECTURE COMPARISON (per-dataset RMSE / CMAPSS / n_windows):"
        )
        for arch_name, results in all_results.items():
            print(f"\n{arch_name.upper()}:")
            print(results["metrics"].to_string(index=False))

    # 2) Final-engine metrics (OVERALL) + collect for best-model selection
    print("\nFINAL-ENGINE METRICS (OVERALL):")
    overall_rows = []

    for arch_name, results in all_results.items():
        fm = results.get("final_metrics")
        if fm is None or fm.empty:
            continue

        overall_row = fm[fm["dataset"] == "OVERALL"]
        if overall_row.empty:
            continue

        # Pretty-print accuracy columns as percentages for console
        pct_cols = [c for c in overall_row.columns if c.startswith("acc_within_")]
        display_overall = overall_row.copy()
        for col in pct_cols:
            display_overall[col] = (
                display_overall[col] * 100
            ).round(1).astype(str) + " %"

        print(f"\n{arch_name.upper()}:")
        print(display_overall.to_string(index=False))

        # Keep raw numeric values for ranking
        overall_rows.append(
            {
                "arch": arch_name,
                **overall_row.iloc[0].to_dict(),
            }
        )

    # 3) If multiple architectures ran (e.g. --arch all), declare best model
    if len(architectures) > 1 and overall_rows:
        summary_df = pd.DataFrame(overall_rows)

        # Sort: lower RMSE_final, lower CMAPSS_final, higher acc_within_10
        sort_cols = ["RMSE_final", "CMAPSS_final"]
        ascending = [True, True]
        if "acc_within_10" in summary_df.columns:
            sort_cols.append("acc_within_10")
            ascending.append(False)

        summary_df_sorted = summary_df.sort_values(
            by=sort_cols,
            ascending=ascending,
            ignore_index=True,
        )
        best = summary_df_sorted.iloc[0]
        best_arch = str(best["arch"])

        # Optional: persist best architecture for downstream scripts
        best_path = output_dir / "best_architecture.txt"
        with best_path.open("w") as f:
            f.write(f"{best_arch}\n")

        print("\n" + "-" * 70)
        print("BEST-PERFORMING ARCHITECTURE (based on OVERALL final-engine metrics)")
        print("-" * 70)
        print(f"Best model: {best_arch.upper()}")

        # Pull key metrics for explanation
        def _get(name, default=float("nan")):
            val = best.get(name, default)
            try:
                return float(val)
            except Exception:
                return default

        best_rmse = _get("RMSE_final")
        best_score = _get("CMAPSS_final")
        best_mape = _get("MAPE_final")
        best_acc_pct = _get("Accuracy_pct_final") * 100 if "Accuracy_pct_final" in best else float("nan")
        best_acc10 = _get("acc_within_10") * 100 if "acc_within_10" in best else float("nan")

        print("\nReasoning (vs. other architectures):")
        for _, other in summary_df_sorted.iterrows():
            other_arch = str(other["arch"])
            if other_arch == best_arch:
                continue

            def _other(name):
                return _get(name) if name in other.index else float("nan")

            other_rmse = float(other.get("RMSE_final", float("nan")))
            other_score = float(other.get("CMAPSS_final", float("nan")))
            other_acc10 = (
                float(other.get("acc_within_10", float("nan"))) * 100
                if "acc_within_10" in other.index
                else float("nan")
            )

            msgs = []
            if not np.isnan(best_rmse) and not np.isnan(other_rmse):
                diff_rmse = other_rmse - best_rmse
                if diff_rmse > 0:
                    msgs.append(
                        f"RMSE_final is lower by {diff_rmse:.3f} "
                        f"({best_rmse:.3f} vs {other_rmse:.3f})"
                    )
            if not np.isnan(best_score) and not np.isnan(other_score):
                diff_score = other_score - best_score
                if diff_score > 0:
                    msgs.append(
                        f"CMAPSS_final penalty is lower by {diff_score:.3f} "
                        f"({best_score:.3f} vs {other_score:.3f})"
                    )
            if not np.isnan(best_acc10) and not np.isnan(other_acc10):
                diff_acc10 = best_acc10 - other_acc10
                if diff_acc10 > 0:
                    msgs.append(
                        f"higher fraction of engines within ±10 cycles "
                        f"by {diff_acc10:.1f} percentage points "
                        f"({best_acc10:.1f}% vs {other_acc10:.1f}%)"
                    )

            if not msgs:
                msgs.append("metrics are comparable; selected by tie-break order")

            print(f"- Compared to {other_arch.upper()}: " + "; ".join(msgs))

        print(f"\n[INFO] Best architecture written to: {best_path.resolve()}")


def save_models_and_metadata(
        trained_models: Dict[str, Any],
        feature_cols: List[str],
        config: Dict[str, Any],
        output_dir: Path,
) -> None:
    """
    Save each trained model (.keras) together with a sidecar metadata JSON.

    This allows inference scripts to know exactly which sequence length and
    feature columns were used during training.
    """
    model_dir = output_dir / "final_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Base metadata common to all architectures
    base_meta = {
        "datasets": list(config.get("datasets", [])),
        "sequence_length": int(config.get("sequence_length")),
        "cap_val": int(config.get("cap_val")),
        "K": int(config.get("K")),
        "val_size": float(config.get("val_size")),
        "use_tuning": bool(config.get("use_tuning")),
        "use_common_sensors": bool(config.get("use_common_sensors")),
        "uncertainty_method": config.get("uncertainty_method"),
        "alpha": float(config.get("alpha")),
        "mc_samples": int(config.get("mc_samples")),
        "clip_pred": bool(config.get("clip_pred")),
        "feature_cols": list(feature_cols),
    }

    for arch, model in trained_models.items():
        arch_stub = f"{arch}_final"
        model_path = model_dir / f"{arch_stub}.keras"
        meta_path = model_dir / f"{arch_stub}.meta.json"

        print(f"[SAVE] Saving final {arch} model to {model_path}")
        model.save(model_path)

        meta = dict(base_meta)
        meta["architecture"] = arch

        with meta_path.open("w") as f:
            json.dump(meta, f, indent=2)

        print(f"[SAVE] Saved {arch} metadata to {meta_path}")


# ============================================================================
# Legacy function for backward compatibility (if needed elsewhere)
# ============================================================================

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
    Run the complete RUL prediction pipeline (legacy wrapper).

    This function is kept for backward compatibility but delegates to
    the individual pipeline step functions.
    """
    if output_dir is None:
        output_dir = Path("./_outputs/results")

    # Preprocess data
    train_data, test_data = preprocess_data(
        train_data, test_data, rul_data, datasets, cap_val, sensors_to_keep
    )

    # Train/val split
    train_df, val_df = train_val_split(train_data, datasets, val_size)

    # Regime clustering and normalization
    train_df, val_df, test_data, setting_cols, sensor_cols = regime_clustering(
        train_df, val_df, test_data, datasets, K
    )
    
    # Sensor analysis (optional)
    sensor_results = None
    if run_sensor_analysis_flag:
        sensor_results = sensor_analysis_step(
            train_df, val_df, sensor_cols, output_dir, run_analysis=True
        )
    
    # Apply sensor selection
    train_df, val_df, test_data, sensor_cols = apply_sensor_selection(
        train_df, val_df, test_data, datasets, sensor_results, sensor_cols
    )

    # Prepare sequences
    sequences_data = sequence_generation(
        train_df, val_df, test_data, datasets,
        sensor_cols, setting_cols, sequence_length, K
    )

    # Train models
    trained_models = train_models(sequences_data, architectures, epochs,
                                  use_tuning)

    # Evaluate
    all_results = test_and_evaluate(trained_models, sequences_data, datasets,
                                    output_dir)

    # Report
    report_results(all_results, output_dir, architectures)

    return all_results