import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from cmapss_rul.config import make_paths, ensure_dirs
from cmapss_rul.download import download_and_extract, DEFAULT_URL
from cmapss_rul.load import load_all
from cmapss_rul.explore import missing_and_dupes_report, non_constant_sensors
from cmapss_rul.preprocess import drop_unwanted_sensors, compute_rul_train, compute_rul_test, cap_rul
from cmapss_rul.regimes import add_dataset_tags, concat_train, fit_kmeans_settings, assign_regime, fit_per_regime_sensor_scalers, transform_sensors_per_regime, scale_settings
from cmapss_rul.sequences import add_regime_onehot, build_feature_cols, create_sequences, build_test_sequences_per_dataset
from cmapss_rul import model_tcn, model_lstm, model_cnn
from cmapss_rul.eval import per_dataset_metrics, build_final_engine_table

ALL_FD = ['FD001','FD002','FD003','FD004']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="CMAPSS/RAW_DATA", help="Where RAW_DATA lives (will contain CMaps after unzip)")
    ap.add_argument("--gdrive-root", type=str, default="", help="If provided, results will mirror your original GDrive layout")
    ap.add_argument("--download", action="store_true", help="Download the dataset zip (uses DEFAULT_URL)")
    ap.add_argument("--github-token", type=str, default=os.getenv("GITHUB_TOKEN",""), help="Optional GitHub token for download")
    ap.add_argument("--datasets", nargs="+", default=ALL_FD, help="Subset of datasets to use")
    ap.add_argument("--sequence-length", type=int, default=50)
    ap.add_argument("--cap", type=int, default=125, help="Cap RUL at this value")
    ap.add_argument("--k", type=int, default=6, help="Operating regime KMeans K")
    ap.add_argument("--skip-eda", action="store_true")
    ap.add_argument("--skip-tuning", action="store_true")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--arch", type=str, default="tcn", choices=["tcn","lstm","cnn"])
    ap.add_argument("--export-csv", type=str, default="final_engine_rul_predictions.csv")
    args = ap.parse_args()

    paths = make_paths(args.data_root, args.gdrive_root)
    ensure_dirs(paths)

    if args.download:
        zip_path = Path(args.data_root).parent / "CMaps.zip"
        print(f"Downloading to {zip_path} and extracting to {paths.raw_data_dir}...")
        download_and_extract(DEFAULT_URL, zip_path, paths.raw_data_dir, github_token=(args.github_token or None))
        print("Download and extraction complete.")

    # Load
    print("Loading datasets:", args.datasets)
    train_data, test_data, rul_data = load_all(paths.user_data_dir, args.datasets)

    # Checks
    report = missing_and_dupes_report(train_data, test_data)
    print("Missing/Duplicates:", report)

    # Sensors to keep: common non-constant across datasets
    sensors_to_keep = non_constant_sensors(train_data)
    print("Common non-constant sensors:", sensors_to_keep if sensors_to_keep else "(none; will keep all)")

    if sensors_to_keep:
        drop_unwanted_sensors(train_data, sensors_to_keep)

    # RUL
    for fd in args.datasets:
        train_data[fd] = compute_rul_train(train_data[fd])
        test_data[fd]  = compute_rul_test(test_data[fd], rul_data[fd])

    # Add dataset tags
    add_dataset_tags(train_data, test_data, args.datasets)

    # Cap RUL
    combined_train = concat_train(train_data, args.datasets)
    combined_train = cap_rul(combined_train, args.cap)
    for fd in args.datasets:
        test_data[fd] = cap_rul(test_data[fd], args.cap)

    # Identify columns
    setting_cols = [c for c in combined_train.columns if c.startswith(('op_setting_', 'setting_'))]
    sensor_cols  = [c for c in combined_train.columns if c.startswith('sensor_')]
    if not setting_cols: raise RuntimeError("No op_setting* columns found.")
    if not sensor_cols:  raise RuntimeError("No sensor* columns found.")

    # Group split (by engine, across datasets)
    combined_train['group_id'] = combined_train['dataset'].astype(str) + "_" + combined_train['engine_id'].astype(str)
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    groups = combined_train['group_id'].values
    train_idx, val_idx = next(gss.split(combined_train, groups=groups))
    train_df = combined_train.iloc[train_idx].copy()
    val_df   = combined_train.iloc[val_idx].copy()

    # Regimes
    from cmapss_rul.regimes import KMeans
    K = args.k
    km = fit_kmeans_settings(train_df, setting_cols, k=K)
    train_df = assign_regime(train_df, km, setting_cols)
    val_df   = assign_regime(val_df,   km, setting_cols)
    for fd in args.datasets:
        test_data[fd] = assign_regime(test_data[fd], km, setting_cols)

    # Scaling per regime (sensors) + global settings
    scalers = fit_per_regime_sensor_scalers(train_df, sensor_cols, 'regime_id', K)
    train_df = transform_sensors_per_regime(train_df, scalers, sensor_cols, 'regime_id')
    val_df   = transform_sensors_per_regime(val_df,   scalers, sensor_cols, 'regime_id')
    for fd in args.datasets:
        test_data[fd] = transform_sensors_per_regime(test_data[fd], scalers, sensor_cols, 'regime_id')

    scale_settings(train_df, [val_df] + [test_data[fd] for fd in args.datasets], setting_cols)

    # One-hot regimes and features
    add_regime_onehot(train_df, K); add_regime_onehot(val_df, K)
    for fd in args.datasets: add_regime_onehot(test_data[fd], K)
    feature_cols = [c for c in sensor_cols] + [c for c in setting_cols] + [f"regime_{r}" for r in range(K)]
    print("Num features:", len(feature_cols))

    # Sequences
    X_tr, y_tr, _ = create_sequences(train_df, feature_cols, args.sequence_length)
    X_val, y_val, _ = create_sequences(val_df,   feature_cols, args.sequence_length)
    print(f"Train windows: {X_tr.shape}, Val windows: {X_val.shape}")

    # Model selection
    if args.arch == "tcn":
        mod = model_tcn
        proj = "cmapss_tcn"
    elif args.arch == "lstm":
        mod = model_lstm
        proj = "cmapss_lstm"
    else:
        mod = model_cnn
        proj = "cmapss_cnn"

    if args.skip_tuning:
        model, _ = mod.train_default(X_tr, y_tr, X_val, y_val, max_epochs=args.epochs)
    else:
        model, _, best_hp, bs = mod.tune_and_train(X_tr, y_tr, X_val, y_val, max_epochs=args.epochs, directory='tcn_tuning', project_name=proj)
        try:
            print("Best hyperparameters:", best_hp.values)
        except Exception:
            pass

    # Test sequences per dataset
    X_te_dict, y_te_dict, engine_ids_te_dict, last_idx_map =         build_test_sequences_per_dataset(test_data, args.sequence_length, feature_cols)

    # Evaluate
    perf = per_dataset_metrics(model, X_te_dict, y_te_dict, args.datasets)
    print("\nPer-dataset test metrics (combined model):")
    print(perf.to_string(index=False))

    # Final engine table
    df_final = build_final_engine_table(model, X_te_dict, y_te_dict, engine_ids_te_dict, last_idx_map, clip_pred=True)
    if not df_final.empty:
        print("\nTop 20 engines by |prediction error|:")
        print(df_final.head(20).to_string(index=False))
        out_csv = Path(args.export_csv)
        df_final.to_csv(out_csv, index=False)
        print(f"\nSaved: {out_csv.resolve()}")
    else:
        print("No final-window predictions found. Check sequence length.")

if __name__ == "__main__":
    main()
