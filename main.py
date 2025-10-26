import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cmapss_rul.config import make_paths, ensure_dirs, TRAINING, ALL_FD
from cmapss_rul import download, load, explore, preprocess, regimes, sequences, eval
from cmapss_rul import model_tcn, model_lstm, model_cnn

def main():
    ap = argparse.ArgumentParser(description="CMAPSS Remaining Useful Life Prediction")
    ap.add_argument("--download", action="store_true", help="Download CMAPSS data if missing")
    ap.add_argument("--github-token", type=str, default=None, help="Optional GitHub token for download")
    # Optional overrides (config.py provides defaults)
    ap.add_argument("--arch", type=str, choices=["tcn","lstm","cnn"], default=None)
    ap.add_argument("--tuning", type=str, choices=["on","off"], default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--sequence-length", type=int, default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--datasets", nargs="+", default=None)

    args = ap.parse_args()

    arch = args.arch or TRAINING.arch
    use_tuning = (TRAINING.use_tuning if args.tuning is None else (args.tuning == "on"))
    epochs = args.epochs if args.epochs is not None else TRAINING.epochs
    sequence_length = args.sequence_length if args.sequence_length is not None else TRAINING.sequence_length
    K = args.k if args.k is not None else TRAINING.k
    cap_val = args.cap if args.cap is not None else TRAINING.cap
    datasets = args.datasets or list(TRAINING.datasets)

    print(f"[CONFIG] arch={arch} | tuning={'on' if use_tuning else 'off'} | epochs={epochs} | "
          f"seq_len={sequence_length} | K={K} | cap={cap_val} | datasets={datasets}")

    # Paths & dirs
    paths = make_paths()
    ensure_dirs(paths)

    # Download if requested
    if args.download:
        download.fetch_cmaps(paths.raw_data_dir, github_token=args.github_token)

    # Load
    train_data, test_data, rul_data = load.load_all(paths.user_data_dir, datasets)

    # Basic inspection (prints)
    explore.inspect(train_data)
    print("Missing/Dupe report:", explore.missing_and_dupes_report(train_data, test_data))

    # ------------ BEGIN: working preprocessing & split pipeline ------------
    sensors_to_keep = explore.non_constant_sensors(train_data)
    print("Common non-constant sensors:", sensors_to_keep if sensors_to_keep else "(kept all)")

    if sensors_to_keep:
        preprocess.drop_unwanted_sensors(train_data, sensors_to_keep)

    for fd in datasets:
        train_data[fd] = preprocess.compute_rul_train(train_data[fd])
        test_data[fd]  = preprocess.compute_rul_test(test_data[fd], rul_data[fd])

    regimes.add_dataset_tags(train_data, test_data, datasets)
    combined_train = regimes.concat_train(train_data, datasets)

    combined_train = preprocess.cap_rul(combined_train, cap_val)
    for fd in datasets:
        test_data[fd] = preprocess.cap_rul(test_data[fd], cap_val)

    setting_cols = [c for c in combined_train.columns if c.startswith(("op_setting_", "setting_"))]
    sensor_cols  = [c for c in combined_train.columns if c.startswith("sensor_")]
    if not setting_cols: raise RuntimeError("No op_setting* columns found.")
    if not sensor_cols:  raise RuntimeError("No sensor* columns found.")

    from sklearn.model_selection import GroupShuffleSplit
    combined_train["group_id"] = combined_train["dataset"].astype(str) + "_" + combined_train["engine_id"].astype(str)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(combined_train, groups=combined_train["group_id"].values))
    train_df = combined_train.iloc[train_idx].copy()
    val_df   = combined_train.iloc[val_idx].copy()

    km = regimes.fit_kmeans_settings(train_df, setting_cols, k=K)
    train_df = regimes.assign_regime(train_df, km, setting_cols)
    val_df   = regimes.assign_regime(val_df,   km, setting_cols)
    for fd in datasets:
        test_data[fd] = regimes.assign_regime(test_data[fd], km, setting_cols)

    scalers = regimes.fit_per_regime_sensor_scalers(train_df, sensor_cols, "regime_id", K)
    train_df = regimes.transform_sensors_per_regime(train_df, scalers, sensor_cols, "regime_id")
    val_df   = regimes.transform_sensors_per_regime(val_df,   scalers, sensor_cols, "regime_id")
    for fd in datasets:
        test_data[fd] = regimes.transform_sensors_per_regime(test_data[fd], scalers, sensor_cols, "regime_id")

    regimes.scale_settings(train_df, [val_df] + [test_data[fd] for fd in datasets], setting_cols)

    sequences.add_regime_onehot(train_df, K)
    sequences.add_regime_onehot(val_df,   K)
    for fd in datasets:
        sequences.add_regime_onehot(test_data[fd], K)

    regime_onehot_cols = [f"regime_{r}" for r in range(K)]
    feature_cols = sensor_cols + setting_cols + regime_onehot_cols
    print("Num features:", len(feature_cols))

    X_tr, y_tr, _ = sequences.create_sequences(train_df, feature_cols, sequence_length)
    X_val, y_val, _ = sequences.create_sequences(val_df,   feature_cols, sequence_length)
    print(f"Train windows: {X_tr.shape} | Val windows: {X_val.shape}")
    # ------------ END: working preprocessing & split pipeline ------------

    # Model selection
    if arch == "tcn":
        mod = model_tcn; proj = "cmapss_tcn"
    elif arch == "lstm":
        mod = model_lstm; proj = "cmapss_lstm"
    else:
        mod = model_cnn; proj = "cmapss_cnn"

    # Train
    if not use_tuning:
        print("[INFO] Training with fixed hyperparameters...")
        model, _ = mod.train_default(X_tr, y_tr, X_val, y_val, epochs=epochs)


    else:
        print("[INFO] Performing hyperparameter tuning...")
        model, _, best_hp, bs = mod.tune_and_train(
            X_tr, y_tr, X_val, y_val,
            max_epochs=epochs, directory='tcn_tuning', project_name=proj
        )
        try:
            print("Best hyperparameters:", best_hp.values)
        except Exception:
            pass

    # Test windows
    X_te_dict, y_te_dict, engine_ids_te_dict, last_idx_map = \
        sequences.build_test_sequences_per_dataset(test_data, sequence_length, feature_cols)

    # Per-dataset metrics (all windows)
    metrics_df = eval.per_dataset_metrics(model, X_te_dict, y_te_dict, datasets)
    print("\nPer-dataset test metrics (combined model):")
    print(metrics_df.to_string(index=False))

    # Final-engine table (one row per engine @ last cycle)
    final_df = eval.build_final_engine_table(model, X_te_dict, y_te_dict, engine_ids_te_dict, last_idx_map, clip_pred=True)
    if not final_df.empty:
        print("\nTop 20 engines by |prediction error|:")
        print(final_df.head(20).to_string(index=False))
        out_csv = Path("./_outputs/results/model/final_engine_rul_predictions.csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(out_csv, index=False)
        print(f"\nSaved: {out_csv.resolve()}")
    else:
        print("No final-window predictions found. Consider lowering --sequence-length.")

if __name__ == "__main__":
    main()
