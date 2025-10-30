import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Imports utilities and defaults from config file
from cmapss_rul.config import make_paths, ensure_dirs, TRAINING, ALL_FD
# Imports each functional module that handles different phases in the pipeline
from cmapss_rul import download, load, explore, preprocess, regimes, sequences, eval
# Imports three model architectures and the uncertainty helpers
from cmapss_rul import model_tcn, model_lstm, model_cnn, uncertainty


def main():
    # ===============================================================
    # COMMAND LINE SETUP
    # Allows passing options like --arch tcn --epochs 60, etc.
    # ===============================================================
    ap = argparse.ArgumentParser(description="CMAPSS Remaining Useful Life Prediction")
    ap.add_argument("--download", action="store_true", help="Download CMAPSS data if missing")
    ap.add_argument("--github-token", type=str, default=None, help="Optional GitHub token for download (if required)")
    ap.add_argument("--arch", choices=["tcn", "lstm", "cnn"], default=None, help="Model architecture to train")
    ap.add_argument("--tuning", choices=["on", "off"], default=None, help="Hyperparameter tuning on/off")
    ap.add_argument("--epochs", type=int, default=None, help="Training epochs")
    ap.add_argument("--sequence-length", type=int, default=None, help="Sliding window length")
    ap.add_argument("--k", type=int, default=None, help="Number of operating regimes (KMeans clusters)")
    ap.add_argument("--cap", type=int, default=None, help="Cap for RUL target")
    ap.add_argument("--datasets", nargs="+", default=None, help="Subset of datasets to use (e.g., FD001 FD003)")

    # Uncertainty controls (can also be set in config.py)
    ap.add_argument("--uncertainty", choices=["none", "conformal", "mc"], default=None,
                    help="Interval method: none, conformal residual quantile, or mc (Monte Carlo dropout)")
    ap.add_argument("--alpha", type=float, default=None,
                    help="(1 - alpha) = target coverage; e.g., alpha=0.1 -> ~90% interval")
    ap.add_argument("--mc-samples", type=int, default=None, help="T: number of MC dropout samples")
    args = ap.parse_args()

    # ===============================================================
    # CONFIGURATION SETUP
    # Uses CLI overrides if provided, otherwise falls back to config.py
    # ===============================================================
    arch = args.arch or TRAINING.arch
    use_tuning = (TRAINING.use_tuning if args.tuning is None else (args.tuning == "on"))
    epochs = args.epochs if args.epochs is not None else TRAINING.epochs
    sequence_length = args.sequence_length if args.sequence_length is not None else TRAINING.sequence_length
    K = args.k if args.k is not None else TRAINING.k
    cap_val = args.cap if args.cap is not None else TRAINING.cap
    datasets = args.datasets or list(TRAINING.datasets)

    # Uncertainty parameters (fall back to TRAINING if not provided)
    unc_method = args.uncertainty if args.uncertainty is not None else getattr(TRAINING, "uncertainty_method", "none")
    alpha = args.alpha if args.alpha is not None else getattr(TRAINING, "alpha", 0.10)
    mc_T = args.mc_samples if args.mc_samples is not None else getattr(TRAINING, "mc_samples", 50)
    clip_pred = getattr(TRAINING, "clip_pred", True)

    print(
        f"[CONFIG] arch={arch} | tuning={'on' if use_tuning else 'off'} | epochs={epochs} | "
        f"seq_len={sequence_length} | K={K} | cap={cap_val} | datasets={datasets} | "
        f"uncertainty={unc_method} | alpha={alpha} | mc_T={mc_T}"
    )

    # ===============================================================
    # PREPARE FOLDERS & OPTIONAL DOWNLOAD
    # Creates required directories; downloads CMaps if requested
    # ===============================================================
    paths = make_paths()
    ensure_dirs(paths)
    if args.download:
        download.fetch_cmaps(paths.raw_data_dir, github_token=args.github_token)

    # ===============================================================
    # LOAD RAW DATA FILES
    # Reads train/test/RUL text files into pandas DataFrames
    # ===============================================================
    train_data, test_data, rul_data = load.load_all(paths.user_data_dir, datasets)

    # Basic inspection and quick data quality checks
    explore.inspect(train_data)
    print("Missing/Dupe report:", explore.missing_and_dupes_report(train_data, test_data))

    # ===============================================================
    # PREPROCESSING PIPELINE
    # Drops constant sensors, computes RUL, caps targets, and prepares features
    # ===============================================================
    # 1) Keep only sensors that vary across all selected datasets
    sensors_to_keep = explore.non_constant_sensors(train_data)
    print("Common non-constant sensors:", sensors_to_keep if sensors_to_keep else "(kept all)")
    if sensors_to_keep:
        preprocess.drop_unwanted_sensors(train_data, sensors_to_keep)

    # 2) Compute RUL targets for train/test
    for fd in datasets:
        train_data[fd] = preprocess.compute_rul_train(train_data[fd])
        test_data[fd] = preprocess.compute_rul_test(test_data[fd], rul_data[fd])

    # 3) Tag dataset name and combine train splits
    regimes.add_dataset_tags(train_data, test_data, datasets)
    combined_train = regimes.concat_train(train_data, datasets)

    # 4) Cap overly large RUL values to reduce target skew
    combined_train = preprocess.cap_rul(combined_train, cap_val)
    for fd in datasets:
        test_data[fd] = preprocess.cap_rul(test_data[fd], cap_val)

    # 5) Identify feature columns (settings + sensors)
    setting_cols = [c for c in combined_train.columns if c.startswith(("op_setting_", "setting_"))]
    sensor_cols = [c for c in combined_train.columns if c.startswith("sensor_")]
    if not setting_cols:
        raise RuntimeError("No op_setting* columns found.")
    if not sensor_cols:
        raise RuntimeError("No sensor* columns found.")

    # ===============================================================
    # TRAIN/VALIDATION SPLIT
    # Splits by entire engines (grouped) to avoid leakage
    # ===============================================================
    from sklearn.model_selection import GroupShuffleSplit
    combined_train["group_id"] = combined_train["dataset"].astype(str) + "_" + combined_train["engine_id"].astype(str)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(combined_train, groups=combined_train["group_id"].values))
    train_df = combined_train.iloc[train_idx].copy()
    val_df = combined_train.iloc[val_idx].copy()

    # ===============================================================
    # OPERATING REGIME CLUSTERING & SCALING
    # Clusters by settings, scales sensors per regime, and scales settings globally
    # ===============================================================
    # 1) Cluster by operating settings (KMeans)
    km = regimes.fit_kmeans_settings(train_df, setting_cols, k=K)
    train_df = regimes.assign_regime(train_df, km, setting_cols)
    val_df = regimes.assign_regime(val_df, km, setting_cols)
    for fd in datasets:
        test_data[fd] = regimes.assign_regime(test_data[fd], km, setting_cols)

    # 2) Per-regime standardization for sensor columns
    scalers = regimes.fit_per_regime_sensor_scalers(train_df, sensor_cols, "regime_id", K)
    train_df = regimes.transform_sensors_per_regime(train_df, scalers, sensor_cols, "regime_id")
    val_df = regimes.transform_sensors_per_regime(val_df, scalers, sensor_cols, "regime_id")
    for fd in datasets:
        test_data[fd] = regimes.transform_sensors_per_regime(test_data[fd], scalers, sensor_cols, "regime_id")

    # 3) Global scaling for the operating settings
    regimes.scale_settings(train_df, [val_df] + [test_data[fd] for fd in datasets], setting_cols)

    # 4) One-hot encode the regime ID so the model "knows" the operating condition
    sequences.add_regime_onehot(train_df, K)
    sequences.add_regime_onehot(val_df, K)
    for fd in datasets:
        sequences.add_regime_onehot(test_data[fd], K)

    # 5) Build sliding-window sequences for modeling
    regime_onehot_cols = [f"regime_{r}" for r in range(K)]
    feature_cols = sensor_cols + setting_cols + regime_onehot_cols
    print("Num features:", len(feature_cols))
    X_tr, y_tr, _ = sequences.create_sequences(train_df, feature_cols, sequence_length)
    X_val, y_val, _ = sequences.create_sequences(val_df, feature_cols, sequence_length)
    print(f"Train windows: {X_tr.shape} | Val windows: {X_val.shape}")

    # ===============================================================
    # MODEL SELECTION & TRAINING
    # Picks TCN/LSTM/CNN; trains with or without tuning
    # ===============================================================
    mod = {"tcn": model_tcn, "lstm": model_lstm, "cnn": model_cnn}[arch]
    if not use_tuning:
        print("[INFO] Training with fixed hyperparameters...")
        model, _ = mod.train_default(X_tr, y_tr, X_val, y_val, epochs=epochs)
    else:
        if hasattr(mod, "tune"):
            print("[INFO] Performing hyperparameter tuning...")
            best_model, best_hp, tuner, history = mod.tune(
                X_tr, y_tr, X_val, y_val,
                max_epochs=epochs,
                directory=f"{arch}_tuning",
                project_name=f"cmapss_{arch}",
            )
            model = best_model
            try:
                print("Best hyperparameters:", best_hp.values)
            except Exception:
                pass
        else:
            print(f"[WARN] Tuning is not implemented for '{arch}'. Training with fixed hyperparameters.")
            model, _ = mod.train_default(X_tr, y_tr, X_val, y_val, epochs=epochs)

    # ===============================================================
    # UNCERTAINTY CALIBRATION (OPTIONAL)
    # Conformal: compute residual quantile on the validation set
    # ===============================================================
    qhat = None
    if unc_method == "conformal":
        print(f"[UNCERTAINTY] Calibrating conformal intervals (alpha={alpha})...")
        qhat = uncertainty.conformal_calibrate(model, X_val, y_val, alpha=alpha, clip_pred=clip_pred)
        print(f"[UNCERTAINTY] qhat = {qhat:.4f}")

    # ===============================================================
    # BUILD TEST WINDOWS
    # Converts test sets into sliding windows for inference
    # ===============================================================
    X_te_dict, y_te_dict, engine_ids_te_dict, last_idx_map = \
        sequences.build_test_sequences_per_dataset(test_data, sequence_length, feature_cols)

    # ===============================================================
    # METRICS (OVER ALL TEST WINDOWS)
    # Prints RMSE and CMAPSS for each dataset using all windows
    # ===============================================================
    metrics_df = eval.per_dataset_metrics(model, X_te_dict, y_te_dict, datasets)
    print("\nPer-dataset test metrics:")
    print(metrics_df.to_string(index=False))

    # ===============================================================
    # FINAL-ENGINE RESULTS + UNCERTAINTY BANDS
    # For each engine, we take the final window only (last cycle).
    # MC-Dropout is run ONLY on those final windows to save memory.
    # ===============================================================
    final_rows = []
    for fd, Xte in X_te_dict.items():
        if Xte.shape[0] == 0 or fd not in last_idx_map:
            continue

        # Indices of the last window per engine (final cycle)
        final_idxs = sorted(last_idx_map[fd].values())

        if unc_method == "mc":
            # Memory-safe MC: sample only the final windows (not every window)
            print(f"[UNCERTAINTY] Running MC dropout on final windows ({mc_T} samples) for {fd}...")
            Xte_final = Xte[final_idxs]
            samples = uncertainty.mc_predict(model, Xte_final, T=mc_T)  # returns shape (T, Nfinal)
            mean, lo_fin, hi_fin = uncertainty.mc_interval_from_samples(samples, alpha=alpha, clip_pred=clip_pred)
            yhat_fin = np.clip(mean, 0, None) if clip_pred else mean
        else:
            # Deterministic prediction for all windows, then slice to finals
            yhat_all = model.predict(Xte, verbose=0).reshape(-1)
            if clip_pred:
                yhat_all = np.clip(yhat_all, 0, None)
            yhat_fin = yhat_all[final_idxs]
            if unc_method == "conformal" and qhat is not None:
                lo_all, hi_all = uncertainty.conformal_interval(yhat_all, qhat, clip_pred=clip_pred)
                lo_fin, hi_fin = lo_all[final_idxs], hi_all[final_idxs]
            else:
                lo_fin = hi_fin = None

        ytrue_all = y_te_dict[fd]
        eids_all = engine_ids_te_dict[fd]
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

    # ===============================================================
    # SAVE RESULTS (CSV)
    # Always writes one-row-per-engine with prediction; bands if available
    # ===============================================================
    if not final_df.empty:
        out_csv = Path("./_outputs/results/model/final_engine_rul_predictions.csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(out_csv, index=False)
        print(f"[SAVE] Final results: {out_csv.resolve()}")
    else:
        print("No final-window predictions found. Consider lowering --sequence-length.")
        return  # Nothing to plot or evaluate further

    # ===============================================================
    # PLOTS
    # Always generate Actual vs Prediction plots; bands overlay if present
    # ===============================================================
    fig_all = Path("./_outputs/results/figures/rul_bands_final_all.svg")
    eval.plot_final_engine_bands(final_df, str(fig_all))
    for fd in sorted(final_df["dataset"].unique()):
        fig_ds = Path(f"./_outputs/results/figures/rul_bands_final_{fd}.svg")
        eval.plot_final_engine_bands(final_df, str(fig_ds), dataset=fd)

    # ===============================================================
    # COVERAGE & INTERVAL WIDTH (ONLY IF BANDS EXIST)
    # Computes band coverage and width stats per dataset
    # ===============================================================
    has_lo = "y_pred_lo" in final_df.columns
    has_hi = "y_pred_hi" in final_df.columns
    some_finite_lo = has_lo and pd.to_numeric(final_df["y_pred_lo"], errors="coerce").notna().any()
    some_finite_hi = has_hi and pd.to_numeric(final_df["y_pred_hi"], errors="coerce").notna().any()

    if has_lo and has_hi and some_finite_lo and some_finite_hi:
        cov_df = eval.interval_accuracy_summary(final_df)
        cov_csv = Path("./_outputs/results/model/interval_coverage_summary.csv")
        cov_df.to_csv(cov_csv, index=False)
        print("\nInterval coverage summary:")
        print(cov_df.to_string(index=False))

        final_df["interval_width"] = (
            pd.to_numeric(final_df["y_pred_hi"], errors="coerce")
            - pd.to_numeric(final_df["y_pred_lo"], errors="coerce")
        )
        by_ds = final_df.groupby("dataset")["interval_width"].agg(["count", "mean", "median"]).reset_index()
        print("\nInterval width summary:")
        print(by_ds.to_string(index=False))
    else:
        # Bands missing or entirely NaN: coverage skipped, but plots were still produced
        print("[INFO] No usable interval bounds found; computed plots without bands.")


if __name__ == "__main__":
    main()
