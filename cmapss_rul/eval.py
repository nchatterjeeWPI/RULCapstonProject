import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error

# =========================
# Utilities
# =========================
def _to_float_array(x):
    """Coerce a sequence/Series to a float numpy array (invalid -> NaN)."""
    if isinstance(x, (pd.Series, pd.Index)):
        s = pd.to_numeric(x, errors="coerce")
    else:
        s = pd.to_numeric(pd.Series(x), errors="coerce")
    return s.to_numpy(dtype=float)


# =========================
# 1) Scores & metrics
# =========================
def cmapss_score(rul_true, rul_pred):
    diff = np.array(rul_pred) - np.array(rul_true)
    return float(np.where(diff < 0, np.exp(-diff/13) - 1, np.exp(diff/10) - 1).sum())

def per_dataset_metrics(model, X_te_dict, y_te_dict, datasets):
    rows = []
    for fd in datasets:
        Xte, yte = X_te_dict.get(fd), y_te_dict.get(fd)
        if Xte is None or Xte.shape[0] == 0:
            rows.append((fd, np.nan, np.nan, 0)); continue
        yhat = np.clip(model.predict(Xte, verbose=0).flatten(), 0, None)
        rmse = float(np.sqrt(mean_squared_error(yte, yhat)))
        score = cmapss_score(yte, yhat)
        rows.append((fd, rmse, score, int(Xte.shape[0])))
    return pd.DataFrame(rows, columns=["dataset","RMSE","CMAPSS","n_windows"])


# =========================
# 2) Final-engine table
# =========================
def build_final_engine_table(model, X_te_dict, y_te_dict, engine_ids_te_dict, last_idx_map, clip_pred=True):
    rows = []
    for fd, Xte in X_te_dict.items():
        if fd not in last_idx_map or Xte.shape[0] == 0: continue
        yhat_all = model.predict(Xte, verbose=0).flatten()
        if clip_pred: yhat_all = np.clip(yhat_all, 0, None)
        ytrue_all = y_te_dict[fd]; eids_all = engine_ids_te_dict[fd]
        final_idxs = sorted(last_idx_map[fd].values())
        for idx in final_idxs:
            y_true = float(ytrue_all[idx]); y_pred = float(yhat_all[idx])
            delta = y_pred - y_true
            rows.append({
                "dataset": fd, "engine_id": int(eids_all[idx]),
                "y_true": y_true, "y_pred": y_pred,
                "delta": float(delta), "abs_delta": float(abs(delta)),
                "pct_abs_error": float(abs(delta)/y_true*100.0) if y_true > 0 else np.nan,
            })
    df = pd.DataFrame(rows)
    if df.empty: return df
    return df.sort_values("abs_delta", ascending=False, ignore_index=True)


def evaluate_per_dataset(model, X_te_dict, y_te_dict, engine_ids_te_dict, last_idx_map):
    # Simple wrapper that could be extended
    return {
        "note": "See per-dataset metrics via per_dataset_metrics() and final table via build_final_engine_table()."
    }


# =========================
# 3) Save helpers
# =========================
def save_results(metrics_df_or_dict, out_path):
    try:
        if isinstance(metrics_df_or_dict, pd.DataFrame):
            metrics_df_or_dict.to_csv(out_path, index=False)
        else:
            import json
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(metrics_df_or_dict, f, indent=2)
        print(f"[SAVE] Wrote results to {out_path}")
    except Exception as e:
        print(f"[SAVE] Could not save results: {e}")


# =========================
# 4) Interval coverage
# =========================
def coverage_from_bands(y_true, y_lo, y_hi):
    """
    Compute interval coverage (fraction of true values inside [lo, hi]).
    Robust to None/strings; they become NaN and are excluded.
    """
    y_true_f = _to_float_array(y_true)
    y_lo_f   = _to_float_array(y_lo)
    y_hi_f   = _to_float_array(y_hi)

    valid = np.isfinite(y_true_f) & np.isfinite(y_lo_f) & np.isfinite(y_hi_f)
    n = int(valid.sum())
    if n == 0:
        return {"n": 0, "covered": 0, "coverage": float("nan"), "violations_idx": []}

    t  = y_true_f[valid]; lo = y_lo_f[valid]; hi = y_hi_f[valid]
    ok = (t >= lo) & (t <= hi)
    covered = int(ok.sum())
    violations_idx = np.where(~ok)[0].tolist()
    return {"n": n, "covered": covered, "coverage": covered / n, "violations_idx": violations_idx}


def interval_accuracy_summary(final_df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by dataset and compute coverage stats.
    Returns a DataFrame with dataset, n, covered, coverage.
    """
    rows = []
    for ds, g in final_df.groupby("dataset"):
        cov = coverage_from_bands(g["y_true"], g["y_pred_lo"], g["y_pred_hi"])
        rows.append({
            "dataset": ds,
            "n": cov["n"],
            "covered": cov["covered"],
            "coverage": cov["coverage"],
        })
    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)


# =========================
# 5) Plots
# =========================
def plot_final_engine_bands(final_df: pd.DataFrame, out_path: str, dataset: str | None = None):
    """
    Plot y_true, y_pred, and [y_lo, y_hi] for final-engine rows.
    Handles NaNs gracefully; fills only where bounds are finite.
    """
    df = final_df.copy()
    if dataset is not None:
        df = df[df["dataset"] == dataset].copy()
    if df.empty:
        print("[PLOT] No data to plot.")
        return

    df = df.sort_values(["dataset", "engine_id"]).reset_index(drop=True)
    x = np.arange(len(df))

    y_true = _to_float_array(df["y_true"])
    y_pred = _to_float_array(df["y_pred"])
    lo = _to_float_array(df["y_pred_lo"]) if "y_pred_lo" in df else np.full_like(y_true, np.nan)
    hi = _to_float_array(df["y_pred_hi"]) if "y_pred_hi" in df else np.full_like(y_true, np.nan)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))

    finite_band = np.isfinite(lo) & np.isfinite(hi)
    if finite_band.any():
        plt.fill_between(x, lo, hi, alpha=0.2, label="Prediction Band (lo–hi)")

    plt.plot(x, y_true, label="RUL Actual")
    plt.plot(x, y_pred, "--o", markersize=3, label="RUL Prediction")

    title = "Final-Engine RUL with Prediction Bands"
    if dataset is not None:
        title += f" – {dataset}"
    plt.title(title)
    plt.xlabel("Engine (sorted)")
    plt.ylabel("RUL")
    plt.legend()
    plt.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Save as SVG
    svg_path = str(Path(out_path).with_suffix(".svg"))
    plt.savefig(svg_path, format="svg")
    plt.close()
    print(f"[PLOT] Saved scalable vector plot: {svg_path}")

