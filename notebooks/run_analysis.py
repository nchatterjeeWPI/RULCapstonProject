# %% [markdown]
# # RUL Model Run Analysis
#
# Loads:
# - run_config.json (what we ran)
# - training histories (loss vs epoch)
# - per-dataset window metrics
# - final engine-level predictions & metrics
#
# and produces plots & tables to help decide how to improve the model.

# %% imports
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from IPython.display import display  # for nicer tables in notebooks
except ImportError:
    # Fallback for plain Python scripts: just alias to print
    def display(x):
        print(x)

plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["axes.grid"] = True

# %% locate project root (walk up until we see _outputs)
def find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "_outputs").exists():
            return p
    return start

# When running as a script, __file__ exists; in a notebook it usually doesn't.
if "__file__" in globals():
    start_path = Path(__file__).resolve().parent
else:
    start_path = Path.cwd()

PROJECT_ROOT = find_project_root(start_path)

OUTPUT_DIR = PROJECT_ROOT / "_outputs" / "results"
MODEL_DIR = OUTPUT_DIR / "model"
TRAIN_LOGS_DIR = PROJECT_ROOT / "_outputs" / "training_logs"
ANALYSIS_FIG_DIR = PROJECT_ROOT / "_outputs" / "figures" / "analysis"
ANALYSIS_FIG_DIR.mkdir(parents=True, exist_ok=True)


# Adjust if you only train a subset
ARCHS = ["tcn", "lstm", "cnn"]

print("Project root:", PROJECT_ROOT)
print("Results dir :", OUTPUT_DIR)
print("Model dir   :", MODEL_DIR)
print("Train logs  :", TRAIN_LOGS_DIR)


def save_and_show(fig, name: str, show: bool = True):
    """Save the current figure to ANALYSIS_FIG_DIR with the given name."""
    path = ANALYSIS_FIG_DIR / name
    fig.savefig(path, bbox_inches="tight")
    if show:
        pass
        # plt.show()
    else:
        plt.close(fig)
    print(f"[INFO] Saved figure: {path}")


# %% 1. Load run_config.json (if you added that in main.py)
run_config_path = OUTPUT_DIR / "run_config.json"
if run_config_path.exists():
    with run_config_path.open() as f:
        run_config = json.load(f)
    print("\n=== Run Config (truncated) ===")
    print(json.dumps(run_config, indent=2)[:2000], "...\n")
else:
    print(f"[WARN] run_config.json not found at {run_config_path}")

# %% 2. Training curves per architecture
def load_history(arch: str) -> pd.DataFrame | None:
    """
    Try to load training history for an architecture.
    Expected filenames (customize as needed):
      - <arch>_train_history.csv
      - <arch>_tuning_history.csv
    in _outputs/training_logs.
    """
    candidates = [
        TRAIN_LOGS_DIR / f"{arch}_train_history.csv",
        TRAIN_LOGS_DIR / f"{arch}_tuning_history.csv",
    ]
    for path in candidates:
        if path.exists():
            print(f"[INFO] Loaded history for {arch.upper()} from {path}")
            return pd.read_csv(path)
    print(f"[WARN] No history CSV found for {arch.upper()}")
    return None


def plot_history(df: pd.DataFrame, arch: str) -> None:
    """Plot train/val loss and (optionally) metrics vs epoch."""
    if df is None or df.empty:
        print(f"[WARN] Empty history for {arch.upper()}")
        return

    df = df.copy()
    if "epoch" not in df.columns:
        df.insert(0, "epoch", np.arange(1, len(df) + 1))

    # Loss
    fig = plt.figure()
    plt.plot(df["epoch"], df["loss"], label="train_loss")
    if "val_loss" in df.columns:
        plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{arch.upper()} – Loss vs Epoch")
    plt.legend()
    save_and_show(fig, f"{arch}_loss_vs_epoch.png")

    # Optional metrics (rmse, mae, mape, etc. if present)
    for metric in ["rmse", "mae", "mape"]:
        train_col = metric
        val_col = f"val_{metric}"
        if train_col in df.columns:
            fig = plt.figure()
            plt.plot(df["epoch"], df[train_col], label=f"train_{metric}")
            if val_col in df.columns:
                plt.plot(df["epoch"], df[val_col], label=f"val_{metric}")
            plt.xlabel("Epoch")
            plt.ylabel(metric.upper())
            plt.title(f"{arch.upper()} – {metric.upper()} vs Epoch")
            plt.legend()
            save_and_show(fig, f"{arch}_{metric}_vs_epoch.png")


for arch in ARCHS:
    hist_df = load_history(arch)
    if hist_df is not None:
        plot_history(hist_df, arch)

# %% 3. Per-dataset window metrics (if you saved them)
def load_per_dataset_metrics(arch: str) -> pd.DataFrame | None:
    path = MODEL_DIR / f"per_dataset_window_metrics_{arch}.csv"
    if not path.exists():
        print(f"[WARN] per_dataset_window_metrics not found for {arch.upper()} at {path}")
        return None
    df = pd.read_csv(path)
    print(f"\n=== {arch.upper()} – Per-dataset window metrics ===")
    try:
        display(df)  # Jupyter
    except NameError:
        print(df.to_string(index=False))
    return df


per_dataset_metrics = {}
for arch in ARCHS:
    per_dataset_metrics[arch] = load_per_dataset_metrics(arch)

# %% 4. Final engine-level metrics (per dataset + OVERALL)
def load_final_engine_metrics(arch: str) -> pd.DataFrame | None:
    path = MODEL_DIR / f"final_engine_metrics_{arch}.csv"
    if not path.exists():
        print(f"[WARN] final_engine_metrics not found for {arch.upper()} at {path}")
        return None
    df = pd.read_csv(path)
    print(f"\n=== {arch.upper()} – Final engine metrics ===")
    try:
        display(df)
    except NameError:
        print(df.to_string(index=False))
    return df


final_metrics = {}
for arch in ARCHS:
    final_metrics[arch] = load_final_engine_metrics(arch)

def extract_overall_row(df: pd.DataFrame, arch: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    # Look for a column that identifies dataset/group
    possible_cols = [c for c in df.columns if c.lower() in ("dataset", "group", "subset")]
    if not possible_cols:
        return pd.DataFrame()
    col = possible_cols[0]
    mask = df[col].astype(str).str.upper() == "OVERALL"
    overall = df[mask].copy()
    overall.insert(0, "arch", arch.upper())
    return overall

overall_rows = [extract_overall_row(df, arch) for arch, df in final_metrics.items()]
overall_rows = [r for r in overall_rows if not r.empty]
if overall_rows:
    overall_cmp = pd.concat(overall_rows, ignore_index=True)
    print("\n=== Architecture Comparison – OVERALL row ===")
    try:
        display(overall_cmp)
    except NameError:
        print(overall_cmp.to_string(index=False))
else:
    print("[WARN] No OVERALL rows found in final metrics.")

# %% 5. Engine-level predictions – error analysis
def load_final_predictions(arch: str) -> pd.DataFrame | None:
    path = MODEL_DIR / f"final_engine_rul_predictions_{arch}.csv"
    if not path.exists():
        print(f"[WARN] final_engine_rul_predictions not found for {arch.upper()} at {path}")
        return None
    df = pd.read_csv(path)
    print(f"\n[INFO] Loaded final predictions for {arch.upper()} from {path}")
    return df


def enrich_predictions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "delta" not in df.columns and {"y_true", "y_pred"} <= set(df.columns):
        df["delta"] = df["y_pred"] - df["y_true"]
    if "abs_delta" not in df.columns and "delta" in df.columns:
        df["abs_delta"] = df["delta"].abs()
    if "pct_abs_error" not in df.columns and {"y_true", "abs_delta"} <= set(df.columns):
        df["pct_abs_error"] = np.where(
            df["y_true"] != 0,
            100.0 * df["abs_delta"] / df["y_true"].abs(),
            np.nan,
        )
    return df


def plot_error_distributions(df: pd.DataFrame, arch: str) -> None:
    if df is None or df.empty:
        print(f"[WARN] Empty predictions for {arch.upper()}")
        return

    df = enrich_predictions(df)

    # Histogram of absolute error
    fig = plt.figure()
    df["abs_delta"].hist(bins=30)
    plt.xlabel("|y_pred - y_true| (cycles)")
    plt.ylabel("count")
    plt.title(f"{arch.upper()} – Histogram of absolute error")
    save_and_show(fig, f"{arch}_abs_error_hist.png")

    # Scatter: true vs predicted
    if {"y_true", "y_pred"} <= set(df.columns):
        fig = plt.figure()
        plt.scatter(df["y_true"], df["y_pred"], alpha=0.6)
        max_val = max(df["y_true"].max(), df["y_pred"].max())
        plt.plot([0, max_val], [0, max_val])  # ideal y = x
        plt.xlabel("True RUL")
        plt.ylabel("Predicted RUL")
        plt.title(f"{arch.upper()} – y_true vs y_pred")
        save_and_show(fig, f"{arch}_ytrue_vs_ypred.png")

    # Error vs true RUL
    if {"y_true", "delta"} <= set(df.columns):
        fig = plt.figure()
        plt.scatter(df["y_true"], df["delta"], alpha=0.6)
        plt.axhline(0, linestyle="--")
        plt.xlabel("True RUL")
        plt.ylabel("Prediction error (y_pred - y_true)")
        plt.title(f"{arch.upper()} – Error vs True RUL")
        save_and_show(fig, f"{arch}_error_vs_true_rul.png")

    # Boxplot of abs error per dataset (if dataset column exists)
    # Boxplot of abs error per dataset (if dataset-like column exists)
    # Try to find something that looks like a dataset column
    ds_candidates = [
        c for c in df.columns
        if "dataset" in c.lower() or c.lower().startswith("fd")
    ]

    if not ds_candidates:
        print(f"[WARN] No dataset-like column found for {arch.upper()} – "
              f"available columns: {list(df.columns)}")
        return

    ds_col = ds_candidates[0]
    df_plot = df[[ds_col, "abs_delta"]].copy()

    # Ensure numeric abs_delta
    df_plot["abs_delta"] = pd.to_numeric(df_plot["abs_delta"], errors="coerce")

    # Show counts per dataset to verify we have data
    counts = df_plot.groupby(ds_col)["abs_delta"].count()
    print(f"\n[{arch.upper()}] abs_delta counts per dataset for boxplot:")
    print(counts)

    if (counts == 0).all():
        print(f"[WARN] All abs_delta values are NaN or missing for {arch.upper()} – "
              "boxplot will be empty.")
        return

    fig, ax = plt.subplots()
    df_plot.boxplot(column="abs_delta", by=ds_col, ax=ax)
    ax.set_ylabel("|y_pred - y_true| (cycles)")
    ax.set_title(f"{arch.upper()} – Absolute error by dataset")
    plt.suptitle("")
    plt.tight_layout()
    save_and_show(fig, f"{arch}_abs_error_by_dataset.png")


for arch in ARCHS:
    preds = load_final_predictions(arch)
    if preds is None:
        continue

    preds = enrich_predictions(preds)

    # Worst engines by mean |error|
    required = {"dataset", "engine_id", "abs_delta"}
    if required <= set(preds.columns):
        worst = (
            preds.groupby(["dataset", "engine_id"])["abs_delta"]
            .mean()
            .reset_index()
            .sort_values("abs_delta", ascending=False)
            .head(10)
        )
        print(f"\n=== {arch.upper()} – Worst 10 engines by mean |error| ===")
        try:
            display(worst)
        except NameError:
            print(worst.to_string(index=False))

    plot_error_distributions(preds, arch)

# %% 6. Accuracy within ±K cycles (computed directly from preds)
def accuracy_within_k(df: pd.DataFrame, k: int) -> float:
    if df is None or df.empty:
        return np.nan
    df = enrich_predictions(df)
    return (df["abs_delta"] <= k).mean()


for arch in ARCHS:
    preds = load_final_predictions(arch)
    if preds is None:
        continue
    preds = enrich_predictions(preds)
    for k in (10, 20):
        acc = accuracy_within_k(preds, k)
        print(f"{arch.upper()} – Accuracy within ±{k} cycles: {acc:.3f}")
