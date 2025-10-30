# ===============================================================
# cmapss_rul/regimes.py
# ===============================================================
# This module groups engine records by operating conditions (“regimes”)
# and applies appropriate scaling:
#   1) Tag each row with its dataset (FD001, FD002, …)
#   2) Concatenate multiple datasets for training
#   3) Cluster operating settings (op_setting_*) with KMeans → regime_id
#   4) Fit per-regime StandardScalers for sensor columns (sensor_*)
#   5) Transform sensors per regime and globally scale operating settings
#
# Why do this?
# Different engines run under different operating conditions. Clustering by
# settings (e.g., op_setting_1..3) lets us normalize sensors *within* each
# operating regime, improving learning stability and accuracy.
# ===============================================================

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ===============================================================
# 1) ADD DATASET TAGS
# ===============================================================
# Adds a 'dataset' column (e.g., 'FD001', 'FD002', …) to every row
# so downstream steps know which dataset each sample came from.
# ---------------------------------------------------------------
def add_dataset_tags(train_data, test_data, datasets):
    """
    Mutates the per-dataset dicts by adding a 'dataset' column to each DataFrame.

    Parameters:
        train_data (dict[str, pd.DataFrame]): maps FD name → training DataFrame
        test_data  (dict[str, pd.DataFrame]): maps FD name → test DataFrame
        datasets   (list[str]): list like ['FD001', 'FD002', ...]
    """
    for fd in datasets:
        # Copy to avoid mutating original references
        train_data[fd] = train_data[fd].copy()
        train_data[fd]['dataset'] = fd

        test_data[fd] = test_data[fd].copy()
        test_data[fd]['dataset'] = fd


# ===============================================================
# 2) CONCATENATE TRAIN SETS
# ===============================================================
# Stacks multiple FDxxx training DataFrames into one big table for modeling.
# We preserve the 'dataset' tag so we can split/evaluate later.
# ---------------------------------------------------------------
def concat_train(train_data, datasets):
    """
    Concatenate training DataFrames across all requested datasets.

    Returns:
        pd.DataFrame: combined training table with a 'dataset' column
    """
    return pd.concat([train_data[fd] for fd in datasets], ignore_index=True)


# ===============================================================
# 3) CLUSTER OPERATING SETTINGS → regime_id
# ===============================================================
# KMeans groups similar operating conditions together using 'op_setting_*' cols.
# Each row is later assigned a 'regime_id' predicted by this fitted model.
# ---------------------------------------------------------------
def fit_kmeans_settings(train_df: pd.DataFrame, setting_cols, k=6):
    """
    Fit a KMeans clustering model on operating setting columns.

    Parameters:
        train_df (pd.DataFrame): combined training data
        setting_cols (list[str]): columns like ['op_setting_1', 'op_setting_2', 'op_setting_3']
        k (int): number of regimes (clusters)

    Returns:
        KMeans: fitted model that can predict regime_id
    """
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(train_df[setting_cols].values)
    return km


def assign_regime(df: pd.DataFrame, km: KMeans, setting_cols):
    """
    Use a fitted KMeans model to assign a 'regime_id' to each row.

    Parameters:
        df (pd.DataFrame): any split (train/val/test) with setting columns
        km (KMeans): fitted KMeans model from fit_kmeans_settings()
        setting_cols (list[str]): same columns used during kmeans fit

    Returns:
        pd.DataFrame: copy of df with a new 'regime_id' column
    """
    out = df.copy()
    out['regime_id'] = km.predict(out[setting_cols].values)
    return out


# ===============================================================
# 4) FIT PER-REGIME SCALERS ON SENSOR COLUMNS
# ===============================================================
# For each regime r (0..K-1), fit a StandardScaler on that regime’s sensor data.
# If no samples exist for a regime in the training split, fall back to
# fitting on all sensors (prevents errors).
# ---------------------------------------------------------------
def fit_per_regime_sensor_scalers(df_train: pd.DataFrame, sensor_cols, regime_col: str, K: int):
    """
    Fit a StandardScaler for each regime on sensor columns.

    Parameters:
        df_train (pd.DataFrame): training data WITH a 'regime_id' (or custom) column
        sensor_cols (list[str]): columns like ['sensor_1', ..., 'sensor_N']
        regime_col (str): column name for regime (e.g., 'regime_id')
        K (int): number of regimes

    Returns:
        dict[int, StandardScaler]: mapping regime → fitted scaler
    """
    scalers = {}
    for r in range(K):
        scaler = StandardScaler()
        mask = (df_train[regime_col] == r)

        # If the split has no samples for this regime, fit on all data to be safe
        to_fit = df_train.loc[mask, sensor_cols] if mask.any() else df_train[sensor_cols]
        scaler.fit(to_fit)

        scalers[r] = scaler
    return scalers


# ===============================================================
# 5) TRANSFORM SENSORS PER REGIME
# ===============================================================
# Apply the corresponding scaler to each row based on its regime_id.
# We copy the DataFrame for safety, and only transform the sensor columns.
# ---------------------------------------------------------------
def transform_sensors_per_regime(df: pd.DataFrame, scalers, sensor_cols, regime_col: str):
    """
    Transform sensor columns using the pre-fitted per-regime scalers.

    Parameters:
        df (pd.DataFrame): any split (train/val/test) with regime_col present
        scalers (dict[int, StandardScaler]): regime → scaler
        sensor_cols (list[str]): sensor columns to scale
        regime_col (str): column containing regime ids

    Returns:
        pd.DataFrame: copy of df with scaled sensor columns
    """
    out = df.copy()

    # Ensure numeric dtype for stable scaling
    out[sensor_cols] = out[sensor_cols].astype(float)

    # Vector of regime ids for fast boolean masking
    rids = out[regime_col].to_numpy()

    # Work on a copy of sensor subframe to avoid SettingWithCopyWarning
    X = out[sensor_cols].copy()

    # For each regime, transform only the rows that belong to it
    for r, scaler in scalers.items():
        mask = (rids == r)
        if mask.any():
            X.loc[mask, :] = scaler.transform(X.loc[mask, :])

    out[sensor_cols] = X
    return out


# ===============================================================
# 6) SCALE OPERATING SETTINGS (GLOBAL)
# ===============================================================
# Fit a single StandardScaler on the training split’s operating settings,
# then apply the same transform to validation and test splits.
# This ensures settings are on a similar scale across all splits.
# ---------------------------------------------------------------
def scale_settings(df_train: pd.DataFrame, df_others, setting_cols):
    """
    Fit a global StandardScaler on setting columns from the training set
    and apply it to other DataFrames (validation + all test sets).

    Parameters:
        df_train (pd.DataFrame): training split
        df_others (list[pd.DataFrame]): other splits to transform (val + tests)
        setting_cols (list[str]): columns to scale (op_setting_*)

    Returns:
        StandardScaler: the fitted scaler (useful for debugging or later reuse)
    """
    scaler = StandardScaler()
    df_train.loc[:, setting_cols] = scaler.fit_transform(df_train[setting_cols])

    for df in df_others:
        df.loc[:, setting_cols] = scaler.transform(df[setting_cols])

    return scaler
