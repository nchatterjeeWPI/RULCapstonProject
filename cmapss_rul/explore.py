# ===============================================================
# cmapss_rul/explore.py
# ===============================================================
# This module performs basic data exploration and integrity checks
# on the CMAPSS datasets before preprocessing or model training.
#
# What’s inside:
#   1) missing_and_dupes_report(): reports missing values and duplicate rows
#   2) non_constant_sensors(): identifies which sensors actually vary over time
#   3) inspect(): prints dataset shapes, columns, and sample records
#
# Why this matters:
#   - Some sensors are constant across all engines (no information gain)
#   - Some datasets may have missing or duplicate rows
#   - Inspecting structure early helps avoid downstream errors
# ===============================================================

import pandas as pd


# ===============================================================
# 1) MISSING AND DUPLICATE VALUE REPORT
# ===============================================================
# Checks each dataset for:
#   • Missing (NaN) values — indicates corrupted or incomplete data
#   • Duplicate rows — can skew statistics or bias training
#
# Returns a summary dictionary with counts for train and test.
# ---------------------------------------------------------------
def missing_and_dupes_report(train_data, test_data):
    """
    Creates a summary of missing values and duplicate counts for each dataset.

    Parameters:
        train_data (dict[str, pd.DataFrame]): training sets per FD00x subset
        test_data (dict[str, pd.DataFrame]): testing sets per FD00x subset

    Returns:
        dict[str, dict[str, int]]: nested dictionary summarizing missing and duplicate entries
    """
    out = {}
    for fd in train_data.keys():
        out[fd] = {
            # Count total missing values across all columns
            "train_missing": int(train_data[fd].isnull().sum().sum()),
            "test_missing": int(test_data[fd].isnull().sum().sum()),

            # Count number of duplicate rows
            "train_dupes": int(train_data[fd].duplicated().sum()),
            "test_dupes": int(test_data[fd].duplicated().sum()),
        }
    return out


# ===============================================================
# 2) IDENTIFY NON-CONSTANT SENSORS
# ===============================================================
# Finds which sensors have meaningful variation (standard deviation
# above a small threshold) across all training subsets.
#
# Why we do this:
#   - Some CMAPSS sensors have constant readings (zero variance).
#   - Removing them saves computation and improves model clarity.
#
# The function returns only those sensor columns that vary in every dataset.
# ---------------------------------------------------------------
def non_constant_sensors(train_data, std_threshold: float = 1e-4):
    """
    Identify sensors that vary across all training datasets.

    Parameters:
        train_data (dict[str, pd.DataFrame]): training sets per FD00x subset
        std_threshold (float): minimum standard deviation required to keep a sensor

    Returns:
        list[str]: sensors that are non-constant across all datasets
    """
    sensor_sets = []

    # Loop through each dataset and check variability
    for fd, df in train_data.items():
        # Include both sensors and op_settings for variability test
        sensor_op_cols = [
            c for c in df.columns if c.startswith("sensor_") or c.startswith("op_setting_")
        ]

        # Compute standard deviation for numeric columns
        std = df[sensor_op_cols].std(numeric_only=True)

        # Keep only columns with variability above the threshold
        non_const = std[std >= std_threshold].index.tolist()

        # Extract only sensor columns from the above list
        non_const_sensors = [c for c in non_const if c.startswith("sensor_")]

        # Save this set for the current dataset
        sensor_sets.append(set(non_const_sensors))

    # If no datasets were provided, return empty list
    if not sensor_sets:
        return []

    # Keep only sensors that are variable in ALL datasets
    common = set.intersection(*sensor_sets)

    # Sort by sensor number (so sensor_1, sensor_2, …)
    return sorted(list(common), key=lambda x: int(x.split("_")[1]))


# ===============================================================
# 3) INSPECT BASIC DATASET STRUCTURE
# ===============================================================
# Prints a summary for each training dataset:
#   • Shape (rows, columns)
#   • Column names (first 10)
#   • Missing and duplicate counts
#   • First 3 sample rows (for quick glance)
#
# This function is mostly for manual inspection — not used by the model.
# ---------------------------------------------------------------
def inspect(train_data_dict):
    """
    Prints quick overview of each training dataset for sanity checks.

    Parameters:
        train_data_dict (dict[str, pd.DataFrame]): training sets per FD00x subset
    """
    print("[INFO] Inspecting training datasets...")
    for name, df in train_data_dict.items():
        print(f"\n--- Dataset: {name} ---")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns[:10])} ...")  # print first 10 columns
        print(f"Missing values: {df.isnull().sum().sum()}")
        print(f"Duplicate rows: {df.duplicated().sum()}")
        print(df.head(3))  # show top 3 rows for a quick preview
