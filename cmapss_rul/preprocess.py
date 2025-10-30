# ===============================================================
# cmapss_rul/preprocess.py
# ===============================================================
# This module performs essential preprocessing steps before training:
#   1) Drops unused or constant sensors
#   2) Computes Remaining Useful Life (RUL) for train and test data
#   3) Caps RUL at a maximum threshold to limit extreme values
#
# These steps clean and standardize the raw CMAPSS dataset so that models
# can learn stable and meaningful relationships between sensor readings
# and engine degradation over time.
# ===============================================================

import numpy as np
import pandas as pd


# ===============================================================
# 1) DROP UNWANTED OR CONSTANT SENSORS
# ===============================================================
# Removes sensors that are either constant (no variation) or not shared
# across all datasets. This reduces noise and avoids overfitting.
# ---------------------------------------------------------------
def drop_unwanted_sensors(train_data, sensors_to_keep):
    """
    Removes unnecessary sensor columns from each training DataFrame.

    Parameters:
        train_data (dict[str, pd.DataFrame]): dictionary mapping dataset name (e.g., 'FD001')
            to its training DataFrame.
        sensors_to_keep (list[str]): list of sensor column names to keep based on variability analysis.

    Notes:
        - This function modifies the train_data dictionary *in place*.
        - Sensors that never change or are missing in other datasets are dropped.
    """
    for fd in train_data.keys():
        # Identify sensors starting with "sensor_" that are not in the keep list
        cols_to_drop = [c for c in train_data[fd].columns if c.startswith("sensor_") and c not in sensors_to_keep]
        # Drop those sensors from the current dataset
        train_data[fd].drop(columns=cols_to_drop, inplace=True)


# ===============================================================
# 2) COMPUTE REMAINING USEFUL LIFE (RUL) FOR TRAINING DATA
# ===============================================================
# In the CMAPSS dataset, each engine runs until failure. The RUL for each
# record is simply how many cycles are left before failure.
#
# Example:
#   If an engine runs for 300 total cycles, then:
#       at cycle 298 → RUL = 2
#       at cycle 299 → RUL = 1
#       at cycle 300 → RUL = 0
# ---------------------------------------------------------------
def compute_rul_train(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Remaining Useful Life (RUL) for training engines.

    Parameters:
        df (pd.DataFrame): raw training data with columns ['engine_id', 'cycle', sensors...]

    Returns:
        pd.DataFrame: same data with an additional 'RUL' column.
    """
    # Find the last cycle number for each engine
    mx = df.groupby('engine_id')['cycle'].max().reset_index().rename(columns={'cycle': 'max_cycle'})

    # Merge the max cycle count back onto the original dataframe
    out = df.merge(mx, on='engine_id', how='left')

    # RUL = (max cycle) - (current cycle)
    out['RUL'] = out['max_cycle'] - out['cycle']

    # Drop temporary column to clean up
    return out.drop(columns=['max_cycle'])


# ===============================================================
# 3) COMPUTE REMAINING USEFUL LIFE (RUL) FOR TEST DATA
# ===============================================================
# The test data is trickier: engines in the test set are *not* run to failure.
# Instead, we have a separate RUL file (RUL_FD00x.txt) telling us how many
# additional cycles each engine would have lasted after the last recorded one.
#
# The goal here is to add that extra RUL to the measured portion, so that
# every test sample has a true RUL value aligned with the training format.
# ---------------------------------------------------------------
def compute_rul_test(test_df: pd.DataFrame, rul_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Remaining Useful Life (RUL) for test engines using the provided RUL file.

    Parameters:
        test_df (pd.DataFrame): test data for one dataset (FD00x)
        rul_df (pd.DataFrame): corresponding RUL file with the final remaining life for each engine

    Returns:
        pd.DataFrame: test data with a calculated 'RUL' column.
    """

    # Work on a copy to avoid modifying the original test_df
    tmp = test_df.copy()

    # Copy RUL reference data (engine_id starts at 1, not 0)
    r = rul_df.copy()
    r.index = r.index + 1  # shift index so engine_id aligns with CMAPSS numbering

    # Merge base RUL info onto test set
    tmp = tmp.merge(r, left_on='engine_id', right_index=True, how='left')

    # Find max cycle observed for each test engine
    mxt = tmp.groupby('engine_id')['cycle'].max().reset_index().rename(columns={'cycle': 'max_cycle_test'})

    # Merge back to calculate cycle offset
    tmp = tmp.merge(mxt, on='engine_id', how='left')

    # Add the remaining cycles after the last observation
    tmp['RUL'] = tmp['RUL'] + (tmp['max_cycle_test'] - tmp['cycle'])

    # Clean up and return
    return tmp.drop(columns=['max_cycle_test'])


# ===============================================================
# 4) CAP MAXIMUM RUL VALUE
# ===============================================================
# RUL can be extremely large at the start of an engine’s life.
# To prevent the model from focusing too heavily on early, high-RUL samples,
# we cap RUL to a maximum value (default: 125). This also helps normalize
# the target range and improve learning stability.
# ---------------------------------------------------------------
def cap_rul(df: pd.DataFrame, cap: int = 125) -> pd.DataFrame:
    """
    Caps Remaining Useful Life (RUL) at a maximum threshold.

    Parameters:
        df (pd.DataFrame): dataset with an 'RUL' column
        cap (int): maximum RUL value to allow (default = 125)

    Returns:
        pd.DataFrame: copy of df with RUL values capped
    """
    if 'RUL' in df.columns:
        df = df.copy()
        df['RUL'] = np.minimum(df['RUL'], cap)
    return df
