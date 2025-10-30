# cmapss_rul/load.py
# ===============================================================
# Purpose:
#   Handles loading of CMAPSS turbofan engine datasets (train, test, RUL).
#   Each dataset (FD001–FD004) comes as plain-text files.
#   This module reads them into pandas DataFrames with proper column names.
#
# Functions:
#   - load_all(): Reads all datasets and returns three dictionaries:
#       train_data, test_data, rul_data
# ===============================================================

from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

# ---------------------------------------------------------------
# Define the standard column layout for CMAPSS datasets.
# Each data file has the same structure:
#   engine_id | cycle | op_setting_1-3 | sensor_1-21
# ---------------------------------------------------------------
COLUMNS = (
    ['engine_id', 'cycle']
    + [f'op_setting_{i}' for i in range(1, 4)]
    + [f'sensor_{i}' for i in range(1, 22)]
)


def load_all(user_data_dir: Path, datasets: List[str]) -> Tuple[
    Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]
]:
    """
    Load all CMAPSS dataset files (train, test, and RUL) into pandas DataFrames.

    Args:
        user_data_dir (Path): Path to folder containing CMAPSS .txt files.
        datasets (List[str]): List of dataset names, e.g. ['FD001', 'FD002'].

    Returns:
        Tuple of three dictionaries:
            - train_data: {FD001: DataFrame, FD002: DataFrame, ...}
            - test_data:  {FD001: DataFrame, FD002: DataFrame, ...}
            - rul_data:   {FD001: DataFrame, FD002: DataFrame, ...}
    """

    # ---------------------------------------------------------------
    # Read training and test data files for each selected dataset.
    # Files are space-separated text with no headers.
    # ---------------------------------------------------------------
    train_data = {
        fd: pd.read_csv(user_data_dir / f"train_{fd}.txt", sep=r"\s+", header=None)
        for fd in datasets
    }
    test_data = {
        fd: pd.read_csv(user_data_dir / f"test_{fd}.txt", sep=r"\s+", header=None)
        for fd in datasets
    }

    # RUL (Remaining Useful Life) values are stored in a separate file.
    # Each row corresponds to one engine in the test set.
    rul_data = {
        fd: pd.read_csv(
            user_data_dir / f"RUL_{fd}.txt", sep=r"\s+", header=None, names=["RUL"]
        )
        for fd in datasets
    }

    # ---------------------------------------------------------------
    # Assign consistent column names to train/test DataFrames.
    # ---------------------------------------------------------------
    for fd in datasets:
        train_data[fd].columns = COLUMNS
        test_data[fd].columns = COLUMNS

    # ---------------------------------------------------------------
    # Return all three as a tuple of dictionaries.
    # ---------------------------------------------------------------
    return train_data, test_data, rul_data
