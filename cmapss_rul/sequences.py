# ===============================================================
# cmapss_rul/sequences.py
# ===============================================================
# This module prepares input sequences for model training and testing.
# In predictive maintenance, we use a fixed-length sliding window (sequence)
# to represent recent sensor readings before predicting the Remaining Useful Life (RUL).
#
# Main functions:
#   1. add_regime_onehot() - Adds one-hot encoded regime indicator columns
#   2. create_sequences()   - Builds sliding-window training/validation sequences
#   3. build_test_sequences_per_dataset() - Prepares test sequences for each dataset
# ===============================================================

import numpy as np
import pandas as pd


# ===============================================================
# 1. ADD REGIME ONE-HOT COLUMNS
# ===============================================================
# Each operating regime (0, 1, 2, …, K-1) is converted into a set of binary columns.
# For example, regime_0 = 1 means the sample belongs to regime 0; otherwise 0.
# This allows the model to learn regime information as part of its input.
# ---------------------------------------------------------------
def add_regime_onehot(df: pd.DataFrame, K: int):
    """
    Add K binary (0/1) columns to represent which regime each sample belongs to.
    Each row in the dataframe gets a 1 in its corresponding regime column and 0 in others.

    Parameters:
        df (pd.DataFrame): The input data containing a 'regime_id' column.
        K (int): Number of regimes (clusters) found during preprocessing.
    """
    for r in range(K):
        col = f"regime_{r}"                    # Create column name (e.g., 'regime_0')
        df[col] = (df['regime_id'] == r).astype(int)  # Set to 1 where regime_id == r


# ===============================================================
# 2. CREATE SEQUENCES FOR TRAINING AND VALIDATION
# ===============================================================
# This function slides a window of fixed length (sequence_length) across each engine’s data.
# Each window becomes one sample for model training, where:
#   - X contains the sequence of features (sensor data, settings, regime one-hot)
#   - y is the RUL at the last time step of that sequence
#
# Example:
#   Suppose sequence_length = 50 → we take 50 consecutive cycles for one training sample.
# ---------------------------------------------------------------
def create_sequences(df: pd.DataFrame, feature_cols, sequence_length: int):
    """
    Create sliding-window sequences from the full training dataframe.

    Parameters:
        df (pd.DataFrame): Combined data containing all engines and cycles.
        feature_cols (list): Columns to use as model inputs (sensor + settings + regimes).
        sequence_length (int): Number of time steps per sequence.

    Returns:
        X (np.ndarray): 3D array of input sequences with shape (samples, sequence_length, features)
        y (np.ndarray): 1D array of target RUL values for each sequence
        eids (np.ndarray): Engine identifiers for each sequence (dataset + engine_id)
    """

    X, y, eids = [], [], []

    # Group by dataset and engine so each engine’s timeline is processed separately
    for (ds, eid), g in df.groupby(['dataset', 'engine_id']):
        # Ensure chronological order
        g = g.sort_values('cycle')

        # Skip engines missing RUL data
        if 'RUL' not in g.columns or g['RUL'].isnull().any():
            continue

        # Slide a window of size 'sequence_length' across the cycles
        for i in range(len(g) - sequence_length + 1):
            # Extract feature matrix for this window
            window = g.iloc[i:i+sequence_length][feature_cols].values
            X.append(window)

            # The target RUL is the RUL at the last cycle in this window
            y.append(g.iloc[i+sequence_length-1]['RUL'])

            # Keep track of which engine this window came from
            eids.append((ds, int(eid)))

    # Convert lists to numpy arrays for efficient model input
    return np.array(X), np.array(y), np.array(eids, dtype=object)


# ===============================================================
# 3. CREATE TEST SEQUENCES (PER DATASET)
# ===============================================================
# Similar to create_sequences(), but this version processes test data for every dataset.
# It returns a dictionary of arrays so we can evaluate each FD00x dataset separately.
# ---------------------------------------------------------------
def build_test_sequences_per_dataset(test_data_dict, seq_len: int, feature_cols):
    """
    Build test sequences for each dataset in a dictionary format.

    Parameters:
        test_data_dict (dict): Dictionary of test DataFrames keyed by dataset name (e.g., 'FD001')
        seq_len (int): Sequence length (number of cycles per window)
        feature_cols (list): List of columns to use as model inputs

    Returns:
        X_te_dict (dict): Maps dataset → 3D array of test input sequences
        y_te_dict (dict): Maps dataset → 1D array of true RUL values
        win_eids_dict (dict): Maps dataset → array of engine IDs per sequence
        last_idx_map (dict): Maps dataset → dictionary showing the final sequence index per engine
    """

    # Initialize output containers for each dataset
    X_te_dict, y_te_dict, win_eids_dict, last_idx_map = {}, {}, {}, {}

    # Loop over each dataset (FD001, FD002, etc.)
    for fd, df in test_data_dict.items():
        # Sort by engine and cycle number to preserve temporal order
        df = df.sort_values(['engine_id', 'cycle']).copy()

        X_list, y_list, eid_list = [], [], []

        # Process each engine individually
        for eid, eng_df in df.groupby('engine_id'):
            eng_df = eng_df.sort_values('cycle')
            n = len(eng_df)

            # Skip short engines (fewer cycles than the sequence length)
            if n < seq_len:
                continue

            # Convert selected features and RUL values to numpy arrays
            mat = eng_df[feature_cols].values
            rul = eng_df['RUL'].values

            # Slide the sequence window across the engine's timeline
            for i in range(n - seq_len + 1):
                X_list.append(mat[i:i+seq_len])       # input sequence
                y_list.append(rul[i+seq_len-1])       # target RUL at end of window
                eid_list.append(int(eid))             # record engine id

        # Handle case: no valid sequences for this dataset
        if len(X_list) == 0:
            X_te_dict[fd] = np.empty((0, seq_len, len(feature_cols)))
            y_te_dict[fd] = np.empty((0,))
            win_eids_dict[fd] = np.empty((0,), dtype=int)
            last_idx_map[fd] = {}
            continue

        # Convert to numpy arrays
        X = np.array(X_list)
        y = np.array(y_list)
        eids = np.array(eid_list, dtype=int)

        # Map each engine ID to the index of its final sequence window
        idx_map = {}
        for i, eid in enumerate(eids):
            idx_map[eid] = i

        # Store results in dictionaries keyed by dataset name
        X_te_dict[fd] = X
        y_te_dict[fd] = y
        win_eids_dict[fd] = eids
        last_idx_map[fd] = idx_map

    # Return all test data sequences, organized per dataset
    return X_te_dict, y_te_dict, win_eids_dict, last_idx_map
