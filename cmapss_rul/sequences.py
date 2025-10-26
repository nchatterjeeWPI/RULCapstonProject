import numpy as np
import pandas as pd

def add_regime_onehot(df: pd.DataFrame, K: int):
    for r in range(K):
        col = f"regime_{r}"
        df[col] = (df['regime_id'] == r).astype(int)

def create_sequences(df: pd.DataFrame, feature_cols, sequence_length: int):
    X, y, eids = [], [], []
    for (ds, eid), g in df.groupby(['dataset', 'engine_id']):
        g = g.sort_values('cycle')
        if 'RUL' not in g.columns or g['RUL'].isnull().any():
            continue
        for i in range(len(g) - sequence_length + 1):
            window = g.iloc[i:i+sequence_length][feature_cols].values
            X.append(window)
            y.append(g.iloc[i+sequence_length-1]['RUL'])
            eids.append((ds, int(eid)))
    return np.array(X), np.array(y), np.array(eids, dtype=object)

def build_test_sequences_per_dataset(test_data_dict, seq_len: int, feature_cols):
    X_te_dict, y_te_dict, win_eids_dict, last_idx_map = {}, {}, {}, {}
    for fd, df in test_data_dict.items():
        df = df.sort_values(['engine_id', 'cycle']).copy()
        X_list, y_list, eid_list = [], [], []
        for eid, eng_df in df.groupby('engine_id'):
            eng_df = eng_df.sort_values('cycle')
            n = len(eng_df)
            if n < seq_len:
                continue
            mat = eng_df[feature_cols].values
            rul = eng_df['RUL'].values
            for i in range(n - seq_len + 1):
                X_list.append(mat[i:i+seq_len])
                y_list.append(rul[i+seq_len-1])
                eid_list.append(int(eid))
        if len(X_list) == 0:
            import numpy as np
            X_te_dict[fd] = np.empty((0, seq_len, len(feature_cols)))
            y_te_dict[fd] = np.empty((0,))
            win_eids_dict[fd] = np.empty((0,), dtype=int)
            last_idx_map[fd] = {}
            continue
        import numpy as np
        X = np.array(X_list); y = np.array(y_list); eids = np.array(eid_list, dtype=int)
        idx_map = {}
        for i, eid in enumerate(eids):
            idx_map[eid] = i
        X_te_dict[fd], y_te_dict[fd], win_eids_dict[fd], last_idx_map[fd] = X, y, eids, idx_map
    return X_te_dict, y_te_dict, win_eids_dict, last_idx_map

