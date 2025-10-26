import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def add_dataset_tags(train_data, test_data, datasets):
    for fd in datasets:
        train_data[fd] = train_data[fd].copy(); train_data[fd]['dataset'] = fd
        test_data[fd]  = test_data[fd].copy();  test_data[fd]['dataset']  = fd

def concat_train(train_data, datasets):
    import pandas as pd
    return pd.concat([train_data[fd] for fd in datasets], ignore_index=True)

def fit_kmeans_settings(train_df: pd.DataFrame, setting_cols, k=6):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(train_df[setting_cols].values)
    return km

def assign_regime(df: pd.DataFrame, km: KMeans, setting_cols):
    out = df.copy()
    out['regime_id'] = km.predict(out[setting_cols].values)
    return out

def fit_per_regime_sensor_scalers(df_train: pd.DataFrame, sensor_cols, regime_col: str, K: int):
    scalers = {}
    for r in range(K):
        scaler = StandardScaler()
        mask = (df_train[regime_col] == r)
        scaler.fit(df_train.loc[mask, sensor_cols] if mask.any() else df_train[sensor_cols])
        scalers[r] = scaler
    return scalers

def transform_sensors_per_regime(df: pd.DataFrame, scalers, sensor_cols, regime_col: str):
    out = df.copy()
    out[sensor_cols] = out[sensor_cols].astype(float)  # ensure float dtype
    rids = out[regime_col].to_numpy()
    X = out[sensor_cols].copy()
    for r, scaler in scalers.items():
        mask = (rids == r)
        if mask.any():
            X.loc[mask, :] = scaler.transform(X.loc[mask, :])
    out[sensor_cols] = X
    return out


def scale_settings(df_train: pd.DataFrame, df_others, setting_cols):
    scaler = StandardScaler()
    df_train.loc[:, setting_cols] = scaler.fit_transform(df_train[setting_cols])
    for df in df_others:
        df.loc[:, setting_cols] = scaler.transform(df[setting_cols])
    return scaler

