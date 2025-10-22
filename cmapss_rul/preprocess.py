from typing import Dict, List
import pandas as pd
import numpy as np

def drop_unwanted_sensors(train_data: Dict[str, pd.DataFrame], sensors_to_keep: List[str]) -> None:
    for fd in train_data.keys():
        cols_to_drop = [c for c in train_data[fd].columns if c.startswith("sensor_") and c not in sensors_to_keep]
        train_data[fd].drop(columns=cols_to_drop, inplace=True)

def compute_rul_train(df: pd.DataFrame) -> pd.DataFrame:
    mx = df.groupby('engine_id')['cycle'].max().reset_index().rename(columns={'cycle':'max_cycle'})
    out = df.merge(mx, on='engine_id', how='left')
    out['RUL'] = out['max_cycle'] - out['cycle']
    return out.drop(columns=['max_cycle'])

def compute_rul_test(test_df: pd.DataFrame, rul_df: pd.DataFrame) -> pd.DataFrame:
    tmp = test_df.copy()
    r = rul_df.copy()
    r.index = r.index + 1  # engine ids start at 1
    tmp = tmp.merge(r, left_on='engine_id', right_index=True, how='left')
    mxt = tmp.groupby('engine_id')['cycle'].max().reset_index().rename(columns={'cycle':'max_cycle_test'})
    tmp = tmp.merge(mxt, on='engine_id', how='left')
    tmp['RUL'] = tmp['RUL'] + (tmp['max_cycle_test'] - tmp['cycle'])
    return tmp.drop(columns=['max_cycle_test'])

def cap_rul(df: pd.DataFrame, cap: int = 125) -> pd.DataFrame:
    if 'RUL' in df.columns:
        df = df.copy()
        df['RUL'] = np.minimum(df['RUL'], cap)
    return df

