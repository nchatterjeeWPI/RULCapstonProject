from typing import Dict, List, Set
import pandas as pd
import numpy as np

def missing_and_dupes_report(train_data: Dict[str, pd.DataFrame], test_data: Dict[str, pd.DataFrame]) -> Dict[str, dict]:
    out = {}
    for fd in train_data.keys():
        out[fd] = {
            "train_missing": int(train_data[fd].isnull().sum().sum()),
            "test_missing": int(test_data[fd].isnull().sum().sum()),
            "train_dupes": int(train_data[fd].duplicated().sum()),
            "test_dupes": int(test_data[fd].duplicated().sum()),
        }
    return out

def non_constant_sensors(train_data: Dict[str, pd.DataFrame], std_threshold: float = 1e-4) -> List[str]:
    sensor_sets: List[Set[str]] = []
    for fd, df in train_data.items():
        sensor_op_cols = [c for c in df.columns if c.startswith("sensor_") or c.startswith("op_setting_")]
        std = df[sensor_op_cols].std(numeric_only=True)
        non_const = std[std >= std_threshold].index.tolist()
        non_const_sensors = [c for c in non_const if c.startswith("sensor_")]
        sensor_sets.append(set(non_const_sensors))
    if not sensor_sets:
        return []
    common = set.intersection(*sensor_sets)
    return sorted(list(common), key=lambda x: int(x.split("_")[1]))

def strong_correlations(df: pd.DataFrame, cols: List[str], thr: float = 0.9) -> List[tuple]:
    corr = df[cols].corr()
    pairs = []
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            v = corr.iloc[i, j]
            if abs(v) > thr:
                pairs.append((cols[i], cols[j], float(v)))
    return pairs
