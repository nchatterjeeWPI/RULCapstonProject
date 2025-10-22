from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple

COLUMNS = ['engine_id','cycle'] + [f'op_setting_{i}' for i in range(1,4)] + [f'sensor_{i}' for i in range(1,22)]

def load_all(user_data_dir: Path, datasets: List[str]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    train_data = {fd: pd.read_csv(user_data_dir / f"train_{fd}.txt", sep=r"\s+", header=None) for fd in datasets}
    test_data  = {fd: pd.read_csv(user_data_dir / f"test_{fd}.txt",  sep=r"\s+", header=None) for fd in datasets}
    rul_data   = {fd: pd.read_csv(user_data_dir / f"RUL_{fd}.txt",   sep=r"\s+", header=None, names=["RUL"]) for fd in datasets}
    for fd in datasets:
        train_data[fd].columns = COLUMNS
        test_data[fd].columns  = COLUMNS
    return train_data, test_data, rul_data