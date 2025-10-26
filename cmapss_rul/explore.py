import pandas as pd

def missing_and_dupes_report(train_data, test_data):
    out = {}
    for fd in train_data.keys():
        out[fd] = {
            "train_missing": int(train_data[fd].isnull().sum().sum()),
            "test_missing": int(test_data[fd].isnull().sum().sum()),
            "train_dupes": int(train_data[fd].duplicated().sum()),
            "test_dupes": int(test_data[fd].duplicated().sum()),
        }
    return out

def non_constant_sensors(train_data, std_threshold: float = 1e-4):
    sensor_sets = []
    for fd, df in train_data.items():
        sensor_op_cols = [c for c in df.columns if c.startswith("sensor_") or c.startswith("op_setting_")]
        std = df[sensor_op_cols].std(numeric_only=True)
        non_const = std[std >= std_threshold].index.tolist()
        non_const_sensors = [c for c in non_const if c.startswith("sensor_")]
        sensor_sets.append(set(non_const_sensors))
    if not sensor_sets: return []
    common = set.intersection(*sensor_sets)
    return sorted(list(common), key=lambda x: int(x.split("_")[1]))

def inspect(train_data_dict):
    print("[INFO] Inspecting training datasets...")
    for name, df in train_data_dict.items():
        print(f"\n--- Dataset: {name} ---")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns[:10])} ...")
        print(f"Missing values: {df.isnull().sum().sum()}")
        print(f"Duplicate rows: {df.duplicated().sum()}")
        print(df.head(3))
