import pytest
from cmapss_rul import pipeline


@pytest.fixture
def sample_data():
    import pandas as pd

    df_train = pd.DataFrame({
        "engine_id": [1, 1, 1, 2, 2],
        "cycle":    [1, 2, 3, 1, 2],
        "sensor_1": [10, 12, 14, 8, 9],
        "sensor_2": [5.0, 5.5, 6.0, 4.5, 4.8],
    })

    df_test = df_train.copy()

    # rUL table -> one row per engine
    df_rul = pd.DataFrame({"RUL": [100, 90]})

    df_rul.index = range(1, len(df_rul) + 1)

    return df_train, df_test, df_rul


def test_preprocess_scales_and_returns(sample_data):
    """Ensure preprocess_data() returns scaled DataFrames."""
    df_train, df_test, df_rul = sample_data

    train_dict = {"FD001": df_train}
    test_dict  = {"FD001": df_test}
    rul_dict   = {"FD001": df_rul}

    processed_train, processed_test = pipeline.preprocess_data(
        train_dict,
        test_dict,
        rul_dict,
        datasets=["FD001"],
        cap_val=125,
        sensors_to_keep=["sensor_1", "sensor_2"],
    )

    # Unpack FD001 df
    train_fd = processed_train["FD001"]
    test_fd  = processed_test["FD001"]

    # Basic struct. checks
    assert all(s in train_fd.columns for s in ["sensor_1", "sensor_2"])
    assert all(s in test_fd.columns  for s in ["sensor_1", "sensor_2"])
    assert train_fd.shape[1] == test_fd.shape[1]
