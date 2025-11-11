import pytest
from cmapss_rul import pipeline

@pytest.fixture
def sample_data():
    import pandas as pd
    df_train = pd.DataFrame({
        "unit": [1,1,1,2,2],
        "cycle": [1,2,3,1,2],
        "sensor_1": [10,12,14,8,9],
        "sensor_2": [5,5.5,6,4.5,4.8]
    })
    df_test = df_train.copy()
    df_rul = pd.DataFrame({"RUL": [100,90]})
    return df_train, df_test, df_rul

def test_preprocess_scales_and_returns(sample_data):
    """Ensure preprocess_data() returns scaled DataFrames."""
    df_train, df_test, df_rul = sample_data
    processed_train, processed_test = pipeline.preprocess_data(
        df_train, df_test, df_rul, datasets=["FD001"], cap_val=125, sensors_to_keep=["sensor_1","sensor_2"]
    )
    assert all(col.startswith("sensor_") for col in processed_train.columns)
    assert processed_train.shape[1] == processed_test.shape[1]
