import pandas as pd
from cmapss_rul import explore

def test_non_constant_sensors_filters_zero_variance():
    df = pd.DataFrame({
        "sensor_1": [1,1,1],
        "sensor_2": [1,2,3],
        "op_setting_1":[1,2,3]
    })
    sensors = explore.non_constant_sensors({"FD001": df})
    assert "sensor_2" in sensors
    assert "sensor_1" not in sensors
