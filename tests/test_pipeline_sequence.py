import numpy as np
from cmapss_rul import pipeline

def test_sequence_generation_creates_consistent_shapes():
    """Validate sequence_generation() returns expected tensor shapes."""
    df = {"train": None, "val": None, "test": None}
    results = pipeline.sequence_generation(
        df, df, df, datasets=["FD001"],
        sensor_cols=["sensor_1","sensor_2"], setting_cols=[],
        seq_length=30, K=3
    )
    assert isinstance(results, dict)
    assert "train" in results
    assert "test" in results
