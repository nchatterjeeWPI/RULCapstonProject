import pytest
from unittest.mock import patch, MagicMock
from cmapss_rul import pipeline

def test_load_datasets_returns_expected_structure():
    """Verify load_datasets() returns train, test, and RUL datasets."""
    with patch("pandas.read_csv") as mock_read:
        mock_read.return_value = MagicMock()
        train, test, rul = pipeline.load_datasets("dummy_path", ["FD001"])
        assert train is not None
        assert test is not None
        assert rul is not None
