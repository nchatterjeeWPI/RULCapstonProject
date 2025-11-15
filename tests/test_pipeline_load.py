from types import SimpleNamespace
from unittest.mock import patch, MagicMock
from cmapss_rul import pipeline


def test_load_datasets_returns_expected_structure():
    fake_paths = SimpleNamespace(user_data_dir="dummy_path")

    fake_train = {"FD001": MagicMock()}
    fake_test = {"FD001": MagicMock()}
    fake_rul = {"FD001": MagicMock()}

    with patch("cmapss_rul.pipeline.load.load_all",
               return_value=(fake_train, fake_test, fake_rul)) as mock_load_all:

        train, test, rul = pipeline.load_datasets(fake_paths, ["FD001"])

        mock_load_all.assert_called_once_with("dummy_path", ["FD001"])
        assert train is fake_train
        assert test is fake_test
        assert rul is fake_rul
