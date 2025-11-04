import pytest
from unittest.mock import patch
from cmapss_rul import pipeline
from main import main

@patch("cmapss_rul.pipeline.parse_arguments")
@patch("cmapss_rul.pipeline.setup_and_download")
@patch("cmapss_rul.pipeline.load_datasets")
@patch("cmapss_rul.pipeline.train_models")
@patch("cmapss_rul.pipeline.test_and_evaluate")
@patch("cmapss_rul.pipeline.report_results")

def test_main_runs_without_errors(mock_report, mock_eval, mock_train, mock_load, mock_setup, mock_parse):
    """Ensures main() runs end-to-end without raising errors when dependencies are mocked."""
    mock_parse.return_value = ({"arg": "val"}, {"datasets": "FD001", "cap_val": 125, "val_size": 0.2,
                                                 "sequence_length": 30, "K": 3, "architectures": ["LSTM"],
                                                 "epochs": 1, "use_tuning": False, "run_sensor_analysis": False})
    mock_setup.return_value = ("paths", "output_dir")
    mock_load.return_value = ("train_data", "test_data", "rul_data")
    mock_train.return_value = {"LSTM": "trained_model"}
    mock_eval.return_value = {"LSTM": {"RMSE": 24.7}}
    mock_report.return_value = None

    # Execute
    main()

    # Verify that report_results was called
    mock_report.assert_called_once()
