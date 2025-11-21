import sys
import pytest
import main as main_module  # main.py at project root


@pytest.mark.system
def test_full_pipeline_smoke(tmp_path, monkeypatch):
    out_dir = tmp_path / "system_test"

    argv = [
        "main.py",
        "--arch", "tcn",
        "--datasets", "FD001",
        "--epochs", "2",
        "--sequence-length", "30",
        "--tuning", "off",
        "--out", str(out_dir),
    ]

    monkeypatch.setattr(sys, "argv", argv)
    main_module.main()

    assert out_dir.exists()

    final_model_dir = out_dir / "final_model"
    assert final_model_dir.exists(), "final_model directory missing"

    tcn_model_path = final_model_dir / "tcn_final.keras"
    tcn_meta_path = final_model_dir / "tcn_final.meta.json"
    assert tcn_model_path.exists()
    assert tcn_meta_path.exists()

    model_dir = out_dir / "model"
    assert model_dir.exists(), "model directory missing"

    preds_csv = model_dir / "final_engine_rul_predictions_tcn.csv"
    assert preds_csv.exists()

    cov_csv = model_dir / "interval_coverage_summary_tcn.csv"
    assert cov_csv.exists()

    figures_dir = out_dir / "figures"
    assert figures_dir.exists()
    assert list(figures_dir.glob("*.svg"))
