from cmapss_rul import config

def test_default_config_has_expected_arch():
    assert config.DEFAULT.arch in ["cnn", "lstm", "tcn"]
