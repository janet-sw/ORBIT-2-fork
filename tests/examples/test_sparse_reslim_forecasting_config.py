from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_MODULE_PATH = (
    REPO_ROOT / "examples" / "sparse_reslim_forecasting" / "config.py"
)
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "sparse_reslim_forecasting.yaml"

SPEC = importlib.util.spec_from_file_location("forecast_config", CONFIG_MODULE_PATH)
forecast_config = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = forecast_config
SPEC.loader.exec_module(forecast_config)


def test_load_default_forecasting_config():
    config = forecast_config.load_forecast_config(DEFAULT_CONFIG_PATH)

    assert config.dataset_name == "ERA5_1"
    assert config.input_vars == ("2m_temperature",)
    assert config.output_vars == ("2m_temperature",)
    assert config.history == 1
    assert config.window == 1
    assert config.pred_range == 120
    assert config.compression["compress_ratio"] == 1
    assert config.keep_ratio == 0.25


def test_rejects_invalid_forecast_history(tmp_path):
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    invalid = deepcopy(raw)
    invalid["data"]["history"]["ERA5_1"] = 0
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(invalid), encoding="utf-8")

    with pytest.raises(forecast_config.ConfigError, match="data.history"):
        forecast_config.load_forecast_config(path)


def test_rejects_output_variable_missing_from_inputs(tmp_path):
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    invalid = deepcopy(raw)
    invalid["data"]["dict_out_variables"]["ERA5_1"] = ["temperature_850"]
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(invalid), encoding="utf-8")

    with pytest.raises(forecast_config.ConfigError, match="residual path"):
        forecast_config.load_forecast_config(path)
