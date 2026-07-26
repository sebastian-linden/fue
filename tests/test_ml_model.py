import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from pyfue import Config
from pyfue.models.ml_model import MLUncertaintyModel


@pytest.fixture
def test_config(tmp_path):
    """Fixture providing a Config object for OpenMeteoClient tests."""
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps(
            {
                "data_path": "data/forecasts.csv",
                "runs_dir": "runs",
                "cities": {"london": {"lat": 51.5, "lon": -0.12}},
                "daily": ["temperature_2m_max"],
                "timezone": "UTC",
                "past_days": 5,
                "forecast_days": 10,
            }
        )
    )
    return Config(path=config_file)


class TestMLUncertaintyModel:
    """Test suite for MLUncertaintyModel, focusing on ensemble logic."""

    def test_init(self, test_config):
        """Test that hyperparameters are correctly assigned during initialization."""
        model = MLUncertaintyModel(
            test_config, hidden_layer_sizes=(64, 32), max_iter=1000, alpha=0.05, ensemble_size=10, seed=123
        )

        assert model.hidden_layer_sizes == (64, 32)
        assert model.max_iter == 1000
        assert model.alpha == 0.05
        assert model.ensemble_size == 10
        assert model.seed == 123
        assert model.models == []

    @patch("pyfue.models.ml_model.MLPRegressor")
    def test_fit_internal(self, MockMLP, test_config):
        """Test that fit_internal initializes the correct number of models with distinct seeds."""
        model = MLUncertaintyModel(test_config, ensemble_size=3, seed=42)

        # Mock instance to track calls
        mock_instance = MockMLP.return_value

        X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        Y = pd.DataFrame({"target": [0.5, 0.6]})

        model._fit_internal(X, Y)

        # Check ensemble size
        assert len(model.models) == 3
        assert MockMLP.call_count == 3

        # Verify random states: should be seed + i (42, 43, 44)
        seeds = [call.kwargs["random_state"] for call in MockMLP.call_args_list]
        assert seeds == [42, 43, 44]

        # Verify fit was called on all instances
        assert mock_instance.fit.call_count == 3

    def test_predict_internal_unfitted_raises_error(self, test_config):
        """Test that predict_internal raises ValueError if no models are trained."""
        model = MLUncertaintyModel(test_config)
        X = pd.DataFrame({"a": [1, 2]})

        with pytest.raises(ValueError, match="Model ensemble must be statefully trained"):
            model._predict_internal(X)

    def test_predict_internal_averaging_and_clipping(self, test_config):
        """Test that predictions are correctly averaged across the ensemble and clipped."""
        model = MLUncertaintyModel(test_config, ensemble_size=2)
        model.target_columns = ["target"]

        # Mock two sub-models
        mock_model_1 = MagicMock()
        mock_model_2 = MagicMock()

        # Define outputs: Model 1 predicts [2.0, -2.0], Model 2 predicts [4.0, 2.0]
        # Average: [3.0, 0.0]. Clipped: [3.0, 0.0]
        mock_model_1.predict.return_value = np.array([[2.0], [-2.0]])
        mock_model_2.predict.return_value = np.array([[4.0], [2.0]])

        model.models = [mock_model_1, mock_model_2]

        X = pd.DataFrame({"a": [1, 2]}, index=[0, 1])
        result = model._predict_internal(X)

        # Verify averaging
        assert result.iloc[0, 0] == 3.0

        # Verify clipping of negative values
        assert result.iloc[1, 0] == 0.0

        # Check output structure
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == model.target_columns
