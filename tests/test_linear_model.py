import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from pyfue.config import Config
from pyfue.models.linear_model import LinearUncertaintyModel


@pytest.fixture
def test_config(tmp_path):
    """Provides a valid Config object for model testing."""
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
                "preprocessing": {},
                "default_feature_columns": ["temperature_2m_max"],
                "default_target_columns": ["abs_diff__temperature_2m_max"],
            }
        )
    )
    return Config(path=config_file)


@pytest.fixture
def linear_model(test_config):
    """Fixture providing a LinearUncertaintyModel instance initialized with config."""
    return LinearUncertaintyModel(config=test_config)


class TestLinearUncertaintyModel:
    """Test suite for LinearUncertaintyModel and its base UncertaintyModel logic."""

    @patch("pyfue.models.base.Preprocessor")
    def test_init_with_custom_config(self, MockPreprocessor):
        """Test initialization when a custom config object is explicitly provided."""
        custom_config = MagicMock(spec=Config)
        custom_config.get_preprocessing_rules.return_value = {"custom": "rule"}

        model = LinearUncertaintyModel(config=custom_config)

        MockPreprocessor.assert_called_once_with({"custom": "rule"})
        assert model.config == custom_config

    def test_fit_target_prefix_validation(self, linear_model):
        """Test that fit raises a ValueError if target columns lack the required prefix."""
        dummy_df = pd.DataFrame({"feature_1": [1.0, 2.0], "target_1": [0.5, 0.8]})

        with pytest.raises(ValueError, match="must be prefixed with 'abs_diff__'"):
            linear_model.fit(df=dummy_df, feature_columns=["feature_1"], target_columns=["target_1"])

    @patch("pyfue.models.base.Preprocessor")
    def test_fit_success(self, MockPreprocessor, test_config):
        """Test the successful execution of the fit pipeline and internal model training."""
        model = LinearUncertaintyModel(config=test_config)

        mock_preproc_instance = MockPreprocessor.return_value
        model.preprocessor = mock_preproc_instance

        mock_transformed_df = pd.DataFrame({"scaled_feature_1": [0.1, 0.5, 0.9], "abs_diff__target_1": [1.2, 0.4, 2.1]})
        mock_preproc_instance.transform.return_value = mock_transformed_df
        mock_preproc_instance.map_feature_names.return_value = ["scaled_feature_1"]

        raw_df = pd.DataFrame({"raw_feature_1": [10, 50, 90], "abs_diff__target_1": [1.2, 0.4, 2.1]})

        returned_model = model.fit(df=raw_df, feature_columns=["raw_feature_1"], target_columns=["abs_diff__target_1"])

        mock_preproc_instance.fit.assert_called_once_with(raw_df)
        mock_preproc_instance.transform.assert_called_once_with(raw_df)

        assert hasattr(model, "model")
        assert returned_model is model

    def test_predict_unfitted_base_state(self, linear_model):
        """Test that predict raises a RuntimeError if the base class attributes are uninitialized."""
        dummy_df = pd.DataFrame({"raw_feature": [1.0]})

        with pytest.raises(
            RuntimeError, match=r"Model pipeline must be statefully `\.fit\(\)` before generating inferences\."
        ):
            linear_model.predict(dummy_df)

    def test_predict_missing_internal_model(self, linear_model):
        """Test that predict raises a ValueError if the concrete internal model was not trained."""
        linear_model.processed_feature_columns = ["scaled_feature"]
        linear_model.target_columns = ["abs_diff__target"]

        linear_model.preprocessor = MagicMock()
        linear_model.preprocessor.transform.return_value = pd.DataFrame({"scaled_feature": [1.0]})

        dummy_df = pd.DataFrame({"raw_feature": [1.0]})

        with pytest.raises(ValueError, match="Model must be trained via .fit\\(\\) before making predictions."):
            linear_model.predict(dummy_df)

    @patch("pyfue.models.base.Preprocessor")
    def test_predict_success(self, MockPreprocessor, test_config):
        """Test the successful execution of the predict pipeline, including clipping and inverse transformation."""
        model = LinearUncertaintyModel(config=test_config)
        model.processed_feature_columns = ["scaled_feature"]
        model.target_columns = ["abs_diff__temperature_2m"]

        mock_preproc_instance = MockPreprocessor.return_value
        model.preprocessor = mock_preproc_instance
        mock_preproc_instance.transform.return_value = pd.DataFrame({"scaled_feature": [0.5, -0.5]})

        def mock_inverse(series, target_name):
            return series * 10.0

        mock_preproc_instance.inverse_transform_target.side_effect = mock_inverse

        model.model = MagicMock()
        model.model.predict.return_value = [[2.0], [-1.0]]

        input_df = pd.DataFrame({"raw_feature": [10.0, 20.0]})
        result_df = model.predict(input_df)

        assert result_df.shape == (2, 1)
        assert result_df["abs_diff__temperature_2m"].iloc[0] == 20.0
        assert result_df["abs_diff__temperature_2m"].iloc[1] == 0.0

    def test_evaluate_unfitted_model(self, linear_model):
        """Test that evaluating an unfitted model raises a RuntimeError."""
        dummy_df = pd.DataFrame({"dummy_feature": [1.0]})

        with pytest.raises(RuntimeError, match=r"Model must be statefully `\.fit\(\)` before running evaluation\."):
            linear_model.evaluate(dummy_df)

    @patch.object(LinearUncertaintyModel, "predict")
    def test_evaluate_success(self, mock_predict, linear_model):
        """Test that evaluate computes correct MAE and RMSE using mocked predictions."""
        linear_model.target_columns = ["abs_diff__temperature_2m", "abs_diff__wind_speed_10m"]

        df_val = pd.DataFrame(
            {"abs_diff__temperature_2m": [2.0, 4.0, 6.0], "abs_diff__wind_speed_10m": [1.0, 3.0, 5.0]}
        )

        mock_predictions_df = pd.DataFrame(
            {
                "abs_diff__temperature_2m": [1.0, 4.0, 5.0],
                "abs_diff__wind_speed_10m": [2.0, 3.0, 7.0],
            }
        )
        mock_predict.return_value = mock_predictions_df

        metrics = linear_model.evaluate(df_val)

        assert metrics["abs_diff__temperature_2m"]["MAE"] == pytest.approx(0.6666, abs=1e-4)
        assert metrics["abs_diff__wind_speed_10m"]["RMSE"] == pytest.approx(1.2909, abs=1e-4)

    def test_save_with_explicit_id(self, linear_model, tmp_path):
        """Test saving a model and its metrics using an explicit run_id in a temporary directory."""
        linear_model.raw_feature_columns = ["delta_days", "day_of_year"]
        linear_model.processed_feature_columns = ["delta_days_sqrt"]
        linear_model.target_columns = ["abs_diff__temperature_2m_max"]
        linear_model.config.runs_dir = str(tmp_path)

        explicit_id = "test_run_linear_01"
        metrics = {"abs_diff__temperature_2m_max": {"MAE": 0.45, "RMSE": 0.58}}

        run_path, run_id = linear_model.save(run_id=explicit_id, metrics=metrics)

        # Assertions updated to match joblib and meta.json file outputs
        assert run_id == explicit_id
        assert Path(run_path).exists()
        assert (Path(run_path) / "model.joblib").is_file()
        assert (Path(run_path) / "meta.json").is_file()

    def test_load_successful(self, test_config, tmp_path):
        """Test round-trip state reconstruction using base wrapper load classmethod."""
        test_config.runs_dir = str(tmp_path)
        original_model = LinearUncertaintyModel(config=test_config)
        original_model.target_columns = ["abs_diff__precipitation_sum"]
        original_model.processed_feature_columns = ["delta_days_sqrt"]

        run_id = "roundtrip_test_id"
        run_path, saved_id = original_model.save(run_id=run_id)

        # Pass both runs_dir and run_id to match load(runs_dir, run_id)
        loaded_model = LinearUncertaintyModel.load(runs_dir=tmp_path, run_id=saved_id)

        # Assertions to verify restored state match
        assert loaded_model.target_columns == original_model.target_columns
        assert loaded_model.processed_feature_columns == original_model.processed_feature_columns
