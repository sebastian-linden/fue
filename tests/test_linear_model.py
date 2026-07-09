from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from fue.config import Config
from fue.models.linear_model import LinearUncertaintyModel


class TestLinearUncertaintyModel:
    """Test suite for LinearUncertaintyModel, which also tests base UncertaintyModel logic."""

    @patch("fue.models.base.Config")
    @patch("fue.models.base.Preprocessor")
    def test_init_default(self, MockPreprocessor, MockConfig):
        """Test initialization without passing a config (ensures default creation)."""
        # Setup the fake config behavior
        mock_config_instance = MockConfig.return_value
        mock_config_instance.get_preprocessing_rules.return_value = {"dummy": "rule"}

        # Execute
        model = LinearUncertaintyModel()

        # Assertions
        # 1. Check that default Config and Preprocessor were instantiated
        MockConfig.assert_called_once()
        MockPreprocessor.assert_called_once_with({"dummy": "rule"})

        # 2. Verify all attributes are properly assigned and initialized to None
        assert model.config == mock_config_instance
        assert model.preprocessor == MockPreprocessor.return_value
        assert model.raw_feature_columns is None
        assert model.processed_feature_columns is None
        assert model.target_columns is None
        assert model.X is None
        assert model.Y is None

    @patch("fue.models.base.Preprocessor")
    def test_init_with_custom_config(self, MockPreprocessor):
        """Test initialization when a custom config object is explicitly provided."""
        # Setup a custom mock config
        custom_config = MagicMock(spec=Config)
        custom_config.get_preprocessing_rules.return_value = {"custom": "rule"}

        # Execute by passing the custom config
        model = LinearUncertaintyModel(config=custom_config)

        # Assertions
        # 1. Check that the preprocessor used the custom config's rules
        MockPreprocessor.assert_called_once_with({"custom": "rule"})

        # 2. Ensure the model stored our custom config, not a new default one
        assert model.config == custom_config

    def test_fit_target_prefix_validation(self):
        """Test that fit raises a ValueError if target columns lack the required prefix."""
        model = LinearUncertaintyModel()
        dummy_df = pd.DataFrame({"feature_1": [1.0, 2.0], "target_1": [0.5, 0.8]})

        with pytest.raises(ValueError, match="must be prefixed with 'abs_diff__'"):
            model.fit(df=dummy_df, feature_columns=["feature_1"], target_columns=["target_1"])

    @patch("fue.models.base.Preprocessor")
    @patch("fue.models.base.Config")
    def test_fit_success(self, MockConfig, MockPreprocessor):
        """Test the successful execution of the fit pipeline and internal model training."""
        model = LinearUncertaintyModel()

        # Override the preprocessor with our mock instance
        mock_preproc_instance = MockPreprocessor.return_value
        model.preprocessor = mock_preproc_instance

        # Define the exact output the mock preprocessor should return when called
        mock_transformed_df = pd.DataFrame({"scaled_feature_1": [0.1, 0.5, 0.9], "abs_diff__target_1": [1.2, 0.4, 2.1]})
        mock_preproc_instance.transform.return_value = mock_transformed_df
        mock_preproc_instance.map_feature_names.return_value = ["scaled_feature_1"]

        # Define the raw input provided to the fit method
        raw_df = pd.DataFrame({"raw_feature_1": [10, 50, 90], "abs_diff__target_1": [1.2, 0.4, 2.1]})

        # Execute the method
        returned_model = model.fit(df=raw_df, feature_columns=["raw_feature_1"], target_columns=["abs_diff__target_1"])

        # Verify the preprocessor was called in the correct sequence with the right data
        mock_preproc_instance.fit.assert_called_once_with(raw_df)
        mock_preproc_instance.transform.assert_called_once_with(raw_df)
        mock_preproc_instance.map_feature_names.assert_called_once_with(["raw_feature_1"])

        # Verify the class state was updated correctly with the segregated matrices
        assert model.X.shape == (3, 1)  # ty: ignore[unresolved-attribute]
        assert model.Y.shape == (3, 1)  # ty: ignore[unresolved-attribute]
        assert "scaled_feature_1" in model.X.columns  # ty: ignore[unresolved-attribute]
        assert "abs_diff__target_1" in model.Y.columns  # ty: ignore[unresolved-attribute]

        # Verify that the internal linear regression model was instantiated and stored
        assert hasattr(model, "model")

        # Verify that the method returns the model instance itself
        assert returned_model is model

    def test_predict_unfitted_base_state(self):
        """Test that predict raises a RuntimeError if the base class attributes are uninitialized."""
        model = LinearUncertaintyModel()
        dummy_df = pd.DataFrame({"raw_feature": [1.0]})

        # Escaping the period and parentheses for the regex match
        with pytest.raises(
            RuntimeError, match=r"Model pipeline must be statefully `\.fit\(\)` before generating inferences\."
        ):
            model.predict(dummy_df)

    def test_predict_missing_internal_model(self):
        """Test that predict raises a ValueError if the concrete internal model was not trained."""
        model = LinearUncertaintyModel()

        # Manually satisfy the base class state requirements
        model.processed_feature_columns = ["scaled_feature"]
        model.target_columns = ["abs_diff__target"]

        # Override preprocessor to bypass base class transform errors
        model.preprocessor = MagicMock()
        model.preprocessor.transform.return_value = pd.DataFrame({"scaled_feature": [1.0]})

        dummy_df = pd.DataFrame({"raw_feature": [1.0]})

        with pytest.raises(ValueError, match="Model must be trained via .fit\\(\\) before making predictions."):
            model.predict(dummy_df)

    @patch("fue.models.base.Preprocessor")
    def test_predict_success(self, MockPreprocessor):
        """Test the successful execution of the predict pipeline, including clipping and inverse transformation."""
        model = LinearUncertaintyModel()

        # Manually satisfy the base class state requirements
        model.processed_feature_columns = ["scaled_feature"]
        model.target_columns = ["abs_diff__temperature_2m"]

        # Setup mock preprocessor to control the data flow
        mock_preproc_instance = MockPreprocessor.return_value
        model.preprocessor = mock_preproc_instance

        # Define the output of the transform method
        mock_preproc_instance.transform.return_value = pd.DataFrame({"scaled_feature": [0.5, -0.5]})

        # Define a mock inverse transformation function to trace the data
        def mock_inverse(series, target_name):
            return series * 10.0

        mock_preproc_instance.inverse_transform_target.side_effect = mock_inverse

        # Setup the internal scikit-learn model and its raw output
        model.model = MagicMock()
        # First row simulates a standard prediction, second row simulates a negative prediction requiring clipping
        model.model.predict.return_value = [[2.0], [-1.0]]

        # Input DataFrame
        input_df = pd.DataFrame({"raw_feature": [10.0, 20.0]})

        # Execute the method
        result_df = model.predict(input_df)

        # Verify the preprocessor was called correctly
        mock_preproc_instance.transform.assert_called_once_with(input_df)
        mock_preproc_instance.inverse_transform_target.assert_called_once()

        # Verify the dimensions of the output
        assert result_df.shape == (2, 1)
        assert "abs_diff__temperature_2m" in result_df.columns

        # Verify clipping and inverse transform mathematical results
        # Row 0: 2.0 is not clipped -> multiplied by 10.0 = 20.0
        assert result_df["abs_diff__temperature_2m"].iloc[0] == 20.0

        # Row 1: -1.0 is clipped to 0.0 -> multiplied by 10.0 = 0.0
        assert result_df["abs_diff__temperature_2m"].iloc[1] == 0.0

    def test_evaluate_unfitted_model(self):
        """Test that evaluating an unfitted model raises a RuntimeError."""
        model = LinearUncertaintyModel()
        dummy_df = pd.DataFrame({"dummy_feature": [1.0]})

        # Escaping the period and parentheses for the regex match
        with pytest.raises(RuntimeError, match=r"Model must be statefully `\.fit\(\)` before running evaluation\."):
            model.evaluate(dummy_df)

    @patch.object(LinearUncertaintyModel, "predict")
    def test_evaluate_success(self, mock_predict):
        """Test that evaluate computes correct MAE and RMSE using mocked predictions."""
        model = LinearUncertaintyModel()
        model.target_columns = ["abs_diff__temperature_2m", "abs_diff__wind_speed_10m"]

        # Define the validation dataset with the actual ground truth values
        df_val = pd.DataFrame(
            {"abs_diff__temperature_2m": [2.0, 4.0, 6.0], "abs_diff__wind_speed_10m": [1.0, 3.0, 5.0]}
        )

        # Define the mocked predictions returned by the model
        mock_predictions_df = pd.DataFrame(
            {
                "abs_diff__temperature_2m": [1.0, 4.0, 5.0],  # Errors: 1.0, 0.0, 1.0
                "abs_diff__wind_speed_10m": [2.0, 3.0, 7.0],  # Errors: -1.0, 0.0, -2.0
            }
        )
        mock_predict.return_value = mock_predictions_df

        # Execute the method
        metrics = model.evaluate(df_val)

        # Verify predict was called with the validation dataframe
        mock_predict.assert_called_once_with(df_val)

        # Assert structure of the returned dictionary
        assert "abs_diff__temperature_2m" in metrics
        assert "abs_diff__wind_speed_10m" in metrics

        # Verify the mathematical accuracy of the computed metrics
        # Temperature MAE: (1 + 0 + 1) / 3 = 0.666...
        # Temperature RMSE: sqrt((1 + 0 + 1) / 3) = 0.816...
        assert metrics["abs_diff__temperature_2m"]["MAE"] == pytest.approx(0.6666, abs=1e-4)
        assert metrics["abs_diff__temperature_2m"]["RMSE"] == pytest.approx(0.8164, abs=1e-4)

        # Wind Speed MAE: (1 + 0 + 2) / 3 = 1.0
        # Wind Speed RMSE: sqrt((1 + 0 + 4) / 3) = 1.2909...
        assert metrics["abs_diff__wind_speed_10m"]["MAE"] == pytest.approx(1.0, abs=1e-4)
        assert metrics["abs_diff__wind_speed_10m"]["RMSE"] == pytest.approx(1.2909, abs=1e-4)

    @patch.object(LinearUncertaintyModel, "evaluate")
    @patch.object(LinearUncertaintyModel, "fit")
    def test_study_data_convergence(self, mock_fit, mock_evaluate):
        """Test that convergence tracking slices data correctly, skips small fractions, and records metrics."""
        model = LinearUncertaintyModel()

        # Define a mock training dataset with exactly 100 rows to easily calculate slice sizes
        train_df = pd.DataFrame({"dummy_feature": range(100)})
        val_df = pd.DataFrame({"dummy_feature": range(20)})
        feature_cols = ["dummy_feature"]
        target_cols = ["abs_diff__dummy_target"]

        # Define test increments:
        # 0.04 * 100 = 4 rows (should be skipped due to < 5 condition)
        # 0.10 * 100 = 10 rows (should be evaluated)
        # 0.50 * 100 = 50 rows (should be evaluated)
        custom_increments = [0.04, 0.10, 0.50]

        # Define the mocked dictionary returned by evaluate()
        mock_evaluate.return_value = {"abs_diff__dummy_target": {"MAE": 0.25, "RMSE": 0.45}}

        # Execute the method
        history = model.study_data_convergence(
            train_df=train_df,
            val_df=val_df,
            feature_columns=feature_cols,
            target_columns=target_cols,
            increments=custom_increments,
        )

        # Verify that fit and evaluate were only called twice (skipping the 0.04 fraction)
        assert mock_fit.call_count == 2
        assert mock_evaluate.call_count == 2

        # Verify the structure and values of the history dictionary
        assert "abs_diff__dummy_target" in history
        target_history = history["abs_diff__dummy_target"]

        assert target_history["sizes"] == [10, 50]
        assert target_history["MAE"] == [0.25, 0.25]
        assert target_history["RMSE"] == [0.45, 0.45]

        # Verify the arguments passed to the fit method during its final call
        last_fit_args, last_fit_kwargs = mock_fit.call_args
        last_train_slice = last_fit_args[0]
        assert len(last_train_slice) == 50
        assert last_fit_args[1] == feature_cols
        assert last_fit_args[2] == target_cols

    def test_plot_learning_curve_invalid_metric(self):
        """Test that the plotting method firmly rejects unsupported metrics."""
        model = LinearUncertaintyModel()

        with pytest.raises(ValueError, match="Metric variant specification must be either 'MAE' or 'RMSE'."):
            # Pass an empty dict and an invalid metric name
            model.plot_learning_curve({}, metric="MAPE")

    @patch("fue.models.base.plt")
    def test_plot_learning_curve_success(self, mock_plt):
        """
        Test the successful orchestration of the matplotlib calls,
        verifying string formatting and ensuring the plot displays without freezing.
        """
        model = LinearUncertaintyModel()

        # Define a mock convergence history matching the expected structure
        mock_history = {
            "abs_diff__temperature_2m": {"sizes": [10, 50, 100], "MAE": [0.8, 0.5, 0.4]},
            "abs_diff__wind_speed_10m": {"sizes": [10, 50, 100], "MAE": [1.5, 1.1, 0.9]},
        }

        # Execute the method
        model.plot_learning_curve(mock_history, metric="MAE")

        # Assertions
        # 1. Verify the plot window was initialized and displayed safely
        mock_plt.figure.assert_called_once()
        mock_plt.show.assert_called_once()

        # 2. Verify that plt.plot was called exactly twice (once for each target in our mock dictionary)
        assert mock_plt.plot.call_count == 2

        # 3. Verify the string manipulation logic (title-casing and removing prefixes)
        # We intercept the arguments passed to the very first plt.plot() call
        first_plot_args, first_plot_kwargs = mock_plt.plot.call_args_list[0]

        # The expected lists of data
        assert first_plot_args[0] == [10, 50, 100]  # sizes
        assert first_plot_args[1] == [0.8, 0.5, 0.4]  # scores

        # The expected label: "abs_diff__temperature_2m" -> "temperature 2m" -> "Temperature 2M"
        assert first_plot_kwargs["label"] == "Temperature 2M (MAE)"

    def test_save_with_explicit_id(self, tmp_path):
        """Test saving a model and its metrics using an explicit run_id in a temporary directory."""
        model = LinearUncertaintyModel()

        # Initialize attributes exactly as they exist in your updated class layout
        model.raw_feature_columns = ["delta_days", "day_of_year"]
        model.processed_feature_columns = ["delta_days_sqrt", "sin_day_of_year", "cos_day_of_year"]
        model.target_columns = ["abs_diff__temperature_2m_max"]

        explicit_id = "test_run_linear_01"
        metrics = {"abs_diff__temperature_2m_max": {"MAE": 0.45, "RMSE": 0.58}}

        # Act: Execute save inside the safe temp sandbox
        run_path, run_id = model.save(run_id=explicit_id, metrics=metrics, runs_dir=str(tmp_path))  # ty: ignore [not-iterable]

        # Assertions
        assert run_path == tmp_path / explicit_id
        assert (run_path / "model.joblib").exists()
        assert (run_path / "meta.json").exists()

        # Verify the meta format fields match your internal instance dictionaries
        import json

        with open(run_path / "meta.json") as f:
            metadata = json.load(f)

        assert metadata["run_id"] == explicit_id
        assert metadata["model_type"] == "LinearUncertaintyModel"
        assert metadata["metrics"] == metrics
        assert metadata["features_used"] == model.processed_feature_columns

    def test_save_with_auto_generated_id(self, tmp_path):
        """Test that leaving run_id empty fallback generates a timestamped alternative."""
        model = LinearUncertaintyModel()
        model.raw_feature_columns = ["delta_days"]

        # Act: Pass run_id=None to trigger automatic identification flow
        run_path, run_id = model.save(run_id=None, metrics=None, runs_dir=str(tmp_path))  # ty: ignore [invalid-argument-type, not-iterable]

        # Assertions
        assert run_path.exists()
        assert (run_path / "model.joblib").exists()
        assert not (run_path / "meta.json").exists()

    def test_load_successful(self, tmp_path):
        """Test round-trip state reconstruction using base wrapper load classmethod."""
        original_model = LinearUncertaintyModel()
        original_model.target_columns = ["abs_diff__precipitation_sum"]
        original_model.processed_feature_columns = ["delta_days_sqrt"]

        run_id = "roundtrip_test_id"
        original_model.save(run_id=run_id, runs_dir=str(tmp_path))

        # Act: Call class method restoration pathway
        loaded_model = LinearUncertaintyModel.load(run_id=run_id, runs_dir=str(tmp_path))

        # Assertions
        assert loaded_model.__class__.__name__ == "LinearUncertaintyModel"
        assert loaded_model.target_columns == ["abs_diff__precipitation_sum"]
        assert loaded_model.processed_feature_columns == ["delta_days_sqrt"]
