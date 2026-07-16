from unittest.mock import MagicMock

import pandas as pd
import pytest

from pyfue.tuner import HyperparameterTuner


# ---------------------------------------------------------
# Pytest Fixtures (Synthetic Data)
# ---------------------------------------------------------
@pytest.fixture
def dummy_data():
    """Generates a trivial dataset to satisfy pandas DataFrame requirements."""
    df = pd.DataFrame(
        {
            "feature_1": [1, 2, 3, 4, 5],
            "feature_2": [5, 4, 3, 2, 1],
            "abs_diff__target_A": [0.1, 0.2, 0.1, 0.3, 0.2],
            "abs_diff__target_B": [0.5, 0.4, 0.5, 0.6, 0.5],
        }
    )
    features = ["feature_1", "feature_2"]
    targets = ["abs_diff__target_A", "abs_diff__target_B"]
    return df, df.copy(), features, targets  # train_df, val_df, features, targets


# ---------------------------------------------------------
# Test Cases
# ---------------------------------------------------------
class TestHyperparameterTuner:
    """Test suite for the ephemeral in-memory HPO Tuner."""

    def test_compute_objective_global(self):
        """Tests that the global target strategy correctly averages the chosen metric."""
        tuner = HyperparameterTuner(
            model_class=MagicMock, param_grid={}, optimization_target="global", optimization_metric="RMSE"
        )

        # Mock evaluation output from an UncertaintyModel
        mock_metrics = {
            "abs_diff__target_A": {"MAE": 0.2, "RMSE": 0.4},
            "abs_diff__target_B": {"MAE": 0.3, "RMSE": 0.6},
        }

        score = tuner._compute_objective(mock_metrics)
        # Global RMSE should be the average of 0.4 and 0.6
        assert score == 0.5

    def test_compute_objective_specific(self):
        """Tests that the tuner can isolate and optimize a single target variable."""
        tuner = HyperparameterTuner(
            model_class=MagicMock, param_grid={}, optimization_target="abs_diff__target_B", optimization_metric="MAE"
        )

        mock_metrics = {
            "abs_diff__target_A": {"MAE": 0.2, "RMSE": 0.4},
            "abs_diff__target_B": {"MAE": 0.9, "RMSE": 1.2},
        }

        score = tuner._compute_objective(mock_metrics)
        # Should cleanly extract only target_B's MAE
        assert score == 0.9

    def test_search_execution_and_persistence(self, dummy_data):
        """Tests that the grid search loops correctly and only saves the absolute best model."""
        train_df, val_df, features, targets = dummy_data

        # 1. Define a tiny 3-iteration grid
        param_grid = {"alpha": [0.1, 0.01, 0.001]}

        # 2. Create a factory function to generate Mock models
        # We want the second model (alpha=0.01) to have the best (lowest) score
        mock_model_instances = []

        def mock_model_factory(**kwargs):
            mock_instance = MagicMock()
            mock_instance.kwargs = kwargs

            # Program the fake evaluation behavior
            if kwargs["alpha"] == 0.1:
                fake_rmse = 1.5  # Worst
            elif kwargs["alpha"] == 0.01:
                fake_rmse = 0.5  # Best!
            else:
                fake_rmse = 1.0  # Middle

            mock_instance.evaluate.return_value = {"abs_diff__dummy": {"RMSE": fake_rmse}}
            mock_model_instances.append(mock_instance)
            return mock_instance

        # 3. Initialize Tuner
        tuner = HyperparameterTuner(
            model_class=mock_model_factory,  # ty: ignore [invalid-argument-type]
            param_grid=param_grid,
            optimization_target="global",
        )

        # 4. Execute Search
        best_model, history = tuner.search(train_df, val_df, features, targets, run_id="test_hpo_run")

        # 5. Assertions
        assert len(history) == 3, "Grid search should execute exactly 3 times."

        # Ensure it actually picked the model with alpha=0.01 as the champion
        assert best_model.kwargs["alpha"] == 0.01  # ty: ignore [unresolved-attribute]

        # Verify exactly ONE model was saved (the ephemeral checkpointing strategy)
        mock_model_instances[0].save.assert_not_called()
        mock_model_instances[1].save.assert_called_once()  # The champion!
        mock_model_instances[2].save.assert_not_called()

        # Verify the history payload stored the scores correctly
        assert history[1]["score"] == 0.5
