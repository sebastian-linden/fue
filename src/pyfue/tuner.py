"""
Hyperparameter optimization (HPO) engine for the pyfue package.

This module provides grid-search capabilities to sweep through various model
configurations, evaluate performance on out-of-sample validation data, and
identify the optimal configuration for uncertainty estimation.
"""

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import ParameterGrid

from .config import Config
from .models import MLUncertaintyModel, UncertaintyModel
from .utils import generate_run_id

# Initialize module-scoped logger
logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """
    Automates the process of finding the best settings for our uncertainty models.

    This class loops through a grid of possible hyperparameter values, runs training
    and evaluation iterations for each setting combination, collapses multi-target
    error scores down to a single optimization value, and selects the champion model.
    """

    def __init__(
        self,
        model_class: type[UncertaintyModel],
        param_grid: dict[str, list],
        optimization_target: str = "global",
        optimization_metric: str = "RMSE",
    ):
        """
        Initializes the tuner with a target model type, parameters grid, and scoring metrics.

        Parameters
        ----------
        model_class : type[UncertaintyModel]
            The uninstantiated class type of the model you want to tune (for example, MLUncertaintyModel).
        param_grid : dict of (str, list)
            A dictionary where keys match the model's setup argument names and values are
            lists of settings options to try.
        optimization_target : str, default="global"
            The target column to optimize for. If set to 'global', the tuner averages scores
            across all weather variables.
        optimization_metric : str, default="RMSE"
            The name of the evaluation error metric to track (for example, 'RMSE' or 'MAE').
        """
        self.model_class = model_class
        self.param_grid = param_grid
        self.optimization_target = optimization_target
        self.optimization_metric = optimization_metric

    def _compute_objective(self, metrics_dict: dict[str, dict[str, float]]) -> float:
        """
        Flattens multi-target error scores into a single number for tracking comparisons.

        If configured for global optimization, it averages the chosen metric across all
        tracked columns. Otherwise, it extracts the score for a single priority target variable.

        Parameters
        ----------
        metrics_dict : dict
            A nested dictionary of evaluation results returned by the model class.

        Returns
        -------
        float
            The calculated scalar summary score used to evaluate this parameter combination.
        """
        if self.optimization_target == "global":
            # Extract the specific metric (e.g., RMSE) for every target variable and average it
            scores = [target_metrics[self.optimization_metric] for target_metrics in metrics_dict.values()]
            return float(np.mean(scores))
        else:
            # Route directly to the user-specified priority target
            return metrics_dict[self.optimization_target][self.optimization_metric]

    def search(
        self,
        config: Config,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_columns: list,
        target_columns: list,
        run_id: str | None = None,
    ) -> tuple[MLUncertaintyModel, list[dict[str, Any]]]:
        """
        Runs the grid search sweep across all setting combinations to find the best model.

        Generates all hyperparameter combinations, trains a model candidate on the training pool,
        evaluates performance against validation records, and prints an updated leaderboard
        status to the console. The overall best model is then automatically saved to disk.

        Parameters
        ----------
        train_df : pd.DataFrame
            The data frame holding historical training records.
        val_df : pd.DataFrame
            The out-of-sample data frame used for validation testing.
        feature_columns : list
            List of column names used as model inputs.
        target_columns : list
            List of absolute error column names the model aims to predict.
        run_id : str or None, default=None
            A custom directory name for storing the results. If None, an ID is generated
            automatically using the project's timestamp utilities.

        Returns
        -------
        tuple of (UncertaintyModel, list of dict)
            A two-element tuple containing:
            - The optimal trained model instance (the champion).
            - A list of dictionary records mapping out the parameter states and scoring
              history for every iteration.
        """
        grid = list(ParameterGrid(self.param_grid))
        total_runs = len(grid)

        if run_id is None:
            run_id = generate_run_id(purpose="hpo")
            logger.debug("No explicit run_id provided for HPO. Auto-generated identifier: %s", run_id)

        logger.info("Starting HPO sweep. Total configurations to evaluate: %d", total_runs)

        best_score = float("inf")
        best_model = None
        best_metrics = None
        history = []

        # Mute convergence warnings only during the tuning process
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)

            for i, params in enumerate(grid):
                # 1. Instantiate the model dynamically with the current grid slice
                model = self.model_class(config=config, **params)

                # 2. Train on the 80% pool
                model.fit(train_df, feature_columns, target_columns)

                # 3. Evaluate on the 20% unseen pool
                metrics = model.evaluate(val_df)

                # 4. Collapse metrics to a single optimization scalar
                current_score = self._compute_objective(metrics)

                # 5. Track leaderboard logic
                is_best = False
                if current_score < best_score:
                    best_score = current_score
                    best_model = model
                    best_metrics = metrics
                    is_best = True

                # 6. One-Liner State Logging
                # Output: "Iter 4/48 | Score (RMSE): 0.8421 | Best: 0.8101 | Params: {'alpha': 0.01, ...}"
                mark = "⭐" if is_best else "  "
                logger.info(
                    "[%s] Iter %d/%d | Score (%s): %.4f | Best: %.4f | Params: %s",
                    mark,
                    i + 1,
                    total_runs,
                    self.optimization_metric,
                    current_score,
                    best_score,
                    params,
                )

                # Store iteration results strictly in memory
                history.append({"iteration": i + 1, "params": params, "score": current_score, "metrics": metrics})

        # 7. Persist the Champion Model to disk
        logger.info("HPO complete. Writing champion model to disk under run_id: '%s'.", run_id)

        # Package the HPO search metadata alongside the standard validation metrics
        tracking_metadata = {
            "hpo_winning_score": best_score,
            "hpo_optimization_metric": self.optimization_metric,
            "hpo_optimization_target": self.optimization_target,
            "validation_metrics": best_metrics,
        }

        # We leverage Phase 1's save logic!
        best_model.save(run_id=run_id, metrics=tracking_metadata)  # ty: ignore [unresolved-attribute]

        return best_model, history  # ty: ignore [invalid-return-type]
