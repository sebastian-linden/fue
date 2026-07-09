import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import ParameterGrid

from .models import MLUncertaintyModel, UncertaintyModel
from .utils import generate_run_id

# Initialize module-scoped logger
logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """
    Orchestrates in-memory hyperparameter optimization for FUE Uncertainty Models.

    This tuner safely respects temporal data splits by taking pre-separated
    training and validation dataframes. It performs an exhaustive grid search
    in memory, tracks validation metrics for all configurations, and writes
    only the highest-performing model to disk.
    """

    def __init__(
        self,
        model_class: type[UncertaintyModel],
        param_grid: dict[str, list],
        optimization_target: str = "global",
        optimization_metric: str = "RMSE",
    ):
        """
        Initializes the tuner with a model class and a search space.

        Parameters
        ----------
        model_class : Type[UncertaintyModel]
            The uninstantiated class of the model to tune (e.g., MLUncertaintyModel).
        param_grid : Dict[str, list]
            Dictionary with parameters names (`str`) as keys and lists of parameter
            settings to try as values.
        optimization_target : str, optional
            The specific target column to optimize for (e.g., 'abs_diff__temperature_2m_max').
            If set to "global" (default), minimizes the average error across all targets.
        optimization_metric : str, optional
            The metric key to minimize. Usually "RMSE" (default) or "MAE".
        """
        self.model_class = model_class
        self.param_grid = param_grid
        self.optimization_target = optimization_target
        self.optimization_metric = optimization_metric

    def _compute_objective(self, metrics_dict: dict[str, dict[str, float]]) -> float:
        """
        Collapses a multi-target validation metrics dictionary into a single float scalar.

        Parameters
        ----------
        metrics_dict : Dict[str, Dict[str, float]]
            The nested dictionary returned by the `evaluate()` method.

        Returns
        -------
        float
            The scalar value to be minimized.
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
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_columns: list,
        target_columns: list,
        run_id: str | None = None,
    ) -> tuple[MLUncertaintyModel, list[dict[str, Any]]]:
        """
        Executes the grid search loop, tracking configurations in memory.

        Parameters
        ----------
        train_df : pd.DataFrame
            The chronologically split training dataset.
        val_df : pd.DataFrame
            The chronologically split validation dataset.
        feature_columns : list
            List of raw feature column names to feed to the model.
        target_columns : list
            List of absolute difference target column names to predict.
        save_run_id : str, optional
            The tracker key under which the champion model will be persisted.

        Returns
        -------
        Tuple[UncertaintyModel, List[Dict[str, Any]]]
            A tuple containing the fitted champion model object and the complete
            history of all evaluated configurations and their scores.
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
                model = self.model_class(**params)

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
