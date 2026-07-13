"""
Abstract base class definition for models within the fue package.

This module defines the skeleton and shared workflows for all uncertainty
estimators. It manages feature preprocessing, coordinate scaling, prediction mapping,
and validation tools like model training convergence checks.
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fue.config import Config
from fue.utils import generate_run_id

from .preprocessor import Preprocessor

# Initialize the module-level logger using lazy formatting conventions
logger = logging.getLogger(__name__)


class UncertaintyModel(ABC):
    """
    Sets up the baseline variables, configuration properties, and preprocessor hooks.

    Parameters
    ----------
    config : Config or None, default=None
        The global settings mapping object. If None is provided, a fresh
        internal configuration default mapping is created.
    """

    def __init__(self, config: Config | None = None) -> None:
        """
        Prepares features statefully and triggers the submodel training calculations.

        Verifies that your target outputs include the explicit absolute difference prefix,
        learns and applies adjustments using the data preprocessor, builds input
        design data frames, and forwards the cleaned states down to your chosen submodel.

        Parameters
        ----------
        df : pd.DataFrame
            The input weather training lines pooled from local source storage.
        feature_columns : list of str
            The explicit names of rows used as features for the models.
        target_columns : list of str
            The calculated forecast deviation columns we want to estimate (must begin with 'abs_diff__').

        Returns
        -------
        UncertaintyModel
            The current class instance with trained tracking weights.
        """
        if config is None:
            logger.debug("No configuration provided; initializing default Config object.")
            self.config = Config()
        else:
            self.config = config
        self.preprocessor = Preprocessor(self.config.get_preprocessing_rules())
        self.raw_feature_columns = None
        self.processed_feature_columns = None
        self.target_columns = None
        self.X = None
        self.Y = None
        self._is_fitted = False

    def fit(self, df: pd.DataFrame, feature_columns: list, target_columns: list) -> "UncertaintyModel":
        """
        Prepares features statefully and triggers the submodel training calculations.

        Verifies that your target outputs include the explicit absolute difference prefix,
        learns and applies adjustments using the data preprocessor, builds input
        design data frames, and forwards the cleaned states down to your chosen submodel.

        Parameters
        ----------
        df : pd.DataFrame
            The input weather training lines pooled from local source storage.
        feature_columns : list of str
            The explicit names of rows used as features for the models.
        target_columns : list of str
            The calculated forecast deviation columns we want to estimate (must begin with 'abs_diff__').

        Returns
        -------
        UncertaintyModel
            The current class instance with trained tracking weights.
        """
        logger.info("Initializing uncertainty model fitting sequence.")
        logger.debug("Input DataFrame shape: %s. Raw feature column count: %s.", df.shape, len(feature_columns))

        self.raw_feature_columns = feature_columns
        # The target columns are pre-fixed with "abs_diff__" indicating that they refer
        # to the absolute difference between the forecast and the actual observation.
        for col in target_columns:
            if not col.startswith("abs_diff__"):
                logger.error("Validation failed: target column '%s' missing required 'abs_diff__' prefix.", col)
                raise ValueError(
                    f"Target column '{col}' must be prefixed with 'abs_diff__' to indicate absolute difference."
                )
        self.target_columns = target_columns

        # 1. Statefully evaluate data and output completed arrays
        logger.info("Executing preprocessing pipeline (fit and transform).")
        self.preprocessor.fit(df)
        scaled_df = self.preprocessor.transform(df)

        # 2. ASK the preprocessor what the true final feature column layout is
        self.processed_feature_columns = self.preprocessor.map_feature_names(self.raw_feature_columns)
        logger.debug(
            "Mapped raw features to processed feature column layout. Final count: %s.",
            len(self.processed_feature_columns),
        )

        # 3. Segregate clean feature arrays and multi-target vectors
        self.X = scaled_df[self.processed_feature_columns]
        self.Y = scaled_df[self.target_columns]
        logger.debug("Segregated matrices created. X shape: %s, Y shape: %s.", self.X.shape, self.Y.shape)

        # 4. Pass matrices down to subclass calculations
        logger.info("Passing matrices down to concrete subclass mathematical operations.")
        self._fit_internal(self.X, self.Y)

        logger.info("Uncertainty model fitting sequence successfully completed.")
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates error magnitude inferences and transforms variables back to normal units.

        Applies the training scaling rules to incoming test metrics, requests the raw
        estimated targets from your submodel calculation layer, and performs exponential
        re-scaling to restore real-world weather units.

        Parameters
        ----------
        df : pd.DataFrame
            Fresh forecast records requiring uncertainty predictions.

        Returns
        -------
        pd.DataFrame
            A clean data frame holding the final estimated absolute error bounds.
        """
        if self.processed_feature_columns is None or self.target_columns is None:
            logger.error("Prediction sequence aborted: model pipeline lacks stateful fit.")
            raise RuntimeError("Model pipeline must be statefully `.fit()` before generating inferences.")

        logger.info("Initiating inference pipeline.")
        logger.debug("Inference DataFrame dimensions: %s.", df.shape)

        # 1. Transform inference features identically
        scaled_df = self.preprocessor.transform(df)
        X = scaled_df[self.processed_feature_columns]

        # 2. Get raw transformed model output arrays from subclass
        raw_predictions_df = self._predict_internal(X)
        logger.debug("Generated raw internal predictions. Output shape: %s.", raw_predictions_df.shape)

        # 3. Symmetrically iterate through target variables and back-scale them to real world units
        inverted_predictions = pd.DataFrame(index=raw_predictions_df.index)
        for col in raw_predictions_df.columns:
            # We map back matching rules using the base target column string names
            # (Subclasses must name output metrics to match target_columns strings)
            base_target_name = next((t for t in self.target_columns if t in col), col)
            inverted_predictions[col] = self.preprocessor.inverse_transform_target(
                raw_predictions_df[col], target_name=base_target_name
            )
            logger.debug("Applied inverse data transformation for prediction target: %s.", base_target_name)

        logger.info("Inference pipeline successfully completed.")
        return inverted_predictions

    def evaluate(self, df_val: pd.DataFrame) -> dict:
        """
        Validates model accuracy against standalone validation data rows.

        Generates real-world error predictions on a test frame and evaluates
        accuracy scores, saving Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)
        per column parameter.

        Parameters
        ----------
        df_val : pd.DataFrame
            The out-of-sample data frame slice used for verification.

        Returns
        -------
        dict
            A summary dictionary mapping target column titles directly to nested
            MAE and RMSE float validation values.
        """
        if self.target_columns is None:
            logger.error("Evaluation sequence aborted: model lacks target column state.")
            raise RuntimeError("Model must be statefully `.fit()` before running evaluation.")

        logger.info("Executing model validation evaluation.")

        # Generate real-world inverted unit predictions
        predictions_df = self.predict(df_val)

        metrics = {}
        for col in self.target_columns:
            actual = df_val[col].astype(float)
            predicted = predictions_df[col].astype(float)

            mae = np.mean(np.abs(actual - predicted))
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))

            metrics[col] = {"MAE": mae, "RMSE": rmse}
            logger.debug("Computed validation metrics for target %s - MAE: %s, RMSE: %s.", col, mae, rmse)

        logger.info("Model validation completed across %s target(s).", len(self.target_columns))
        return metrics

    def study_data_convergence(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_columns: list,
        target_columns: list,
        increments: list | None = None,
    ) -> dict:
        """
        Evaluates model skill improvements over expanding sample size slices.

        Repeatedly trains the model on increasing fractions of your data pool
        and evaluates skill against validation rows. This records how error metrics
        evolve as the model receives more training rows.

        Parameters
        ----------
        train_df : pd.DataFrame
            The clean training data footprint.
        val_df : pd.DataFrame
            The unseen validation split used for tracking convergence scores.
        feature_columns : list of str
            List of tracking elements to use as inputs.
        target_columns : list of str
            List of forecast difference vectors to calculate.
        increments : list of float or None, default=None
            A series of sample size percentages to check (for example: `[0.1, 0.5, 1.0]`).
            If None, applies the default schedule.

        Returns
        -------
        dict
            A dictionary structure tracking MAE, RMSE, and row counts over every iteration.
        """
        logger.info("Starting data convergence study over fractional sample increments.")
        history = {target: {"MAE": [], "RMSE": [], "sizes": []} for target in target_columns}

        if increments is None:
            increments = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
            logger.debug("No fractional increments provided. Applying default schedule: %s.", increments)

        for fraction in increments:
            slice_size = int(len(train_df) * fraction)
            if slice_size < 5:  # Skip trivial small slices
                logger.warning(
                    "Convergence training slice critically small (size %s < 5). Skipping increment.", slice_size
                )
                continue

            logger.info("Evaluating convergence data slice: %s samples (fraction %s).", slice_size, fraction)
            train_slice = train_df.iloc[:slice_size]

            # Reset pipeline states and fit on the subset slice
            self.fit(train_slice, feature_columns, target_columns)
            scores = self.evaluate(val_df)

            for target in target_columns:
                history[target]["MAE"].append(scores[target]["MAE"])
                history[target]["RMSE"].append(scores[target]["RMSE"])
                history[target]["sizes"].append(slice_size)

        logger.info("Data convergence study successfully concluded.")
        return history

    def plot_learning_curve(self, convergence_history: dict, metric: str = "MAE") -> None:
        """
        Draws a diagnostic learning curve plot using your convergence data results.

        Plots validation error rates against training row counts on a log-log axis system.
        This graph helps confirm if your model scales effectively with additional rows.

        Parameters
        ----------
        convergence_history : dict
            The multi-target validation summary dictionary generated by your tuner class layers.
        metric : str, default="MAE"
            The specific score variable line you want to visualize ('MAE' or 'RMSE').

        Returns
        -------
        None
        """
        if metric not in ["MAE", "RMSE"]:
            logger.error("Plot generation aborted: incompatible metric variant '%s'.", metric)
            raise ValueError("Metric variant specification must be either 'MAE' or 'RMSE'.")

        logger.info("Compiling learning curve plot for diagnostic metric: %s.", metric)
        plt.figure(figsize=(10, 6))

        for target, data in convergence_history.items():
            sizes = data["sizes"]
            scores = data[metric]
            clean_label = target.replace("abs_diff__", "").replace("_", " ").title()

            logger.debug("Plotting %s data points for target label: %s.", len(sizes), clean_label)
            plt.plot(sizes, scores, marker="o", linewidth=2, label=f"{clean_label} ({metric})")

        plt.title(f"Model Convergence Analysis (Validation {metric} vs. Training Samples)")
        plt.xlabel("Number of Processed Training Data Points")
        plt.ylabel(f"Validation Set Error Vector ({metric})")
        plt.xscale("log")
        plt.yscale("log")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="upper right")
        plt.tight_layout()

        logger.info("Learning curve visualization rendered and dispatched to graphical backend.")
        plt.show()

    def save(self, run_id: str | None = None, metrics: dict | None = None, runs_dir: str = "runs") -> tuple[Path, str]:
        """
        Serializes the active model instance state and tracks metrics inside a run folder.

        Dumps the entire pipeline structure via joblib and exports an auxiliary `meta.json`
        manifest capturing timing attributes, target names, feature sets, and scoring matrices.

        Parameters
        ----------
        run_id : str or None, default=None
            A custom experiment tracking label token name. If None, resolves a token
            dynamically via generation hooks.
        metrics : dict or None, default=None
            Leaderboard error values to save inside the metadata file sheet.
        runs_dir : str, default="runs"
            The base directory location name on your filesystem.

        Returns
        -------
        tuple of (pathlib.Path, str)
            A two-element tuple containing:
            - The full absolute path pointing to the created run folder on disk.
            - The tracking run string token name.
        """
        # 1. Fallback to automatic timestamped naming if no ID is passed explicitly
        if not run_id:
            run_id = generate_run_id()
            logger.debug("No explicit run_id provided. Auto-generated identifier: %s", run_id)

        run_path = Path(runs_dir) / run_id

        try:
            # 2. Synchronize directory generation
            run_path.mkdir(parents=True, exist_ok=True)
            logger.info("Tracking experiment artifacts in run directory: %s", run_path)

            # 3. Dump the complete stateful wrapper object via joblib
            model_file = run_path / "model.joblib"
            logger.debug("Serializing complete wrapper instance structure via joblib...")
            joblib.dump(self, model_file)
            logger.info("Successfully serialized model state to %s", model_file.name)

            # 4. Process tracking metrics to a structured json file
            if metrics is not None:
                meta_file = run_path / "meta.json"
                metadata = {
                    "run_id": run_id,
                    "timestamp": datetime.now().isoformat(),
                    "model_type": self.__class__.__name__,
                    "features_used": self.processed_feature_columns,
                    "targets_tracked": self.target_columns,
                    "metrics": metrics,
                }

                logger.debug("Writing validation metric norms to tracking manifest...")
                with open(meta_file, "w") as f:
                    json.dump(metadata, f, indent=4)
                logger.info("Successfully documented validation performance run sheets in %s", meta_file.name)

            return run_path, run_id

        except Exception as e:
            logger.error("Failed to commit run directory tracking for run_id %s. Trace error: %s", run_id, e)
            raise

    @classmethod
    def load(cls, run_id: str, runs_dir: str = "runs") -> "UncertaintyModel":
        """
        Restores a trained estimator model from a persistent binary checkpoint file.

        Climbs structural parent folders to anchor an absolute look-up reference
        and reconstructs the joblib model state safely. This bypasses folder breaks
        caused by moving between notebook spaces and terminal folders.

        Parameters
        ----------
        run_id : str
            The tracking string folder label name to restore.
        runs_dir : str, default="runs"
            The repository destination folder name where experiment assets are stored.

        Returns
        -------
        UncertaintyModel
            The restored instance, ready to generate forward inferences immediately.
        """
        # 1. Establish an absolute anchor path back to your project root folder
        # (src/fue/models/base.py is 3 levels deep from the package root directory)
        package_src_dir = Path(__file__).resolve().parents[3]

        # 2. Build absolute paths to avoid context breaks in Jupyter Notebooks
        model_file = package_src_dir / runs_dir / run_id / "model.joblib"

        # Fallback check for hidden folder structures if written by an alternative tuner layer
        if not model_file.exists():
            alternative_path = package_src_dir / ".fue" / "runs" / run_id / "model.joblib"
            if alternative_path.exists():
                model_file = alternative_path

        logger.info("Attempting to restore model checkpoint state for run tracking key: %s", run_id)
        logger.debug("Target file lookup path resolved to: %s", model_file)

        if not model_file.exists():
            error_msg = f"Reconstruction failed. No binary checkpoint file found at location: {model_file}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            loaded_model = joblib.load(model_file)
            logger.info(
                "Restoration complete. Successfully loaded instance type: %s (Fitted state: %s)",
                loaded_model.__class__.__name__,
                getattr(loaded_model, "_is_fitted", "Unknown"),
            )
            return loaded_model

        except Exception as e:
            logger.error("Binary deserialization error encountered during joblib parsing of %s: %s", model_file, e)
            raise

    @abstractmethod
    def _fit_internal(self, X: pd.DataFrame, Y: pd.DataFrame) -> None:
        """
        Subclass calculation hook to fit the underlying model algorithm parameters.
        """
        pass

    @abstractmethod
    def _predict_internal(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Subclass calculation hook to compute point predictions from input features.
        """
        pass
