from abc import ABC, abstractmethod
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fue.config import Config

from .preprocessor import Preprocessor

# Initialize the module-level logger using lazy formatting conventions
logger = logging.getLogger(__name__)


class UncertaintyModel(ABC):
    """Abstract Base Class orchestrating data preprocessing, pipeline coordination,
    and inverse metric mapping for multi-target forecast uncertainty estimation models.
    """

    def __init__(self, config: Config | None = None):
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

    def fit(self, df: pd.DataFrame, feature_columns: list, target_columns: list) -> "UncertaintyModel":
        """Fits preprocessor metadata, maps dynamic layout adjustments,
        and trains concrete multi-target subclasses.
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
        logger.debug("Mapped raw features to processed feature column layout. Final count: %s.", len(self.processed_feature_columns))

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
        """Applies learned feature transformations and automatically rescales predictions."""
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
        """Evaluates the model's prediction accuracy against a validation dataset.

        Returns a dictionary containing MAE and RMSE for each target variable.
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
        """Studies model validation error trajectories across increasing dataset sizes."""
        logger.info("Starting data convergence study over fractional sample increments.")
        history = {target: {"MAE": [], "RMSE": [], "sizes": []} for target in target_columns}

        if increments is None:
            increments = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
            logger.debug("No fractional increments provided. Applying default schedule: %s.", increments)

        for fraction in increments:
            slice_size = int(len(train_df) * fraction)
            if slice_size < 5:  # Skip trivial small slices
                logger.warning("Convergence training slice critically small (size %s < 5). Skipping increment.", slice_size)
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
        """Generates clear diagnostic validation plots tracking convergence against sample size."""
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

    @abstractmethod
    def _fit_internal(self, X: pd.DataFrame, Y: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def _predict_internal(self, X: pd.DataFrame) -> pd.DataFrame:
        pass
