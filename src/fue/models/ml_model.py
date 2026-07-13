"""
Ensemble Multi-Layer Perceptron (MLP) network model implementation for the fue package.

This module provides a concrete subclass that fits an ensemble of neural network
regressors to estimate non-linear absolute deviations of forecast weather parameters
while minimizing random initialization variance.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

from .base import UncertaintyModel

# Initialize module-scoped logger
logger = logging.getLogger(__name__)


class MLUncertaintyModel(UncertaintyModel):
    """
    A machine learning neural network ensemble used to estimate forecast error bounds.

    This model inherits from the base UncertaintyModel. Instead of relying on a single
    neural network, it trains an ensemble of multiple Multi-Layer Perceptrons (MLPs).
    Averaging predictions across this group helps smooth out random variations caused
    by network weight initialization, delivering reliable, non-linear error ceiling estimates.
    """

    def __init__(
        self,
        config=None,
        hidden_layer_sizes: tuple = (32, 16),
        max_iter: int = 500,
        alpha: float = 0.01,
        ensemble_size: int = 5,
        seed: int = 42,
    ):
        """
        Initializes the neural network ensemble model with custom training parameters.

        Parameters
        ----------
        config : Config or None, default=None
            The global settings mapping object. If None, resolves a default pipeline setup.
        hidden_layer_sizes : tuple of int, default=(32, 16)
            The structural configuration mapping hidden layers and node counts.
        max_iter : int, default=500
            The maximum number of training epochs allowed during backpropagation sweeps.
        alpha : float, default=0.01
            L2 regularization penalty value to protect network weights from overfitting.
        ensemble_size : int, default=5
            The total number of individual neural network models to train inside the ensemble pool.
        seed : int, default=42
            The starting random seed used to generate unique, reproducible states for each member.
        """

        super().__init__(config=config)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.alpha = alpha
        self.ensemble_size = ensemble_size
        self.seed = seed

        self.models = []

        logger.debug(
            "Initialized MLUncertaintyModel: hidden_layers=%s, max_iter=%d, alpha=%s, ensemble_size=%d",
            hidden_layer_sizes,
            max_iter,
            alpha,
            ensemble_size,
        )

    def _fit_internal(self, X: pd.DataFrame, Y: pd.DataFrame) -> None:
        """
        Trains each Multi-Layer Perceptron member inside the model ensemble.

        Loops through the configured ensemble size, sets a unique random seed for each submodel,
        initializes scikit-learn MLP Regressors with early stopping to prevent overfitting,
        and statefully tracks the completed sub-models in memory.

        Parameters
        ----------
        X : pd.DataFrame
            The preprocessed and scaled design matrix containing feature values.
        Y : pd.DataFrame
            The target data frame containing true historical absolute deviations.

        Returns
        -------
        None
        """

        logger.info("Training an Ensemble of %d MLP Regressors...", self.ensemble_size)
        logger.debug("Training features shape: %s, targets shape: %s", X.shape, Y.shape)

        self.models = []

        for i in range(self.ensemble_size):
            member_seed = self.seed + i
            logger.debug("Training ensemble sub-model %d/%d with seed %d", i + 1, self.ensemble_size, member_seed)

            model = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation="relu",
                solver="adam",
                alpha=self.alpha,
                max_iter=self.max_iter,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=member_seed,
            )
            model.fit(X, Y)

            # Log individual sub-model convergence metrics if early stopping triggered
            logger.debug(
                "Sub-model %d converged after %d iterations. Final loss: %.4f", i + 1, model.n_iter_, model.loss_
            )
            self.models.append(model)

        logger.info("Successfully trained all %d models in the ensemble.", self.ensemble_size)

    def _predict_internal(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generates aggregated error scale predictions from all trained neural networks.

        Gathers individual predictions from every submodel in our ensemble pool, computes
        their arithmetic mean, maps the results to your target labels, and clips any final
        negative values to 0.0 to respect real-world physical boundaries.

        Parameters
        ----------
        X : pd.DataFrame
            The design matrix holding preprocessed inputs for our prediction tracking.

        Returns
        -------
        pd.DataFrame
            A data frame containing the averaged absolute deviation predictions, with all
            values guaranteed to be 0 or higher.

        Raises
        ------
        ValueError
            If you try to run predictions before the model ensemble has been trained.
        """

        if not self.models:
            error_msg = "Model ensemble must be statefully trained via .fit() before prediction."
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("Generating ensemble predictions for %d records...", len(X))

        # Collect predictions from all sub-models
        ensemble_predictions = []
        for i, model in enumerate(self.models):
            logger.debug("Computing inference path for ensemble sub-model %d/%d", i + 1, len(self.models))
            ensemble_predictions.append(model.predict(X))

        # Average predictions across the 3D array axis
        mean_predictions = np.mean(ensemble_predictions, axis=0)

        output_df = pd.DataFrame(mean_predictions, columns=self.target_columns, index=X.index)

        # Monitor structural adjustments (e.g., physical target lower bounding adjustments)
        clipped_values = (output_df < 0.0).sum().sum()
        if clipped_values > 0:
            logger.debug(
                "Enforced physical constraint boundary: Clipped %d negative predictions to 0.0", clipped_values
            )

        return output_df.clip(lower=0.0)
