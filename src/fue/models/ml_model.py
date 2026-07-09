import logging

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

from .base import UncertaintyModel

# Initialize module-scoped logger
logger = logging.getLogger(__name__)


class MLUncertaintyModel(UncertaintyModel):
    """Concrete subclass implementing an Ensemble of Multi-Layer Perceptrons (MLPs)
    to mitigate random initialization variance on small datasets.
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
        """Initializes the Ensemble ML model.

        Args:
            ensemble_size: Number of individual MLPs to train with different seeds.
            alpha: Increased default L2 regularization to aggressively penalize variance.
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
        """Trains multiple independent MLPs with distinct initializations."""
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
        """Generates simultaneous estimates by averaging predictions across the ensemble."""
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
