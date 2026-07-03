import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

from .base import UncertaintyModel


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

    def _fit_internal(self, X: pd.DataFrame, Y: pd.DataFrame) -> None:
        """Trains multiple independent MLPs with distinct initializations."""
        print(f"Training an Ensemble of {self.ensemble_size} MLP Regressors...")
        self.models = []

        for i in range(self.ensemble_size):
            # Generate a unique, deterministic seed for each ensemble member
            member_seed = self.seed + i

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
            self.models.append(model)

        print(f"Successfully trained all {self.ensemble_size} models in the ensemble.")

    def _predict_internal(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generates simultaneous estimates by averaging predictions across the ensemble."""
        if not self.models:
            raise ValueError("Model ensemble must be statefully trained via .fit() before prediction.")

        # Collect predictions from all sub-models
        ensemble_predictions = []
        for model in self.models:
            ensemble_predictions.append(model.predict(X))

        # Average predictions across the 3D array axis
        mean_predictions = np.mean(ensemble_predictions, axis=0)

        output_df = pd.DataFrame(mean_predictions, columns=self.target_columns, index=X.index)
        return output_df.clip(lower=0.0)
