import logging
import pandas as pd
from sklearn.linear_model import LinearRegression

from .base import UncertaintyModel

# Initialize the module-level logger
logger = logging.getLogger(__name__)


class LinearUncertaintyModel(UncertaintyModel):
    """Concrete subclass implementing a Multi-Output Linear Regression model

    to directly predict the absolute deviations (real-world uncertainty scale)
    of numerical weather forecasts.
    """

    def _fit_internal(self, X: pd.DataFrame, Y: pd.DataFrame) -> None:
        """Fits the underlying linear regression model directly to the

        pre-computed absolute deviation targets.
        """
        logger.info("Training linear regression model to predict absolute deviations...")
        logger.debug("Linear regression fit initiated with X shape: %s and Y shape: %s.", X.shape, Y.shape)

        # Instantiate and fit the scikit-learn multi-output linear model
        self.model = LinearRegression()
        self.model.fit(X, Y)
        
        logger.debug("Linear regression model fitting completed.")

    def _predict_internal(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generates point predictions representing the expected physical error

        magnitude for each weather variable.
        """
        if not hasattr(self, "model"):
            logger.error("Prediction sequence aborted: internal linear model is not trained.")
            raise ValueError("Model must be trained via .fit() before making predictions.")

        logger.info("Generating predictions for absolute deviations using the trained linear model...")
        logger.debug("Linear model prediction initiated for input matrix shape: %s.", X.shape)

        # Generate the primary point predictions (which represent the uncertainty scale)
        raw_predictions = self.model.predict(X)

        # Build the output DataFrame mapped exactly to your target columns
        output_df = pd.DataFrame(raw_predictions, columns=self.target_columns, index=X.index)

        logger.debug("Applying lower bound clipping (0) to linear model output DataFrame.")

        return output_df.clip(lower=0)