"""
Multi-output linear regression model implementation for the fue package.

This module provides a concrete subclass that fits a standard multi-output
linear regression to estimate the absolute prediction deviations of target
meteorological variables.
"""

import logging

import pandas as pd
from sklearn.linear_model import LinearRegression

from .base import UncertaintyModel

# Initialize the module-level logger
logger = logging.getLogger(__name__)


class LinearUncertaintyModel(UncertaintyModel):
    """
    A baseline linear regression model used to estimate forecast error bounds.

    This model inherits from the base UncertaintyModel. It uses scikit-learn's
    standard linear regression to find straight-line trends between our preprocessed
    weather features and historical forecast error scales.
    """

    def _fit_internal(self, X: pd.DataFrame, Y: pd.DataFrame) -> None:
        """
        Fits the scikit-learn multi-output linear model on the provided data matrices.

        Parameters
        ----------
        X : pd.DataFrame
            The cleaned and scaled design matrix containing feature values.
        Y : pd.DataFrame
            The target data frame containing true historical absolute deviations.

        Returns
        -------
        None
        """

        logger.info("Training linear regression model to predict absolute deviations...")
        logger.debug("Linear regression fit initiated with X shape: %s and Y shape: %s.", X.shape, Y.shape)

        # Instantiate and fit the scikit-learn multi-output linear model
        self.model = LinearRegression()
        self.model.fit(X, Y)

        logger.debug("Linear regression model fitting completed.")

    def _predict_internal(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generates error scale predictions and maps them to a formatted data frame.

        Calculates straight-line trends from the input features and crops any
        resulting negative predictions at zero to prevent physically impossible
        uncertainty ranges.

        Parameters
        ----------
        X : pd.DataFrame
            The design matrix holding preprocessed inputs for our prediction tracking.

        Returns
        -------
        pd.DataFrame
            A data frame containing the predicted absolute deviations, with all values
            guaranteed to be 0 or higher.

        Raises
        ------
        ValueError
            If you try to run predictions before the model has been trained.
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
