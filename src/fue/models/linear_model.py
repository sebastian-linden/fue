import pandas as pd
from sklearn.linear_model import LinearRegression

from .base import UncertaintyModel


class LinearUncertaintyModel(UncertaintyModel):
    """Concrete subclass implementing a Multi-Output Linear Regression model

    to directly predict the absolute deviations (real-world uncertainty scale)
    of numerical weather forecasts.
    """

    def _fit_internal(self, X: pd.DataFrame, Y: pd.DataFrame) -> None:
        """Fits the underlying linear regression model directly to the

        pre-computed absolute deviation targets.
        """
        print("Training linear regression model to predict absolute deviations...")

        # Instantiate and fit the scikit-learn multi-output linear model
        self.model = LinearRegression()
        self.model.fit(X, Y)

    def _predict_internal(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generates point predictions representing the expected physical error

        magnitude for each weather variable.
        """
        if not hasattr(self, "model"):
            raise ValueError("Model must be trained via .fit() before making predictions.")

        print("Generating predictions for absolute deviations using the trained linear model...")

        # Generate the primary point predictions (which represent the uncertainty scale)
        raw_predictions = self.model.predict(X)

        # Build the output DataFrame mapped exactly to your target columns
        output_df = pd.DataFrame(raw_predictions, columns=self.target_columns, index=X.index)

        return output_df.clip(lower=0)
