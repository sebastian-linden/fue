from abc import ABC, abstractmethod

import pandas as pd

from fue.config import Config

from .preprocessor import Preprocessor


class UncertaintyModel(ABC):
    """Abstract Base Class orchestrating data preprocessing, pipeline coordination,
    and inverse metric mapping for multi-target forecast uncertainty estimation models.
    """

    def __init__(self, config: Config | None = None):
        if config is None:
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
        self.raw_feature_columns = feature_columns
        # The target columns are pre-fixed with "abs_diff__" indicating that they refer
        # to the absolute difference between the forecast and the actual observation.
        for col in target_columns:
            if not col.startswith("abs_diff__"):
                raise ValueError(
                    f"Target column '{col}' must be prefixed with 'abs_diff__' to indicate absolute difference."
                )
        self.target_columns = target_columns

        # 1. Statefully evaluate data and output completed arrays
        self.preprocessor.fit(df)
        scaled_df = self.preprocessor.transform(df)

        # 2. ASK the preprocessor what the true final feature column layout is
        self.processed_feature_columns = self.preprocessor.map_feature_names(self.raw_feature_columns)

        # 3. Segregate clean feature arrays and multi-target vectors
        self.X = scaled_df[self.processed_feature_columns]
        self.Y = scaled_df[self.target_columns]

        # 4. Pass matrices down to subclass calculations
        self._fit_internal(self.X, self.Y)

        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies learned feature transformations and automatically rescales predictions."""
        if self.processed_feature_columns is None or self.target_columns is None:
            raise RuntimeError("Model pipeline must be statefully `.fit()` before generating inferences.")

        # 1. Transform inference features identically
        scaled_df = self.preprocessor.transform(df)
        X = scaled_df[self.processed_feature_columns]

        # 2. Get raw transformed model output arrays from subclass
        raw_predictions_df = self._predict_internal(X)

        # 3. Symmetrically iterate through target variables and back-scale them to real world units
        inverted_predictions = pd.DataFrame(index=raw_predictions_df.index)
        for col in raw_predictions_df.columns:
            # We map back matching rules using the base target column string names
            # (Subclasses must name output metrics to match target_columns strings)
            base_target_name = next((t for t in self.target_columns if t in col), col)
            inverted_predictions[col] = self.preprocessor.inverse_transform_target(
                raw_predictions_df[col], target_name=base_target_name
            )

        return inverted_predictions

    @abstractmethod
    def _fit_internal(self, X: pd.DataFrame, Y: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def _predict_internal(self, X: pd.DataFrame) -> pd.DataFrame:
        pass
