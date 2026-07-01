import pandas as pd

from .base import UncertaintyModel


class MLUncertaintyModel(UncertaintyModel):
    def _fit_internal(self, X: pd.DataFrame, Y: pd.DataFrame) -> None:
        print("Training ML model...")

    def _predict_internal(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"forecast": [10.2], "uncertainty": [0.2]})
