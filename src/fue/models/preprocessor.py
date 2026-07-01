import numpy as np
import pandas as pd
from scipy.stats import boxcox
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Preprocessor:
    """Stateful preprocessor that learns transformations on training data,
    handles cyclical feature expansions, and maps tracking column names dynamically.
    """

    def __init__(self, rules: dict):
        """
        Args:
            rules: Dict mapping column names to transformation strings.
                   e.g., {"wind_direction_10m_dominant": "sin-cos"}
        """
        self.rules = rules
        self.scalers = {}
        self.boxcox_lambdas = {}

    def map_feature_names(self, raw_features: list) -> list:
        """Transforms a list of raw feature names into the names of the columns
        that will exist after calling `.transform()`.

        Args:
            raw_features: List of original string column names.

        Returns:
            list: A new list with tracking adjustments (e.g., sin-cos splits).
        """
        processed_features = []
        for col in raw_features:
            method = self.rules.get(col)
            if method == "sin-cos":
                # Expand single name into the dual geometric vectors
                processed_features.extend([f"{col}_sin", f"{col}_cos"])
            else:
                # All other transformations stay in-place to preserve name tracking
                processed_features.append(col)
        return processed_features

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        """Learn scaling parameters from the training dataframe layout."""
        for col, method in self.rules.items():
            if col not in df.columns:
                continue

            series_data = df[col].to_numpy().reshape(-1, 1)

            if method == "standard":
                scaler = StandardScaler()
                scaler.fit(series_data)
                self.scalers[col] = scaler

            elif method == "min-max":
                scaler = MinMaxScaler()
                scaler.fit(series_data)
                self.scalers[col] = scaler

            elif method == "box-cox":
                min_val = df[col].min()
                shift = max(0, -min_val) + 1e-5 if min_val <= 0 else 0
                shifted_data = df[col] + shift
                _, lmbda = boxcox(shifted_data)
                self.boxcox_lambdas[col] = {"lambda": lmbda, "shift": shift}

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply transformations and structural column expansions. Returns a new DataFrame."""
        df_out = df.copy()

        for col, method in self.rules.items():
            if col not in df_out.columns:
                continue

            if method == "standard" or method == "min-max":
                scaler = self.scalers.get(col)
                if scaler is None:
                    raise RuntimeError(f"Preprocessor must be fitted before transforming: {col}")
                arr = df_out[col].to_numpy().reshape(-1, 1)
                df_out[col] = scaler.transform(arr).flatten()

            elif method == "log":
                df_out[col] = np.log1p(df_out[col] + 1e-1)

            elif method == "box-cox":
                params = self.boxcox_lambdas.get(col)
                if params is None:
                    raise RuntimeError(f"Preprocessor must be fitted before transforming: {col}")
                shifted_data = df_out[col] + params["shift"]
                df_out[col] = boxcox(shifted_data, lmbda=params["lambda"])

            elif method == "sin-cos":
                # Convert circular degree vectors to stable coordinates
                radians = np.radians(df_out[col])
                df_out[f"{col}_sin"] = np.sin(radians)
                df_out[f"{col}_cos"] = np.cos(radians)
                # Drop original raw degrees column so models don't ingest it duplicated
                df_out.drop(columns=[col], inplace=True)

        return df_out

    def inverse_transform_target(self, series: pd.Series, target_name: str) -> pd.Series:
        """Reverses the transformation for predictions or uncertainty metrics back to real units."""
        arr = series.to_numpy()
        method = self.rules.get(target_name)

        if not method or method == "sin-cos":
            return series  # sin-cos is lossy/periodic and shouldn't ever be a target model output

        if method == "standard" or method == "min-max":
            scaler = self.scalers.get(target_name)
            if scaler is None:
                raise RuntimeError(f"Preprocessor not fitted for target column: {target_name}")
            arr_rescaled = scaler.inverse_transform(arr.reshape(-1, 1))
            return pd.Series(arr_rescaled.flatten(), index=series.index, name=series.name)

        elif method == "log":
            return pd.Series(np.expm1(arr) - 1e-1, index=series.index, name=series.name)

        elif method == "box-cox":
            params = self.boxcox_lambdas.get(target_name)
            if params is None:
                raise RuntimeError(f"Preprocessor not fitted for target column: {target_name}")
            lmbda = params["lambda"]
            shift = params["shift"]

            if lmbda == 0:
                inv_box = np.exp(arr)
            else:
                inv_box = np.power(arr * lmbda + 1.0, 1.0 / lmbda)

            return pd.Series(inv_box - shift, index=series.index, name=series.name)

        return series
