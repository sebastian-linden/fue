import logging
import numpy as np
import pandas as pd
from scipy.stats import boxcox
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Initialize the module-level logger
logger = logging.getLogger(__name__)

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
        logger.debug("Initialized Preprocessor with %s registered transformation rules.", len(self.rules))

    def map_feature_names(self, raw_features: list) -> list:
        """Transforms a list of raw feature names into the names of the columns
        that will exist after calling `.transform()`.

        Args:
            raw_features: List of original string column names.

        Returns:
            list: A new list with tracking adjustments (e.g., sin-cos splits).
        """
        logger.debug("Mapping feature names for %s raw features.", len(raw_features))
        processed_features = []
        
        for col in raw_features:
            method = self.rules.get(col)
            if method == "sin-cos":
                # Expand single name into the dual geometric vectors
                processed_features.extend([f"{col}_sin", f"{col}_cos"])
                logger.debug("Expanded cyclical feature '%s' into dual sin-cos vectors.", col)
            else:
                # All other transformations stay in-place to preserve name tracking
                processed_features.append(col)
                
        logger.debug("Feature mapping complete. Expanded dimension from %s to %s.", len(raw_features), len(processed_features))
        return processed_features

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        """Learn scaling parameters from the training dataframe layout."""
        logger.info("Executing preprocessor fit routine.")
        logger.debug("Fitting against DataFrame with shape: %s.", df.shape)

        for col, method in self.rules.items():
            if col not in df.columns:
                logger.warning("Target column '%s' defined in rules is missing from DataFrame. Skipping.", col)
                continue

            series_data = df[col].to_numpy().reshape(-1, 1)

            if method == "standard":
                scaler = StandardScaler()
                scaler.fit(series_data)
                self.scalers[col] = scaler
                logger.debug("Fitted StandardScaler state for feature: %s.", col)

            elif method == "min-max":
                scaler = MinMaxScaler()
                scaler.fit(series_data)
                self.scalers[col] = scaler
                logger.debug("Fitted MinMaxScaler state for feature: %s.", col)

            elif method == "box-cox":
                min_val = df[col].min()
                shift = max(0, -min_val) + 1e-5 if min_val <= 0 else 0
                shifted_data = df[col] + shift
                _, lmbda = boxcox(shifted_data)
                self.boxcox_lambdas[col] = {"lambda": lmbda, "shift": shift}
                logger.debug("Fitted Box-Cox parameters for feature: %s (lambda: %s, shift: %s).", col, lmbda, shift)

            elif method in ["log", "square", "sqrt", "sin-cos"]:
                logger.debug("Acknowledged stateless transformation '%s' for feature: %s.", method, col)
                pass  # These are stateless and don't need fitting, but are valid
            else:
                logger.error("Fit sequence aborted: unhandled transformation method '%s' for column '%s'.", method, col)
                raise ValueError(f"Unknown preprocessing method '{method}' specified for column '{col}'.")

        logger.info("Preprocessor fit routine completed successfully.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply transformations and structural column expansions. Returns a new DataFrame."""
        logger.info("Executing preprocessing transform routine.")
        logger.debug("Input DataFrame shape for transform: %s.", df.shape)
        
        df_out = df.copy()

        for col, method in self.rules.items():
            if col not in df_out.columns:
                logger.warning("Target column '%s' defined in rules is missing from DataFrame. Skipping transform.", col)
                continue
                
            logger.debug("Applying '%s' transformation to column '%s'.", method, col)

            if method == "standard" or method == "min-max":
                scaler = self.scalers.get(col)
                if scaler is None:
                    logger.error("Transform sequence aborted: Preprocessor lacks fitted scaler state for '%s'.", col)
                    raise RuntimeError(f"Preprocessor must be fitted before transforming: {col}")
                arr = df_out[col].to_numpy().reshape(-1, 1)
                df_out[col] = scaler.transform(arr).flatten()

            elif method == "log":
                df_out[col] = np.log1p(df_out[col] + 1e-1)

            elif method == "square":
                df_out[col] = np.square(df_out[col])

            elif method == "sqrt":
                # Ensure we handle negative values gracefully if they ever slip in
                df_out[col] = np.sqrt(np.maximum(df_out[col], 0))

            elif method == "box-cox":
                params = self.boxcox_lambdas.get(col)
                if params is None:
                    logger.error("Transform sequence aborted: Preprocessor lacks fitted Box-Cox parameters for '%s'.", col)
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
                logger.debug("Dropped original circular column '%s' after sin-cos expansion.", col)

            else:
                logger.error("Transform sequence aborted: unhandled transformation method '%s' for column '%s'.", method, col)
                raise ValueError(f"Unknown preprocessing method '{method}' specified for column '{col}'.")

        logger.info("Preprocessing transform routine completed successfully. Output shape: %s.", df_out.shape)
        return df_out

    def inverse_transform_target(self, series: pd.Series, target_name: str) -> pd.Series:
        """Reverses the transformation for predictions or uncertainty metrics back to real units."""
        arr = series.to_numpy()
        method = self.rules.get(target_name)
        
        logger.debug("Executing inverse transform for target '%s' using method '%s'.", target_name, method)

        if not method or method == "sin-cos":
            logger.debug("No valid inverse mathematical transformation required for target '%s'. Returning raw series.", target_name)
            return series  # sin-cos is lossy/periodic and shouldn't ever be a target model output

        if method == "standard" or method == "min-max":
            scaler = self.scalers.get(target_name)
            if scaler is None:
                logger.error("Inverse transform aborted: Preprocessor lacks fitted scaler state for target '%s'.", target_name)
                raise RuntimeError(f"Preprocessor not fitted for target column: {target_name}")
            arr_rescaled = scaler.inverse_transform(arr.reshape(-1, 1))
            return pd.Series(arr_rescaled.flatten(), index=series.index, name=series.name)

        elif method == "log":
            return pd.Series(np.expm1(arr) - 1e-1, index=series.index, name=series.name)

        elif method == "square":
            # The inverse of squaring is taking the square root
            return pd.Series(np.sqrt(np.maximum(arr, 0)), index=series.index, name=series.name)

        elif method == "sqrt":
            # The inverse of a square root is squaring
            return pd.Series(np.square(arr), index=series.index, name=series.name)

        elif method == "box-cox":
            params = self.boxcox_lambdas.get(target_name)
            if params is None:
                logger.error("Inverse transform aborted: Preprocessor lacks Box-Cox parameters for target '%s'.", target_name)
                raise RuntimeError(f"Preprocessor not fitted for target column: {target_name}")
            lmbda = params["lambda"]
            shift = params["shift"]

            if lmbda == 0:
                inv_box = np.exp(arr)
            else:
                inv_box = np.power(arr * lmbda + 1.0, 1.0 / lmbda)

            return pd.Series(inv_box - shift, index=series.index, name=series.name)

        else:
            logger.error("Inverse transform aborted: unhandled preprocessing method '%s' for target '%s'.", method, target_name)
            raise ValueError(f"Unknown preprocessing method '{method}' specified for target '{target_name}'.")

        return series
