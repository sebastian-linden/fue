"""
Data preprocessing and transformation utilities for the pyfue package.

This module provides stateful and stateless tools to prepare raw weather data
for machine learning models. It supports scaling (Standard/Min-Max), mathematical
adjustments (Log, Square Root, Box-Cox), and expanding cyclical columns into sine
and cosine components.
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import boxcox
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Initialize the module-level logger
logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Transforms data fields statefully before training and reverses them after predicting.

    This class reads transformation choices from a rules dictionary. It remembers
    parameters learned during training (like scaling ranges or Box-Cox lambda values)
    so it can process incoming data identically or translate model error outputs
    back into real-world units.
    """

    def __init__(self, rules: dict):
        """
        Sets up empty storage containers for scalers and lambda parameters.

        Parameters
        ----------
        rules : dict
            A dictionary mapping column name strings directly to their preferred
            transformation style keywords (for example: `{'precipitation_sum': 'log'}`).
        """
        self.rules = rules
        self.scalers = {}
        self.boxcox_lambdas = {}
        logger.debug("Initialized Preprocessor with %s registered transformation rules.", len(self.rules))

    def map_feature_names(self, raw_features: list) -> list:
        """
        Adjusts a list of column names to account for columns that expand during scaling.

        Loops through your feature names and checks if any are flagged for a 'sin-cos'
        split. If they are, it replaces that single name with twin '_sin' and '_cos'
        labels so the model's design matrix columns align correctly.

        Parameters
        ----------
        raw_features : list of str
            The starting column name strings before any scaling changes take place.

        Returns
        -------
        list of str
            An updated list of column names reflecting the final preprocessed table layout.
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

        logger.debug(
            "Feature mapping complete. Expanded dimension from %s to %s.", len(raw_features), len(processed_features)
        )
        return processed_features

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        """
        Learns scaling statistics and normalization shapes from a training data frame.

        Iterates through the registered columns list, extracts data series matrices,
        and calculates baseline parameters like means, variances, ranges, or Box-Cox shifts.
        Stateless operations (like a standard square root) are skipped but acknowledged.

        Parameters
        ----------
        df : pd.DataFrame
            The input training dataset used to calculate preprocessing limits.

        Returns
        -------
        Preprocessor
            The current class instance loaded with the newly learned parameters.

        Raises
        ------
        ValueError
            If an unrecognized transformation method keyword is encountered in your rules dictionary.
        """
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
        """
        Applies learned or stateless transformation rules to a data frame.

        Creates a copy of your table, processes configured columns according to their
        respective rules, and cleanly splits circular values (like degrees or angles)
        into standalone sine and cosine columns while dropping the raw original field.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset containing columns that need cleaning or reshaping.

        Returns
        -------
        pd.DataFrame
            A copy of the input data frame with all transformation adjustments applied.

        Raises
        ------
        RuntimeError
            If a stateful scaler or parameters block is requested before running `.fit()`.
        ValueError
            If a column uses an unknown transformation keyword string.
        """
        logger.info("Executing preprocessing transform routine.")
        logger.debug("Input DataFrame shape for transform: %s.", df.shape)

        df_out = df.copy()

        for col, method in self.rules.items():
            if col not in df_out.columns:
                logger.warning(
                    "Target column '%s' defined in rules is missing from DataFrame. Skipping transform.", col
                )
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
                    logger.error(
                        "Transform sequence aborted: Preprocessor lacks fitted Box-Cox parameters for '%s'.", col
                    )
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
                logger.error(
                    "Transform sequence aborted: unhandled transformation method '%s' for column '%s'.", method, col
                )
                raise ValueError(f"Unknown preprocessing method '{method}' specified for column '{col}'.")

        logger.info("Preprocessing transform routine completed successfully. Output shape: %s.", df_out.shape)
        return df_out

    def inverse_transform_target(self, series: pd.Series, target_name: str) -> pd.Series:
        """
        Reverses scaling adjustments on model outputs to restore real-world weather units.

        Takes raw numbers generated by an uncertainty estimator model and performs the
        symmetrical opposite operation (such as exponentiating a log, or squaring a
        square root) to turn abstract numbers back into normal physical units like °C or mm.

        Parameters
        ----------
        series : pd.Series
            The raw predicted error outputs coming from your trained model layer.
        target_name : str
            The dictionary lookup label representing the base column name before scaling.

        Returns
        -------
        pd.Series
            A fresh data series containing values translated back into real-world units.

        Raises
        ------
        RuntimeError
            If you try to reverse standard, min-max, or Box-Cox columns before the
            preprocessor has been statefully fitted.
        ValueError
            If the targeted variable maps back to an unsupported transformation rule.
        """
        arr = series.to_numpy()
        method = self.rules.get(target_name)

        logger.debug("Executing inverse transform for target '%s' using method '%s'.", target_name, method)

        if not method or method == "sin-cos":
            logger.debug(
                "No valid inverse mathematical transformation required for target '%s'. Returning raw series.",
                target_name,
            )
            return series  # sin-cos is lossy/periodic and shouldn't ever be a target model output

        if method == "standard" or method == "min-max":
            scaler = self.scalers.get(target_name)
            if scaler is None:
                logger.error(
                    "Inverse transform aborted: Preprocessor lacks fitted scaler state for target '%s'.", target_name
                )
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
                logger.error(
                    "Inverse transform aborted: Preprocessor lacks Box-Cox parameters for target '%s'.", target_name
                )
                raise RuntimeError(f"Preprocessor not fitted for target column: {target_name}")
            lmbda = params["lambda"]
            shift = params["shift"]

            if lmbda == 0:
                inv_box = np.exp(arr)
            else:
                inv_box = np.power(arr * lmbda + 1.0, 1.0 / lmbda)

            return pd.Series(inv_box - shift, index=series.index, name=series.name)

        else:
            logger.error(
                "Inverse transform aborted: unhandled preprocessing method '%s' for target '%s'.", method, target_name
            )
            raise ValueError(f"Unknown preprocessing method '{method}' specified for target '{target_name}'.")
