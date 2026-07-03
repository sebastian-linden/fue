import os
from pathlib import Path

import pandas as pd

from .openmeteoclient import OpenMeteoClient


class Data:
    """This class manages all of the data related operations."""

    def __init__(self):

        self.PATH_TO_DATA = Path(__file__).resolve().parent.parent.parent / "data"
        self.PATH_TO_RAW = os.path.join(self.PATH_TO_DATA, "raw", "forecasts.csv")

        # This is the data as fetched from the open-meteo API.
        self.raw = pd.DataFrame()

        self.meta_variables = ["location_name", "latitude", "longitude", "forecasted_on", "forecast_for"]
        self.weather_variables = []
        self.numeric_variables = []
        return None

    def read_raw(self, path=None):
        """This method reads the raw data available in the directory:
        data/raw/
        """
        if path is not None:
            self.PATH_TO_RAW = path
        try:
            self.raw = pd.read_csv(self.PATH_TO_RAW)
            self.raw = self.convert_to_best_dtypes(self.raw)
            self.weather_variables = [c for c in self.raw.columns if c not in self.meta_variables]
            self.numeric_variables = self.weather_variables + ["latitude", "longitude"]

        except Exception as e:
            raise FileNotFoundError("forecasts.csv was not found") from e
        return None

    def convert_to_best_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """_summary_

        Args:
            df (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """

        datetime_cols = ["forecasted_on", "forecast_for"]
        for col in df.columns:
            if col == "location_name":
                df[col] = df[col].astype(str)
            elif col in datetime_cols:
                df[col] = pd.to_datetime(df[col])
            else:
                df[col] = pd.to_numeric(df[col], errors="raise")
        # Ingest units safely for both training matrices and live client responses
        if "sunshine_duration" in df.columns:
            df["sunshine_duration"] = df["sunshine_duration"] / 3600.0
        return df

    def generate_dataset(self, location_name="aachen", val_fraction=0.2, random_state=42):
        """Filters raw forecasting data by city, splits entries into look-ahead forecasts
        and ground-truth measurements proxies using a custom 12-hour boundary window,
        pairs them together on calendar dates to compute target absolute error metrics,
        and optionally outputs reproducible training and validation splits.

        Args:
            location_name (str, optional): The name of the geographic city filter
                defined in the configuration file. Defaults to "aachen".
            val_fraction (float, optional): The proportional fraction of the processed
                paired dataset to split off into a separate validation subset.
                Must be between 0.0 and 1.0. Defaults to 0.2.
            random_state (int, optional): Fixed seed value used to control the random
                shuffling permutations before splitting, ensuring reproducible datasets.
                Defaults to 42.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: A two-element tuple containing:
                - train_df: The primary DataFrame used to train uncertainty models.
                - val_df: The validation subset DataFrame (empty if val_fraction=0.0).
        """

        # 1. Separate data by city
        city_raw = self.raw[self.raw["location_name"] == location_name].copy()

        # Enforce explicit datetime tracking
        city_raw["forecast_for"] = pd.to_datetime(city_raw["forecast_for"])
        city_raw["forecasted_on"] = pd.to_datetime(city_raw["forecasted_on"])

        # 2. Separate forecasts from measurements using your custom 12-hour boundary rule
        delta_seconds = (city_raw["forecast_for"] - city_raw["forecasted_on"]).dt.total_seconds()
        forecast_split = city_raw[delta_seconds > 12 * 60 * 60].copy()
        measured_split = city_raw[delta_seconds <= 12 * 60 * 60].copy()

        # 3. Inject a normalized date matching key to bypass intraday timestamp mismatches
        forecast_split["_match_date"] = forecast_split["forecast_for"].dt.normalize()
        measured_split["_match_date"] = measured_split["forecast_for"].dt.normalize()

        # Deduplicate measurements to ensure 1 unique ground truth row per calendar day
        measured_split = measured_split.drop_duplicates(subset=["_match_date"])

        # Isolate the validation ground truth targets and rename to prevent column collisions
        measured_split = measured_split[["_match_date"] + self.weather_variables].rename(
            columns={var: f"true_{var}" for var in self.weather_variables}
        )

        # 4. Vectorized Inner Join on the normalized calendar day
        dataset = pd.merge(forecast_split, measured_split, on="_match_date", how="inner")

        # 5. Extract engineered features and apply standard unit scaling
        dataset["day_of_year"] = dataset["forecast_for"].dt.day_of_year
        dataset["delta_days"] = (dataset["forecast_for"] - dataset["forecasted_on"]).dt.total_seconds() / (60 * 60 * 24)

        # 6. Calculate continuous absolute delta error matrices
        abs_diff_cols = []
        for var in self.weather_variables:
            diff_col = f"abs_diff__{var}"
            dataset[diff_col] = (dataset[var] - dataset[f"true_{var}"]).abs()
            abs_diff_cols.append(diff_col)

        # 7. Drop incomplete records and clean out structural helper variables
        dataset.dropna(subset=abs_diff_cols, inplace=True)
        dataset.drop(
            columns=["_match_date"] + [f"true_{var}" for var in self.weather_variables], errors="ignore", inplace=True
        )

        # Align design matrix to project layout expectations
        final_features = ["location_name", "latitude", "longitude", "day_of_year", "delta_days"]
        final_features += self.weather_variables + abs_diff_cols
        dataset = dataset[final_features].reset_index(drop=True)

        # 8. Deterministic Validation Split Strategy
        if val_fraction > 0:
            shuffled_dataset = dataset.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
            split_idx = int(len(shuffled_dataset) * (1.0 - val_fraction))

            train_df = shuffled_dataset.iloc[:split_idx].reset_index(drop=True)
            val_df = shuffled_dataset.iloc[split_idx:].reset_index(drop=True)
            return train_df, val_df

        return dataset, pd.DataFrame()

    def remove_duplicates(self, df):
        """This method removes redundant data entries. This might happen when two
        forecasts are fetched which contain identical forecasts in all variables.
        This indicates, that that forecast likely stems from the same weather model
        prediction cycle and is therefore a redundant data point."""

        TOL_DECIMALS = 4

        df[self.numeric_variables] = df[self.numeric_variables].round(decimals=TOL_DECIMALS)
        df.drop_duplicates(inplace=True, keep="first", subset=self.weather_variables)

        return df

    def fetch_forecast(self, config=None) -> pd.DataFrame:
        """Uses the OpenMeteoClient class to fetch current forecasts.

        Args:
            config (_type_, optional): Optionally pass a custom config object. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the fetched forecast
        """
        if config is None:
            client = OpenMeteoClient()
        else:
            client = OpenMeteoClient(config=config)
        current_forecasts = client.fetch_forecast()
        current_forecasts = self.convert_to_best_dtypes(current_forecasts)
        return current_forecasts

    def combine_and_store_forecasts(self, current_forecasts: pd.DataFrame) -> None:
        """Uses the OpenMeteoClient to fetch current forecasts and combines
        with historic forecasts in the corresponding .csv file.

        Args:
            current_forecasts (pd.DataFrame): DataFrame containing the fetched forecast

        Returns:
            None
        """

        # Read existing data
        if self.raw.empty:
            self.read_raw()
        self.raw = self.convert_to_best_dtypes(self.raw)

        # Combine data
        self.raw = pd.concat([self.raw, current_forecasts])
        self.remove_duplicates(self.raw)
        self.raw.to_csv(self.PATH_TO_RAW, index=False)

        return None
