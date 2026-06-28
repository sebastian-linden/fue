import os
from pathlib import Path

import pandas as pd

from .openmeteoclient import OpenMeteoClient  # When imported as part of package


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

        return df

    def summarize(self):
        """This method outputs a summary of all data currently available"""
        pass

    def generate_dataset(self, location_name="aachen"):
        """This method filters dataset by city, separates forecasts from
        measured data, computes uncertainty distributions from given
        data sub-set and optionally separates data into training and
        validation data.
        1. Separate data by city
        2. Separate forecasts from measurements
        .. Prepare target dataframe (add/modify columns)
        3. Match forecasts to their measurements
        4. Compute absolute differences
        5. Return DataFrame
        """

        # 1. Separate data by city
        local_raw = self.raw[self.raw["location_name"] == location_name]

        # 2. Separate forecasts from measurements
        forecast_split = local_raw[
            (local_raw["forecast_for"] - local_raw["forecasted_on"]).dt.total_seconds() > 12 * 60 * 60
        ]
        measured_split = local_raw[
            (local_raw["forecast_for"] - local_raw["forecasted_on"]).dt.total_seconds() <= 12 * 60 * 60
        ]

        # .. Prepare target dataframe (add/modify columns)
        dataset_columns = ["location_name", "latitude", "longitude", "day_of_year", "delta_days"]
        dataset_columns += self.weather_variables
        for var in self.weather_variables:
            dataset_columns.append(f"abs_diff__{var}")
        dataset = pd.DataFrame(columns=dataset_columns)
        dataset[["location_name", "latitude", "longitude"]] = forecast_split[["location_name", "latitude", "longitude"]]
        dataset[self.weather_variables] = forecast_split[self.weather_variables]
        dataset["day_of_year"] = forecast_split["forecast_for"].dt.day_of_year
        dataset["delta_days"] = (
            forecast_split["forecast_for"] - forecast_split["forecasted_on"]
        ).dt.total_seconds() / (60 * 60 * 24)

        # 3. Match forecasts to their measurements and compute absolute differences
        """ For every forecast entry,
                I need to find the matching measurement entry,
                compute abs(forecast-measured) for every variable"""
        for row in measured_split.iterrows():
            # Find match(es)
            measurement = row[1]
            day = measurement["forecast_for"].day_of_year
            match_condition = forecast_split["forecast_for"].dt.day_of_year == day
            # Compute absolute difference
            for var in self.weather_variables:
                dataset.loc[match_condition, f"abs_diff__{var}"] = (
                    dataset[match_condition][var] - measurement[var]
                ).abs()

        # 4. Remove unmatched forecasts
        # This can happen, when in one fetches for example a 14-day forecast,
        # but in the following days doesn't download all the corresponding measurements.
        # Then, we're left with forecasts, for which we don't know the ground truth.
        # When there's a NAN entry for the absolute difference of a variable, then
        # this indicates an unmatched forecast.
        some_abs_diff__weather_variable = f"abs_diff__{self.weather_variables[0]}"
        nonNAN_indices = pd.Index.notna(dataset[some_abs_diff__weather_variable])
        dataset = dataset[nonNAN_indices]

        return dataset

    def remove_duplicates(self, df):
        """This method removes redundant data entries. This might happen when two
        forecasts are fetched which contain identical forecasts in all variables.
        This indicates, that that forecast likely stems from the same weather model
        prediction cycle and is therefore a redundant data point."""

        # MIN_FETCH_INTERVAL = pd.Timedelta(hours=3) # 3h is a common update frequency of the weather models
        TOL_DECIMALS = 4

        df[self.numeric_variables] = df[self.numeric_variables].round(decimals=TOL_DECIMALS)
        df.drop_duplicates(inplace=True, keep="first", subset=self.weather_variables)

        return df

    def fetch_and_store_forecasts(self):
        """Uses the OpenMeteoClient to fetch current forecasts and combines
        with historic forecasts in the corresponding .csv file.
        """

        # Fetch new data
        client = OpenMeteoClient()
        current_forecasts = client.fetch_forecast()
        current_forecasts = self.convert_to_best_dtypes(current_forecasts)

        # Read existing data
        if self.raw.empty:
            self.read_raw()
        self.raw = self.convert_to_best_dtypes(self.raw)

        # Combine data
        self.raw = pd.concat([self.raw, current_forecasts])
        self.remove_duplicates(self.raw)
        self.raw.to_csv(self.PATH_TO_RAW, index=False)

        return None


if __name__ == "__main__":
    data = Data()
    data.fetch_and_store_forecasts()
    print(data.generate_dataset(location_name="aachen"))
