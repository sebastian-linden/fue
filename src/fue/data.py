import logging
import os
from pathlib import Path

import pandas as pd
import numpy as np

from .openmeteoclient import OpenMeteoClient
from .utils import pair_cities_by_proximity

logger = logging.getLogger(__name__)


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

        logger.debug("Initialized Data handler with raw data path %s", self.PATH_TO_RAW)
        return None

    def read_raw(self, path=None):
        """This method reads the raw data available in the directory:
        data/raw/
        """
        if path is not None:
            self.PATH_TO_RAW = path
            logger.debug("Using custom raw data path %s", self.PATH_TO_RAW)

        try:
            self.raw = pd.read_csv(self.PATH_TO_RAW)
            self.raw = self.convert_to_best_dtypes(self.raw, sun_duration_to_hours=True)
            self.weather_variables = [c for c in self.raw.columns if c not in self.meta_variables]
            self.numeric_variables = self.weather_variables + ["latitude", "longitude"]
        except Exception as exc:
            logger.error("Failed to read raw forecasts from %s", self.PATH_TO_RAW, exc_info=True)
            raise FileNotFoundError("forecasts.csv was not found") from exc

        logger.info(f"Raw data read from {self.PATH_TO_RAW}. Total records: {len(self.raw)}")
        return None

    def convert_to_best_dtypes(self, df: pd.DataFrame, sun_duration_to_hours: bool = False) -> pd.DataFrame:
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

        if sun_duration_to_hours and "sunshine_duration" in df.columns:
            df["sunshine_duration"] = df["sunshine_duration"] / 3600.0

        logger.debug(
            "Converted DataFrame to best dtypes with %d rows and %d columns",
            len(df),
            len(df.columns),
        )
        return df

    def generate_dataset(self):
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
        raw_copy = self.raw.copy()
        raw_copy["forecast_for"] = pd.to_datetime(raw_copy["forecast_for"])
        raw_copy["forecasted_on"] = pd.to_datetime(raw_copy["forecasted_on"])

        # 2. Separate forecasts from measurements using your custom 12-hour boundary rule
        delta_seconds = (raw_copy["forecast_for"] - raw_copy["forecasted_on"]).dt.total_seconds()
        forecast_split = raw_copy[delta_seconds > 12 * 60 * 60].copy()
        measured_split = raw_copy[delta_seconds <= 12 * 60 * 60].copy()

        logger.debug(
            "Applied 12-hour split: %d forecast rows and %d measurement rows",
            len(forecast_split),
            len(measured_split),
        )

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
        if dataset.empty:
            logger.warning("Dataset generation produced no usable records after merging forecasts and measurements")

        # 5. Extract engineered features and apply standard unit scaling
        dataset["day_of_year"] = dataset["forecast_for"].dt.day_of_year
        dataset["delta_days"] = (
            dataset["forecast_for"] - dataset["forecasted_on"]
        ).dt.total_seconds() / (60 * 60 * 24)

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

        logger.info("Dataset generation completed with %d rows and %d columns", len(dataset), len(dataset.columns))
        return dataset
    
    def split_dataset(
        self, 
        dataset: pd.DataFrame, 
        val_fraction: float = 0.2, 
        random_state: int = 42,
        min_entries_per_city: int = 100
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Splits a unified dataframe into training and validation subsets using an
        adaptive, completely dynamic Stratified Climatic Block strategy.

        Deduces available locations and coordinates entirely from the unique combinations 
        found in the input dataset. Filters out cities with insufficient historical 
        records to prevent severe volumetric imbalance during cross-validation.

        Args:
            dataset (pd.DataFrame): The pooled data containing 'location_name', 'latitude', 
                                    and 'longitude' tracking columns.
            val_fraction (float): The targeted ratio of total data assigned to validation. Defaults to 0.2.
            random_state (int): Seed used to guarantee reproducible split logic. Defaults to 42.
            min_entries_per_city (int): Minimum rows required for a city to be included in the split. Defaults to 100.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: (train_df, val_df) split structures.
        """
        import numpy as np
        from .utils import pair_cities_by_proximity

        # 0. Filter out cities with insufficient data records
        city_counts = dataset["location_name"].value_counts()
        valid_cities = city_counts[city_counts >= min_entries_per_city].index.tolist()

        if not valid_cities:
            logger.warning(
                "No cities met the minimum entry threshold %d; returning empty splits",
                min_entries_per_city,
            )
            return dataset.iloc[0:0].copy(), dataset.iloc[0:0].copy()

        filtered_dataset = dataset[dataset["location_name"].isin(valid_cities)].copy()

        # 1. Deduce city coordinate mappings directly from the filtered DataFrame rows
        coord_df = filtered_dataset[["location_name", "latitude", "longitude"]].drop_duplicates()
        city_dict = {
            row["location_name"]: {"lat": row["latitude"], "lon": row["longitude"]}
            for _, row in coord_df.iterrows()
        }

        # 2. DYNAMIC SPATIAL PAIRING (Delegated to utils using deduced mapping dictionary)
        groups = pair_cities_by_proximity(city_dict)

        # 3. Extract accurate data footprints based on frequency distribution in filtered dataset
        city_row_counts = filtered_dataset["location_name"].value_counts().to_dict()
        total_rows_in_data = len(filtered_dataset)
        target_val_rows = total_rows_in_data * val_fraction

        valid_paired_groups = [g_id for g_id, c_list in groups.items() if len(c_list) >= 2]
        
        # Instantiate localized generator for strict workflow tracking
        rng = np.random.default_rng(random_state)
        rng.shuffle(valid_paired_groups)

        train_cities = []
        val_cities = []
        current_val_rows = 0

        # 4. Incrementally fulfill split fraction using the dynamically generated pairs
        for g_id in valid_paired_groups:
            pair_cities = groups[g_id]
            
            # If target threshold reached, remaining groups fall back entirely to training
            if current_val_rows >= target_val_rows or val_fraction <= 0.0:
                train_cities.extend(pair_cities)
                continue

            # Randomly elect one city from the pair to act as validation target
            idx_for_val = rng.choice([0, 1])
            v_city = pair_cities[idx_for_val]
            t_city = pair_cities[1 - idx_for_val]

            val_cities.append(v_city)
            train_cities.append(t_city)

            # Accumulate exact row footprints into tracking metrics
            current_val_rows += city_row_counts.get(v_city, 0)

        # 5. Handle leftovers (unpaired cities or row mismatches)
        unique_dataset_cities = list(city_dict.keys())
        for city in unique_dataset_cities:
            if city not in train_cities and city not in val_cities:
                train_cities.append(city)

        # 6. Extract split datasets back to data execution routines
        train_df = filtered_dataset[filtered_dataset["location_name"].isin(train_cities)].copy()
        val_df = filtered_dataset[filtered_dataset["location_name"].isin(val_cities)].copy()

        logger.info("Dataset split completed with %d training rows and %d validation rows", len(train_df), len(val_df))
        return train_df, val_df

    def remove_duplicates(self, df):
        """This method removes redundant data entries. This might happen when two
        forecasts are fetched which contain identical forecasts in all variables.
        This indicates, that that forecast likely stems from the same weather model
        prediction cycle and is therefore a redundant data point."""

        TOL_DECIMALS = 4

        rows_before = len(df)
        df[self.numeric_variables] = df[self.numeric_variables].round(decimals=TOL_DECIMALS)
        df.drop_duplicates(inplace=True, keep="first", subset=self.weather_variables)

        logger.debug("Duplicate removal reduced rows from %d to %d", rows_before, len(df))
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

        try:
            current_forecasts = client.fetch_forecast()
        except Exception:
            logger.error("Forecast fetch failed while calling OpenMeteoClient", exc_info=True)
            raise

        current_forecasts = self.convert_to_best_dtypes(current_forecasts)
        logger.debug("Fetched %d forecast rows from Open-Meteo", len(current_forecasts))
        return current_forecasts

    def combine_and_store_forecasts(self, current_forecasts: pd.DataFrame) -> None:
        """Combine newly fetched forecasts with historic forecasts and store the result."""
        logger.info("Combining newly fetched forecasts with stored data")

        if self.raw.empty:
            self.read_raw()
        self.raw = self.convert_to_best_dtypes(self.raw)

        if current_forecasts.empty:
            logger.warning("No fresh forecast rows were provided; nothing will be added to storage")

        try:
            self.raw = pd.concat([self.raw, current_forecasts])
            self.remove_duplicates(self.raw)
            self.raw.to_csv(self.PATH_TO_RAW, index=False)
        except Exception:
            logger.error("Failed to combine and store forecasts in %s", self.PATH_TO_RAW, exc_info=True)
            raise

        logger.info("Forecasts stored to %s", self.PATH_TO_RAW)
        return None

    def get_collection_summary(self, threshold: int = 100) -> pd.DataFrame:
        """
        Computes a statistical breakdown of the collected weather records per city,
        identifying which locations have crossed the training threshold.

        Args:
            threshold (int): Minimum rows required for active status. Defaults to 100.

        Returns:
            pd.DataFrame: A sorted summary DataFrame containing record counts and readiness status.
        """
        logger.debug("Generating collection summary with threshold %d", threshold)
        # Generate the unified dataset to get final valid target pairings
        dataset = self.generate_dataset()
        
        # Calculate counts and build the summary structure
        counts = dataset["location_name"].value_counts()
        
        summary_data = []
        for city, count in counts.items():
            status = "ACTIVE" if count >= threshold else "WAITING"
            summary_data.append({
                "location_name": city,
                "valid_records": count,
                "status": status
            })
            
        # Return as a DataFrame for maximum downstream flexibility
        return pd.DataFrame(summary_data)