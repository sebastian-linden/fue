"""
Data pipeline management system for the pyfue package.

This module handles reading raw forecast csv files, sorting incoming data, parsing
units, applying our custom 12-hour boundary rule to separate predictions from ground-truth
measurements, and splitting data cleanly by city proximity groups for machine learning.
"""

import logging
import os

import numpy as np
import pandas as pd

from .config import Config
from .openmeteoclient import OpenMeteoClient
from .utils import pair_cities_by_proximity

logger = logging.getLogger(__name__)


class Data:
    """
    Manages loading, cleaning, assembling, and splitting our project's weather datasets.

    This class coordinates the data-shaping pipeline. It reads from local storage files,
    calculates physical absolute forecast error columns, handles temporal-safe data partitions,
    and packages data frames so they are structured perfectly for downstream training models.
    """

    def __init__(self, config: Config) -> None:
        """Sets up default file paths and tracking names for different categories of variables."""

        # Get path
        if config is None:
            logger.error("Configuration argument is None.")
            raise TypeError("Configuration argument is None.")
        else:
            self.config = config
            self.path = self.config.data_path

        # This is the data as fetched from the open-meteo API.
        self.raw = pd.DataFrame()

        self.meta_variables = ["location_name", "latitude", "longitude", "forecasted_on", "forecast_for"]
        self.weather_variables = self.config.params["daily"]
        self.columns = self.meta_variables + self.weather_variables
        self.numeric_variables = self.weather_variables + ["latitude", "longitude"]

        logger.debug("Initialized Data handler with raw data path %s", self.path)
        return None

    def read_raw(self) -> bool:
        """
        Reads the local raw forecast CSV file from disk into memory.

        After loading the file, it cleans up types, automatically parses units,
        and updates list indices containing column headers for meta and weather parameters.

        Parameters
        ----------
        path : str or pathlib.Path or None, default=None
            Custom path pointing to a raw forecast file. If None, uses the
            default location folder inside your project workspace root.

        Returns
        -------
        bool
            True if the file was read successfully, False otherwise.

        Raises
        ------
        FileNotFoundError
            If no forecast csv source matches the determined path target.
        """
        if os.path.exists(self.path) is False:
            print(f"Couldn't find {self.path}")
            logger.error("Couldn't find %s", self.path, exc_info=True)
            raise FileNotFoundError("%s was not found", self.path)
        elif os.stat(self.path).st_size == 0:
            print(f"File {self.path} is empty; no data will be loaded")
            logger.warning("File %s is empty; no data will be loaded", self.path)
            self.raw = pd.DataFrame(columns=self.columns)
            return False
        else:
            try:
                self.raw = pd.read_csv(self.path)
                self.raw = self.convert_to_best_dtypes(self.raw, sun_duration_to_hours=False)
            except Exception as exc:
                logger.error("Failed to read raw forecasts correctly from %s", self.path, exc_info=True)
                raise BaseException("%s was found, but couldn't be read successfully", self.path) from exc
        logger.info(f"Raw data read from {self.path}. Total records: {len(self.raw)}")
        return True

    def convert_to_best_dtypes(self, df: pd.DataFrame, sun_duration_to_hours: bool = False) -> pd.DataFrame:
        """
        Ensures columns use optimal pandas types and converts metrics to correct units.

        Forces city strings, timestamps, and numbers into their proper pandas formats.
        It also scales raw sunshine seconds into hours to maintain mathematical sanity
        across model boundaries.

        Parameters
        ----------
        df : pd.DataFrame
            The input weather records needing column type adjustments.
        sun_duration_to_hours : bool, default=False
            If True, converts 'sunshine_duration' from raw seconds to hours.

        Returns
        -------
        pd.DataFrame
            The freshly typed data frame with identical rows.
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

    def generate_dataset(self) -> pd.DataFrame:
        """
        assembles our target training matrix by matching forecasts against ground truth.

        This method executes several steps:
        1. Splits chronological rows using a 12-hour lead window (older rows become
           forecast entries, while immediate horizons act as actual observation proxies).
        2. Deduplicates matched dates to secure exactly one true target per calendar day.
        3. Runs an inner merge, extracts seasonal timelines, and evaluates the absolute
           residual differences (|predicted - observed|) to create our final training targets.

        Returns
        -------
        pd.DataFrame
            The consolidated design matrix containing core feature columns, lead timelines,
            and target absolute difference values.
        """

        if self.raw.empty:
            self.read_raw()

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

        logger.info("Dataset generation completed with %d rows and %d columns", len(dataset), len(dataset.columns))
        return dataset

    def split_dataset(
        self, dataset: pd.DataFrame, val_fraction: float = 0.2, random_state: int = 42, min_entries_per_city: int = 100
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits data chronologically and geographically for training & validation.

        First, drops rows belonging to newer cities that haven't collected enough
        records. Next, pairs remaining cities using spatial distance metrics.
        It randomly assigns one city from each pair to training and the other to
        validation, ensuring an entire regional microclimate is fully quarantined
        from the training group.

        Parameters
        ----------
        dataset : pd.DataFrame
            The raw constructed input dataset from our pipeline layer.
        val_fraction : float, default=0.2
            The target percentage size of our validation dataset chunk (e.g. 0.2 for 20%).
        random_state : int, default=42
            The initialization token to ensure repeatable city-flipping choices.
        min_entries_per_city : int, default=100
            The minimum row cut-off length required for a city to join the active pipeline.

        Returns
        -------
        tuple of pd.DataFrame
            A two-element tuple containing our (train_df, val_df) split subsets.
        """

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
            row["location_name"]: {"lat": row["latitude"], "lon": row["longitude"]} for _, row in coord_df.iterrows()
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

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Purges matching overlapping data entries based on weather metric states.

        Rounds the position data down to a tiny decimal tolerance limit, identifies exact
        duplicate weather attribute rows, and preserves the first occurrence while popping out
        the trailing duplicates.

        Parameters
        ----------
        df : pd.DataFrame
            The target data frame tracking rows that might possess duplicates.

        Returns
        -------
        pd.DataFrame
            A duplicate-free row-subset of our original input frame.
        """
        TOL_DECIMALS = 4

        rows_before = len(df)
        df[self.numeric_variables] = df[self.numeric_variables].round(decimals=TOL_DECIMALS)
        df.drop_duplicates(inplace=True, keep="first", subset=self.weather_variables)

        logger.debug("Duplicate removal reduced rows from %d to %d", rows_before, len(df))
        return df

    def fetch_forecast(self) -> pd.DataFrame:
        """
        Uses the internal OpenMeteoClient to download live forward-looking forecasts.

        Returns
        -------
        pd.DataFrame
            Live weather point predictions structured into typed pandas rows.
        """
        client = OpenMeteoClient(config=self.config)

        try:
            current_forecasts = client.fetch_forecast()
        except Exception:
            logger.error("Forecast fetch failed while calling OpenMeteoClient", exc_info=True)
            raise

        current_forecasts = self.convert_to_best_dtypes(current_forecasts, sun_duration_to_hours=True)
        logger.debug("Fetched %d forecast rows from Open-Meteo", len(current_forecasts))
        return current_forecasts

    def combine_and_store_forecasts(self, current_forecasts: pd.DataFrame) -> None:
        """
        Merges newly scraped forecast entries into our main persistent CSV archive file.

        Loads your existing baseline historical rows, glues the fresh API scrape rows
        directly to the bottom, applies automatic duplicate row removal rules, and
        re-saves the structured updates back to disk storage.

        Parameters
        ----------
        current_forecasts : pd.DataFrame
            Fresh weather predictions scraped directly from the current API download session.

        Returns
        -------
        None
        """
        logger.info("Combining newly fetched forecasts with stored data")

        if self.raw.empty:
            status = self.read_raw()
            print("Read raw data status:", status)
            if status is False:
                logger.warning("No existing raw data was found; starting fresh with new forecasts")
                self.raw = pd.DataFrame()

        if current_forecasts.empty:
            logger.warning("No fresh forecast rows were provided; nothing will be added to storage")

        try:
            self.raw = pd.concat([self.raw, current_forecasts])
            self.remove_duplicates(self.raw)
            self.raw.to_csv(self.path, index=False)
        except Exception:
            logger.error("Failed to combine and store forecasts in %s", self.path, exc_info=True)
            raise

        logger.info("Forecasts stored to %s", self.path)
        return None

    def get_collection_summary(self, threshold: int = 100) -> pd.DataFrame:
        """
        Generates a quick diagnostic status report for all our monitored cities.

        Evaluates rows through our active dataset assembly engine, counts valid
        completed pairings per town, and tags each city name as 'ACTIVE' or 'WAITING'
        based on your custom validation threshold length.

        Parameters
        ----------
        threshold : int, default=100
            The line count limit required for a city to graduate to active model training.

        Returns
        -------
        pd.DataFrame
            A tabular summary tracking three columns: 'location_name', 'valid_records',
            and 'status'. Ideal for displaying a clean report inside our terminal interface.
        """
        logger.debug("Generating collection summary with threshold %d", threshold)
        # Generate the unified dataset to get final valid target pairings
        dataset = self.generate_dataset()

        # Calculate counts and build the summary structure
        counts = dataset["location_name"].value_counts()

        summary_data = []
        for city, count in counts.items():
            status = "ACTIVE" if count >= threshold else "WAITING"
            summary_data.append({"location_name": city, "valid_records": count, "status": status})

        # Return as a DataFrame for maximum downstream flexibility
        return pd.DataFrame(summary_data)
