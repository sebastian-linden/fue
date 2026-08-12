import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# Adjust the import based on your exact project structure
from pyfue import Config, Data


@pytest.fixture
def test_config(tmp_path):
    """Creates a complete temporary config.json for Data class testing."""
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps(
            {
                "data_path": "data/forecasts.csv",
                "runs_dir": "runs",
                "cities": {"london": {"lat": 51.5, "lon": -0.12}},
                "daily": ["temperature_2m_max"],
                "timezone": "UTC",
                "past_days": 5,
                "forecast_days": 10,
                "preprocessing": {},
                "default_feature_columns": ["temperature_2m_max"],
                "default_target_columns": ["abs_diff__temperature_2m_max"],
            }
        )
    )
    return Config(path=config_file)


class TestData:
    """Test suite for the Data class handling forecast and measurement data."""

    def test_init(self, test_config):
        """Test the initialization of the Data class to ensure attributes are set correctly."""
        data = Data(test_config)

        # Check that the raw DataFrame is initialized and empty
        assert isinstance(data.raw, pd.DataFrame)
        assert data.raw.empty

        # Check default variables lists
        assert data.meta_variables == ["location_name", "latitude", "longitude", "forecasted_on", "forecast_for"]

        # Check path resolutions
        assert isinstance(data.path, Path)

    def test_read_raw_success(self, test_config):
        """Test reading a valid raw data file updates paths and variables correctly."""
        data = Data(test_config)
        data.path = "tests/test_forecasts.csv"

        data.read_raw()

        # Assertions
        assert not data.raw.empty

        # Check that weather and numeric variables were populated
        assert len(data.weather_variables) > 0
        assert "latitude" in data.numeric_variables
        assert "longitude" in data.numeric_variables

        # Check that meta variables are NOT in weather variables
        for meta_var in data.meta_variables:
            assert meta_var not in data.weather_variables

    def test_convert_to_best_dtypes(self, test_config):
        """Test that the dtype conversion method enforces strings, datetimes, and numerics, and handles units."""
        data = Data(test_config)

        # Create a mock DataFrame with incorrect, raw types
        mock_df = pd.DataFrame(
            {
                "location_name": [123, 456],  # Integers simulating bad string data
                "forecasted_on": ["2026-07-01 10:00:00", "2026-07-02 10:00:00"],  # Strings simulating datetimes
                "forecast_for": ["2026-07-01 12:00:00", "2026-07-02 12:00:00"],
                "temperature_2m": ["20.5", "22.1"],  # Strings simulating floats
                "sunshine_duration": [3600.0, 7200.0],  # Seconds that need conversion to hours
            }
        )

        # Execute
        cleaned_df = data.convert_to_best_dtypes(mock_df)

        # Assertions for Datatypes
        assert pd.api.types.is_object_dtype(cleaned_df["location_name"]) or pd.api.types.is_string_dtype(
            cleaned_df["location_name"]
        )
        assert cleaned_df["location_name"].iloc[0] == "123"

        assert pd.api.types.is_datetime64_any_dtype(cleaned_df["forecasted_on"])
        assert pd.api.types.is_datetime64_any_dtype(cleaned_df["forecast_for"])

        assert pd.api.types.is_numeric_dtype(cleaned_df["temperature_2m"])
        assert cleaned_df["temperature_2m"].iloc[0] == 20.5

    def test_generate_dataset(self, test_config):
        """
        Verifies that the dataset generation pipeline successfully processes
        multi-city raw data into a single, unified DataFrame with the correct
        tracking and feature columns.
        """
        data = Data(test_config)
        data.path = "tests/test_forecasts.csv"

        data.read_raw()

        # Generate the dataset (now processes all available data at once)
        dataset = data.generate_dataset()

        # 1. Type and structure checks
        assert isinstance(dataset, pd.DataFrame), "generate_dataset must return a single DataFrame"
        assert len(dataset) == 489, "Row count mismatch: expected 489 rows from test dataset"

        # 2. Feature engineering checks
        expected_columns = ["location_name", "day_of_year", "delta_days", "temperature_2m_max"]
        for col in expected_columns:
            assert col in dataset.columns, f"Missing engineered column: {col}"

        # 3. Completeness check
        unique_cities = dataset["location_name"].unique()
        assert len(unique_cities) == 6, "Dataset should contain exactly 6 pooled cities"
        assert set(unique_cities) == {"london", "berlin", "aachen", "paris", "rome", "madrid"}

    def test_split_dataset(self, test_config):
        """
        Verifies the Stratified Climatic Block strategy.
        Mathematically asserts that the Haversine pairing algorithm successfully
        prevents spatial data leakage between the training and validation subsets.
        """
        data = Data(test_config)
        data.path = "tests/test_forecasts.csv"

        data.read_raw()

        # Split the dataset using the deterministic random state
        train_df, val_df = data.split_dataset(data.raw, val_fraction=0.3, random_state=42)

        # 1. Type and Size checks
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(val_df, pd.DataFrame)
        assert len(train_df) == 800, "Train row count mismatch"
        assert len(val_df) == 400, "Validation row count mismatch"
        assert len(train_df) + len(val_df) == len(data.raw), "Data was lost during the split"

        # 2. Extract city lists
        train_cities = set(train_df["location_name"].unique())
        val_cities = set(val_df["location_name"].unique())

        # 3. THE CRITICAL V&V CHECK: Absolute Disjointness (No Leakage)
        assert train_cities.isdisjoint(val_cities), (
            f"DATA LEAKAGE DETECTED! Overlapping cities: {train_cities.intersection(val_cities)}"
        )

        # 4. Deterministic Placement Check (Ensures random_state seeding works)
        assert val_cities == {"berlin", "rome"}, "Validation split distribution drifted"
        assert train_cities == {"london", "aachen", "paris", "madrid"}, "Train split distribution drifted"

    def test_remove_duplicates(self, test_config):
        """
        Test that duplicate forecasts are removed based on weather variables,
        respecting the 4-decimal place rounding tolerance.
        """
        data = Data(test_config)

        # Manually set the variable lists to isolate this test from read_raw()
        data.weather_variables = ["temperature_2m", "wind_speed_10m"]
        data.numeric_variables = ["temperature_2m", "wind_speed_10m", "latitude", "longitude"]

        # Create a mock DataFrame
        mock_data = pd.DataFrame(
            {
                "meta_id": ["A", "B", "C"],
                "latitude": [50.11111, 50.11114, 50.11119],  # idx 0 & 1 round to 50.1111, idx 2 rounds to 50.1112
                "longitude": [8.2222, 8.2222, 8.2222],
                "temperature_2m": [25.12341, 25.12344, 25.9999],  # idx 0 & 1 round to 25.1234
                "wind_speed_10m": [5.0, 5.0, 5.0],
            }
        )

        # Execute
        cleaned_df = data.remove_duplicates(mock_data)

        # Assertions
        # Row 0 and Row 1 are identical in weather_variables after 4-decimal rounding.
        # Row 1 should be dropped, leaving exactly 2 rows.
        assert len(cleaned_df) == 2

        # Ensure the index was maintained correctly (row "B" / idx 1 is gone)
        assert cleaned_df["meta_id"].tolist() == ["A", "C"]

        # Verify that the rounding actually occurred across the numeric variables
        assert cleaned_df.iloc[0]["temperature_2m"] == 25.1234
        assert cleaned_df.iloc[0]["latitude"] == 50.1111
        assert cleaned_df.iloc[1]["latitude"] == 50.1112

    @patch("pyfue.data.OpenMeteoClient")
    def test_fetch_forecast(self, MockClient, test_config):
        """
        Test that fetch_forecast initializes the client, calls the API,
        and correctly processes the data types without making real network requests.
        """
        data = Data(test_config)

        # Setup the fake OpenMeteoClient and dictate what it should return
        mock_instance = MockClient.return_value
        mock_raw_api_response = pd.DataFrame(
            {
                "location_name": ["berlin"],
                "forecasted_on": ["2026-07-02 10:00:00"],
                "forecast_for": ["2026-07-02 12:00:00"],
                "temperature_2m": ["22.5"],  # String to test type conversion
                "sunshine_duration": [3600.0],  # Seconds to test unit conversion
            }
        )
        mock_instance.fetch_forecast.return_value = mock_raw_api_response

        # Execute
        result_df = data.fetch_forecast()

        # Assertions
        # 1. Verify the client was instantiated correctly by asserting it receives the config
        MockClient.assert_called_once_with(config=test_config)

        # 2. Verify the fetch method was actually called
        mock_instance.fetch_forecast.assert_called_once()

        # 3. Verify that convert_to_best_dtypes was successfully applied to the API response
        assert pd.api.types.is_numeric_dtype(result_df["temperature_2m"])
        assert result_df["temperature_2m"].iloc[0] == 22.5

    def test_combine_and_store_forecasts(self, test_config, tmp_path):
        """
        Test combining existing forecast data with new fetched rows,
        ensuring duplicates are handled and mutations are cleanly written to disk.
        """
        data = Data(test_config)

        # 1. Setup our sandboxed temporary file paths
        temp_csv = tmp_path / "temp_forecasts.csv"
        data.path = str(temp_csv)

        # 2. Extract a real baseline dataframe from your test file to seed the test
        real_sample_df = pd.read_csv("tests/test_forecasts.csv").head(2)

        # Save this baseline slice to our temp workspace file
        real_sample_df.to_csv(temp_csv, index=False)

        # 3. Build a "newly fetched" forecast block using the exact same structure
        # Row 0: An exact duplicate of the first row (should be ignored)
        # Row 1: A brand new row (we alter weather parameters slightly to mock a new horizon)
        new_forecasts_df = real_sample_df.copy()
        new_forecasts_df.loc[1, "temperature_2m_max"] = 99.0  # Unique marker for the new forecast
        new_forecasts_df.loc[1, "forecast_for"] = "2026-12-31 23:59:59"  # Change unique timeframe

        # Execute the pipeline
        data.combine_and_store_forecasts(new_forecasts_df)

        # 4. Read back the updated file from our temp disk to run checks
        saved_df = pd.read_csv(temp_csv)

        # We began with 2 rows. We added 2 rows, but 1 was an exact duplicate.
        # The resulting storage file should have exactly 3 rows.
        assert len(saved_df) == 3

        # Ensure that our modified data made it safely into the file
        assert 99.0 in saved_df["temperature_2m_max"].values
