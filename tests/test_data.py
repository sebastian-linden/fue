from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# Adjust the import based on your exact project structure
from fue import Data


class TestData:
    """Test suite for the Data class handling forecast and measurement data."""

    def test_init(self):
        """Test the initialization of the Data class to ensure attributes are set correctly."""
        data = Data()

        # Check that the raw DataFrame is initialized and empty
        assert isinstance(data.raw, pd.DataFrame)
        assert data.raw.empty

        # Check default variables lists
        assert data.meta_variables == ["location_name", "latitude", "longitude", "forecasted_on", "forecast_for"]
        assert data.weather_variables == []
        assert data.numeric_variables == []

        # Check path resolutions
        assert isinstance(data.PATH_TO_DATA, Path)
        assert data.PATH_TO_DATA.name == "data"
        assert isinstance(data.PATH_TO_RAW, str)
        assert data.PATH_TO_RAW.endswith("forecasts.csv")

    def test_read_raw_success(self):
        """Test reading a valid raw data file updates paths and variables correctly."""
        data = Data()

        # Path to the specific subset you created for testing
        test_file_path = "tests/test_forecasts.csv"

        # Execute the method
        data.read_raw(path=test_file_path)

        # Assertions
        assert data.PATH_TO_RAW == test_file_path
        assert not data.raw.empty

        # Check that weather and numeric variables were populated
        assert len(data.weather_variables) > 0
        assert "latitude" in data.numeric_variables
        assert "longitude" in data.numeric_variables

        # Check that meta variables are NOT in weather variables
        for meta_var in data.meta_variables:
            assert meta_var not in data.weather_variables

    def test_read_raw_file_not_found(self):
        """Test that the correct exception is raised when the raw data file is missing."""
        data = Data()

        # Using pytest.raises to catch the specific exception and error message
        with pytest.raises(FileNotFoundError, match="forecasts.csv was not found"):
            data.read_raw(path="does_not_exist_123.csv")

    def test_convert_to_best_dtypes(self):
        """Test that the dtype conversion method enforces strings, datetimes, and numerics, and handles units."""
        data = Data()

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

        # Assertion for Unit Conversion (sunshine_duration / 3600.0)
        assert cleaned_df["sunshine_duration"].iloc[0] == 1.0
        assert cleaned_df["sunshine_duration"].iloc[1] == 2.0

    def test_generate_dataset(self):
        """
        Test the generate_dataset method against known snapshots to ensure
        reproducibility and prevent regressions in the data pipeline.
        """
        data = Data()
        data.read_raw(path="tests/test_forecasts.csv")

        #  1: Baseline Default Parameters
        train_df, val_df = data.generate_dataset(location_name="london", val_fraction=0.2, random_state=42)
        assert train_df.shape == (336, 23)
        assert val_df.shape == (85, 23)

        # Verify exact row permutations to ensure random_state is respected
        assert train_df.iloc[0]["day_of_year"] == 146
        assert train_df.iloc[0]["delta_days"] == pytest.approx(5.50, abs=0.01)
        assert val_df.iloc[0]["day_of_year"] == 175
        assert val_df.iloc[0]["delta_days"] == pytest.approx(2.21, abs=0.01)

        # TEST 2: No Validation Split
        train_df_no_val, val_df_no_val = data.generate_dataset(
            location_name="london", val_fraction=0.0, random_state=42
        )
        assert train_df_no_val.shape == (421, 23)
        assert val_df_no_val.empty

        # TEST 3: Reproducibility via Different Random State
        train_df_rs, val_df_rs = data.generate_dataset(location_name="london", val_fraction=0.2, random_state=99)
        assert train_df_rs.shape == (336, 23)
        assert val_df_rs.shape == (85, 23)

        # The top row should be different due to the new seed
        assert train_df_rs.iloc[0]["day_of_year"] == 183
        assert train_df_rs.iloc[0]["delta_days"] == pytest.approx(6.61, abs=0.01)

        # TEST 4: Edge Case - Missing City
        train_df_missing, val_df_missing = data.generate_dataset(
            location_name="unknown_city", val_fraction=0.2, random_state=42
        )
        # Should return safely without errors, yielding empty sets
        assert train_df_missing.empty
        assert val_df_missing.empty

    def test_remove_duplicates(self):
        """
        Test that duplicate forecasts are removed based on weather variables,
        respecting the 4-decimal place rounding tolerance.
        """
        data = Data()

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

    @patch("fue.data.OpenMeteoClient")
    def test_fetch_forecast(self, MockClient):
        """
        Test that fetch_forecast initializes the client, calls the API,
        and correctly processes the data types without making real network requests.
        """
        data = Data()

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
        # 1. Verify the client was instantiated correctly (no config passed)
        MockClient.assert_called_once_with()
        # 2. Verify the fetch method was actually called
        mock_instance.fetch_forecast.assert_called_once()

        # 3. Verify that convert_to_best_dtypes was successfully applied to the API response
        assert pd.api.types.is_numeric_dtype(result_df["temperature_2m"])
        assert result_df["temperature_2m"].iloc[0] == 22.5
        assert result_df["sunshine_duration"].iloc[0] == 1.0  # 3600 / 3600

    def test_combine_and_store_forecasts(self, tmp_path):
        """
        Test combining existing forecast data with new fetched rows,
        ensuring duplicates are handled and mutations are cleanly written to disk.
        """
        data = Data()

        # 1. Setup our sandboxed temporary file paths
        temp_csv = tmp_path / "temp_forecasts.csv"
        data.PATH_TO_RAW = str(temp_csv)

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
