import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from fue.config import Config

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fue.openmeteoclient import OpenMeteoClient


class TestOpenMeteoClientInit:
    """Test suite for OpenMeteoClient initialization"""

    def test_client_initialization(self):
        """Test that OpenMeteoClient initializes without errors"""
        client = OpenMeteoClient()
        assert client is not None
        assert hasattr(client, "openmeteo")
        assert hasattr(client, "config")
        assert hasattr(client, "url")

    def test_url_is_correct(self):
        """Test that the API URL is set correctly"""
        client = OpenMeteoClient()
        assert client.url == "https://api.open-meteo.com/v1/forecast"

    def test_cache_session_initialized(self):
        """Test that cache session is initialized"""
        client = OpenMeteoClient()
        assert client.cache_session is not None

    def test_config_loaded(self):
        """Test that config is loaded"""
        client = OpenMeteoClient()
        assert client.config is not None
        assert hasattr(client.config, "params")
        assert hasattr(client.config, "cities")


class TestOpenMeteoClientIntegration:
    """Integration tests for OpenMeteoClient"""

    def test_client_setup_complete(self):
        """Test that all components are properly set up"""
        client = OpenMeteoClient()

        # Verify all necessary attributes exist
        assert hasattr(client, "cache_session")
        assert hasattr(client, "retry_session")
        assert hasattr(client, "openmeteo")
        assert hasattr(client, "url")
        assert hasattr(client, "config")

        # Verify config has necessary attributes
        assert hasattr(client.config, "params")
        assert hasattr(client.config, "cities")

        # Verify config.params has necessary keys
        assert "daily" in client.config.params
        assert "timezone" in client.config.params


class TestOpenMeteoClientLogic:
    """Test suite for the logic and data processing of the OpenMeteoClient."""

    def test_init_with_custom_config(self):
        """Test that passing a custom config skips default initialization (Line 26)."""
        custom_config = Config()
        client = OpenMeteoClient(config=custom_config)
        assert client.config is custom_config

    @patch("fue.openmeteoclient.openmeteo_requests.Client")
    def test_fetch_forecast_data_processing(self, MockAPIClient):
        """
        Test that fetch_forecast correctly requests data from the API and
        properly formats the deeply nested response into a flat Pandas DataFrame.
        """
        # 1. Setup a controlled config so we know exactly what is being requested
        test_config = Config()
        test_config.cities = ["Aachen"]
        test_config.city_coordinates = {"Aachen": {"lat": 50.77, "lon": 6.08}}
        test_config.params["latitude"] = [50.77]
        test_config.params["longitude"] = [6.08]
        test_config.params["daily"] = ["temperature_2m_max", "precipitation_sum"]

        # Instantiate client with our controlled config
        client = OpenMeteoClient(config=test_config)

        # 2. Build the complex mock response object
        # Open-Meteo returns a list of response objects (one per city)
        mock_response = MagicMock()
        mock_response.Latitude.return_value = 50.77
        mock_response.Longitude.return_value = 6.08

        # Mock the 'Daily' sub-object and its time attributes
        mock_daily = MagicMock()
        # UNIX timestamps for 2 days
        mock_daily.Time.return_value = 1685577600  # e.g., June 1, 2023
        mock_daily.TimeEnd.return_value = 1685750400  # e.g., June 3, 2023
        mock_daily.Interval.return_value = 86400  # 1 day in seconds

        # Mock the variables array (temperature and precipitation)
        mock_var_0 = MagicMock()
        mock_var_0.ValuesAsNumpy.return_value = [22.5, 24.1]  # Temp mock
        mock_var_1 = MagicMock()
        mock_var_1.ValuesAsNumpy.return_value = [0.0, 5.2]  # Precip mock

        # Map the requested variables to our mocks
        def side_effect_variables(index):
            return mock_var_0 if index == 0 else mock_var_1

        mock_daily.Variables.side_effect = side_effect_variables
        mock_response.Daily.return_value = mock_daily

        # Attach our fake response to the mocked API client
        mock_api_instance = MockAPIClient.return_value
        mock_api_instance.weather_api.return_value = [mock_response]

        # Override the client's internal openmeteo object with our mock
        client.openmeteo = mock_api_instance

        # 3. Execute the method
        result_df = client.fetch_forecast()

        # 4. Assertions
        # Verify the API was actually called with the correct URL
        mock_api_instance.weather_api.assert_called_once()
        args, kwargs = mock_api_instance.weather_api.call_args
        assert args[0] == "https://api.open-meteo.com/v1/forecast"

        # Verify the DataFrame shape and structure
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 2  # 2 days of data
        assert "location_name" in result_df.columns
        assert "forecast_for" in result_df.columns
        assert "temperature_2m_max" in result_df.columns
        assert "precipitation_sum" in result_df.columns

        # Verify the data was mapped correctly from the API to the columns
        assert result_df["location_name"].iloc[0] == "Aachen"
        assert result_df["temperature_2m_max"].iloc[0] == 22.5
        assert result_df["precipitation_sum"].iloc[1] == 5.2
