import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pyfue.config import Config
from pyfue.openmeteoclient import OpenMeteoClient


@pytest.fixture
def test_config(tmp_path):
    """Fixture providing a Config object for OpenMeteoClient tests."""
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
            }
        )
    )
    return Config(path=config_file)


@pytest.fixture
def client(test_config):
    """Fixture providing OpenMeteoClient initialized with test_config."""
    return OpenMeteoClient(config=test_config)


class TestOpenMeteoClientInit:
    """Test suite for OpenMeteoClient initialization"""

    def test_client_initialization(self, test_config):
        """Test that OpenMeteoClient initializes without errors"""
        client = OpenMeteoClient(config=test_config)
        assert client is not None
        assert hasattr(client, "openmeteo")
        assert hasattr(client, "config")
        assert hasattr(client, "url")

    def test_url_is_correct(self, test_config):
        """Test that the API URL is set correctly"""
        client = OpenMeteoClient(config=test_config)
        assert client.url == "https://api.open-meteo.com/v1/forecast"

    def test_cache_session_initialized(self, test_config):
        """Test that cache session is initialized"""
        client = OpenMeteoClient(config=test_config)
        assert client.cache_session is not None

    def test_config_loaded(self, test_config):
        """Test that config is loaded"""
        client = OpenMeteoClient(config=test_config)
        assert client.config is not None
        assert hasattr(client.config, "params")
        assert hasattr(client.config, "cities")


class TestOpenMeteoClientIntegration:
    """Integration tests for OpenMeteoClient"""

    def test_client_setup_complete(self, test_config):
        """Test that all components are properly set up"""
        client = OpenMeteoClient(config=test_config)

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

    @patch("pyfue.openmeteoclient.openmeteo_requests.Client")
    def test_fetch_forecast_data_processing(self, MockAPIClient, test_config):
        """
        Test that fetch_forecast correctly requests data from the API and
        properly formats the deeply nested response into a flat Pandas DataFrame.
        """
        # 1. Setup a controlled config so we know exactly what is being requested
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
        mock_daily.Time.return_value = 1685577600  # June 1, 2023
        mock_daily.TimeEnd.return_value = 1685750400  # June 3, 2023
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
