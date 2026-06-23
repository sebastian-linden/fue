import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fue.config import Config


@pytest.fixture
def config():
    """Fixture to provide a Config instance for tests."""
    return Config()


@pytest.fixture
def temp_config_dir():
    """Fixture to create a temporary directory with a test config.json."""
    temp_dir = tempfile.mkdtemp()
    original_cwd = Path.cwd()

    # Create a minimal test config file
    test_config = {
        "cities": {
            "test_city1": {"lat": 50.0, "lon": 10.0},
            "test_city2": {"lat": 51.0, "lon": 11.0},
        },
        "daily": ["temperature_2m_max", "temperature_2m_min"],
        "timezone": "UTC",
        "past_days": 5,
        "forecast_days": 10,
    }

    config_path = Path(temp_dir) / "config.json"
    with open(config_path, "w") as f:
        json.dump(test_config, f)

    # Change to temp directory
    import os

    os.chdir(temp_dir)

    yield temp_dir

    # Cleanup
    os.chdir(original_cwd)
    shutil.rmtree(temp_dir)


class TestConfigInit:
    """Tests for Config initialization."""

    def test_config_loads_successfully(self, config):
        """Test that Config loads the config.json file successfully."""
        assert config.params is not None
        assert len(config.cities) > 0
        assert len(config.city_coordinates) > 0

    def test_config_extracts_cities(self, config):
        """Test that cities are extracted correctly."""
        assert isinstance(config.cities, list)
        assert "aachen" in config.cities
        assert "berlin" in config.cities

    def test_config_reformats_coordinates(self, config):
        """Test that coordinates are reformatted for Open-Meteo API."""
        assert "latitude" in config.params
        assert "longitude" in config.params
        assert len(config.params["latitude"]) == len(config.cities)
        assert len(config.params["longitude"]) == len(config.cities)
        assert all(isinstance(lat, (int, float)) for lat in config.params["latitude"])
        assert all(isinstance(lon, (int, float)) for lon in config.params["longitude"])

    def test_config_cities_removed_from_params(self, config):
        """Test that cities dict is removed from params."""
        assert "cities" not in config.params


class TestConfigRepr:
    """Tests for Config __repr__ method."""

    def test_repr_contains_cities(self, config):
        """Test that __repr__ includes city information."""
        repr_str = config.__repr__()
        assert "cities=" in repr_str
        assert "num_cities=" in repr_str

    def test_repr_contains_config_params(self, config):
        """Test that __repr__ includes configuration parameters."""
        repr_str = config.__repr__()
        assert "timezone=" in repr_str
        assert "past_days=" in repr_str
        assert "forecast_days=" in repr_str
        assert "daily_variables=" in repr_str


class TestSetters:
    """Tests for individual setter methods."""

    def test_set_timezone(self, config):
        """Test setting timezone."""
        config.set_timezone("America/New_York")
        assert config.params["timezone"] == "America/New_York"

    def test_set_past_days(self, config):
        """Test setting past_days."""
        config.set_past_days(14)
        assert config.params["past_days"] == 14

    def test_set_past_days_invalid_negative(self, config):
        """Test that negative past_days raises error."""
        with pytest.raises(ValueError):
            config.set_past_days(-1)

    def test_set_past_days_invalid_type(self, config):
        """Test that non-integer past_days raises error."""
        with pytest.raises(ValueError):
            config.set_past_days("5")

    def test_set_forecast_days(self, config):
        """Test setting forecast_days."""
        config.set_forecast_days(21)
        assert config.params["forecast_days"] == 21

    def test_set_forecast_days_invalid_negative(self, config):
        """Test that negative forecast_days raises error."""
        with pytest.raises(ValueError):
            config.set_forecast_days(-5)

    def test_set_daily_variables(self, config):
        """Test setting daily variables."""
        new_vars = ["temperature_2m_max", "wind_speed_10m_max"]
        config.set_daily_variables(new_vars)
        assert config.params["daily"] == new_vars

    def test_set_daily_variables_invalid_type(self, config):
        """Test that non-list daily variables raise error."""
        with pytest.raises(ValueError):
            config.set_daily_variables("temperature_2m_max")


class TestUpdate:
    """Tests for the update method."""

    def test_update_single_param(self, config):
        """Test updating a single parameter."""
        config.update(timezone="Europe/Paris")
        assert config.params["timezone"] == "Europe/Paris"

    def test_update_multiple_params(self, config):
        """Test updating multiple parameters at once."""
        config.update(timezone="UTC", past_days=10, forecast_days=20)
        assert config.params["timezone"] == "UTC"
        assert config.params["past_days"] == 10
        assert config.params["forecast_days"] == 20

    def test_update_invalid_parameter(self, config):
        """Test that invalid parameter raises error."""
        with pytest.raises(KeyError):
            config.update(invalid_param="value")

    def test_update_validates_parameters(self, config):
        """Test that update validates parameters."""
        with pytest.raises(ValueError):
            config.update(past_days=-5)


class TestAddCity:
    """Tests for add_city method."""

    def test_add_city_success(self, config):
        """Test adding a new city."""
        initial_count = len(config.cities)
        config.add_city("test_city", 45.0, 15.0)

        assert "test_city" in config.cities
        assert len(config.cities) == initial_count + 1
        assert config.city_coordinates["test_city"] == {"lat": 45.0, "lon": 15.0}
        assert 45.0 in config.params["latitude"]
        assert 15.0 in config.params["longitude"]

    def test_add_city_maintains_alignment(self, config):
        """Test that adding city maintains alignment of lists."""
        config.add_city("new_city", 48.0, 12.0)

        # Check that all lists have same length
        assert len(config.cities) == len(config.params["latitude"])
        assert len(config.cities) == len(config.params["longitude"])
        assert len(config.cities) == len(config.city_coordinates)

    def test_add_city_duplicate_error(self, config):
        """Test that adding duplicate city raises error."""
        config.add_city("duplicate", 50.0, 10.0)
        with pytest.raises(ValueError, match="already exists"):
            config.add_city("duplicate", 51.0, 11.0)

    def test_add_city_invalid_coordinates(self, config):
        """Test that invalid coordinates raise error."""
        with pytest.raises(ValueError):
            config.add_city("invalid", "not_a_number", 10.0)

        with pytest.raises(ValueError):
            config.add_city("invalid", 50.0, "not_a_number")


class TestRemoveCity:
    """Tests for remove_city method."""

    def test_remove_city_success(self, config):
        """Test removing a city."""
        city_to_remove = config.cities[0]
        initial_count = len(config.cities)

        config.remove_city(city_to_remove)

        assert city_to_remove not in config.cities
        assert len(config.cities) == initial_count - 1
        assert city_to_remove not in config.city_coordinates

    def test_remove_city_maintains_alignment(self, config):
        """Test that removing city maintains alignment of lists."""
        config.remove_city(config.cities[0])

        # Check that all lists have same length
        assert len(config.cities) == len(config.params["latitude"])
        assert len(config.cities) == len(config.params["longitude"])
        assert len(config.cities) == len(config.city_coordinates)

    def test_remove_city_nonexistent_error(self, config):
        """Test that removing nonexistent city raises error."""
        with pytest.raises(ValueError, match="not found"):
            config.remove_city("nonexistent_city")

    def test_remove_city_removes_from_coordinates(self, config):
        """Test that removing city also removes its coordinates."""
        city_to_remove = config.cities[0]
        config.remove_city(city_to_remove)

        assert city_to_remove not in config.city_coordinates
        # Check that latitude/longitude arrays no longer contain the city's coordinates
        if len(config.cities) > 0:
            # At least verify structure is consistent
            assert len(config.params["latitude"]) == len(config.cities)
            assert len(config.params["longitude"]) == len(config.cities)


class TestSave:
    def test_save_creates_valid_json(self, temp_config_dir):
        """Test that save creates a valid JSON file."""
        from fue.config import Config

        config = Config()
        # Retrieve existing settings
        try:
            prev_timezone = config.params["timezone"]
        except Exception as e:
            prev_timezone = "Europe/Berlin"
            print("Test Config Save(): ", e)
        # Change the time zone
        config.set_timezone("America/Los_Angeles")
        config.save()
        # Read the saved file and verify
        config = Config()
        # Test
        assert config.params["timezone"] == "America/Los_Angeles"
        # Reset timezone back to previous value
        config.set_timezone(prev_timezone)
        config.save()

    def test_save_can_be_reloaded(self, temp_config_dir):
        """Test that saved config can be reloaded."""
        from fue.config import Config

        config = Config()
        config.set_timezone("Europe/London")
        config.add_city("save_test", 51.5, -0.1)
        config.save()

        # Create new instance and verify changes were saved
        config2 = Config()
        assert config2.params["timezone"] == "Europe/London"
        assert "save_test" in config2.cities

        config.remove_city("save_test")
