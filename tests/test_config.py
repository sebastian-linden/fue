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
    temp_dir = Path(tempfile.mkdtemp())

    # Build a specific test config file that includes baseline preprocessing configurations
    test_config = {
        "preprocessing": {"precipitation": "log", "humidity": "min-max"},
        "cities": {
            "test_city1": {"lat": 50.0, "lon": 10.0},
            "test_city2": {"lat": 51.0, "lon": 11.0},
        },
        "daily": ["temperature_2m_max", "temperature_2m_min"],
        "timezone": "UTC",
        "past_days": 5,
        "forecast_days": 10,
    }

    config_path = temp_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(test_config, f)

    # Hand the Path object containing our isolated file over to the test context
    yield config_path

    # Clean up the isolated test sandbox directory entirely after execution
    shutil.rmtree(temp_dir)


class TestConfigInit:
    """Tests for Config initialization using isolated configuration files."""

    def test_config_loads_successfully(self, temp_config_dir):
        """Test that Config loads the config.json file successfully."""
        from fue.config import Config

        config = Config(path=temp_config_dir)

        assert config.params is not None
        assert len(config.cities) > 0
        assert len(config.city_coordinates) > 0

    def test_config_extracts_cities(self, temp_config_dir):
        """Test that cities defined in the fixture are extracted correctly."""
        from fue.config import Config

        config = Config(path=temp_config_dir)

        assert isinstance(config.cities, list)
        assert "test_city1" in config.cities
        assert "test_city2" in config.cities

    def test_config_reformats_coordinates(self, temp_config_dir):
        """Test that coordinates are reformatted for Open-Meteo API."""
        from fue.config import Config

        config = Config(path=temp_config_dir)

        assert "latitude" in config.params
        assert "longitude" in config.params
        assert len(config.params["latitude"]) == len(config.cities)
        assert len(config.params["longitude"]) == len(config.cities)
        assert all(isinstance(lat, (int, float)) for lat in config.params["latitude"])
        assert all(isinstance(lon, (int, float)) for lon in config.params["longitude"])

    def test_config_cities_removed_from_params(self, temp_config_dir):
        """Test that the 'cities' key is removed from params after initialization."""
        from fue.config import Config

        config = Config(path=temp_config_dir)

        assert "cities" not in config.params


class TestConfigRepr:
    """Tests for Config __repr__ method using isolated configuration files."""

    def test_repr_contains_cities(self, temp_config_dir):
        """Test that __repr__ includes city information."""
        from fue.config import Config

        config = Config(path=temp_config_dir)

        repr_str = config.__repr__()
        assert "cities=" in repr_str
        assert "num_cities=" in repr_str

    def test_repr_contains_config_params(self, temp_config_dir):
        """Test that __repr__ includes configuration parameters."""
        from fue.config import Config

        config = Config(path=temp_config_dir)

        repr_str = config.__repr__()
        assert "timezone=" in repr_str
        assert "past_days=" in repr_str
        assert "forecast_days=" in repr_str
        assert "daily_variables=" in repr_str


class TestSetters:
    """Tests for individual setter methods using isolated configuration files."""

    def test_set_timezone(self, temp_config_dir):
        """Test setting timezone."""
        from fue.config import Config

        config = Config(path=temp_config_dir)

        config.set_timezone("America/New_York")
        assert config.params["timezone"] == "America/New_York"

    def test_set_past_days(self, temp_config_dir):
        """Test setting past_days."""
        from fue.config import Config

        config = Config(path=temp_config_dir)

        config.set_past_days(14)
        assert config.params["past_days"] == 14

    def test_set_past_days_invalid_negative(self, temp_config_dir):
        """Test that negative past_days raises error."""
        from fue.config import Config

        config = Config(path=temp_config_dir)

        with pytest.raises(ValueError, match="must be a non-negative integer"):
            config.set_past_days(-1)

    def test_set_past_days_invalid_type(self, temp_config_dir):
        """Test that non-integer past_days raises error."""
        from fue.config import Config

        config = Config(path=temp_config_dir)

        with pytest.raises(ValueError, match="must be a non-negative integer"):
            config.set_past_days("5")  # ty: ignore[invalid-argument-type]

    def test_set_forecast_days(self, temp_config_dir):
        """Test setting forecast_days."""
        from fue.config import Config

        config = Config(path=temp_config_dir)

        config.set_forecast_days(21)
        assert config.params["forecast_days"] == 21

    def test_set_forecast_days_invalid_negative(self, temp_config_dir):
        """Test that negative forecast_days raises error."""
        from fue.config import Config

        config = Config(path=temp_config_dir)

        with pytest.raises(ValueError, match="must be a non-negative integer"):
            config.set_forecast_days(-5)

    def test_set_daily_variables(self, temp_config_dir):
        """Test setting daily variables."""
        from fue.config import Config

        config = Config(path=temp_config_dir)

        new_vars = ["temperature_2m_max", "wind_speed_10m_max"]
        config.set_daily_variables(new_vars)
        assert config.params["daily"] == new_vars

    def test_set_daily_variables_invalid_type(self, temp_config_dir):
        """Test that non-list daily variables raise error."""
        from fue.config import Config

        config = Config(path=temp_config_dir)

        with pytest.raises(ValueError, match="must be a list"):
            config.set_daily_variables("temperature_2m_max")  # ty: ignore[invalid-argument-type]


class TestUpdate:
    """Tests for the update method using isolated configuration files."""

    def test_update_single_param(self, temp_config_dir):
        """Test updating a single parameter."""
        from fue.config import Config

        config = Config(path=temp_config_dir)

        config.update(timezone="Europe/Paris")
        assert config.params["timezone"] == "Europe/Paris"

    def test_update_multiple_params(self, temp_config_dir):
        """Test updating multiple parameters at once."""
        from fue.config import Config

        config = Config(path=temp_config_dir)

        config.update(timezone="UTC", past_days=10, forecast_days=20)
        assert config.params["timezone"] == "UTC"
        assert config.params["past_days"] == 10
        assert config.params["forecast_days"] == 20

    def test_update_invalid_parameter(self, temp_config_dir):
        """Test that an invalid parameter name raises a KeyError."""
        from fue.config import Config

        config = Config(path=temp_config_dir)

        with pytest.raises(KeyError, match="Unknown configuration parameter"):
            config.update(invalid_param="value")

    def test_update_validates_parameters(self, temp_config_dir):
        """Test that update correctly triggers validation logic."""
        from fue.config import Config

        config = Config(path=temp_config_dir)

        with pytest.raises(ValueError, match="must be a non-negative integer"):
            config.update(past_days=-5)


class TestAddCity:
    """Tests for add_city method using isolated configuration files."""

    def test_add_city_success(self, temp_config_dir):
        """Test adding a new city to an isolated config."""
        from fue.config import Config

        config = Config(path=temp_config_dir)

        initial_count = len(config.cities)
        config.add_city("test_city", 45.0, 15.0)

        assert "test_city" in config.cities
        assert len(config.cities) == initial_count + 1
        assert config.city_coordinates["test_city"] == {"lat": 45.0, "lon": 15.0}
        assert 45.0 in config.params["latitude"]
        assert 15.0 in config.params["longitude"]

    def test_add_city_maintains_alignment(self, temp_config_dir):
        """Test that adding a city maintains alignment of internal data structures."""
        from fue.config import Config

        config = Config(path=temp_config_dir)

        config.add_city("new_city", 48.0, 12.0)

        # Check that all data structures are updated in unison
        assert len(config.cities) == len(config.params["latitude"])
        assert len(config.cities) == len(config.params["longitude"])
        assert len(config.cities) == len(config.city_coordinates)

    def test_add_city_duplicate_error(self, temp_config_dir):
        """Test that adding a duplicate city name raises a ValueError."""
        from fue.config import Config

        config = Config(path=temp_config_dir)

        config.add_city("duplicate", 50.0, 10.0)
        with pytest.raises(ValueError, match="already exists"):
            config.add_city("duplicate", 51.0, 11.0)

    def test_add_city_invalid_coordinates(self, temp_config_dir):
        """Test that non-numeric coordinates raise a ValueError."""
        from fue.config import Config

        config = Config(path=temp_config_dir)

        with pytest.raises(ValueError):
            config.add_city("invalid", 50.0, 4.0)


class TestRemoveCity:
    """Tests for remove_city method using isolated configuration files."""

    def test_remove_city_success(self, temp_config_dir):
        """Test removing a city from an isolated config."""
        from fue.config import Config

        config = Config(path=temp_config_dir)

        city_to_remove = config.cities[0]
        initial_count = len(config.cities)

        config.remove_city(city_to_remove)

        assert city_to_remove not in config.cities
        assert len(config.cities) == initial_count - 1
        assert city_to_remove not in config.city_coordinates

    def test_remove_city_maintains_alignment(self, temp_config_dir):
        """Test that removing city maintains alignment of internal data structures."""
        from fue.config import Config

        config = Config(path=temp_config_dir)

        config.remove_city(config.cities[0])

        assert len(config.cities) == len(config.params["latitude"])
        assert len(config.cities) == len(config.params["longitude"])
        assert len(config.cities) == len(config.city_coordinates)

    def test_remove_city_nonexistent_error(self, temp_config_dir):
        """Test that removing nonexistent city raises a ValueError."""
        from fue.config import Config

        config = Config(path=temp_config_dir)

        with pytest.raises(ValueError, match="not found"):
            config.remove_city("nonexistent_city")

    def test_remove_city_removes_from_coordinates(self, temp_config_dir):
        """Test that removing city also correctly cleans up coordinates arrays."""
        from fue.config import Config

        config = Config(path=temp_config_dir)

        city_to_remove = config.cities[0]
        config.remove_city(city_to_remove)

        assert city_to_remove not in config.city_coordinates
        # Validate that the coordinate lists were shrunk to match the cities list
        assert len(config.params["latitude"]) == len(config.cities)
        assert len(config.params["longitude"]) == len(config.cities)


class TestSave:
    def test_save_creates_valid_json(self, temp_config_dir):
        """Test that save creates a valid JSON file using the temporary path."""
        from fue.config import Config

        # Initialize with the temporary config path provided by the fixture
        config = Config(path=temp_config_dir)

        # Change the time zone
        config.set_timezone("America/Los_Angeles")
        config.save()

        # Reload from the same temporary path and verify
        config_reloaded = Config(path=temp_config_dir)
        assert config_reloaded.params["timezone"] == "America/Los_Angeles"

        # No reset logic needed; the temp_config_dir fixture will delete
        # this file entirely once the test finishes.

    def test_save_can_be_reloaded(self, temp_config_dir):
        """Test that saved config additions can be reloaded."""
        from fue.config import Config

        # Initialize with the temporary config path
        config = Config(path=temp_config_dir)
        config.set_timezone("Europe/London")
        config.add_city("save_test", 51.5, -0.1)
        config.save()

        # Create new instance referencing the same temporary path
        config2 = Config(path=temp_config_dir)
        assert config2.params["timezone"] == "Europe/London"
        assert "save_test" in config2.cities


class TestPreprocessing:
    def test_preprocessing_rule_save_and_load(self, temp_config_dir):
        """Test that preprocessing rules are saved and loaded correctly in isolation."""
        from fue.config import Config

        # Load the configuration from our isolated sandbox path
        config = Config(path=temp_config_dir)

        # Test targets are dynamically loaded from our temporary file definition
        metrics = list(config.get_preprocessing_rules().keys())
        test_metric = metrics[0] if metrics else "temperature_2m_max"

        # Mutate and save modifications directly into the temporary workspace file
        config.set_preprocessing_rule(test_metric, "standardize")
        config.save()

        # Reload a separate configuration instance referencing the same temporary path
        config2 = Config(path=temp_config_dir)
        assert config2.preprocessing.get(test_metric) == "standardize"
