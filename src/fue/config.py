import json
from pathlib import Path


class Config:
    """This class handles writing to and reading from the configuration file.
    It also implements methods to interact with the configuration.
    """

    def __init__(self):
        """Initializes parameter dictionary and reads from config.json file"""

        self.params = {}
        # Use __file__ to create absolute path to config.json in project root
        self.path = Path(__file__).parent.parent.parent / "config.json"
        if not self.path.exists():
            raise FileNotFoundError(f"The file 'config.json' was not found at: {self.path}")
        else:
            with open(self.path) as file:
                self.params = json.load(file)
            # Reformat dictionary to suit the format that Open-Meteo needs
            self.city_coordinates = self.params["cities"]
            self.cities = list(self.city_coordinates.keys())
            del self.params["cities"]
            self.params["latitude"] = [item["lat"] for item in self.city_coordinates.values()]
            self.params["longitude"] = [item["lon"] for item in self.city_coordinates.values()]

    def __repr__(self):
        return (
            f"Config(\n"
            f"  cities={self.cities},\n"
            f"  num_cities={len(self.cities)},\n"
            f"  timezone={self.params.get('timezone')},\n"
            f"  past_days={self.params.get('past_days')},\n"
            f"  forecast_days={self.params.get('forecast_days')},\n"
            f"  daily_variables={len(self.params.get('daily', []))}\n"
            f")"
        )

    def set_timezone(self, timezone: str) -> None:
        """Set the timezone parameter.

        Args:
            timezone: Timezone string (e.g., 'Europe/Berlin')
        """
        self.params["timezone"] = timezone

    def set_past_days(self, past_days: int) -> None:
        """Set the number of past days for historical data.

        Args:
            past_days: Number of past days (must be non-negative)
        """
        if not isinstance(past_days, int) or past_days < 0:
            raise ValueError(f"past_days must be a non-negative integer, got {past_days}")
        self.params["past_days"] = past_days

    def set_forecast_days(self, forecast_days: int) -> None:
        """Set the number of forecast days.

        Args:
            forecast_days: Number of forecast days (must be non-negative)
        """
        if not isinstance(forecast_days, int) or forecast_days < 0:
            raise ValueError(f"forecast_days must be a non-negative integer, got {forecast_days}")
        self.params["forecast_days"] = forecast_days

    def set_daily_variables(self, daily: list) -> None:
        """Set the daily weather variables to fetch.

        Args:
            daily: List of variable names to fetch
        """
        if not isinstance(daily, list):
            raise ValueError(f"daily must be a list, got {type(daily)}")
        self.params["daily"] = daily

    def update(self, **kwargs) -> None:
        """Update multiple configuration parameters at once.

        Args:
            timezone: Timezone string
            past_days: Number of past days
            forecast_days: Number of forecast days
            daily: List of daily variables
        """
        for key, value in kwargs.items():
            if key == "timezone":
                self.set_timezone(value)
            elif key == "past_days":
                self.set_past_days(value)
            elif key == "forecast_days":
                self.set_forecast_days(value)
            elif key == "daily":
                self.set_daily_variables(value)
            else:
                raise KeyError(f"Unknown configuration parameter: {key}")

    def add_city(self, name: str, lat: float, lon: float) -> None:
        """Add a new city to the configuration.

        Args:
            name: City name (must be unique)
            lat: Latitude coordinate
            lon: Longitude coordinate
        """
        if name in self.cities:
            raise ValueError(f"City '{name}' already exists in configuration")
        if not isinstance(name, str):
            raise ValueError("City name must be a string")
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            raise ValueError("Latitude and longitude must be numeric values")

        self.cities.append(name)
        self.city_coordinates[name] = {"lat": lat, "lon": lon}
        self.params["latitude"].append(lat)
        self.params["longitude"].append(lon)

    def remove_city(self, name: str) -> None:
        """Remove a city from the configuration.

        Args:
            name: City name to remove
        """
        if name not in self.cities:
            raise ValueError(f"City '{name}' not found in configuration")

        # Find the index of the city
        index = self.cities.index(name)

        # Remove from all data structures
        self.cities.remove(name)
        del self.city_coordinates[name]
        self.params["latitude"].pop(index)
        self.params["longitude"].pop(index)

    def save(self) -> None:
        """Save the current configuration back to config.json."""
        config_data = {"cities": self.city_coordinates, **self.params}
        del config_data["latitude"]
        del config_data["longitude"]
        with open(self.path, "w") as file:
            json.dump(config_data, file, indent=4)


if __name__ == "__main__":
    config = Config()
    print(config.__repr__)
