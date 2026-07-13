"""
Configuration management system for the fue package.

This module handles reading and writing the 'config.json' settings file. It keeps
track of the cities we want to download, API keys, timezones, and the specific
rules used to clean and transform our data before training.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Config:
    """
    Manages reading, modifying, and saving the project's settings.

    This class loads settings from config.json, structures city coordinates for
    the weather API client, and provides simple methods to add or remove cities,
    change timezones, or adjust forecasting time windows.
    """

    def __init__(self, path: str | Path | None = None):
        """
        Loads the config file and sets up internal variables for cities and parameters.

        It parses the JSON file, extracts city lists into a separate tracking variable,
        and extracts lists of latitudes and longitudes so that they match what the
        Open-Meteo API expects.

        Parameters
        ----------
        path : str or pathlib.Path or None, default=None
            The file path to config.json. If None, it automatically finds the file
            by looking up from the folder where this code is running.

        Raises
        ------
        FileNotFoundError
            If config.json cannot be found at the default or specified path.
        """

        self.params = {}

        # Use __file__ to create absolute path to config.json in project root
        # If a custom path is given, use it; otherwise, fall back to default project root
        if path is not None:
            self.path = Path(path)
        else:
            self.path = Path(__file__).parent.parent.parent / "config.json"

        if not self.path.exists():
            raise FileNotFoundError(f"The file 'config.json' was not found at: {self.path}")
        else:
            with open(self.path) as file:
                self.params = json.load(file)
            # Reformat dictionary to suit the format that Open-Meteo needs
            self.city_coordinates = self.params.get("cities", {})
            self.params["latitude"] = [item["lat"] for item in self.city_coordinates.values()]
            self.params["longitude"] = [item["lon"] for item in self.city_coordinates.values()]

            # Pull out some of the configuration variables into separate variables and delete
            # them from the self.params dictionary, as this is the one, that will be passed to
            # Open-Meteo API
            if "cities" in self.params:
                self.cities = list(self.city_coordinates.keys())
                del self.params["cities"]

            if "preprocessing" in self.params:
                self.preprocessing = self.params.get("preprocessing", {})
                del self.params["preprocessing"]
            else:
                self.preprocessing = {}

            if "default_feature_columns" in self.params:
                self.default_feature_columns = self.params.get("default_feature_columns", [])
                del self.params["default_feature_columns"]
            if "default_target_columns" in self.params:
                self.default_target_columns = self.params.get("default_target_columns", [])
                del self.params["default_target_columns"]

        logger.info(f"Configuration loaded from {self.path}")

    def __repr__(self):
        """
        Creates a clean summary text block showing the current state of the config.

        Returns
        -------
        str
            A multi-line text representation showing the loaded cities, timezone,
            past/forecast days, and active variables.
        """
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
        """
        Updates the target timezone parameter in our settings.

        Parameters
        ----------
        timezone : str
            A standard timezone name string (for example, 'Europe/Berlin').

        Returns
        -------
        None
        """
        self.params["timezone"] = timezone
        logger.info(f"Timezone set to {timezone}")

    def set_past_days(self, past_days: int) -> None:
        """
        Sets how many days of past historical weather data we want to request.

        Parameters
        ----------
        past_days : int
            The number of history days to request. Must be 0 or a positive integer.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the value provided is negative or not a whole number.
        """
        if not isinstance(past_days, int) or past_days < 0:
            raise ValueError(f"past_days must be a non-negative integer, got {past_days}")
        self.params["past_days"] = past_days
        logger.info(f"Past days set to {past_days}")

    def set_forecast_days(self, forecast_days: int) -> None:
        """
        Sets how many days into the future our weather forecasts should reach.

        Parameters
        ----------
        forecast_days : int
            The length of the forecast window in days. Must be 0 or a positive integer.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the value provided is negative or not a whole number.
        """
        if not isinstance(forecast_days, int) or forecast_days < 0:
            raise ValueError(f"forecast_days must be a non-negative integer, got {forecast_days}")
        self.params["forecast_days"] = forecast_days
        logger.info(f"Forecast days set to {forecast_days}")

    def set_daily_variables(self, daily: list) -> None:
        """
        Updates the list of specific weather variables we want to pull from the API.

        Parameters
        ----------
        daily : list
            A list of weather variable strings (like `['temperature_2m_max', 'precipitation_sum']`).

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the provided value is not structured as a Python list.
        """
        if not isinstance(daily, list):
            raise ValueError(f"daily must be a list, got {type(daily)}")
        self.params["daily"] = daily
        logger.info(f"Daily variables set to {daily}")

    def update(self, **kwargs) -> None:
        """
        Updates multiple configuration parameters at the same time.

        Parameters
        ----------
        **kwargs : dict
            Named arguments matching existing parameters (`timezone`, `past_days`,
            `forecast_days`, or `daily`).

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If you pass a setting name that the configuration class doesn't recognize.
        """
        # No logger is needed, because each individual setter method
        # already logs the changes made to the configuration.
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
        """
        Adds a new city and its coordinates to our tracking list.

        This method verifies that the city doesn't already exist and that coordinates
        are valid numbers before appending them to the configuration lists.

        Parameters
        ----------
        name : str
            The name of the city (e.g., 'aachen'). Must be unique.
        lat : float
            The latitude value of the city location.
        lon : float
            The longitude value of the city location.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the city name already exists in the current settings.
        TypeError
            If the city name is not a string, or if coordinates are not numbers.
        """
        if name in self.cities:
            raise ValueError(f"City '{name}' already exists in configuration")
        if not isinstance(name, str):
            raise TypeError("City name must be a string")
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            raise TypeError("Latitude and longitude must be numeric values")

        self.cities.append(name)
        self.city_coordinates[name] = {"lat": lat, "lon": lon}
        self.params["latitude"].append(lat)
        self.params["longitude"].append(lon)
        logger.info(f"City '{name}' added with coordinates (lat: {lat}, lon: {lon})")

    def remove_city(self, name: str) -> None:
        """
        Removes a city and its coordinates from our tracking list.

        It finds the position of the city in our internal variables and removes it
        completely so that our parallel list dimensions stay lined up.

        Parameters
        ----------
        name : str
            The name string of the city to delete.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the specified city name cannot be found in our settings.
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
        logger.info(f"City '{name}' removed from configuration.")

    def save(self) -> None:
        """
        Writes all current setting variables back into our config.json file on disk.

        This re-assembles the dictionary elements, strips out temporary processing lists
        like latitudes and longitudes, and saves a clean formatted JSON file.

        Returns
        -------
        None
        """
        config_data = self.params
        config_data["cities"] = self.city_coordinates
        config_data["preprocessing"] = self.preprocessing
        config_data["default_feature_columns"] = self.default_feature_columns
        config_data["default_target_columns"] = self.default_target_columns
        del config_data["latitude"]
        del config_data["longitude"]
        with open(self.path, "w") as file:
            json.dump(config_data, file, indent=4)
        logger.info(f"Configuration saved to {self.path}")

    def get_preprocessing_rules(self) -> dict:
        """
        Returns the data transformation rules currently configured in our settings.

        Returns
        -------
        dict
            A dictionary mapping weather variables directly to their transformation
            methods (for example: `{'abs_diff__precipitation_sum': 'log'}`).
        """
        return self.preprocessing

    def set_preprocessing_rule(self, variable: str, method: str) -> None:
        """
        Sets or changes a data cleaning/transformation rule for a specific weather variable.

        Parameters
        ----------
        variable : str
            The specific weather column or target variable name.
        method : str
            The name of the scaling method to apply (such as 'min-max', 'standard', or 'log').

        Returns
        -------
        None
        """
        self.preprocessing[variable] = method
        logger.info(f"Preprocessing rule set for '{variable}' to '{method}'")
        return None


if __name__ == "__main__":
    config = Config()
    print(config.__repr__)
