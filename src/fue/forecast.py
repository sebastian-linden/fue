import matplotlib.pyplot as plt

from .config import Config
from .data import Data


class Forecast:
    """A class, that applies a fitted uncertainty model to a forecast, that can be either
    given or fetched from the Open-Meteo API via the OpenMeteoClient() class. This class also provides
    methods to visualize the forecast and its uncertainty predictions, if a model is given.
    """

    def __init__(self):
        self.forecast = None
        self.uncertainty_model = None
        self.uncertainty_predictions = None

    def fetch_forecast(self, location_name: str, forecast_days: int = 14) -> None:
        """Fetches the latest forecast for the given location and number of days directly from the Open-Meteo API.

        Args:
            location_name (str): The name of the location. Must be one of the cities defined in the configuration file.
            forecast_days (int, optional): The number of days to fetch. Defaults to 14.

        Returns:
            None: Sets the forecast DataFrame directly in the class instance.
        """

        # Create a custom Config() object and fetch data using that
        one_city_config = Config()
        if location_name not in one_city_config.cities:
            raise ValueError(f"Location '{location_name}' is not defined in the configuration file.")
        one_city_config.params["latitude"] = one_city_config.city_coordinates[location_name]["lat"]
        one_city_config.params["longitude"] = one_city_config.city_coordinates[location_name]["lon"]
        one_city_config.set_forecast_days(forecast_days)
        one_city_config.set_past_days(0)
        self.forecast = Data().fetch_forecast(config=one_city_config)

        # Here are two lines that are copied from the Data() class:
        self.forecast["day_of_year"] = self.forecast["forecast_for"].dt.day_of_year
        self.forecast["delta_days"] = (
            self.forecast["forecast_for"] - self.forecast["forecasted_on"]
        ).dt.total_seconds() / (60 * 60 * 24)

        return None

    def compute_uncertainties(self, uncertainty_model) -> None:
        """Applies a fitted uncertainty model to the forecast DataFrame to generate uncertainty predictions.

        Args:
            uncertainty_model: A fitted instance of an UncertaintyModel subclass.

        Returns:
            None: Sets the uncertainty predictions DataFrame directly in the class instance.
        """
        self.uncertainty_predictions = uncertainty_model.predict(self.forecast)
        return None

    def plot(self, target_variables: str | list) -> None:

        # Value and Type Checks
        if target_variables is None:
            raise ValueError("target_variables must be provided for plotting.")
        if self.forecast is None:
            raise ValueError("Forecast data has not been fetched. Call 'fetch_forecast()' first.")
        if self.uncertainty_predictions is None:
            raise ValueError("Uncertainty predictions have not been computed. Call 'compute_uncertainties()' first.")

        match target_variables:
            case str():
                target_variables = [target_variables]
            case list():
                pass
            case _:
                raise TypeError("target_variables must be either a string or a list of strings.")
        for var in target_variables:
            if not var.startswith("abs_diff__"):
                raise ValueError("Target variables need to start with 'abs_diff__'")

        # Plot specifications
        COLORS = {
            # Raw features
            "temperature_2m_max": "crimson",  # Warm red for maximum temperature
            "temperature_2m_min": "royalblue",  # Cool blue for minimum temperature
            "precipitation_sum": "teal",  # Greenish-blue for rain accumulation
            "sunshine_duration": "gold",  # Sunny yellow for hours of sun
            "wind_speed_10m_mean": "slategray",  # Muted slate for wind speed vectors
            "precipitation_probability_mean": "cornflowerblue",  # Soft blue for probability percentiles
            # Target variables
            "abs_diff__temperature_2m_max": "darkred",  # Deep red for max temperature inaccuracy
            "abs_diff__temperature_2m_min": "navy",  # Deep navy for min temperature inaccuracy
            "abs_diff__precipitation_sum": "darkcyan",  # Dark cyan for rain forecast error
            "abs_diff__sunshine_duration": "darkorange",  # Rich orange for sunshine duration error
            "abs_diff__wind_speed_10m_mean": "dimgray",  # Dark neutral gray for wind speed error
            "abs_diff__precipitation_probability_mean": "mediumpurple",  # Distinct purple accent for probability error
        }
        UNITS = {
            # Raw features
            "temperature_2m_max": "°C",
            "temperature_2m_min": "°C",
            "precipitation_sum": "mm",
            "sunshine_duration": "s",  # Seconds (or 'h' if parsed to hours)
            "wind_speed_10m_mean": "m/s",  # Standard meteorological velocity
            "precipitation_probability_mean": "%",
            # Target variables (same units as raw features)
            "abs_diff__temperature_2m_max": "°C",
            "abs_diff__temperature_2m_min": "°C",
            "abs_diff__precipitation_sum": "mm",
            "abs_diff__sunshine_duration": "s",
            "abs_diff__wind_speed_10m_mean": "m/s",
            "abs_diff__precipitation_probability_mean": "%",
        }
        TITLES = {
            # Raw features
            "temperature_2m_max": "Maximum Temperature (2m)",
            "temperature_2m_min": "Minimum Temperature (2m)",
            "precipitation_sum": "Total Daily Precipitation",
            "sunshine_duration": "Daily Sunshine Duration",
            "wind_speed_10m_mean": "Mean Wind Speed (10m)",
            "precipitation_probability_mean": "Mean Precipitation Probability",
            # Target variables
            "abs_diff__temperature_2m_max": "Max Temperature Forecast Absolute Error",
            "abs_diff__temperature_2m_min": "Min Temperature Forecast Absolute Error",
            "abs_diff__precipitation_sum": "Precipitation Forecast Absolute Error",
            "abs_diff__sunshine_duration": "Sunshine Duration Forecast Absolute Error",
            "abs_diff__wind_speed_10m_mean": "Mean Wind Speed Forecast Absolute Error",
            "abs_diff__precipitation_probability_mean": "Precipitation Probability Forecast Absolute Error",
        }
        days = self.forecast["forecast_for"].dt.strftime("%a, %d-%m-%Y")

        for abs_diff__var in target_variables:
            var = abs_diff__var[10:]  # "abs_diff__" has 10 letters
            if var not in self.forecast.columns:
                raise ValueError(f"Target variable '{var}' is not present in the forecast DataFrame.")
            if abs_diff__var not in self.uncertainty_predictions.columns:
                raise ValueError(f"Uncertainty predictions for target variable '{var}' are not available.")

            var_series = self.forecast[var]
            var_diff = self.uncertainty_predictions[abs_diff__var]
            plt.plot(days, var_series, color=COLORS[var], label=TITLES[var])
            plt.fill_between(
                x=days,
                y1=var_series - var_diff,
                y2=var_series + var_diff,
                alpha=0.2,
                color=COLORS[abs_diff__var],
                label=TITLES[abs_diff__var],
            )
            plt.grid(True)
            plt.xticks(rotation=90)
            plt.title(TITLES[var])
            plt.ylabel(f"{TITLES[var]} in {UNITS[var]}")
            plt.legend()
            plt.tight_layout()
            plt.show()
