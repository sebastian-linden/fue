"""
Forecast inference and visualization pipeline for the pyfue package.

This module maps fitted machine learning and linear uncertainty estimators onto
freshly pulled weather forecasts. It applies physical boundary safeguards (such as
clipping metrics at zero or capping sunshine hours via astronomical day length)
and renders localized error-bound graphs.
"""

import logging
import warnings

import matplotlib.pyplot as plt

from .config import Config
from .data import Data
from .utils import daylength

logger = logging.getLogger(__name__)


class Forecast:
    """
    Applies trained error-prediction models to weather forecasts and handles plotting.

    This class tracks live point predictions alongside their calculated error ranges,
    manages lookup dictionaries for plot labels, units, and colors, and overlays
    shaded confidence bands ($\\pm \text{error}$) on meteorological lines.
    """

    # Class-level plot specifications for known variables
    _KNOWN_COLORS = {
        # Raw features
        "temperature_2m_max": "crimson",
        "temperature_2m_min": "royalblue",
        "precipitation_sum": "teal",
        "sunshine_duration": "gold",
        "wind_speed_10m_mean": "slategray",
        "precipitation_probability_mean": "cornflowerblue",
        # Target variables
        "abs_diff__temperature_2m_max": "darkred",
        "abs_diff__temperature_2m_min": "navy",
        "abs_diff__precipitation_sum": "darkcyan",
        "abs_diff__sunshine_duration": "darkorange",
        "abs_diff__wind_speed_10m_mean": "dimgray",
        "abs_diff__precipitation_probability_mean": "mediumpurple",
    }
    _KNOWN_UNITS = {
        # Raw features
        "temperature_2m_max": "°C",
        "temperature_2m_min": "°C",
        "precipitation_sum": "mm",
        "sunshine_duration": "h",
        "wind_speed_10m_mean": "m/s",
        "precipitation_probability_mean": "%",
        # Target variables
        "abs_diff__temperature_2m_max": "°C",
        "abs_diff__temperature_2m_min": "°C",
        "abs_diff__precipitation_sum": "mm",
        "abs_diff__sunshine_duration": "h",
        "abs_diff__wind_speed_10m_mean": "m/s",
        "abs_diff__precipitation_probability_mean": "%",
    }
    _KNOWN_TITLES = {
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

    def __init__(self) -> None:
        """
        Initializes an empty forecast profile container with default tracking states.
        """
        self.forecast = None
        self.uncertainty_model = None
        self.uncertainty_predictions = None
        self.past_days = None

    def fetch_forecast(self, location_name: str, forecast_days: int = 14, past_days: int = 0) -> None:
        """
        Downloads a fresh forecast profile for a single city from the data client layer.

        Validates user-supplied timing bounds, creates a localized temporary city config,
        fetches raw predictive rows, and appends structural index features such as
        seasonal times and ahead-horizon day fractions.

        Parameters
        ----------
        location_name : str
            The name string matching a tracked target city in our coordinates configurations.
        forecast_days : int, default=14
            The total look-ahead horizon window length to download, specified in full days.
        past_days : int, default=0
            The number of back-dated historical timeline records to include alongside the forecast.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If any passed argument fails basic validation check rules for matching data types.
        ValueError
            If timing bounds fall into negative counts or if the city name is not registered.
        """
        # Input validation
        if not isinstance(location_name, str):
            raise TypeError(f"location_name must be a string, got {type(location_name).__name__}")
        if not isinstance(forecast_days, int):
            raise TypeError(f"forecast_days must be an integer, got {type(forecast_days).__name__}")
        if not isinstance(past_days, int):
            raise TypeError(f"past_days must be an integer, got {type(past_days).__name__}")
        if forecast_days <= 0:
            raise ValueError(f"forecast_days must be > 0, got {forecast_days}")
        if past_days < 0:
            raise ValueError(f"past_days must be >= 0, got {past_days}")

        logger.info("Fetching forecast for %s", location_name)

        self.past_days = past_days
        # Create a custom Config() object and fetch data using that
        one_city_config = Config()
        if location_name not in one_city_config.cities:
            available_cities = ", ".join(sorted(one_city_config.cities))
            raise ValueError(
                f"Location '{location_name}' is not defined in the configuration file. "
                f"Available cities: {available_cities}"
            )
        one_city_config.params["latitude"] = one_city_config.city_coordinates[location_name]["lat"]
        one_city_config.params["longitude"] = one_city_config.city_coordinates[location_name]["lon"]
        one_city_config.set_forecast_days(forecast_days)
        one_city_config.set_past_days(self.past_days)
        self.forecast = Data().fetch_forecast(config=one_city_config)

        # Validate that forecast has required columns
        required_columns = {"forecast_for", "forecasted_on"}
        missing_columns = required_columns - set(self.forecast.columns)
        if missing_columns:
            raise ValueError(f"Forecast DataFrame missing required columns: {missing_columns}")

        # Here are two lines that are copied from the Data() class:
        self.forecast["day_of_year"] = self.forecast["forecast_for"].dt.day_of_year
        self.forecast["delta_days"] = (
            self.forecast["forecast_for"] - self.forecast["forecasted_on"]
        ).dt.total_seconds() / (60 * 60 * 24)

        logger.debug(
            "Forecast fetched with %d rows and %d columns",
            len(self.forecast),
            len(self.forecast.columns),
        )

        return None

    def compute_uncertainties(self, uncertainty_model) -> None:
        """
        Runs the active weather forecast through our estimator to predict absolute error margins.

        Checks that our forecast lines are loaded and that the selected model possesses
        a callable predict method before populating our data frame arrays with raw calculated deviations.

        Parameters
        ----------
        uncertainty_model : object
            A statefully trained uncertainty estimator model subclass instance containing a `.predict()` method.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If forecast rows haven't been pulled yet, or if the model predictions come back empty.
        AttributeError
            If the supplied model lacks a callable predict function interface.
        """
        # Validate state
        if self.forecast is None:
            raise ValueError("Forecast data has not been fetched. Call 'fetch_forecast()' first.")

        # Validate uncertainty_model
        if uncertainty_model is None:
            raise ValueError("uncertainty_model cannot be None")
        if not hasattr(uncertainty_model, "predict"):
            raise AttributeError(
                f"uncertainty_model must have a 'predict' method. Got {type(uncertainty_model).__name__}"
            )
        if not callable(uncertainty_model.predict):
            raise AttributeError(
                f"uncertainty_model.predict must be callable. Got {type(uncertainty_model.predict).__name__}"
            )

        logger.info("Computing uncertainty predictions")
        self.uncertainty_model = uncertainty_model
        self.uncertainty_predictions = uncertainty_model.predict(self.forecast)

        # Validate output
        if self.uncertainty_predictions is None:
            raise ValueError("uncertainty_model.predict() returned None")

        logger.debug("Uncertainty predictions computed with shape %s", self.uncertainty_predictions.shape)

        return None

    def _get_plot_spec(self, variable: str, spec_type: str) -> str:
        """
        Looks up visual properties like color shades, labels, or units for a specific weather type.

        If a column header is completely unknown, it defaults to a neutral fallback state
        and triggers a clear terminal log alert so your pipeline doesn't crash during presentation.

        Parameters
        ----------
        variable : str
            The column header dictionary key name to evaluate (e.g., 'temperature_2m_max').
        spec_type : str
            The property layout block to open. Must be exactly 'color', 'unit', or 'title'.

        Returns
        -------
        str
            The text string holding the matched hex name, metric character, or title text block.

        Raises
        ------
        ValueError
            If an invalid configuration string type token is requested.
        """
        if spec_type == "color":
            specs = self._KNOWN_COLORS
        elif spec_type == "unit":
            specs = self._KNOWN_UNITS
        elif spec_type == "title":
            specs = self._KNOWN_TITLES
        else:
            raise ValueError(f"Invalid spec_type: {spec_type}. Must be 'color', 'unit', or 'title'")

        if variable in specs:
            return specs[variable]
        else:
            # Unknown variable - warn but provide sensible default
            warnings.warn(
                f"Variable '{variable}' is not in predefined {spec_type}s. \
                    Using default value. Known variables: {sorted(specs.keys())}",
                UserWarning,
                stacklevel=3,
            )
            logger.warning("Unknown variable '%s' for %s", variable, spec_type)

            # Provide sensible defaults for unknown variables
            if spec_type == "color":
                return "gray"
            elif spec_type == "unit":
                return "[unknown]"
            else:  # title
                return variable

    def plot(self, target_variables: str | list) -> None:
        """
        Renders line plots showing forecast tracks with shaded error ranges.

        Extracts matching data paths for each requested column, shifts back-dated records
        down to zero variance bounds, and builds upper and lower bounds. It also
        clips values at zero for non-negative tracks and implements physics-based caps on
        sunshine parameters based on geographical day lengths.

        Parameters
        ----------
        target_variables : str or list of str
            A single metric key or group list of error keys to draw (must begin with 'abs_diff__').

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If data elements are uninitialized or missing, or if variable string layouts match poorly.
        TypeError
            If target inputs use container forms other than strings or string lists.
        """
        # Value and Type Checks
        if target_variables is None:
            raise ValueError("target_variables must be provided for plotting.")
        if self.forecast is None:
            raise ValueError("Forecast data has not been fetched. Call 'fetch_forecast()' first.")
        if self.uncertainty_predictions is None:
            raise ValueError("Uncertainty predictions have not been computed. Call 'compute_uncertainties()' first.")
        if self.past_days is None:
            raise ValueError("past_days is not set. This should have been set during fetch_forecast().")

        match target_variables:
            case str():
                target_variables = [target_variables]
            case list():
                if len(target_variables) == 0:
                    raise ValueError("target_variables list cannot be empty")
                pass
            case _:
                raise TypeError("target_variables must be either a string or a list of strings.")

        # Validate that all variables start with 'abs_diff__'
        for var in target_variables:
            if not isinstance(var, str):
                raise TypeError(f"All target_variables must be strings, got {type(var).__name__}")
            if not var.startswith("abs_diff__"):
                raise ValueError(f"Target variable '{var}' must start with 'abs_diff__'")

        logger.info("Plotting %d target variable(s)", len(target_variables))

        days = self.forecast["forecast_for"].dt.strftime("%a, %d-%m-%Y")

        for abs_diff__var in target_variables:
            var = abs_diff__var[10:]  # "abs_diff__" has 10 letters

            # Validate that both the raw and diff variables exist
            if var not in self.forecast.columns:
                raise ValueError(
                    f"Target variable '{var}' is not present in the forecast DataFrame. "
                    f"Available columns: {sorted(self.forecast.columns)}"
                )
            if abs_diff__var not in self.uncertainty_predictions.columns:
                raise ValueError(
                    f"Uncertainty predictions for target variable '{var}' are not available. "
                    f"Available uncertainty columns: {sorted(self.uncertainty_predictions.columns)}"
                )

            var_series = self.forecast[var].copy()
            # Non-destructive: make a copy before modifying
            var_diff = self.uncertainty_predictions[abs_diff__var].copy()
            var_diff.iloc[: self.past_days + 1] = 0

            ub = var_series + var_diff
            lb = var_series - var_diff

            # Apply bounds based on variable type
            if "temperature" not in var:
                ub = ub.clip(lower=0)
                lb = lb.clip(lower=0)

            # Handle sunshine duration specifically - validate latitude column exists
            if "sunshine_duration" in var:
                if "latitude" not in self.forecast.columns:
                    raise ValueError(
                        "Forecast DataFrame is missing 'latitude' column required for sunshine_duration bounds"
                    )
                daylengths = daylength(self.forecast["day_of_year"], self.forecast["latitude"])
                ub = ub.clip(upper=daylengths)
                lb = lb.clip(upper=daylengths)

            # Get plot specifications with fallback for unknown variables
            color = self._get_plot_spec(var, "color")
            error_color = self._get_plot_spec(abs_diff__var, "color")
            unit = self._get_plot_spec(var, "unit")
            title = self._get_plot_spec(var, "title")
            error_title = self._get_plot_spec(abs_diff__var, "title")

            plt.plot(days, var_series, color=color, label=title)
            plt.fill_between(
                x=days,
                y1=lb,
                y2=ub,
                alpha=0.2,
                color=error_color,
                label=error_title,
            )
            plt.grid(True)
            plt.xticks(rotation=90)
            plt.title(title)
            plt.ylabel(f"{title} in {unit}")
            plt.legend()
            plt.tight_layout()
            plt.show()

            logger.debug("Plotted %s", abs_diff__var)
