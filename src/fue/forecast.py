"""Forecast class for managing weather forecast data and error estimation."""

from typing import Any

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

from .utils import store_forecast


class Forecast:
    """Class to manage weather forecasts and compute error estimates over time."""

    def __init__(self) -> None:
        """Initialize a new Forecast instance."""
        self.city: str | list[str] | None = None
        self.latitude: float | list[float] | None = None
        self.longitude: float | list[float] | None = None
        self.parameters: list[str] = []
        self.current_forecast: pd.DataFrame | None = None

    def set_location(self, city: str | list[str], lat: float | list[float], lon: float | list[float]) -> None:
        """Set the location(s) for the forecast."""
        self.city = city
        self.latitude = lat
        self.longitude = lon

    def set_metrics(self, metrics: list[str]) -> None:
        """Set the weather metrics to forecast.

        Args:
            metrics: List of parameter names (e.g., ["temperature_2m_max", "temperature_2m_min"]).
        """
        self.parameters = metrics

    def download(self) -> str:  # type: ignore[no-untyped-def]
        """Download the current forecast for multiple locations and parameters."""
        # Validate that location, parameters, and city are set and are lists
        if not all([self.latitude, self.longitude, self.city]):
            raise ValueError("Locations or cities not set. Call set_location() first.")
        if (
            not isinstance(self.latitude, list)
            or not isinstance(self.longitude, list)
            or not isinstance(self.city, list)
        ):
            raise ValueError("Latitude, longitude, and city must be provided as lists for batch processing.")

        if not self.parameters:
            raise ValueError("Parameters not set. Call set_parameters() first.")

        # 1. Setup API Client
        cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)  # type: ignore

        # 2. Prepare API Call
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "daily": self.parameters,
            "timezone": "auto",
	        "forecast_days": 14,
	        "past_days": 5,
        }

        try:
            # responses will contain one object per coordinate pair
            responses = openmeteo.weather_api(url, params=params)
        except Exception as e:
            return f"Error fetching forecast: {e}"

        # 3. Dynamic Data Extraction per Response
        for index, response in enumerate(responses):
            current_city = self.city[index]
            daily = response.Daily()

            # Get the UTC offset provided by the API for the specific location
            offset_seconds = response.UtcOffsetSeconds()
            offset_delta = pd.Timedelta(seconds=offset_seconds)

            # Generate the UTC date range
            utc_dates = pd.date_range(
                start=pd.to_datetime(daily.Time(), unit="s", utc=True),  # type: ignore
                end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),  # type: ignore
                freq=pd.Timedelta(seconds=daily.Interval()),  # type: ignore
                inclusive="left",
            )

            # Shift dates and format
            local_date_strings = (utc_dates + offset_delta).strftime("%Y-%m-%d")

            # Create base dictionary
            daily_data: dict[str, Any] = {"date": local_date_strings}

            # Extract values by index
            for i, var_name in enumerate(self.parameters):
                daily_data[var_name] = daily.Variables(i).ValuesAsNumpy()  # type: ignore

            # 4. Store individually
            forecast_df = pd.DataFrame(data=daily_data)

            # Store the current city's forecast
            store_forecast(city=current_city, data=forecast_df)

        return f"Forecast for {len(responses)} cities successfully downloaded and stored."

    def compute_errors(self) -> str:
        """Compute error estimates based on historical forecast data."""
        
        return "Hello from compute_errors()"

    def show(self) -> str:
        """Display the current forecast with estimated errors."""
        print(self.current_forecast)
        return "Hello from show()"
