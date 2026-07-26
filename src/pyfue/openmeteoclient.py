"""
API Client for fetching weather forecasts from Open-Meteo.

This module communicates with the Open-Meteo API. It includes automatic data
caching to prevent hitting rate limits and a retry system to handle network drops.
"""

import logging
from datetime import datetime

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

from .config import Config  # When imported as part of package

logger = logging.getLogger(__name__)


class OpenMeteoClient:
    """
    A client used to fetch and format weather data from the Open-Meteo API.

    This class handles the underlying network connection, sessions, and data caching.
    It downloads weather metrics for all tracked cities simultaneously and maps the
    raw API responses directly into pandas DataFrames for our pipeline.
    """

    def __init__(self, config: Config):
        """
        Sets up the API endpoint URL, a cached network session, and configuration settings.

        Parameters
        ----------
        config : Config or None, default=None
            The shared project configuration object containing tracking parameters.
            If None, a default instance of the Config class is automatically created.
        """

        # Setup the Open-Meteo API client with cache and retry on error
        self.cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
        self.retry_session = retry(self.cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=self.retry_session)  # ty: ignore[invalid-argument-type]

        # Make sure all required weather variables are listed here
        # The order of variables in hourly or daily is important to assign them correctly below
        self.url = "https://api.open-meteo.com/v1/forecast"
        self.config = config
        logger.info("OpenMeteoClient initialized with configuration.")

    def fetch_forecast(self) -> pd.DataFrame:
        """
        Downloads forward-looking weather forecast values for all target cities.

        Queries the Open-Meteo endpoint using parameters from our configuration file,
        loops over each city's geographic response, aligns the time indices to
        the final second of the day, and aggregates everything into a single
        unified data frame.

        Returns
        -------
        pd.DataFrame
            A stacked data frame containing forecast records for all cities.
            Columns include: 'location_name', 'latitude', 'longitude',
            'forecasted_on', 'forecast_for', and the specific weather variables.
        """

        # Fetch responses from API
        responses = self.openmeteo.weather_api(self.url, params=self.config.params)

        columns = ["location_name", "latitude", "longitude", "forecasted_on", "forecast_for"] + self.config.params[
            "daily"
        ]
        forecast_data = pd.DataFrame(columns=columns)

        # Process each location
        for index, response in enumerate(responses):
            location_dict = {
                "location_name": self.config.cities[index],
                "latitude": response.Latitude(),
                "longitude": response.Longitude(),
            }

            location_dict["forecasted_on"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            """ Process daily data """
            daily = response.Daily()

            # Time series - shifted to end of day: 23:59:59
            location_dict["forecast_for"] = pd.date_range(
                start=pd.to_datetime(daily.Time(), unit="s"),  # ty: ignore[unresolved-attribute]
                end=pd.to_datetime(daily.TimeEnd(), unit="s"),  # ty: ignore[unresolved-attribute]
                freq=pd.Timedelta(seconds=daily.Interval()),  # ty: ignore[unresolved-attribute]
                inclusive="left",
            ) + pd.Timedelta(days=1, hours=1, seconds=-1)

            # Predicted Variables
            for i, metric in enumerate(self.config.params["daily"]):
                location_dict[metric] = daily.Variables(i).ValuesAsNumpy()  # type: ignore

            # Concatenate DataFrames
            location_data = pd.DataFrame(location_dict, columns=columns)
            forecast_data = pd.concat([forecast_data, location_data])

        n_cities = len(self.config.cities)
        n_forecast_days = self.config.params.get("forecast_days", 0)
        logger.debug(f"Forecasting data fetched for {n_cities} cities. Total forecasts: {n_cities * n_forecast_days}")

        return forecast_data
