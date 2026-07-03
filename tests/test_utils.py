import numpy as np
import pandas as pd

from fue.utils import daylength


class TestUtils:
    """Test suite for utility functions, ensuring mathematical and physical accuracy."""

    def test_daylength_equator(self):
        """Test that the equator always experiences exactly 12 hours of daylight."""
        # Testing random days throughout the year
        days = pd.Series([1, 80, 172, 264, 355])
        lat = 0.0

        result = daylength(days, lat)

        # The equator should always be ~12 hours regardless of the day of year
        np.testing.assert_allclose(result, 12.0, atol=1e-5)

    def test_daylength_northern_hemisphere_summer_winter(self):
        """Test summer solstice (longest day) and winter solstice (shortest day) at mid-latitudes."""
        # Day 172 is approx June 21 (Summer Solstice)
        # Day 355 is approx Dec 21 (Winter Solstice)
        days = pd.Series([172, 355])
        lat = 50.77  # e.g., Aachen

        result = daylength(days, lat)

        # Summer solstice should be significantly longer than 12 hours
        assert result.iloc[0] > 16.0
        # Winter solstice should be significantly shorter than 12 hours
        assert result.iloc[1] < 9.0

    def test_daylength_polar_day_and_night(self):
        """Test the np.clip logic ensuring polar regions safely cap at 24h or 0h daylight."""
        days = pd.Series([172, 355])
        lat = 80.0  # High up in the Arctic Circle (e.g., Svalbard region)

        result = daylength(days, lat)

        # Summer: Polar day (Midnight Sun) -> arccos(-1) -> 24 hours
        np.testing.assert_allclose(result.iloc[0], 24.0, atol=1e-5)
        # Winter: Polar night -> arccos(1) -> 0 hours
        np.testing.assert_allclose(result.iloc[1], 0.0, atol=1e-5)

    def test_daylength_series_latitude(self):
        """Test that the function successfully broadcasts when latitude is passed as a pd.Series."""
        # Testing the Summer Solstice across three different latitudes simultaneously
        days = pd.Series([172, 172, 172])
        lats = pd.Series([0.0, 50.0, 80.0])  # Equator, Mid-Lat, Arctic

        result = daylength(days, lats)

        # Equator
        np.testing.assert_allclose(result.iloc[0], 12.0, atol=1e-5)
        # Mid-lat summer
        assert result.iloc[1] > 12.0
        # Polar summer
        np.testing.assert_allclose(result.iloc[2], 24.0, atol=1e-5)
