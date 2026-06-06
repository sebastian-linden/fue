"""Tests for `fue` package."""

import fue


def test_import():
    """Verify the package can be imported."""
    assert fue


def test_forecast_init():
    """Verify the correct initialization of a forecast object"""
    assert fue.Forecast()


def test_forecast_set_location():
    """Test whether setting the location works."""
    forecast = fue.Forecast()
    # Test set_location
    forecast.set_location(["aachen", "munich"], lat=50, lon=6)
    assert forecast.latitude == 50
    assert forecast.longitude == 6


def test_forecast_set_metrics():
    """Test whether specifying the metrics works."""
    forecast = fue.Forecast()
    # Test set_metrics
    weather_metrics = ["temperature_2m_max", "temperature_2m_min"]
    forecast.set_metrics(weather_metrics)
    assert forecast.parameters == weather_metrics


def test_forecast_download():
    """Test download function."""
    forecast = fue.Forecast()

    # Must set location and parameters before downloading
    forecast.set_location(["aachen", "munich"], lat=[50, 48], lon=[6, 7])
    forecast.set_metrics(["temperature_2m_max", "temperature_2m_min"])

    # Test download
    result = forecast.download()
    assert "successfully" in result.lower() or "downloaded" in result.lower()


def test_forecast_compute_errors():
    """Test download function."""
    forecast = fue.Forecast()

    # Test download
    assert forecast.compute_errors() == "Hello from compute_errors()"


def test_forecast_show():
    """Test download function."""
    forecast = fue.Forecast()

    # Test download
    assert forecast.show() == "Hello from show()"
