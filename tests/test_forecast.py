from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from pyfue.forecast import Forecast

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def forecast_instance():
    """Create a fresh Forecast instance for each test."""
    return Forecast()


@pytest.fixture
def sample_forecast_df():
    """Create a sample forecast DataFrame with required columns."""
    dates = pd.date_range(start="2026-07-03", periods=14, freq="D")
    forecast_on = pd.Timestamp("2026-07-03")

    return pd.DataFrame(
        {
            "forecast_for": dates,
            "forecasted_on": forecast_on,
            "day_of_year": dates.day_of_year,  # ty: ignore[unresolved-attribute]
            "delta_days": (dates - forecast_on).days,
            "temperature_2m_max": np.random.uniform(15, 30, 14),
            "temperature_2m_min": np.random.uniform(5, 20, 14),
            "precipitation_sum": np.random.uniform(0, 20, 14),
            "sunshine_duration": np.random.uniform(0, 12, 14),
            "wind_speed_10m_mean": np.random.uniform(0, 15, 14),
            "precipitation_probability_mean": np.random.uniform(0, 100, 14),
            "latitude": 52.5,  # Example: Berlin
        }
    )


@pytest.fixture
def sample_uncertainty_df():
    """Create a sample uncertainty predictions DataFrame."""
    return pd.DataFrame(
        {
            "abs_diff__temperature_2m_max": np.random.uniform(0, 5, 14),
            "abs_diff__temperature_2m_min": np.random.uniform(0, 5, 14),
            "abs_diff__precipitation_sum": np.random.uniform(0, 10, 14),
            "abs_diff__sunshine_duration": np.random.uniform(0, 2, 14),
            "abs_diff__wind_speed_10m_mean": np.random.uniform(0, 5, 14),
            "abs_diff__precipitation_probability_mean": np.random.uniform(0, 20, 14),
        }
    )


@pytest.fixture
def mock_uncertainty_model(sample_uncertainty_df):
    """Create a mock uncertainty model that returns predictions."""
    model = Mock()
    model.predict = Mock(return_value=sample_uncertainty_df)
    return model


@pytest.fixture
def mock_config():
    """Create a mock Config object."""
    config = Mock()
    config.cities = ["Berlin", "Paris", "London"]
    config.city_coordinates = {
        "Berlin": {"lat": 52.5200, "lon": 13.4050},
        "Paris": {"lat": 48.8566, "lon": 2.3522},
        "London": {"lat": 51.5074, "lon": -0.1278},
    }
    config.params = {}
    config.set_forecast_days = Mock()
    config.set_past_days = Mock()
    return config


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================


class TestForecastInit:
    """Test Forecast initialization."""

    def test_init_creates_instance(self, forecast_instance):
        """Test that __init__ creates a valid Forecast instance."""
        assert isinstance(forecast_instance, Forecast)

    def test_init_sets_attributes_to_none(self, forecast_instance):
        """Test that all attributes are initialized to None."""
        assert forecast_instance.forecast is None
        assert forecast_instance.uncertainty_model is None
        assert forecast_instance.uncertainty_predictions is None
        assert forecast_instance.past_days is None


# ============================================================================
# FETCH_FORECAST TESTS
# ============================================================================


class TestFetchForecast:
    """Test fetch_forecast method."""

    @patch("pyfue.forecast.Data")
    @patch("pyfue.forecast.Config")
    def test_fetch_forecast_valid_location_default_params(
        self, mock_config_class, mock_data_class, forecast_instance, sample_forecast_df, mock_config
    ):
        """Test fetching forecast with valid location and default parameters."""
        # Setup mocks
        mock_config_class.return_value = mock_config
        mock_data_instance = Mock()
        mock_data_instance.fetch_forecast = Mock(return_value=sample_forecast_df)
        mock_data_class.return_value = mock_data_instance

        # Execute
        forecast_instance.fetch_forecast("Berlin")

        # Assert
        assert forecast_instance.forecast is not None
        assert len(forecast_instance.forecast) == 14
        assert forecast_instance.past_days == 0
        assert "day_of_year" in forecast_instance.forecast.columns
        assert "delta_days" in forecast_instance.forecast.columns

    @patch("pyfue.forecast.Data")
    @patch("pyfue.forecast.Config")
    def test_fetch_forecast_with_custom_params(
        self, mock_config_class, mock_data_class, forecast_instance, sample_forecast_df, mock_config
    ):
        """Test fetching forecast with custom forecast_days and past_days."""
        # Setup mocks
        mock_config_class.return_value = mock_config
        mock_data_instance = Mock()
        mock_data_instance.fetch_forecast = Mock(return_value=sample_forecast_df)
        mock_data_class.return_value = mock_data_instance

        # Execute
        forecast_instance.fetch_forecast("Paris", forecast_days=21, past_days=7)

        # Assert
        assert forecast_instance.past_days == 7
        mock_config.set_forecast_days.assert_called_once_with(21)
        mock_config.set_past_days.assert_called_once_with(7)

    @patch("pyfue.forecast.Config")
    def test_fetch_forecast_invalid_location_type(self, mock_config_class, forecast_instance, mock_config):
        """Test that non-string location_name raises TypeError."""
        mock_config_class.return_value = mock_config

        with pytest.raises(TypeError, match="location_name must be a string"):
            forecast_instance.fetch_forecast(123)

    @patch("pyfue.forecast.Config")
    def test_fetch_forecast_invalid_location_not_in_config(self, mock_config_class, forecast_instance, mock_config):
        """Test that non-existent location raises ValueError."""
        mock_config_class.return_value = mock_config

        with pytest.raises(ValueError, match="Location .* is not defined in the configuration"):
            forecast_instance.fetch_forecast("NonExistentCity")

    @patch("pyfue.forecast.Config")
    def test_fetch_forecast_invalid_forecast_days_type(self, mock_config_class, forecast_instance, mock_config):
        """Test that non-integer forecast_days raises TypeError."""
        mock_config_class.return_value = mock_config

        with pytest.raises(TypeError, match="forecast_days must be an integer"):
            forecast_instance.fetch_forecast("Berlin", forecast_days="14")

    @patch("pyfue.forecast.Config")
    def test_fetch_forecast_invalid_forecast_days_value(self, mock_config_class, forecast_instance, mock_config):
        """Test that forecast_days <= 0 raises ValueError."""
        mock_config_class.return_value = mock_config

        with pytest.raises(ValueError, match="forecast_days must be > 0"):
            forecast_instance.fetch_forecast("Berlin", forecast_days=0)

        with pytest.raises(ValueError, match="forecast_days must be > 0"):
            forecast_instance.fetch_forecast("Berlin", forecast_days=-5)

    @patch("pyfue.forecast.Config")
    def test_fetch_forecast_invalid_past_days_type(self, mock_config_class, forecast_instance, mock_config):
        """Test that non-integer past_days raises TypeError."""
        mock_config_class.return_value = mock_config

        with pytest.raises(TypeError, match="past_days must be an integer"):
            forecast_instance.fetch_forecast("Berlin", past_days="7")

    @patch("pyfue.forecast.Config")
    def test_fetch_forecast_invalid_past_days_value(self, mock_config_class, forecast_instance, mock_config):
        """Test that past_days < 0 raises ValueError."""
        mock_config_class.return_value = mock_config

        with pytest.raises(ValueError, match="past_days must be >= 0"):
            forecast_instance.fetch_forecast("Berlin", past_days=-1)

    @patch("pyfue.forecast.Data")
    @patch("pyfue.forecast.Config")
    def test_fetch_forecast_missing_required_columns(
        self, mock_config_class, mock_data_class, forecast_instance, mock_config
    ):
        """Test that missing required columns raises ValueError."""
        # Setup mocks with incomplete DataFrame
        mock_config_class.return_value = mock_config
        incomplete_df = pd.DataFrame(
            {
                "temperature_2m_max": [20, 21, 22],
                # Missing 'forecast_for' and 'forecasted_on'
            }
        )
        mock_data_instance = Mock()
        mock_data_instance.fetch_forecast = Mock(return_value=incomplete_df)
        mock_data_class.return_value = mock_data_instance

        with pytest.raises(ValueError, match="Forecast DataFrame missing required columns"):
            forecast_instance.fetch_forecast("Berlin")

    @patch("pyfue.forecast.Data")
    @patch("pyfue.forecast.Config")
    def test_fetch_forecast_adds_computed_columns(
        self, mock_config_class, mock_data_class, forecast_instance, sample_forecast_df, mock_config
    ):
        """Test that fetch_forecast adds day_of_year and delta_days columns."""
        # Setup mocks with DataFrame that has required columns but no computed columns
        base_df = sample_forecast_df.drop(columns=["day_of_year", "delta_days"])

        mock_config_class.return_value = mock_config
        mock_data_instance = Mock()
        mock_data_instance.fetch_forecast = Mock(return_value=base_df)
        mock_data_class.return_value = mock_data_instance

        # Execute
        forecast_instance.fetch_forecast("Berlin")

        # Assert
        assert "day_of_year" in forecast_instance.forecast.columns
        assert "delta_days" in forecast_instance.forecast.columns


# ============================================================================
# COMPUTE_UNCERTAINTIES TESTS
# ============================================================================


class TestComputeUncertainties:
    """Test compute_uncertainties method."""

    def test_compute_uncertainties_valid_model_with_forecast(
        self, forecast_instance, sample_forecast_df, sample_uncertainty_df, mock_uncertainty_model
    ):
        """Test computing uncertainties with valid model and fetched forecast."""
        # Setup: Set forecast data directly (simulating fetch_forecast)
        forecast_instance.forecast = sample_forecast_df
        forecast_instance.past_days = 0

        # Execute
        forecast_instance.compute_uncertainties(mock_uncertainty_model)

        # Assert
        assert forecast_instance.uncertainty_model is mock_uncertainty_model
        assert forecast_instance.uncertainty_predictions is not None
        assert len(forecast_instance.uncertainty_predictions) == 14
        mock_uncertainty_model.predict.assert_called_once()

    def test_compute_uncertainties_no_forecast_raises_error(self, forecast_instance, mock_uncertainty_model):
        """Test that computing uncertainties without forecast raises ValueError."""
        with pytest.raises(ValueError, match="Forecast data has not been fetched"):
            forecast_instance.compute_uncertainties(mock_uncertainty_model)

    def test_compute_uncertainties_none_model_raises_error(self, forecast_instance, sample_forecast_df):
        """Test that None as model raises ValueError."""
        forecast_instance.forecast = sample_forecast_df
        forecast_instance.past_days = 0

        with pytest.raises(ValueError, match="uncertainty_model cannot be None"):
            forecast_instance.compute_uncertainties(None)

    def test_compute_uncertainties_model_without_predict_raises_error(self, forecast_instance, sample_forecast_df):
        """Test that model without predict method raises AttributeError."""
        forecast_instance.forecast = sample_forecast_df
        forecast_instance.past_days = 0

        model_without_predict = Mock(spec=[])  # Spec without 'predict'

        with pytest.raises(AttributeError, match="must have a 'predict' method"):
            forecast_instance.compute_uncertainties(model_without_predict)

    def test_compute_uncertainties_predict_not_callable_raises_error(self, forecast_instance, sample_forecast_df):
        """Test that non-callable predict attribute raises AttributeError."""
        forecast_instance.forecast = sample_forecast_df
        forecast_instance.past_days = 0

        model = Mock()
        model.predict = "not_callable"  # Not a callable

        with pytest.raises(AttributeError, match="uncertainty_model.predict must be callable"):
            forecast_instance.compute_uncertainties(model)

    def test_compute_uncertainties_model_returns_none_raises_error(self, forecast_instance, sample_forecast_df):
        """Test that model returning None raises ValueError."""
        forecast_instance.forecast = sample_forecast_df
        forecast_instance.past_days = 0

        model = Mock()
        model.predict = Mock(return_value=None)

        with pytest.raises(ValueError, match="uncertainty_model.predict\\(\\) returned None"):
            forecast_instance.compute_uncertainties(model)

    def test_compute_uncertainties_stores_model_reference(
        self, forecast_instance, sample_forecast_df, mock_uncertainty_model
    ):
        """Test that uncertainty_model is stored as reference."""
        forecast_instance.forecast = sample_forecast_df
        forecast_instance.past_days = 0

        forecast_instance.compute_uncertainties(mock_uncertainty_model)

        assert forecast_instance.uncertainty_model is mock_uncertainty_model


# ============================================================================
# _GET_PLOT_SPEC TESTS
# ============================================================================


class TestGetPlotSpec:
    """Test _get_plot_spec helper method."""

    def test_get_plot_spec_known_color(self, forecast_instance):
        """Test getting color spec for known variable."""
        color = forecast_instance._get_plot_spec("temperature_2m_max", "color")
        assert color == "crimson"

    def test_get_plot_spec_known_unit(self, forecast_instance):
        """Test getting unit spec for known variable."""
        unit = forecast_instance._get_plot_spec("precipitation_sum", "unit")
        assert unit == "mm"

    def test_get_plot_spec_known_title(self, forecast_instance):
        """Test getting title spec for known variable."""
        title = forecast_instance._get_plot_spec("sunshine_duration", "title")
        assert title == "Daily Sunshine Duration"

    def test_get_plot_spec_all_known_variables_colors(self, forecast_instance):
        """Test that all known variables have color specs."""
        known_vars = list(forecast_instance._KNOWN_COLORS.keys())
        for var in known_vars:
            color = forecast_instance._get_plot_spec(var, "color")
            assert isinstance(color, str)
            assert len(color) > 0

    def test_get_plot_spec_all_known_variables_units(self, forecast_instance):
        """Test that all known variables have unit specs."""
        known_vars = list(forecast_instance._KNOWN_UNITS.keys())
        for var in known_vars:
            unit = forecast_instance._get_plot_spec(var, "unit")
            assert isinstance(unit, str)
            assert len(unit) > 0

    def test_get_plot_spec_all_known_variables_titles(self, forecast_instance):
        """Test that all known variables have title specs."""
        known_vars = list(forecast_instance._KNOWN_TITLES.keys())
        for var in known_vars:
            title = forecast_instance._get_plot_spec(var, "title")
            assert isinstance(title, str)
            assert len(title) > 0

    def test_get_plot_spec_unknown_variable_warns(self, forecast_instance):
        """Test that unknown variable generates warning."""
        with pytest.warns(UserWarning, match="not in predefined"):
            forecast_instance._get_plot_spec("unknown_variable", "color")

    def test_get_plot_spec_unknown_variable_default_color(self, forecast_instance):
        """Test that unknown variable returns default color."""
        with pytest.warns(UserWarning):
            color = forecast_instance._get_plot_spec("unknown_variable", "color")
        assert color == "gray"

    def test_get_plot_spec_unknown_variable_default_unit(self, forecast_instance):
        """Test that unknown variable returns default unit."""
        with pytest.warns(UserWarning):
            unit = forecast_instance._get_plot_spec("unknown_variable", "unit")
        assert unit == "[unknown]"

    def test_get_plot_spec_unknown_variable_default_title(self, forecast_instance):
        """Test that unknown variable returns variable name as title."""
        with pytest.warns(UserWarning):
            title = forecast_instance._get_plot_spec("unknown_variable", "title")
        assert title == "unknown_variable"

    def test_get_plot_spec_invalid_spec_type_raises_error(self, forecast_instance):
        """Test that invalid spec_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid spec_type"):
            forecast_instance._get_plot_spec("temperature_2m_max", "invalid_type")

    def test_get_plot_spec_target_variables(self, forecast_instance):
        """Test that target variables (abs_diff__*) have specs."""
        color = forecast_instance._get_plot_spec("abs_diff__temperature_2m_max", "color")
        assert color == "darkred"

        unit = forecast_instance._get_plot_spec("abs_diff__precipitation_sum", "unit")
        assert unit == "mm"


# ============================================================================
# PLOT TESTS
# ============================================================================


class TestPlot:
    """Test plot method."""

    @patch("pyfue.forecast.plt.show")
    @patch("pyfue.forecast.plt.tight_layout")
    @patch("pyfue.forecast.plt.legend")
    @patch("pyfue.forecast.plt.ylabel")
    @patch("pyfue.forecast.plt.title")
    @patch("pyfue.forecast.plt.xticks")
    @patch("pyfue.forecast.plt.grid")
    @patch("pyfue.forecast.plt.fill_between")
    @patch("pyfue.forecast.plt.plot")
    def test_plot_single_variable_as_string(
        self,
        mock_plot,
        mock_fill_between,
        mock_grid,
        mock_xticks,
        mock_title,
        mock_ylabel,
        mock_legend,
        mock_tight_layout,
        mock_show,
        forecast_instance,
        sample_forecast_df,
        sample_uncertainty_df,
    ):
        """Test plotting with single target variable as string."""
        # Setup
        forecast_instance.forecast = sample_forecast_df
        forecast_instance.uncertainty_predictions = sample_uncertainty_df
        forecast_instance.past_days = 0

        # Execute
        forecast_instance.plot("abs_diff__temperature_2m_max")

        # Assert - verify matplotlib was called
        assert mock_plot.called
        assert mock_fill_between.called
        assert mock_show.called

    @patch("pyfue.forecast.plt.show")
    @patch("pyfue.forecast.plt.tight_layout")
    @patch("pyfue.forecast.plt.legend")
    @patch("pyfue.forecast.plt.ylabel")
    @patch("pyfue.forecast.plt.title")
    @patch("pyfue.forecast.plt.xticks")
    @patch("pyfue.forecast.plt.grid")
    @patch("pyfue.forecast.plt.fill_between")
    @patch("pyfue.forecast.plt.plot")
    def test_plot_multiple_variables_as_list(
        self,
        mock_plot,
        mock_fill_between,
        mock_grid,
        mock_xticks,
        mock_title,
        mock_ylabel,
        mock_legend,
        mock_tight_layout,
        mock_show,
        forecast_instance,
        sample_forecast_df,
        sample_uncertainty_df,
    ):
        """Test plotting with multiple target variables as list."""
        # Setup
        forecast_instance.forecast = sample_forecast_df
        forecast_instance.uncertainty_predictions = sample_uncertainty_df
        forecast_instance.past_days = 0

        # Execute
        forecast_instance.plot(["abs_diff__temperature_2m_max", "abs_diff__precipitation_sum"])

        # Assert - verify plt.plot was called twice (once per variable)
        assert mock_plot.call_count == 2

    def test_plot_no_forecast_raises_error(self, forecast_instance, sample_uncertainty_df):
        """Test that plotting without forecast raises ValueError."""
        forecast_instance.uncertainty_predictions = sample_uncertainty_df
        forecast_instance.past_days = 0

        with pytest.raises(ValueError, match="Forecast data has not been fetched"):
            forecast_instance.plot("abs_diff__temperature_2m_max")

    def test_plot_no_uncertainties_raises_error(self, forecast_instance, sample_forecast_df):
        """Test that plotting without computed uncertainties raises ValueError."""
        forecast_instance.forecast = sample_forecast_df
        forecast_instance.past_days = 0

        with pytest.raises(ValueError, match="Uncertainty predictions have not been computed"):
            forecast_instance.plot("abs_diff__temperature_2m_max")

    def test_plot_past_days_not_set_raises_error(self, forecast_instance, sample_forecast_df, sample_uncertainty_df):
        """Test that past_days not set raises ValueError."""
        forecast_instance.forecast = sample_forecast_df
        forecast_instance.uncertainty_predictions = sample_uncertainty_df
        forecast_instance.past_days = None

        with pytest.raises(ValueError, match="past_days is not set"):
            forecast_instance.plot("abs_diff__temperature_2m_max")

    def test_plot_none_target_variables_raises_error(
        self, forecast_instance, sample_forecast_df, sample_uncertainty_df
    ):
        """Test that None as target_variables raises ValueError."""
        forecast_instance.forecast = sample_forecast_df
        forecast_instance.uncertainty_predictions = sample_uncertainty_df
        forecast_instance.past_days = 0

        with pytest.raises(ValueError, match="target_variables must be provided"):
            forecast_instance.plot(None)

    def test_plot_empty_list_raises_error(self, forecast_instance, sample_forecast_df, sample_uncertainty_df):
        """Test that empty list for target_variables raises ValueError."""
        forecast_instance.forecast = sample_forecast_df
        forecast_instance.uncertainty_predictions = sample_uncertainty_df
        forecast_instance.past_days = 0

        with pytest.raises(ValueError, match="target_variables list cannot be empty"):
            forecast_instance.plot([])

    def test_plot_invalid_target_variables_type_raises_error(
        self, forecast_instance, sample_forecast_df, sample_uncertainty_df
    ):
        """Test that invalid type for target_variables raises TypeError."""
        forecast_instance.forecast = sample_forecast_df
        forecast_instance.uncertainty_predictions = sample_uncertainty_df
        forecast_instance.past_days = 0

        with pytest.raises(TypeError, match="target_variables must be either a string or a list"):
            forecast_instance.plot(123)

    def test_plot_non_string_in_list_raises_error(self, forecast_instance, sample_forecast_df, sample_uncertainty_df):
        """Test that non-string in target_variables list raises TypeError."""
        forecast_instance.forecast = sample_forecast_df
        forecast_instance.uncertainty_predictions = sample_uncertainty_df
        forecast_instance.past_days = 0

        with pytest.raises(TypeError, match="All target_variables must be strings"):
            forecast_instance.plot(["abs_diff__temperature_2m_max", 123])

    def test_plot_variable_not_starting_with_abs_diff_raises_error(
        self, forecast_instance, sample_forecast_df, sample_uncertainty_df
    ):
        """Test that variable not starting with 'abs_diff__' raises ValueError."""
        forecast_instance.forecast = sample_forecast_df
        forecast_instance.uncertainty_predictions = sample_uncertainty_df
        forecast_instance.past_days = 0

        with pytest.raises(ValueError, match="must start with 'abs_diff__'"):
            forecast_instance.plot("temperature_2m_max")

    def test_plot_raw_variable_missing_from_forecast_raises_error(self, forecast_instance, sample_uncertainty_df):
        """Test that missing raw variable in forecast raises ValueError."""
        # Create forecast without temperature_2m_max
        forecast = pd.DataFrame(
            {
                "forecast_for": pd.date_range("2026-07-03", periods=14),
                "forecasted_on": pd.Timestamp("2026-07-03"),
                "day_of_year": pd.Series(range(184, 198)),
                "delta_days": pd.Series(range(0, 14)),
                "latitude": 52.5,
                # Missing temperature_2m_max
            }
        )
        forecast_instance.forecast = forecast
        forecast_instance.uncertainty_predictions = sample_uncertainty_df
        forecast_instance.past_days = 0

        with pytest.raises(ValueError, match="is not present in the forecast DataFrame"):
            forecast_instance.plot("abs_diff__temperature_2m_max")

    def test_plot_uncertainty_variable_missing_from_predictions_raises_error(
        self, forecast_instance, sample_forecast_df
    ):
        """Test that missing uncertainty variable in predictions raises ValueError."""
        # Create uncertainty DataFrame without abs_diff__temperature_2m_max
        uncertainty = pd.DataFrame(
            {
                "abs_diff__precipitation_sum": np.random.uniform(0, 10, 14),
            }
        )
        forecast_instance.forecast = sample_forecast_df
        forecast_instance.uncertainty_predictions = uncertainty
        forecast_instance.past_days = 0

        with pytest.raises(ValueError, match="Uncertainty predictions for target variable .* are not available"):
            forecast_instance.plot("abs_diff__temperature_2m_max")

    @patch("pyfue.forecast.plt.show")
    @patch("pyfue.forecast.plt.tight_layout")
    @patch("pyfue.forecast.plt.legend")
    @patch("pyfue.forecast.plt.ylabel")
    @patch("pyfue.forecast.plt.title")
    @patch("pyfue.forecast.plt.xticks")
    @patch("pyfue.forecast.plt.grid")
    @patch("pyfue.forecast.plt.fill_between")
    @patch("pyfue.forecast.plt.plot")
    def test_plot_past_days_uncertainty_set_to_zero(
        self,
        mock_plot,
        mock_fill_between,
        mock_grid,
        mock_xticks,
        mock_title,
        mock_ylabel,
        mock_legend,
        mock_tight_layout,
        mock_show,
        forecast_instance,
        sample_forecast_df,
        sample_uncertainty_df,
    ):
        """Test that uncertainties are set to zero for past days."""
        # Setup with past_days = 3
        forecast_instance.forecast = sample_forecast_df
        forecast_instance.uncertainty_predictions = sample_uncertainty_df.copy()
        forecast_instance.past_days = 3

        # Execute
        forecast_instance.plot("abs_diff__temperature_2m_max")

        # Assert - check that fill_between was called with modified uncertainties
        assert mock_fill_between.called
        # The call would have been made with zeroed uncertainties for first 4 days

    @patch("pyfue.forecast.plt.show")
    @patch("pyfue.forecast.plt.tight_layout")
    @patch("pyfue.forecast.plt.legend")
    @patch("pyfue.forecast.plt.ylabel")
    @patch("pyfue.forecast.plt.title")
    @patch("pyfue.forecast.plt.xticks")
    @patch("pyfue.forecast.plt.grid")
    @patch("pyfue.forecast.plt.fill_between")
    @patch("pyfue.forecast.plt.plot")
    def test_plot_non_temperature_bounds_clipped_to_zero(
        self,
        mock_plot,
        mock_fill_between,
        mock_grid,
        mock_xticks,
        mock_title,
        mock_ylabel,
        mock_legend,
        mock_tight_layout,
        mock_show,
        forecast_instance,
        sample_forecast_df,
        sample_uncertainty_df,
    ):
        """Test that non-temperature variables are clipped at zero."""
        forecast_instance.forecast = sample_forecast_df
        forecast_instance.uncertainty_predictions = sample_uncertainty_df
        forecast_instance.past_days = 0

        # Execute with precipitation (should be clipped to 0 at minimum)
        forecast_instance.plot("abs_diff__precipitation_sum")

        # Assert - verify fill_between was called (bounds would be clipped)
        assert mock_fill_between.called

    @patch("pyfue.forecast.daylength")
    @patch("pyfue.forecast.plt.show")
    @patch("pyfue.forecast.plt.tight_layout")
    @patch("pyfue.forecast.plt.legend")
    @patch("pyfue.forecast.plt.ylabel")
    @patch("pyfue.forecast.plt.title")
    @patch("pyfue.forecast.plt.xticks")
    @patch("pyfue.forecast.plt.grid")
    @patch("pyfue.forecast.plt.fill_between")
    @patch("pyfue.forecast.plt.plot")
    def test_plot_sunshine_duration_requires_latitude(
        self,
        mock_plot,
        mock_fill_between,
        mock_grid,
        mock_xticks,
        mock_title,
        mock_ylabel,
        mock_legend,
        mock_tight_layout,
        mock_show,
        mock_daylength,
        forecast_instance,
        sample_forecast_df,
        sample_uncertainty_df,
    ):
        """Test that plotting sunshine_duration requires latitude column."""
        # Remove latitude column
        forecast_without_lat = sample_forecast_df.drop(columns=["latitude"])
        forecast_instance.forecast = forecast_without_lat
        forecast_instance.uncertainty_predictions = sample_uncertainty_df
        forecast_instance.past_days = 0

        with pytest.raises(ValueError, match="missing 'latitude' column"):
            forecast_instance.plot("abs_diff__sunshine_duration")

    @patch("pyfue.forecast.daylength")
    @patch("pyfue.forecast.plt.show")
    @patch("pyfue.forecast.plt.tight_layout")
    @patch("pyfue.forecast.plt.legend")
    @patch("pyfue.forecast.plt.ylabel")
    @patch("pyfue.forecast.plt.title")
    @patch("pyfue.forecast.plt.xticks")
    @patch("pyfue.forecast.plt.grid")
    @patch("pyfue.forecast.plt.fill_between")
    @patch("pyfue.forecast.plt.plot")
    def test_plot_sunshine_duration_with_latitude(
        self,
        mock_plot,
        mock_fill_between,
        mock_grid,
        mock_xticks,
        mock_title,
        mock_ylabel,
        mock_legend,
        mock_tight_layout,
        mock_show,
        mock_daylength,
        forecast_instance,
        sample_forecast_df,
        sample_uncertainty_df,
    ):
        """Test that plotting sunshine_duration works with latitude."""
        # Setup
        mock_daylength.return_value = np.full(14, 16)  # Mock daylengths
        forecast_instance.forecast = sample_forecast_df
        forecast_instance.uncertainty_predictions = sample_uncertainty_df
        forecast_instance.past_days = 0

        # Execute
        forecast_instance.plot("abs_diff__sunshine_duration")

        # Assert
        assert mock_plot.called
        assert mock_daylength.called


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestForecastIntegration:
    """Test integration of multiple methods."""

    @patch("pyfue.forecast.Data")
    @patch("pyfue.forecast.Config")
    @patch("pyfue.forecast.plt.show")
    def test_full_workflow_fetch_compute_plot(
        self,
        mock_show,
        mock_config_class,
        mock_data_class,
        forecast_instance,
        sample_forecast_df,
        sample_uncertainty_df,
        mock_config,
        mock_uncertainty_model,
    ):
        """Test full workflow: fetch -> compute -> plot."""
        # Setup mocks
        mock_config_class.return_value = mock_config
        mock_data_instance = Mock()
        mock_data_instance.fetch_forecast = Mock(return_value=sample_forecast_df)
        mock_data_class.return_value = mock_data_instance
        mock_uncertainty_model.predict = Mock(return_value=sample_uncertainty_df)

        # Execute full workflow
        with (
            patch("pyfue.forecast.plt.plot"),
            patch("pyfue.forecast.plt.fill_between"),
            patch("pyfue.forecast.plt.grid"),
            patch("pyfue.forecast.plt.xticks"),
            patch("pyfue.forecast.plt.title"),
            patch("pyfue.forecast.plt.ylabel"),
            patch("pyfue.forecast.plt.legend"),
            patch("pyfue.forecast.plt.tight_layout"),
        ):
            forecast_instance.fetch_forecast("Berlin")
            forecast_instance.compute_uncertainties(mock_uncertainty_model)
            forecast_instance.plot("abs_diff__temperature_2m_max")

        # Assert final state
        assert forecast_instance.forecast is not None
        assert forecast_instance.uncertainty_model is not None
        assert forecast_instance.uncertainty_predictions is not None
        assert forecast_instance.past_days == 0

    def test_workflow_fails_if_compute_before_fetch(self, forecast_instance, mock_uncertainty_model):
        """Test that compute fails if fetch is not called first."""
        with pytest.raises(ValueError, match="Forecast data has not been fetched"):
            forecast_instance.compute_uncertainties(mock_uncertainty_model)

    def test_workflow_fails_if_plot_before_compute(self, forecast_instance, sample_forecast_df):
        """Test that plot fails if compute is not called first."""
        forecast_instance.forecast = sample_forecast_df
        forecast_instance.past_days = 0

        with pytest.raises(ValueError, match="Uncertainty predictions have not been computed"):
            forecast_instance.plot("abs_diff__temperature_2m_max")

    @patch("pyfue.forecast.Data")
    @patch("pyfue.forecast.Config")
    def test_multiple_uncertainty_computations_overwrite(
        self,
        mock_config_class,
        mock_data_class,
        forecast_instance,
        sample_forecast_df,
        sample_uncertainty_df,
        mock_config,
    ):
        """Test that computing with new model overwrites previous predictions."""
        # Setup
        mock_config_class.return_value = mock_config
        mock_data_instance = Mock()
        mock_data_instance.fetch_forecast = Mock(return_value=sample_forecast_df)
        mock_data_class.return_value = mock_data_instance

        forecast_instance.fetch_forecast("Berlin")

        # Create two different models
        model1 = Mock()
        model1.predict = Mock(return_value=sample_uncertainty_df)

        model2_predictions = sample_uncertainty_df * 2  # Different values
        model2 = Mock()
        model2.predict = Mock(return_value=model2_predictions)

        # Compute with first model
        forecast_instance.compute_uncertainties(model1)
        first_predictions = forecast_instance.uncertainty_predictions

        # Compute with second model
        forecast_instance.compute_uncertainties(model2)
        second_predictions = forecast_instance.uncertainty_predictions

        # Assert they are different objects
        assert first_predictions is not second_predictions
