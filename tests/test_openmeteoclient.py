import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fue.openmeteoclient import OpenMeteoClient


class TestOpenMeteoClientInit:
    """Test suite for OpenMeteoClient initialization"""

    def test_client_initialization(self):
        """Test that OpenMeteoClient initializes without errors"""
        client = OpenMeteoClient()
        assert client is not None
        assert hasattr(client, 'openmeteo')
        assert hasattr(client, 'config')
        assert hasattr(client, 'url')

    def test_url_is_correct(self):
        """Test that the API URL is set correctly"""
        client = OpenMeteoClient()
        assert client.url == "https://api.open-meteo.com/v1/forecast"

    def test_cache_session_initialized(self):
        """Test that cache session is initialized"""
        client = OpenMeteoClient()
        assert client.cache_session is not None

    def test_config_loaded(self):
        """Test that config is loaded"""
        client = OpenMeteoClient()
        assert client.config is not None
        assert hasattr(client.config, 'params')
        assert hasattr(client.config, 'cities')


class TestOpenMeteoClientIntegration:
    """Integration tests for OpenMeteoClient"""

    def test_client_setup_complete(self):
        """Test that all components are properly set up"""
        client = OpenMeteoClient()

        # Verify all necessary attributes exist
        assert hasattr(client, 'cache_session')
        assert hasattr(client, 'retry_session')
        assert hasattr(client, 'openmeteo')
        assert hasattr(client, 'url')
        assert hasattr(client, 'config')

        # Verify config has necessary attributes
        assert hasattr(client.config, 'params')
        assert hasattr(client.config, 'cities')

        # Verify config.params has necessary keys
        assert 'daily' in client.config.params
        assert 'timezone' in client.config.params
