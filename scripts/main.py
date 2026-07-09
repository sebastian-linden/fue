"""This file is meant to run any of my package code."""

import logging

from fue import Forecast

# Define standard formatters, levels, and routing destinations at the application entrypoint
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s - (%(name)s)",
    handlers=[
        logging.StreamHandler(),  # Outputs clean logs to the terminal console
        logging.FileHandler("fue_pipeline.log"),  # Continuously appends execution traces to a local disk file
    ],
)

# Run pipeline execution safely tracked by the system
logger = logging.getLogger("__main__")

F = Forecast()
F.fetch_forecast(location_name="berlin", forecast_days=14)
logger.info("Forecast fetched.")
