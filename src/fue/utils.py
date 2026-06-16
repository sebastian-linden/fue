import datetime
import json
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

# Define the relative path to your config
CONFIG_PATH = Path("data/user/.fue_config.json")
RAW_DATA_PATH = Path("data/raw/")


def load_config() -> dict:
    """Loads configuration and validates contents."""
    if not CONFIG_PATH.exists():
        print(f"Error: Configuration file not found at {CONFIG_PATH}")
        sys.exit(1)

    try:
        with open(CONFIG_PATH) as f:
            config = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: {CONFIG_PATH} contains invalid JSON.")
        sys.exit(1)

    # Validation logic
    cities = config.get("cities", {})
    metrics = config.get("metrics", [])

    if not cities:
        print("Error: No cities defined in config.json.")
        sys.exit(1)

    if not metrics:
        print("Error: No metrics defined in config.json.")
        sys.exit(1)

    return config


def store_forecast(city: str | None = None, data: pd.DataFrame | None = None) -> None:
    """Store raw forecasting data

    Args:
        data (pandas.DataFrame): DataFrame containing the forecasting data

    Returns:
        None
    """
    if city is None:
        print("No city name was passed.")
        return None
    elif data is None:
        print("No data was passed.")
        return None
    else:
        # check today's date
        today = datetime.date.today()
        today_str = today.strftime("%Y%m%d")
        file_name = f"{city}_{today_str}.csv"
        data.to_csv(f"{RAW_DATA_PATH}/{file_name}", index=False)
        print("Successfully stored today's forecasting data.")


def set_schedule(action: Annotated[str, typer.Argument(help="Either 'activate' or 'deactivate'")]) -> None:
    """Automates the setup or removal of the systemd timer."""

    # Constants for paths
    SYSTEMD_USER_DIR = Path.home() / ".config/systemd/user"
    SERVICE_NAME = "fue.service"
    TIMER_NAME = "fue.timer"

    SERVICE_CONTENT = f"""[Unit]
    Description=Download weather forecast data

    [Service]
    Type=oneshot
    WorkingDirectory={Path.cwd()}
    Environment=PATH=/snap/bin:$PATH
    ExecStart=/snap/bin/uv run fue download

    [Install]
    WantedBy=default.target
    """

    TIMER_CONTENT = """[Unit]
    Description=Run fue download daily at 9am

    [Timer]
    OnCalendar=*-*-* 09:00:00
    Persistent=true

    [Install]
    WantedBy=timers.target
    """

    if action == "activate":
        # Create directory if it doesn't exist
        SYSTEMD_USER_DIR.mkdir(parents=True, exist_ok=True)

        # Write files
        (SYSTEMD_USER_DIR / SERVICE_NAME).write_text(SERVICE_CONTENT)
        (SYSTEMD_USER_DIR / TIMER_NAME).write_text(TIMER_CONTENT)

        # Enable and start
        try:
            subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
            subprocess.run(["systemctl", "--user", "enable", "--now", TIMER_NAME], check=True)
            print("Timer activated successfully for 09:00 AM daily.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to activate timer: {e}")

    elif action == "deactivate":
        # Stop and disable
        try:
            subprocess.run(["systemctl", "--user", "stop", TIMER_NAME], check=False)
            subprocess.run(["systemctl", "--user", "disable", TIMER_NAME], check=False)

            # Remove files
            (SYSTEMD_USER_DIR / SERVICE_NAME).unlink(missing_ok=True)
            (SYSTEMD_USER_DIR / TIMER_NAME).unlink(missing_ok=True)

            subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
            subprocess.run(["systemctl", "--user", "reset-failed"], check=True)
            print("Timer deactivated and files removed.")
        except subprocess.CalledProcessError as e:
            print(f"Error during deactivation: {e}")
    else:
        print("Invalid action. Use 'activate' or 'deactivate'.")
