"""Console script for fue."""

from typing import Annotated

import typer

app = typer.Typer()


@app.command()
def help() -> None:
    """ Displays helpful information"""
    help_message = """
    [Placeholder help message]
    """
    print(help_message)
    return None


@app.command()
def download() -> None:
    """Initiates download using settings from data/config.json."""

    # # Read configuration file
    # from .utils import load_config

    # config = load_config()

    # cities_dict = config["cities"]
    # weather_metrics = config["metrics"]

    # # Initiate forecast object
    # from .forecast import Forecast

    # fc = Forecast()

    # # Prepare lists for batch processing
    # city_names: list[str] = list(cities_dict.keys())
    # latitudes: list[float] = [float(info["lat"]) for info in cities_dict.values()]
    # longitudes: list[float] = [float(info["lon"]) for info in cities_dict.values()]

    # # Set locations as lists to support batch download
    # fc.set_location(city=city_names, lat=latitudes, lon=longitudes)

    # # Set metrics
    # fc.set_metrics(metrics=weather_metrics)

    # # Download forecast data for all cities
    # result_message = fc.download()

    # print(result_message)

    pass


@app.command()
def schedule(action: Annotated[str, typer.Argument(help="Either 'activate' or 'deactivate'")]) -> None:
    """Automates the setup or removal of the systemd timer."""

    pass
