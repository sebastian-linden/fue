"""Console script for fue."""

from typing import Annotated

import typer

app = typer.Typer()


@app.command()
def help() -> None:
    """Displays helpful information"""
    help_message = """
    [Placeholder help message]
    """
    print(help_message)
    return None


@app.command()
def download() -> None:
    """Initiates download using settings from data/config.json."""

    from fue import Data

    D = Data()
    D.combine_and_store_forecasts(D.fetch_forecast())
    print("Successfully stored forecasts.")


@app.command()
def schedule(action: Annotated[str, typer.Argument(help="Either 'activate' or 'deactivate'")]) -> None:
    """Automates the setup or removal of the systemd timer."""

    pass
