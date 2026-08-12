"""
Command-line interface (CLI) for the Forecast Uncertainty Estimation (pyfue) package.

This module utilizes Typer to provide terminal-based execution of core pipeline
tasks, including data ingestion, dataset summarization, and model hyperparameter tuning.
"""

import json
import logging
from pathlib import Path
from typing import Annotated

import joblib
import typer

# Initialize module-scoped logger
logger = logging.getLogger(__name__)

# The callback handles the top-level app description
app = typer.Typer(help="pyfue: Forecast Uncertainty Estimation CLI.")


@app.callback()
def main():
    """
    Main entry point for the pyfue Command Line Interface.

    This function acts as the parent callback for the Typer application,
    establishing the root `pyfue` command from which all sub-commands
    (e.g., `init`, `download`, `train`) are executed.

    Returns
    -------
    None
    """
    pass


@app.command()
def init():
    """
    Initializes a new pyfue workspace in the current directory.

    Creates a local `config.json` and a `runs/` directory, allowing you to
    isolate your data and experiments cleanly per project. It interactively
    prompts the user to define their local workspace preferences.

    Returns
    -------
    None

    Raises
    ------
    typer.Abort
        If the packaged default configuration template cannot be located.
    """
    import os

    from .defaults import CONFIGURATION

    typer.secho("Initializing pyfue project workspace...", fg=typer.colors.CYAN, bold=True)

    # Prompt user for workspace preferences
    config_file = typer.prompt(
        "Where should configuration data be stored? (relative to this folder)", default="config.json"
    )
    data_file = typer.prompt("Where should forecast data be stored? (relative to this folder)", default="forecasts.csv")
    runs_dir_input = typer.prompt("Where should training runs be stored? (relative to this folder)", default="runs")

    config_dest = Path(config_file)
    runs_dir = Path(runs_dir_input)
    data_dest = Path(data_file)

    # Create config file
    with open(config_dest, "w", encoding="utf-8") as file:
        config_data = CONFIGURATION
        config_data["data_path"] = str(data_dest)
        config_data["runs_dir"] = str(runs_dir)
        json.dump(config_data, file, indent=4)
    logger.info(f"Configuration saved to {config_dest}")

    # Create data file
    with open(data_dest, "w") as _:
        logger.info(f"Data file created at {data_dest}")

    # Create runs directory
    os.mkdir(runs_dir)
    logger.info(f"run/ directory created at {runs_dir}")

    # Output success message
    typer.secho("\nWorkspace initialized successfully!", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"Config created at: {config_dest.name}")
    typer.echo(f"Runs directory created at: {runs_dir.name}/")
    typer.echo(f"Data target set to: {data_file}")
    typer.echo("\nYou can now run 'pyfue download' to fetch your first data points.")


# --- Helper Functions ---


def get_latest_run_id(runs_dir: Path) -> str:
    """
    Finds the most recently created run directory within the specified folder.

    Parameters
    ----------
    runs_dir : pathlib.Path
        The path to the directory containing model run folders.

    Returns
    -------
    str or None
        The name of the most recently modified run folder, or None if the directory
        does not exist or contains no valid subdirectories.
    """
    if not runs_dir.exists() or not runs_dir.is_dir():
        return ""

    # Get all subdirectories in the runs folder
    run_folders = [d for d in runs_dir.iterdir() if d.is_dir()]
    if not run_folders:
        return ""

    # Return the name of the most recently modified folder
    latest_run = max(run_folders, key=lambda d: d.stat().st_mtime)
    return latest_run.name


# --- Typer Commands ---


@app.command()
def download(
    config_path: Annotated[str, typer.Option("--config", help="Path to custom config.json")] = "",
):
    """
    Fetches and stores raw weather forecast data via the Open-Meteo API.

    This command reads the target cities and geographical parameters from the
    local config settings, requests the corresponding data using the Data
    layer, and persistently stores the combined results on disk.

    Parameters
    ----------
    config_path : pathlib.Path, optional
        Path to a custom config.json file.
    data_path : pathlib.Path, optional
        Path to a custom forecast data tracking file.

    Returns
    -------
    None

    Raises
    ------
    typer.Abort
        If the workspace configuration cannot be found.
    """
    from pyfue.config import Config
    from pyfue.data import Data

    try:
        config = Config(path=Path(config_path))
        print(config.data_path)
        D = Data(config)

        typer.echo(f"Downloading data for {len(config.cities)} cities...")
        forecast = D.fetch_forecast()
        typer.echo(f"Successfully fetched new forecast records.")
        D.combine_and_store_forecasts(forecast)
        typer.secho("✅ Successfully stored forecasts.", fg=typer.colors.GREEN)
    except FileNotFoundError as e:
        typer.secho(f"❌ Initialization Error: {e}", fg=typer.colors.RED)
        typer.echo("Did you forget to run 'pyfue init' in this directory?")
        raise typer.Abort() from e


@app.command()
def dataset_summary(
    threshold: Annotated[
        int, typer.Option("--t", min=0, help="Minimum records required for a city to be 'active'.")
    ] = 100,
    config_path: Annotated[str, typer.Option("--config", help="Path to custom config.json")] = "",
):
    """
    Generates and prints a summary report of the current local data inventory.

    Evaluates the historical records assembled by the pipeline and outputs a
    color-coded terminal table detailing which cities have gathered enough data
    points to be actively used during model training splits.

    Parameters
    ----------
    threshold : int, default=100
        The minimum number of valid records required for a city to be active.
    config_path : pathlib.Path, optional
        Path to a custom config.json file.

    Returns
    -------
    None
    """
    from pyfue.config import Config
    from pyfue.data import Data

    try:
        config = Config(Path(config_path))
        data = Data(config)

        # Fetch the calculation matrix from the data layer
        summary_df = data.get_collection_summary(threshold=threshold)

        total_cities = len(summary_df)
        active_cities = len(summary_df[summary_df["status"] == "ACTIVE"])
        waiting_cities = total_cities - active_cities

        # Format the terminal presentation block
        print("\n=== pyfue DATA INVENTORY STATUS ===")
        print(f"Total Unique Cities Tracked: {total_cities}")
        print(f"Active in Pipeline (>= {threshold} rows): {active_cities}")
        print(f"Awaiting Graduation (< {threshold} rows): {waiting_cities}")
        print("=" * 50)
        print(f"{'City Name':<20} | {'Valid Records':<13} | {'Pipeline Status'}")
        print("-" * 50)

        for _, row in summary_df.iterrows():
            icon = "✅ ACTIVE" if row["status"] == "ACTIVE" else "⏳ WAITING"
            print(f"{row['location_name']:<20} | {row['valid_records']:<13} | {icon}")
        print("=" * 50 + "\n")

    except Exception as e:
        typer.secho(f"❌ Error: {e}", fg=typer.colors.RED)
        raise typer.Abort() from e


@app.command()
def train(
    model_type: Annotated[str, typer.Option("--model", help="Type of model to train: 'ml' or 'linear'")] = "ml",
    alpha: Annotated[float, typer.Option(help="Regularization strength (ML only)")] = 0.01,
    layers: Annotated[str, typer.Option(help="Hidden layer sizes, e.g., '16,8' (ML only)")] = "16,8",
    config_path: Annotated[str, typer.Option("--config", help="Path to custom config.json")] = "",
):
    """
    Executes the training pipeline for a specified uncertainty estimation model.

    Ingests the local dataset, performs a safe chronological train/validation split,
    instantiates the requested estimator, fits the data statefully, evaluates out-of-sample
    metrics, and saves the trained model checkpoint to the runs directory.

    Parameters
    ----------
    model_type : str, default="ml"
        The class of model to train ('ml' for MLP, 'linear' for regression).
    alpha : float, optional, default=0.01
        The L2 penalty regularization parameter for MLP models.
    layers : str, optional, default="16,8"
        Comma-separated string denoting hidden layer neurons for MLP models.
    config_path : pathlib.Path, optional
        Path to a custom config.json file.

    Returns
    -------
    None

    Raises
    ------
    typer.Abort
        If an unrecognized `model_type` is passed or an uncaught exception drops
        during the pipeline ingestion and serialization phases.
    """
    from pyfue.config import Config
    from pyfue.data import Data
    from pyfue.models import LinearUncertaintyModel, MLUncertaintyModel
    from pyfue.utils import generate_run_id

    typer.echo(f"Initializing training pipeline for model type: {model_type.upper()}")

    try:
        config = Config(Path(config_path))
        data = Data(config)
        dataset = data.generate_dataset()
        train_df, val_df = data.split_dataset(dataset, val_fraction=0.2)

        # Instantiate correct model
        if model_type.lower() == "ml":
            parsed_layers = tuple(int(x.strip()) for x in layers.split(","))
            model = MLUncertaintyModel(config, hidden_layer_sizes=parsed_layers, alpha=alpha)
            model = LinearUncertaintyModel(config=config)
        else:
            typer.secho(f"Error: Unknown model type '{model_type}'. Choose 'ml' or 'linear'.", fg=typer.colors.RED)
            raise typer.Abort()

        typer.echo("Fitting model...")
        model.fit(train_df, config.feature_columns, config.target_columns)

        # Save state
        run_id = generate_run_id(purpose=f"train_{model_type}")
        run_dir = Path("runs") / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        model_path = run_dir / "model.joblib"
        meta_path = run_dir / "meta.json"

        joblib.dump(model, model_path)

        # Save metadata
        meta = {"run_id": run_id, "model_type": model_type, "train_size": len(train_df), "val_size": len(val_df)}
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=4)

        typer.secho(f"✅ Successfully trained and saved model to {run_dir}", fg=typer.colors.GREEN)

    except Exception as e:
        typer.secho(f"❌ Training failed: {e}", fg=typer.colors.RED)
        raise typer.Abort() from e


@app.command()
def evaluate(
    run_id: Annotated[str, typer.Option(help="Specific run ID to evaluate. Defaults to latest.")] = "",
    config_path: Annotated[str, typer.Option("--config", help="Path to custom config.json")] = "",
):
    """
    Validates a saved model's predictive skill using historical validation holdouts.

    Loads a trained checkpoint from the local disk and passes the out-of-sample
    validation fraction through its predictive layer, outputting scoring metrics
    (like MAE and RMSE) directly to the terminal.

    Parameters
    ----------
    run_id : str, optional
        The specific directory ID string inside the runs folder. If None, the
        most recently modified run is used.
    config_path : pathlib.Path, optional
        Path to a custom config.json file.
    data_path : pathlib.Path, optional
        Path to a custom forecast data tracking file.

    Returns
    -------
    None

    Raises
    ------
    typer.Abort
        If no serialized model checkpoints are present on disk, if the specified
        `run_id` folder does not exist, or if data streaming fails.
    """
    import os

    from pyfue.data import Config, Data

    runs_dir = Path.cwd() / "runs"
    if run_id is None:
        run_id = get_latest_run_id(Path.cwd() / "runs")
        if not run_id:
            typer.secho("❌ No runs found in the local ./runs directory.", fg=typer.colors.RED)
            raise typer.Abort()
    model_path = os.path.join(runs_dir, run_id, "model.joblib")

    try:
        typer.echo(f"Loading model from run: {run_id}")
        model = joblib.load(model_path)

        config = Config(Path(config_path))
        data = Data(config)
        dataset = data.generate_dataset()
        _, val_df = data.split_dataset(dataset, val_fraction=0.2)

        typer.echo(f"Evaluating against {len(val_df)} validation records...")
        metrics = model.evaluate(val_df)

        print("\n=== EVALUATION RESULTS ===")
        for target, scores in metrics.items():
            print(f"{target:<40} | MAE: {scores['MAE']:.4f} | RMSE: {scores['RMSE']:.4f}")

    except Exception as e:
        typer.secho(f"❌ Evaluation failed: {e}", fg=typer.colors.RED)
        raise typer.Abort() from e


@app.command()
def forecast(
    loc: Annotated[str, typer.Option("--city", help="City name, e.g., 'aachen'")],
    days: Annotated[int, typer.Option("--days", help="Days to forecast")] = 7,
    run_id: Annotated[str, typer.Option("--id", help="Specific run ID to use. Defaults to latest.")] = "",
    plot: Annotated[
        bool, typer.Option("--plot", help="Pop open a matplotlib window instead of terminal table")
    ] = False,
    config_path: Annotated[str, typer.Option("--config", help="Path to custom config.json")] = "",
):
    """
    Generates forward operational forecasts decorated with custom uncertainty bounds.

    Pulls live weather predictions using the Open-Meteo API for a requested city
    coordinate space over an arbitrary horizon window, runs the point predictions
    through a selected checkpoint uncertainty model, and maps out a tabular daily
    terminal layout or populates an interactive matplotlib visualization highlighting
    expected physical forecast fluctuations ($\\pm \text{error}$).

    Parameters
    ----------
    loc : str
        The designated city location query key matching an item tracked in the
        coordinates system dictionary config (e.g., 'aachen', 'london').
    days : int, default=7
        The horizontal lead length of the forward operational prediction window on
        the target API request.
    run_id : str or None, default=None
        The tracking folder token for the uncertainty estimator checkpoint model to load.
        If None, targets the latest matching local run directory.
    plot : bool, default=False
        Toggle flag governing display. If True, maps out the metric tracking sequences
        via an external matplotlib display window. If False, streams formatted string
        blocks directly into standard output.

    Returns
    -------
    None

    Raises
    ------
    typer.Abort
        If model files are unreadable, API communication errors manifest, or index mapping
        mismatches interrupt cell formatting steps.
    """
    from pyfue.config import Config
    from pyfue.defaults import UNITS
    from pyfue.forecast import Forecast

    runs_dir = Path("runs")
    if not run_id:
        run_id = get_latest_run_id(runs_dir)
        if run_id == "":
            typer.secho("❌ No models found. Please run 'pyfue train' first.", fg=typer.colors.RED)
            raise typer.Abort()

    model_path = runs_dir / run_id / "model.joblib"
    if not model_path.exists():
        typer.secho(f"❌ Model file not found at {model_path}", fg=typer.colors.RED)
        raise typer.Abort()

    try:
        typer.echo(f"Loading model from run: {run_id}")
        model = joblib.load(model_path)

        config = Config(Path(config_path))
        F = Forecast(config)
        typer.echo(f"Fetching {days}-day forecast for {loc.capitalize()}...")
        F.fetch_forecast(location_name=loc.lower(), forecast_days=days)

        typer.echo("Computing uncertainties...")
        F.compute_uncertainties(model)

        if plot:
            typer.echo("Rendering plot window...")

            F.plot(config.target_columns)
        else:
            # Map target columns to their base forecast keys and cleaner header labels
            mapping = {
                "abs_diff__temperature_2m_max": ("temperature_2m_max", "t_max"),
                "abs_diff__temperature_2m_min": ("temperature_2m_min", "t_min"),
                "abs_diff__precipitation_sum": ("precipitation_sum", "prcp_sum"),
                "abs_diff__wind_speed_10m_mean": ("wind_speed_10m_mean", "wind_mean"),
                "abs_diff__precipitation_probability_mean": ("precipitation_probability_mean", "prcp_prob"),
            }

            # Print clean terminal headers
            typer.secho(f"\n=== WEATHER FORECAST WITH UNCERTAINTY BOUNDS: {loc.upper()} ===", fg=typer.colors.CYAN)
            header_str = f"{'Date':<12} | " + " | ".join([f"{lbl:<22}" for _, lbl in mapping.values()])
            print(header_str)
            print("-" * len(header_str))

            # Iterate daily through rows (assuming index contains dates or aligned frames)
            # Open-Meteo dataframes generated by fetch_forecast contain 'date' or a time-index
            df_base = F.forecast
            df_unc = F.uncertainty_predictions

            for idx in range(len(df_base)):  # ty: ignore [invalid-argument-type]
                # Handle dates safely if it's a Column or a Pandas Index
                date_val = str(df_base.index[idx]) if "date" not in df_base.columns else str(df_base["date"].iloc[idx])  # ty: ignore [not-subscriptable, unresolved-attribute]
                # Truncate timestamp strings to just YYYY-MM-DD if needed
                date_str = date_val.split(" ")[0][:10]

                row_cells = []
                for target_col, (base_col, _) in mapping.items():
                    val = df_base[base_col].iloc[idx]  # ty: ignore [not-subscriptable]
                    err = df_unc[target_col].iloc[idx]  # ty: ignore [not-subscriptable]
                    unit = UNITS.get(target_col, "")

                    # Format as: value (± error) unit
                    cell_text = f"{val:.1f} (± {err:.1f}) {unit}"
                    row_cells.append(f"{cell_text:<22}")

                print(f"{date_str:<12} | " + " | ".join(row_cells))
            print()

    except Exception as e:
        typer.secho(f"❌ Forecasting failed: {e}", fg=typer.colors.RED)
        raise typer.Abort() from e


@app.command()
def tune(
    config_path: Annotated[str, typer.Option("--config", help="Path to custom config.json")] = "",
):
    """
    Executes hyperparameter optimization (HPO) across the active dataset.

    Ingests the locally stored raw data, applies a strict chronological and
    city-stratified split to prevent temporal leakage, and runs a grid search
    using the HyperparameterTuner. The best performing model (champion) and
    its training history are automatically serialized to the runs directory.

    Returns
    -------
    None
    """
    from pathlib import Path

    from pyfue.config import Config
    from pyfue.data import Data
    from pyfue.models import MLUncertaintyModel
    from pyfue.tuner import HyperparameterTuner
    from pyfue.utils import generate_run_id

    run_id = generate_run_id(purpose="tune")
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Send detailed logs to the file, while keeping the terminal quiet
    log_file = run_dir / "tune_debug.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,  # INFO captures your Tuner's iteration outputs
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    typer.echo(f"Starting HPO pipeline. Detailed iterations logged to: {log_file}")

    try:
        config = Config(Path(config_path))
        feature_columns = config.feature_columns
        target_columns = config.target_columns

        data = Data(config)
        dataset = data.generate_dataset()

        typer.echo("Splitting dataset chronologically...")
        train_df, val_df = data.split_dataset(dataset, val_fraction=0.2, min_entries_per_city=100)

        param_grid = {
            "hidden_layer_sizes": [(32, 16), (16, 8), (16,)],
            "alpha": [0.001, 0.01, 0.1],
            "ensemble_size": [3, 5],
            "max_iter": [1000],
        }

        tuner = HyperparameterTuner(
            model_class=MLUncertaintyModel,  # Works perfectly with generalized Tuner
            param_grid=param_grid,
            optimization_target="global",
            optimization_metric="RMSE",
        )

        typer.echo("Searching for optimal hyperparameters (Warnings suppressed)...")

        # Unpack correctly matching the tuple returned by search()
        best_model, history = tuner.search(
            config=config,
            train_df=train_df,
            val_df=val_df,
            feature_columns=feature_columns,
            target_columns=target_columns,
            run_id=run_id,
        )

        # Grab the best score from the history list for terminal display
        best_score = min([run["score"] for run in history])
        typer.echo(f"Optimal parameters determined with validation RMSE of {best_score:.4f}")
        typer.secho(f"✅ Tuning complete! Champion model saved to {run_dir}", fg=typer.colors.GREEN)

    except Exception as e:
        typer.secho(f"❌ Tuning failed: {e}", fg=typer.colors.RED)
        raise typer.Abort() from e


if __name__ == "__main__":
    app()
