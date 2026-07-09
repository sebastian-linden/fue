import typer
from typing_extensions import Annotated

import json
import logging
from pathlib import Path
from typing import Optional

import joblib

# The callback handles the top-level app description
app = typer.Typer(help="FUE: Forecast Uncertainty Estimation CLI.")

@app.callback()
def main():
    """
    Main entry point for the FUE CLI. Use a sub-command to execute actions.
    """
    pass

@app.command()
def download():
    """
    Fetches raw forecast data from the Open-Meteo API and stores it 
    based on the current config.json settings.
    """
    from fue import Data
    D = Data()
    D.combine_and_store_forecasts(D.fetch_forecast())
    print("Successfully stored forecasts.")

@app.command()
def dataset_summary(
    threshold: Annotated[
        int, 
        typer.Option(
            min=0, 
            help="The minimum number of valid records required for a city to be considered 'active' in the pipeline."
        )
    ] = 100
):
    """
    Displays a summary report of the current FUE data inventory, including:
    - Total unique cities tracked
    - Number of cities active in the pipeline (>= threshold rows)
    - Number of cities awaiting graduation (< threshold rows)
    - A detailed table of each city's valid records and pipeline status.
    Args:
        threshold (int): The minimum number of valid records required for a city to be considered "active" in the pipeline. Default is 100.
    
    """
    from fue.data import Data
    
    data = Data()
    data.read_raw()
    
    # Fetch the calculation matrix from the data layer
    summary_df = data.get_collection_summary(threshold=threshold)
    
    total_cities = len(summary_df)
    active_cities = len(summary_df[summary_df["status"] == "ACTIVE"])
    waiting_cities = total_cities - active_cities
    
    # Format the terminal presentation block
    print("\n=== FUE DATA INVENTORY STATUS ===")
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


# --- Helper Functions ---

def get_latest_run_id(runs_dir: Path) -> Optional[str]:
    """Finds the most recently created run directory."""
    if not runs_dir.exists() or not runs_dir.is_dir():
        return None
    
    # List all subdirectories, sort by creation/modification time (newest last)
    subdirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    
    latest_dir = max(subdirs, key=lambda d: d.stat().st_mtime)
    return latest_dir.name

# --- Typer Commands ---

@app.command()
def train(
    model_type: Annotated[str, typer.Option("--model", help="Type of model to train: 'ml' or 'linear'")] = "ml",
    alpha: Annotated[Optional[float], typer.Option(help="Regularization strength (ML only)")] = 0.01,
    layers: Annotated[Optional[str], typer.Option(help="Hidden layer sizes, e.g., '16,8' (ML only)")] = "16,8"
):
    """
    Trains a new Uncertainty Model and saves it to the runs/ directory.
    """
    from fue.config import Config
    from fue.data import Data
    from fue.models import LinearUncertaintyModel, MLUncertaintyModel
    from fue.utils import generate_run_id

    typer.echo(f"Initializing training pipeline for model type: {model_type.upper()}")
    
    try:
        config = Config()
        data = Data()
        data.read_raw()
        dataset = data.generate_dataset()
        
        typer.echo("Splitting dataset chronologically...")
        train_df, val_df = data.split_dataset(dataset, val_fraction=0.2)
        
        # Instantiate correct model
        if model_type.lower() == "ml":
            parsed_layers = tuple(int(x.strip()) for x in layers.split(","))
            model = MLUncertaintyModel(hidden_layer_sizes=parsed_layers, alpha=alpha)
        elif model_type.lower() == "linear":
            model = LinearUncertaintyModel(config)
        else:
            typer.secho(f"Error: Unknown model type '{model_type}'. Choose 'ml' or 'linear'.", fg=typer.colors.RED)
            raise typer.Abort()

        typer.echo("Fitting model...")
        model.fit(train_df, config.default_feature_columns, config.default_target_columns)

        # Save state
        run_id = generate_run_id(purpose=f"train_{model_type}")
        run_dir = Path("runs") / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = run_dir / "model.joblib"
        meta_path = run_dir / "meta.json"
        
        joblib.dump(model, model_path)
        
        # Save metadata
        meta = {
            "run_id": run_id,
            "model_type": model_type,
            "train_size": len(train_df),
            "val_size": len(val_df)
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=4)
            
        typer.secho(f"✅ Successfully trained and saved model to {run_dir}", fg=typer.colors.GREEN)
        
    except Exception as e:
        typer.secho(f"❌ Training failed: {e}", fg=typer.colors.RED)
        raise typer.Abort()

@app.command()
def evaluate(
    run_id: Annotated[Optional[str], typer.Option(help="Specific run ID to evaluate. Defaults to latest.")] = None
):
    """
    Evaluates a saved model against the current validation dataset.
    """
    from fue.config import Config
    from fue.data import Data
    
    runs_dir = Path("runs")
    if not run_id:
        run_id = get_latest_run_id(runs_dir)
        if not run_id:
            typer.secho("❌ No models found. Please run 'fue train' first.", fg=typer.colors.RED)
            raise typer.Abort()
            
    model_path = runs_dir / run_id / "model.joblib"
    if not model_path.exists():
        typer.secho(f"❌ Model file not found at {model_path}", fg=typer.colors.RED)
        raise typer.Abort()
        
    try:
        typer.echo(f"Loading model from run: {run_id}")
        model = joblib.load(model_path)
        
        data = Data()
        data.read_raw()
        dataset = data.generate_dataset()
        _, val_df = data.split_dataset(dataset, val_fraction=0.2)
        
        typer.echo(f"Evaluating against {len(val_df)} validation records...")
        metrics = model.evaluate(val_df)
        
        print("\n=== EVALUATION RESULTS ===")
        for target, scores in metrics.items():
            print(f"{target:<40} | MAE: {scores['MAE']:.4f} | RMSE: {scores['RMSE']:.4f}")
            
    except Exception as e:
        typer.secho(f"❌ Evaluation failed: {e}", fg=typer.colors.RED)
        raise typer.Abort()

@app.command()
def forecast(
    loc: Annotated[str, typer.Option("--city", help="City name, e.g., 'aachen'")],
    days: Annotated[int, typer.Option("--days", help="Days to forecast")] = 7,
    run_id: Annotated[Optional[str], typer.Option(help="Specific run ID to use. Defaults to latest.")] = None,
    plot: Annotated[bool, typer.Option("--plot", help="Pop open a matplotlib window instead of terminal table")] = False
):
    """
    Fetches a forecast for a city and estimates uncertainty bounds using a trained model.
    """
    from fue.forecast import Forecast
    
    runs_dir = Path("runs")
    if not run_id:
        run_id = get_latest_run_id(runs_dir)
        if not run_id:
            typer.secho("❌ No models found. Please run 'fue train' first.", fg=typer.colors.RED)
            raise typer.Abort()

    model_path = runs_dir / run_id / "model.joblib"
    if not model_path.exists():
        typer.secho(f"❌ Model file not found at {model_path}", fg=typer.colors.RED)
        raise typer.Abort()

    try:
        typer.echo(f"Loading model from run: {run_id}")
        model = joblib.load(model_path)
        
        F = Forecast()
        typer.echo(f"Fetching {days}-day forecast for {loc.capitalize()}...")
        F.fetch_forecast(location_name=loc.lower(), forecast_days=days)
        
        typer.echo("Computing uncertainties...")
        F.compute_uncertainties(model)
        
        if plot:
            typer.echo("Rendering plot window...")
            F.plot()
        else:
            # Map target columns to their base forecast keys and cleaner header labels
            mapping = {
                "abs_diff__temperature_2m_max": ("temperature_2m_max", "t_max"),
                "abs_diff__temperature_2m_min": ("temperature_2m_min", "t_min"),
                "abs_diff__precipitation_sum": ("precipitation_sum", "prcp_sum"),
                "abs_diff__wind_speed_10m_mean": ("wind_speed_10m_mean", "wind_mean"),
                "abs_diff__precipitation_probability_mean": ("precipitation_probability_mean", "prcp_prob")
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
            
            for idx in range(len(df_base)):
                # Handle dates safely if it's a Column or a Pandas Index
                date_val = str(df_base.index[idx]) if "date" not in df_base.columns else str(df_base["date"].iloc[idx])
                # Truncate timestamp strings to just YYYY-MM-DD if needed
                date_str = date_val.split(" ")[0][:10]
                
                row_cells = []
                for target_col, (base_col, _) in mapping.items():
                    val = df_base[base_col].iloc[idx]
                    err = df_unc[target_col].iloc[idx]
                    unit = F._KNOWN_UNITS.get(target_col, "")
                    
                    # Format as: value (± error) unit
                    cell_text = f"{val:.1f} (± {err:.1f}) {unit}"
                    row_cells.append(f"{cell_text:<22}")
                
                print(f"{date_str:<12} | " + " | ".join(row_cells))
            print()
            
    except Exception as e:
        typer.secho(f"❌ Forecasting failed: {e}", fg=typer.colors.RED)
        raise typer.Abort()


@app.command()
def tune():
    """
    Runs Hyperparameter Optimization (HPO) for the ML model, logs metrics,
    and statefully stores the optimal model configuration and weights.
    """
    import logging
    from pathlib import Path
    
    from fue.config import Config
    from fue.data import Data
    from fue.models import MLUncertaintyModel
    from fue.tuner import HyperparameterTuner
    from fue.utils import generate_run_id

    run_id = generate_run_id(purpose="tune")
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Send detailed logs to the file, while keeping the terminal quiet
    log_file = run_dir / "tune_debug.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO, # INFO captures your Tuner's iteration outputs
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    typer.echo(f"Starting HPO pipeline. Detailed iterations logged to: {log_file}")
    
    try:
        config = Config()
        feature_columns = config.default_feature_columns
        target_columns = config.default_target_columns

        data = Data()
        data.read_raw()
        dataset = data.generate_dataset()
        
        typer.echo("Splitting dataset chronologically...")
        train_df, val_df = data.split_dataset(dataset, val_fraction=0.2, min_entries_per_city=100)

        param_grid = {
            "hidden_layer_sizes": [(32, 16), (16, 8), (16,)],
            "alpha": [0.001, 0.01, 0.1],
            "ensemble_size": [3, 5],
            "max_iter": [1000]
        }

        tuner = HyperparameterTuner(
            model_class=MLUncertaintyModel, # Works perfectly with generalized Tuner
            param_grid=param_grid,
            optimization_target="global",
            optimization_metric="RMSE"
        )

        typer.echo("Searching for optimal hyperparameters (Warnings suppressed)...")
        
        # Unpack correctly matching the tuple returned by search()
        best_model, history = tuner.search(
            train_df=train_df, 
            val_df=val_df, 
            feature_columns=feature_columns, 
            target_columns=target_columns,
            run_id=run_id
        )
        
        # Grab the best score from the history list for terminal display
        best_score = min([run["score"] for run in history])
        typer.echo(f"Optimal parameters determined with validation RMSE of {best_score:.4f}")
        typer.secho(f"✅ Tuning complete! Champion model saved to {run_dir}", fg=typer.colors.GREEN)
        
    except Exception as e:
        typer.secho(f"❌ Tuning failed: {e}", fg=typer.colors.RED)
        raise typer.Abort()


if __name__ == "__main__":
    app()