import typer
from typing_extensions import Annotated

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

if __name__ == "__main__":
    app()