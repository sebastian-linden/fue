from pathlib import Path

from fue import Data


def run_exploration():
    # 1. Initialize and load test data
    data = Data()

    # Dynamically resolve path to ensure it runs from anywhere
    test_dir = Path(__file__).resolve().parent
    test_csv_path = str(test_dir / "test_forecasts.csv")

    print(f"Loading raw data from: {test_csv_path}")
    data.read_raw(path=test_csv_path)

    print("\n" + "=" * 50)
    print("TEST 1: Baseline Default Parameters")
    print("location='london', val_fraction=0.2, random_state=42")
    print("=" * 50)
    train_df, val_df = data.generate_dataset(location_name="london", val_fraction=0.2, random_state=42)
    print(f"Train Shape: {train_df.shape} | Val Shape: {val_df.shape}")
    if not train_df.empty:
        print(
            f"Train Row 0 - day_of_year: {train_df.iloc[0]['day_of_year']}, delta_days: {
                train_df.iloc[0]['delta_days']:.2f}"
        )
    if not val_df.empty:
        print(
            f"Val Row 0   - day_of_year: {val_df.iloc[0]['day_of_year']}, delta_days: {
                val_df.iloc[0]['delta_days']:.2f}"
        )

    print("\n" + "=" * 50)
    print("TEST 2: No Validation Split")
    print("location='london', val_fraction=0.0, random_state=42")
    print("=" * 50)
    train_df_no_val, val_df_no_val = data.generate_dataset(location_name="london", val_fraction=0.0, random_state=42)
    print(f"Train Shape: {train_df_no_val.shape} | Val Shape: {val_df_no_val.shape}")

    print("\n" + "=" * 50)
    print("TEST 3: Reproducibility via Different Random State")
    print("location='london', val_fraction=0.2, random_state=99")
    print("=" * 50)
    train_df_rs, val_df_rs = data.generate_dataset(location_name="london", val_fraction=0.2, random_state=99)
    print(f"Train Shape: {train_df_rs.shape} | Val Shape: {val_df_rs.shape}")
    if not train_df_rs.empty:
        print(
            f"Train Row 0 - day_of_year: {train_df_rs.iloc[0]['day_of_year']}, delta_days: {
                train_df_rs.iloc[0]['delta_days']:.2f}"
        )

    print("\n" + "=" * 50)
    print("TEST 4: Edge Case - Missing City")
    print("location='unknown_city', val_fraction=0.2, random_state=42")
    print("=" * 50)
    train_df_missing, val_df_missing = data.generate_dataset(
        location_name="unknown_city", val_fraction=0.2, random_state=42
    )
    print(f"Train Shape: {train_df_missing.shape} | Val Shape: {val_df_missing.shape}")


if __name__ == "__main__":
    run_exploration()
