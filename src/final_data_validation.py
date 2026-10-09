
from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/features/aep_hourly_features.csv")


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)
    df["Datetime"] = pd.to_datetime(
        df["Datetime"], errors="raise"
    )

    required_columns = [
        "Datetime",
        "AEP_MW",
        "lag_1",
        "lag_24",
        "target_next_hour",
    ]

    # Check required columns
    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Run quality checks
    missing_values = int(
        df[required_columns].isna().sum().sum()
    )
    duplicate_timestamps = int(
        df["Datetime"].duplicated().sum()
    )
    negative_demand = int(
        (df[["AEP_MW", "lag_1", "lag_24",
             "target_next_hour"]] < 0).sum().sum()
    )
    unsorted_timestamps = not df["Datetime"].is_monotonic_increasing

    print("\n--- FINAL DATA VALIDATION REPORT ---")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Missing values in required columns: {missing_values}")
    print(f"Duplicate timestamps: {duplicate_timestamps}")
    print(f"Negative demand values: {negative_demand}")
    print(f"Timestamps sorted: {not unsorted_timestamps}")

    print("\nTarget summary:")
    print(df["target_next_hour"].describe())

    if (
        missing_values == 0
        and duplicate_timestamps == 0
        and negative_demand == 0
        and not unsorted_timestamps
    ):
        print("\nRESULT: Basic data quality checks passed.")
    else:
        print("\nRESULT: Review the failed checks before training.")


if __name__ == "__main__":
    main()