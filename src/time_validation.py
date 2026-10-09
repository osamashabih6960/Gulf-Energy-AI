
from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/processed/aep_hourly_clean.csv")


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {INPUT_PATH}. "
            "Run data_cleaning.py first."
        )

    # Load the cleaned dataset
    df = pd.read_csv(INPUT_PATH)
    df["Datetime"] = pd.to_datetime(
        df["Datetime"], errors="raise"
    )
    df = df.sort_values("Datetime").reset_index(drop=True)

    # Calculate the time difference between consecutive rows
    df["time_difference"] = df["Datetime"].diff()

    expected_interval = pd.Timedelta(hours=1)

    # Find intervals that are not exactly one hour
    irregular = df[
        df["time_difference"].notna()
        & (df["time_difference"] != expected_interval)
    ].copy()

    # Count missing hourly timestamps for gaps longer than one hour
    long_gaps = irregular[
        irregular["time_difference"] > expected_interval
    ].copy()

    long_gaps["missing_hours"] = (
        long_gaps["time_difference"] / expected_interval
    ).astype(int) - 1

    print("\n--- TIME VALIDATION REPORT ---")
    print(f"Total rows: {len(df)}")
    print(f"First timestamp: {df['Datetime'].min()}")
    print(f"Last timestamp: {df['Datetime'].max()}")
    print(f"Irregular time intervals: {len(irregular)}")
    print(
        "Gaps longer than one hour: "
        f"{len(long_gaps)}"
    )
    print(
        "Estimated missing hourly timestamps: "
        f"{long_gaps['missing_hours'].sum()}"
    )

    if not irregular.empty:
        print("\nSample irregular intervals:")
        print(
            irregular[
                ["Datetime", "time_difference"]
            ].head(10).to_string(index=False)
        )
    else:
        print("\nAll consecutive timestamps are one hour apart.")

    print("\nTime validation completed.")


if __name__ == "__main__":
    main()