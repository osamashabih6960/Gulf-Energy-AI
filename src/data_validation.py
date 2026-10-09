
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/processed/aep_hourly.csv")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_PATH}. "
            "Run data_ingestion.py first."
        )

    df = pd.read_csv(DATA_PATH)
    df["Datetime"] = pd.to_datetime(
        df["Datetime"], errors="coerce"
    )
    df["AEP_MW"] = pd.to_numeric(
        df["AEP_MW"], errors="coerce"
    )

    print("\n--- DATA VALIDATION REPORT ---")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    print("\nMissing values:")
    print(df.isnull().sum())

    print(
        "\nDuplicate timestamps:",
        df["Datetime"].duplicated().sum()
    )

    print(
        "Negative demand values:",
        (df["AEP_MW"] < 0).sum()
    )

    valid_dates = df["Datetime"].dropna()
    is_sorted = valid_dates.is_monotonic_increasing
    print(f"Timestamps in chronological order: {is_sorted}")

    print("\nDemand statistics:")
    print(df["AEP_MW"].describe())

    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()