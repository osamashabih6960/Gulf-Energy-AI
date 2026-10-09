
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

    # 1. Check missing values
    print("\nMissing values:")
    print(df.isnull().sum())

    # 2. Check duplicate timestamps
    print(
        "\nDuplicate timestamps:",
        df["Datetime"].duplicated().sum()
    )

    # 3. Display all rows with duplicate timestamps
    duplicate_rows = df[
        df["Datetime"].duplicated(keep=False)
    ].sort_values("Datetime")

    print("\nDuplicate timestamp records:")
    if duplicate_rows.empty:
        print("No duplicate timestamps found.")
    else:
        print(duplicate_rows.to_string(index=False))

    # 4. Check exact duplicate rows
    print(
        "\nExact duplicate rows:",
        df.duplicated().sum()
    )

    # 5. Check negative demand values
    print(
        "\nNegative demand values:",
        (df["AEP_MW"] < 0).sum()
    )

    # 6. Check chronological order
    valid_dates = df["Datetime"].dropna()
    is_sorted = valid_dates.is_monotonic_increasing
    print(
        f"\nTimestamps in chronological order: {is_sorted}"
    )

    # 7. Demand statistics
    print("\nDemand statistics:")
    print(df["AEP_MW"].describe())

    # 8. Display first five rows
    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()