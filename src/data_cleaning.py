
from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/processed/aep_hourly.csv")
OUTPUT_PATH = Path("data/processed/aep_hourly_clean.csv")


def main():
    # 1. Check input file
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH}. "
            "Run data_ingestion.py first."
        )

    # 2. Load dataset
    df = pd.read_csv(INPUT_PATH)
    initial_rows = len(df)

    # 3. Convert columns to correct data types
    df["Datetime"] = pd.to_datetime(
        df["Datetime"], errors="coerce"
    )
    df["AEP_MW"] = pd.to_numeric(
        df["AEP_MW"], errors="coerce"
    )

    # 4. Remove rows with invalid or missing values
    df = df.dropna(subset=["Datetime", "AEP_MW"])

    # 5. Remove exact duplicate records only
    df = df.drop_duplicates()

    # 6. Average demand values for repeated timestamps
    df = (
        df.groupby("Datetime", as_index=False)
        ["AEP_MW"]
        .mean()
    )

    # 7. Sort data chronologically
    df = df.sort_values("Datetime").reset_index(drop=True)

    # 8. Save cleaned dataset
    OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )
    df.to_csv(OUTPUT_PATH, index=False)

    # 9. Print cleaning report
    print("\n--- DATA CLEANING REPORT ---")
    print(f"Rows before cleaning: {initial_rows}")
    print(f"Rows after cleaning: {len(df)}")
    print(f"Rows removed/merged: {initial_rows - len(df)}")
    print(
        "Remaining duplicate timestamps:",
        df["Datetime"].duplicated().sum()
    )
    print(
        "Missing values remaining:",
        df.isnull().sum().sum()
    )
    print(f"Cleaned dataset saved to: {OUTPUT_PATH}")

    print("\nFirst 5 cleaned rows:")
    print(df.head())


if __name__ == "__main__":
    main()