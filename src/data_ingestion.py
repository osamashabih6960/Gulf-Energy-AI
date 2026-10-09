
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/AEP_hourly.csv")
PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = PROCESSED_DIR / "aep_hourly.csv"


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)

    required_columns = ["Datetime", "AEP_MW"]
    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    df = df[required_columns].copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df["AEP_MW"] = pd.to_numeric(df["AEP_MW"], errors="coerce")

    if df.isnull().any().any():
        raise ValueError("Dataset contains invalid or missing values.")

    df = df.sort_values("Datetime").reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("Data ingestion completed!")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Saved to: {OUTPUT_PATH}")
    print(df.head())


if __name__ == "__main__":
    main()