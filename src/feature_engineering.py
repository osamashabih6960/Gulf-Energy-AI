
from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/processed/aep_hourly_clean.csv")
OUTPUT_PATH = Path("data/features/aep_hourly_features.csv")


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {INPUT_PATH}"
        )

    # 1. Load and sort the cleaned data
    df = pd.read_csv(INPUT_PATH)
    df["Datetime"] = pd.to_datetime(
        df["Datetime"], errors="raise"
    )
    df = df.sort_values("Datetime").reset_index(drop=True)

    # 2. Create a timestamp-to-demand lookup
    demand_by_time = (
        df.set_index("Datetime")["AEP_MW"].to_dict()
    )

    # 3. Create time-based features
    df["year"] = df["Datetime"].dt.year
    df["month"] = df["Datetime"].dt.month
    df["day"] = df["Datetime"].dt.day
    df["hour"] = df["Datetime"].dt.hour
    df["day_of_week"] = df["Datetime"].dt.dayofweek
    df["is_weekend"] = (
        df["Datetime"].dt.dayofweek >= 5
    ).astype(int)

    # 4. Look up demand at exact previous timestamps
    df["lag_1"] = (
        df["Datetime"] - pd.Timedelta(hours=1)
    ).map(demand_by_time)

    df["lag_24"] = (
        df["Datetime"] - pd.Timedelta(hours=24)
    ).map(demand_by_time)

    # 5. Target: demand at the exact next-hour timestamp
    df["target_next_hour"] = (
        df["Datetime"] + pd.Timedelta(hours=1)
    ).map(demand_by_time)

    # 6. Remove rows where required timestamps are unavailable
    required = ["lag_1", "lag_24", "target_next_hour"]
    df = df.dropna(subset=required).reset_index(drop=True)

    # 7. Save the final feature dataset
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    # 8. Report
    print("\n--- FEATURE ENGINEERING REPORT ---")
    print(f"Rows after validation: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Missing values in required features: "
          f"{df[required].isna().sum().sum()}")
    print(f"Saved to: {OUTPUT_PATH}")

    print("\nSample rows:")
    print(
        df[
            ["Datetime", "AEP_MW", "lag_1",
             "lag_24", "target_next_hour"]
        ].head()
    )


if __name__ == "__main__":
    main()