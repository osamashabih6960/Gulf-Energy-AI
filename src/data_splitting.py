
from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/features/aep_hourly_features.csv")
OUTPUT_DIR = Path("data/splits")

TRAIN_RATIO = 0.80


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {INPUT_PATH}"
        )

    # 1. Load the validated feature dataset
    df = pd.read_csv(INPUT_PATH)
    df["Datetime"] = pd.to_datetime(
        df["Datetime"], errors="raise"
    )

    # 2. Confirm chronological order
    df = df.sort_values("Datetime").reset_index(drop=True)

    if df["Datetime"].duplicated().any():
        raise ValueError("Duplicate timestamps found.")

    if df.isna().any().any():
        raise ValueError(
            "Missing values found. Validate the dataset first."
        )

    # 3. Split by time, without shuffling
    split_index = int(len(df) * TRAIN_RATIO)

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    # 4. Save the datasets
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = OUTPUT_DIR / "train.csv"
    test_path = OUTPUT_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # 5. Print the report
    print("\n--- CHRONOLOGICAL DATA SPLIT REPORT ---")
    print(f"Total rows: {len(df)}")
    print(f"Training rows: {len(train_df)}")
    print(f"Testing rows: {len(test_df)}")

    print("\nTraining period:")
    print(f"Start: {train_df['Datetime'].min()}")
    print(f"End:   {train_df['Datetime'].max()}")

    print("\nTesting period:")
    print(f"Start: {test_df['Datetime'].min()}")
    print(f"End:   {test_df['Datetime'].max()}")

    print("\nSaved files:")
    print(train_path)
    print(test_path)

    print("\nChronological split completed.")


if __name__ == "__main__":
    main()