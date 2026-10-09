
from pathlib import Path
import pandas as pd

INPUT_DIR = Path("data/splits")
OUTPUT_DIR = Path("data/model_data")

TARGET = "target_next_hour"

FEATURES = [
    "AEP_MW",
    "lag_1",
    "lag_24",
    "year",
    "month",
    "day",
    "hour",
    "day_of_week",
    "is_weekend",
]


def main():
    train_path = INPUT_DIR / "train.csv"
    test_path = INPUT_DIR / "test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Train/test files not found. "
            "Run src/data_splitting.py first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    required_columns = FEATURES + [TARGET]

    for name, df in [
        ("training", train_df),
        ("testing", test_df),
    ]:
        missing = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{name} dataset is missing: {missing}"
            )

        if df[required_columns].isna().any().any():
            raise ValueError(
                f"Missing values found in {name} data."
            )

    # Separate inputs and prediction targets
    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    # Save the prepared datasets
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
    y_train.to_frame().to_csv(
        OUTPUT_DIR / "y_train.csv", index=False
    )

    X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
    y_test.to_frame().to_csv(
        OUTPUT_DIR / "y_test.csv", index=False
    )

    print("\n--- TRAINING DATA PREPARATION REPORT ---")
    print(f"Training input shape: {X_train.shape}")
    print(f"Training target shape: {y_train.shape}")
    print(f"Testing input shape: {X_test.shape}")
    print(f"Testing target shape: {y_test.shape}")
    print(f"\nFeatures: {FEATURES}")
    print(f"Target: {TARGET}")
    print(f"Saved to: {OUTPUT_DIR}")
    print("\nData preparation completed.")


if __name__ == "__main__":
    main()