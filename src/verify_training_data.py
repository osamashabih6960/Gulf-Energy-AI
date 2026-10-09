
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/model_data")

FILES = [
    "X_train.csv",
    "y_train.csv",
    "X_test.csv",
    "y_test.csv",
]


def main():
    # Check that all prepared files exist
    for filename in FILES:
        path = DATA_DIR / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv")

    # Verify input and target row counts
    assert len(X_train) == len(y_train), (
        "Training input and target row counts differ."
    )
    assert len(X_test) == len(y_test), (
        "Testing input and target row counts differ."
    )

    # Verify feature columns match
    assert list(X_train.columns) == list(X_test.columns), (
        "Training and testing features do not match."
    )

    # Verify no missing values
    assert not X_train.isna().any().any()
    assert not y_train.isna().any().any()
    assert not X_test.isna().any().any()
    assert not y_test.isna().any().any()

    print("\n--- TRAINING DATA VERIFICATION ---")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print(f"y_test shape:  {y_test.shape}")
    print(f"Feature columns match: {list(X_train.columns) == list(X_test.columns)}")
    print("Missing values: 0")
    print("\nRESULT: All verification checks passed.")


if __name__ == "__main__":
    main()