
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/model_data")


def test_training_files_exist():
    """Check that all training and testing files exist."""
    files = [
        "X_train.csv",
        "y_train.csv",
        "X_test.csv",
        "y_test.csv",
    ]

    for filename in files:
        assert (DATA_DIR / filename).is_file(), (
            f"Missing file: {filename}"
        )


def test_training_data_has_no_missing_values():
    """Check training features and target for missing values."""
    for filename in ["X_train.csv", "y_train.csv"]:
        df = pd.read_csv(DATA_DIR / filename)
        assert not df.isnull().values.any(), (
            f"Missing values found in {filename}"
        )


def test_train_test_features_match():
    """Check training and testing feature columns match."""
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")

    assert list(X_train.columns) == list(X_test.columns)


def test_test_target_has_no_missing_values():
    """Check the test target for missing values."""
    y_test = pd.read_csv(DATA_DIR / "y_test.csv")

    assert not y_test.isnull().values.any()


def test_training_and_testing_data_are_not_empty():
    """Check that all datasets contain rows."""
    files = [
        "X_train.csv",
        "y_train.csv",
        "X_test.csv",
        "y_test.csv",
    ]

    for filename in files:
        df = pd.read_csv(DATA_DIR / filename)
        assert len(df) > 0, f"{filename} is empty"
