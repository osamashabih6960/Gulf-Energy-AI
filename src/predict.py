
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


DATA_DIR = Path("data/model_data")
TEST_SPLIT_PATH = Path("data/splits/test.csv")
MODEL_PATH = Path("models/xgboost_energy_model.pkl")

OUTPUT_DIR = Path("data/predictions")
OUTPUT_PATH = OUTPUT_DIR / "xgboost_predictions.csv"


def main():
    # 1. Check required files
    for path in [MODEL_PATH, TEST_SPLIT_PATH]:
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    # 2. Load the model and test data
    model = joblib.load(MODEL_PATH)

    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")
    test_split = pd.read_csv(TEST_SPLIT_PATH)

    # 3. Validate row alignment
    if len(X_test) != len(y_test) or len(X_test) != len(test_split):
        raise ValueError("Test feature, target, and split row counts differ.")

    if list(X_test.columns) != list(model.feature_names_in_):
        raise ValueError("Test feature columns do not match model features.")

    if "Datetime" not in test_split.columns:
        raise ValueError("Datetime column is missing from test split.")

    if "target_next_hour" not in test_split.columns:
        raise ValueError("target_next_hour is missing from test split.")

    # Ensure the saved target rows match the source split
    if not np.allclose(
        y_test.to_numpy(),
        test_split["target_next_hour"].to_numpy(),
    ):
        raise ValueError("Target rows do not align with the test split.")

    # 4. Generate predictions
    predictions = model.predict(X_test)

    if not np.isfinite(predictions).all():
        raise ValueError("Predictions contain NaN or infinite values.")

    # 5. Calculate metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\nPrediction Test Successful!")
    print("---------------------------")
    print(f"Number of predictions: {len(predictions)}")
    print(f"MAE: {mae:.3f} MW")
    print(f"RMSE: {rmse:.3f} MW")
    print(f"R2 Score: {r2:.4f}")

    # 6. Build results with the actual target timestamp
    results = pd.DataFrame({
        "Target_Datetime": (
            pd.to_datetime(test_split["Datetime"])
            + pd.Timedelta(hours=1)
        ),
        "Actual_MW": y_test.to_numpy(),
        "Predicted_MW": predictions,
    })

    results["Absolute_Error_MW"] = np.abs(
        results["Actual_MW"] - results["Predicted_MW"]
    )

    # 7. Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print(f"\nPredictions saved to: {OUTPUT_PATH}")
    print("\nFirst five predictions:")
    print(results.head().to_string(index=False))

    # 8. Display the largest-error case
    largest_error_index = results["Absolute_Error_MW"].idxmax()

    print("\nLargest-error prediction:")
    print(results.loc[largest_error_index].to_string())


if __name__ == "__main__":
    main()
