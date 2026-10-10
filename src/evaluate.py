
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


PREDICTIONS_PATH = Path("data/predictions/xgboost_predictions.csv")
REPORT_DIR = Path("reports")
REPORT_PATH = REPORT_DIR / "evaluation_report.txt"


def main():
    # 1. Check prediction file
    if not PREDICTIONS_PATH.is_file():
        raise FileNotFoundError(
            f"Predictions file not found: {PREDICTIONS_PATH}. "
            "Run python src/predict.py first."
        )

    # 2. Load predictions
    df = pd.read_csv(PREDICTIONS_PATH)

    required_columns = [
        "Actual_MW",
        "Predicted_MW",
        "Absolute_Error_MW",
    ]

    if not all(column in df.columns for column in required_columns):
        raise ValueError("Prediction file is missing required columns.")

    if df.empty:
        raise ValueError("Prediction file contains no rows.")

    if not np.isfinite(df[required_columns].to_numpy()).all():
        raise ValueError("Prediction data contains invalid numbers.")

    # 3. Extract actual and predicted values
    actual = df["Actual_MW"]
    predicted = df["Predicted_MW"]

    # 4. Calculate evaluation metrics
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    mean_error = (predicted - actual).mean()
    max_absolute_error = df["Absolute_Error_MW"].max()

    # 5. Create evaluation report
    report = (
        "GULF ENERGY AI - MODEL EVALUATION REPORT\n"
        "=======================================\n\n"
        f"Number of predictions: {len(df)}\n"
        f"MAE: {mae:.3f} MW\n"
        f"RMSE: {rmse:.3f} MW\n"
        f"R2 Score: {r2:.4f}\n"
        f"Mean Error (Predicted - Actual): {mean_error:.3f} MW\n"
        f"Maximum Absolute Error: {max_absolute_error:.3f} MW\n\n"
        "Dataset note: Results are based on the AEP historical "
        "energy-demand dataset, not actual Gulf-region data.\n"
    )

    print(report)

    # 6. Save report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"Evaluation report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()