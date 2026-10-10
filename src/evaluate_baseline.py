
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = Path("data/model_data")
PREDICTIONS_PATH = Path("data/predictions/xgboost_predictions.csv")
REPORT_PATH = Path("reports/baseline_comparison.txt")


def calculate_metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
    }


def main():
    x_path = DATA_DIR / "X_test.csv"
    y_path = DATA_DIR / "y_test.csv"

    for path in [x_path, y_path, PREDICTIONS_PATH]:
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    X_test = pd.read_csv(x_path)
    y_test = pd.read_csv(y_path).squeeze("columns")
    predictions = pd.read_csv(PREDICTIONS_PATH)

    if len(X_test) != len(y_test) or len(y_test) != len(predictions):
        raise ValueError("Test data and prediction row counts do not match.")

    required = {"Actual_MW", "Predicted_MW"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"Prediction file must contain: {required}")

    if "AEP_MW" not in X_test.columns:
        raise ValueError("AEP_MW is missing from X_test.csv.")

    actual = y_test.to_numpy()
    baseline_predictions = X_test["AEP_MW"].to_numpy()
    xgb_actual = predictions["Actual_MW"].to_numpy()
    xgb_predictions = predictions["Predicted_MW"].to_numpy()

    if not np.allclose(actual, xgb_actual):
        raise ValueError("Prediction actual values do not match y_test.")

    if not np.isfinite(baseline_predictions).all():
        raise ValueError("Baseline predictions contain invalid values.")

    baseline = calculate_metrics(actual, baseline_predictions)
    xgboost = calculate_metrics(actual, xgb_predictions)

    print("\nGULF ENERGY AI - BASELINE COMPARISON")
    print("=" * 42)
    print(f"{'Metric':<12}{'Baseline':>14}{'XGBoost':>14}")
    print("-" * 40)

    for metric in ["MAE", "RMSE", "R2"]:
        print(
            f"{metric:<12}"
            f"{baseline[metric]:>14.3f}"
            f"{xgboost[metric]:>14.3f}"
        )

    mae_improvement = (
        (baseline["MAE"] - xgboost["MAE"]) / baseline["MAE"] * 100
        if baseline["MAE"] != 0
        else 0.0
    )

    print(f"\nXGBoost MAE improvement: {mae_improvement:.2f}%")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "GULF ENERGY AI - BASELINE COMPARISON\n"
        "====================================\n"
        "Baseline: next-hour demand equals current-hour demand.\n"
        "Dataset: AEP historical demand; not actual Gulf-region data.\n\n"
        f"Baseline MAE: {baseline['MAE']:.3f} MW\n"
        f"Baseline RMSE: {baseline['RMSE']:.3f} MW\n"
        f"Baseline R2: {baseline['R2']:.4f}\n\n"
        f"XGBoost MAE: {xgboost['MAE']:.3f} MW\n"
        f"XGBoost RMSE: {xgboost['RMSE']:.3f} MW\n"
        f"XGBoost R2: {xgboost['R2']:.4f}\n\n"
        f"XGBoost MAE improvement: {mae_improvement:.2f}%\n",
        encoding="utf-8",
    )

    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
