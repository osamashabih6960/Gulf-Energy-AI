
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


PREDICTIONS_PATH = Path("data/predictions/xgboost_predictions.csv")
REPORT_DIR = Path("reports")
REPORT_PATH = REPORT_DIR / "error_analysis_report.txt"


def main():
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {PREDICTIONS_PATH}"
        )

    df = pd.read_csv(PREDICTIONS_PATH)

    required = [
        "Target_Datetime",
        "Actual_MW",
        "Predicted_MW",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Target_Datetime"] = pd.to_datetime(
        df["Target_Datetime"], errors="raise"
    )

    if df[required].isna().any().any():
        raise ValueError("Missing values found in prediction data.")

    df["Error_MW"] = df["Actual_MW"] - df["Predicted_MW"]
    df["Absolute_Error_MW"] = df["Error_MW"].abs()
    df["Hour"] = df["Target_Datetime"].dt.hour
    df["Month"] = df["Target_Datetime"].dt.month

    mae = mean_absolute_error(df["Actual_MW"], df["Predicted_MW"])
    rmse = mean_squared_error(
        df["Actual_MW"], df["Predicted_MW"]
    ) ** 0.5

    hourly = (
        df.groupby("Hour")
        .agg(
            MAE_MW=("Absolute_Error_MW", "mean"),
            Mean_Error_MW=("Error_MW", "mean"),
            Samples=("Error_MW", "size"),
        )
        .sort_values("MAE_MW", ascending=False)
    )

    monthly = (
        df.groupby("Month")
        .agg(
            MAE_MW=("Absolute_Error_MW", "mean"),
            Mean_Error_MW=("Error_MW", "mean"),
            Samples=("Error_MW", "size"),
        )
        .sort_values("MAE_MW", ascending=False)
    )

    highest_errors = df.nlargest(10, "Absolute_Error_MW")[
        [
            "Target_Datetime",
            "Actual_MW",
            "Predicted_MW",
            "Error_MW",
            "Absolute_Error_MW",
        ]
    ]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        file.write("XGBOOST ERROR ANALYSIS REPORT\n")
        file.write("=============================\n\n")
        file.write(f"Prediction rows: {len(df)}\n")
        file.write(f"Overall MAE: {mae:.3f} MW\n")
        file.write(f"Overall RMSE: {rmse:.3f} MW\n\n")

        file.write("HOURLY ERROR ANALYSIS (highest MAE first)\n")
        file.write(hourly.to_string(float_format=lambda x: f"{x:.3f}"))
        file.write("\n\nMONTHLY ERROR ANALYSIS (highest MAE first)\n")
        file.write(monthly.to_string(float_format=lambda x: f"{x:.3f}"))
        file.write("\n\nTOP 10 LARGEST ERRORS\n")
        file.write(
            highest_errors.to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}",
            )
        )
        file.write("\n")

    print("\n--- ERROR ANALYSIS ---")
    print(f"Prediction rows: {len(df)}")
    print(f"Overall MAE: {mae:.3f} MW")
    print(f"Overall RMSE: {rmse:.3f} MW")

    print("\nHours with highest MAE:")
    print(hourly.head(5).round(3))

    print("\nMonths with highest MAE:")
    print(monthly.head(5).round(3))

    print("\nTop 10 largest errors:")
    print(highest_errors.to_string(index=False))

    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
