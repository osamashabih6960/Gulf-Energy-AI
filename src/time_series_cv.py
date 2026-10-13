
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor


DATA_DIR = Path("data/model_data")
REPORT_DIR = Path("reports")
REPORT_PATH = REPORT_DIR / "time_series_cv_report.txt"

N_SPLITS = 5


def create_model():
    return XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1,
    )


def main():
    # Load only the training data for cross-validation.
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train row counts do not match.")

    if X_train.isna().any().any() or y_train.isna().any():
        raise ValueError("Missing values found in training data.")

    # Chronological cross-validation: no shuffling.
    cv = TimeSeriesSplit(n_splits=N_SPLITS)
    results = []

    print("\n--- TIME SERIES CROSS-VALIDATION ---")
    print(f"Training rows: {len(X_train)}")
    print(f"Number of folds: {N_SPLITS}")

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X_train), start=1
    ):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]

        X_val = X_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]

        model = create_model()
        model.fit(X_fold_train, y_fold_train)

        predictions = model.predict(X_val)

        mae = mean_absolute_error(y_val, predictions)
        rmse = mean_squared_error(y_val, predictions) ** 0.5
        r2 = r2_score(y_val, predictions)

        results.append(
            {
                "fold": fold,
                "train_rows": len(train_idx),
                "validation_rows": len(val_idx),
                "MAE_MW": mae,
                "RMSE_MW": rmse,
                "R2": r2,
            }
        )

        print(
            f"Fold {fold}: "
            f"Train={len(train_idx)}, "
            f"Validation={len(val_idx)}, "
            f"MAE={mae:.3f} MW, "
            f"RMSE={rmse:.3f} MW, "
            f"R2={r2:.4f}"
        )

    results_df = pd.DataFrame(results)

    summary = results_df[["MAE_MW", "RMSE_MW", "R2"]].agg(
        ["mean", "std"]
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        file.write("TIME SERIES CROSS-VALIDATION REPORT\n")
        file.write("===================================\n\n")
        file.write(f"Folds: {N_SPLITS}\n")
        file.write(f"Training rows: {len(X_train)}\n\n")
        file.write("Fold-wise results:\n")
        file.write(results_df.to_string(index=False))
        file.write("\n\nMean and standard deviation:\n")
        file.write(summary.to_string())
        file.write("\n")

    print("\n--- CROSS-VALIDATION SUMMARY ---")
    print(summary.round(4))
    print(f"\nReport saved to: {REPORT_PATH}")
    print("\nCross-validation completed.")


if __name__ == "__main__":
    main()
