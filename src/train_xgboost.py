
from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "model_data"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "xgboost_energy_model.pkl"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"

# Use the same absolute database path for consistent tracking
TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("Gulf-Energy-Forecasting")


def main():
    # Check that all required files exist
    required_files = [
        DATA_DIR / "X_train.csv",
        DATA_DIR / "X_test.csv",
        DATA_DIR / "y_train.csv",
        DATA_DIR / "y_test.csv",
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")

    # Load datasets
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")

    if X_train.empty or X_test.empty:
        raise ValueError("Training or testing data is empty.")

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Training and testing feature columns do not match.")

    # Model parameters
    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "random_state": 42,
        "n_jobs": -1,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print("Training XGBoost model...")

    with mlflow.start_run(run_name="XGBoost-Energy-Forecasting") as run:
        mlflow.log_params(params)
        mlflow.log_param("training_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("number_of_features", X_train.shape[1])
        mlflow.log_param("tracking_database", str(MLFLOW_DB_PATH))

        model = XGBRegressor(**params)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions)

        mlflow.log_metrics({
            "MAE_MW": float(mae),
            "RMSE_MW": float(rmse),
            "R2": float(r2),
        })

        # Save local model
        joblib.dump(model, MODEL_PATH)

        # Log model artifact to MLflow
        mlflow.xgboost.log_model(
            xgb_model=model,
            name="xgboost_energy_model",
            input_example=X_train.head(3),
        )

        # Log feature names
        mlflow.log_dict(
            {"feature_names": X_train.columns.tolist()},
            "feature_names.json",
        )

        print("\nModel Evaluation Results")
        print("------------------------")
        print(f"MAE:  {mae:.3f} MW")
        print(f"RMSE: {rmse:.3f} MW")
        print(f"R2:   {r2:.4f}")
        print(f"Local model: {MODEL_PATH}")
        print(f"MLflow Run ID: {run.info.run_id}")
        print(f"MLflow Run Status: {run.info.status}")
        print("Parameters, metrics, and model artifact logged successfully.")


if __name__ == "__main__":
    main()
