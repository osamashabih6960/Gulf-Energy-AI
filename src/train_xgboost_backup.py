
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


DATA_DIR = Path("data/model_data")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "xgboost_energy_model.pkl"


def main():
    # Load training and testing data
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")

    # Create the model
    model = XGBRegressor(
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

    # Train the model
    print("Training XGBoost model...")
    model.fit(X_train, y_train)

    # Generate predictions on the test set
    predictions = model.predict(X_test)

    # Evaluate predictions
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print("\nModel Evaluation Results")
    print("------------------------")
    print(f"MAE:  {mae:.3f} MW")
    print(f"RMSE: {rmse:.3f} MW")
    print(f"R2 Score: {r2:.4f}")

    # Save the trained model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved successfully: {MODEL_PATH}")


if __name__ == "__main__":
    main()
    