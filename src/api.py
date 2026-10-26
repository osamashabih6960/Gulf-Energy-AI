
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "xgboost_energy_model.pkl"

model = joblib.load(MODEL_PATH)
FEATURES = list(model.feature_names_in_)

app = FastAPI(title="Gulf Energy AI API", version="1.0.0")


class PredictionInput(BaseModel):
    AEP_MW: float = Field(gt=0)
    lag_1: float = Field(gt=0)
    lag_24: float = Field(gt=0)
    year: int
    month: int = Field(ge=1, le=12)
    day: int = Field(ge=1, le=31)
    hour: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    is_weekend: int = Field(ge=0, le=1)


@app.get("/")
def home():
    return {"message": "Gulf Energy AI API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: PredictionInput):
    values = data.model_dump()
    features = pd.DataFrame(
        [[values[name] for name in FEATURES]],
        columns=FEATURES,
    )
    result = float(model.predict(features)[0])
    return {"predicted_energy_MW": round(result, 3), "unit": "MW"}
 