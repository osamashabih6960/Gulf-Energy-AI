from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Gulf Energy AI API is running"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_prediction():
    payload = {
        "AEP_MW": 16000,
        "lag_1": 15800,
        "lag_24": 15000,
        "year": 2018,
        "month": 7,
        "day": 15,
        "hour": 14,
        "day_of_week": 6,
        "is_weekend": 1,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predicted_energy_MW" in response.json()
    assert response.json()["unit"] == "MW"
