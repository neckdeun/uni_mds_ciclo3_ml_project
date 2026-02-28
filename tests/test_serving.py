from __future__ import annotations

from fastapi.testclient import TestClient

from src.serving import app


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_predict_endpoint() -> None:
    payload = {
        "gender": "Male",
        "age": 67,
        "hypertension": 1,
        "heart_disease": 0,
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
        "avg_glucose_level": 228.69,
        "bmi": 36.6,
        "smoking_status": "formerly smoked",
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert "prediction" in body
        assert "stroke_risk_probability" in body
        assert body["prediction"] in [0, 1]
        assert 0 <= body["stroke_risk_probability"] <= 1

