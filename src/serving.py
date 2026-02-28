from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline


MODEL_PATH = Path("models/stroke_model.joblib")

app = FastAPI(
    title="API de prediccion de riesgo de ACV",
    description="Estimacion de probabilidad de ACV a partir de variables del paciente.",
    version="1.0.0",
)
model: Pipeline | None = None


class StrokeFeatures(BaseModel):
    gender: Literal["Male", "Female", "Other"]
    age: float = Field(..., ge=0, le=120)
    hypertension: int = Field(..., ge=0, le=1)
    heart_disease: int = Field(..., ge=0, le=1)
    ever_married: Literal["Yes", "No"]
    work_type: Literal["children", "Govt_job", "Never_worked", "Private", "Self-employed"]
    Residence_type: Literal["Urban", "Rural"]  # noqa: N815
    avg_glucose_level: float = Field(..., ge=0)
    bmi: float = Field(..., ge=0)
    smoking_status: Literal["formerly smoked", "never smoked", "smokes", "Unknown"]


@app.on_event("startup")
def load_model() -> None:
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"No se encontro el modelo en '{MODEL_PATH}'. Ejecuta src/train.py antes de iniciar la API."
        )
    model = joblib.load(MODEL_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: StrokeFeatures) -> dict[str, float | int]:
    if model is None:
        raise HTTPException(status_code=500, detail="El modelo no esta cargado.")

    row = pd.DataFrame([payload.model_dump()])
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0, 1])
    return {
        "prediction": prediction,
        "stroke_risk_probability": round(probability, 6),
    }

