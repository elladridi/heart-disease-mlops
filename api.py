"""FastAPI service that serves the saved heart disease model."""

import os
from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model_pipeline import ZERO_MEANS_MISSING, load_model

MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
state = {}


@asynccontextmanager
async def lifespan(app):
    if os.path.exists(MODEL_PATH):
        state["model"] = load_model(MODEL_PATH)
    yield
    state.clear()


app = FastAPI(title="Heart disease prediction API", lifespan=lifespan)


class Patient(BaseModel):
    Age: int = Field(ge=1, le=120)
    Sex: Literal["M", "F"]
    ChestPainType: Literal["TA", "ATA", "NAP", "ASY"]
    RestingBP: float = Field(ge=0)
    Cholesterol: float = Field(ge=0)
    FastingBS: Literal[0, 1]
    RestingECG: Literal["Normal", "ST", "LVH"]
    MaxHR: int = Field(gt=0)
    ExerciseAngina: Literal["Y", "N"]
    Oldpeak: float
    ST_Slope: Literal["Up", "Flat", "Down"]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict")
def predict(patient: Patient):
    model = state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run: make train")
    df = pd.DataFrame([patient.model_dump()])
    for col in ZERO_MEANS_MISSING:  # same cleaning rule as in training
        df[col] = df[col].replace(0, float("nan"))
    proba = float(model.predict_proba(df)[0, 1])
    return {
        "prediction": int(proba >= 0.5),
        "probability_heart_disease": round(proba, 4),
        "label": "heart disease" if proba >= 0.5 else "no heart disease",
        "note": "Educational project, not a medical tool.",
    }
