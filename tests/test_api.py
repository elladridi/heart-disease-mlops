import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api
from model_pipeline import train_model

PATIENT = {
    "Age": 54, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 140,
    "Cholesterol": 239, "FastingBS": 0, "RestingECG": "Normal",
    "MaxHR": 160, "ExerciseAngina": "N", "Oldpeak": 1.2, "ST_Slope": "Flat",
}


def make_model():
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({
        "Age": rng.integers(30, 80, n),
        "Sex": rng.choice(["M", "F"], n),
        "ChestPainType": rng.choice(["TA", "ATA", "NAP", "ASY"], n),
        "RestingBP": rng.integers(100, 180, n).astype(float),
        "Cholesterol": rng.integers(150, 350, n).astype(float),
        "FastingBS": rng.integers(0, 2, n),
        "RestingECG": rng.choice(["Normal", "ST", "LVH"], n),
        "MaxHR": rng.integers(80, 200, n),
        "ExerciseAngina": rng.choice(["Y", "N"], n),
        "Oldpeak": rng.random(n) * 3,
        "ST_Slope": rng.choice(["Up", "Flat", "Down"], n),
    })
    y = pd.Series(rng.integers(0, 2, n))
    return train_model(X, y, n_estimators=20)


@pytest.fixture
def client():
    with TestClient(api.app) as c:
        api.state["model"] = make_model()
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict(client):
    r = client.post("/predict", json=PATIENT)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability_heart_disease"] <= 1.0


def test_zero_cholesterol_is_accepted(client):
    r = client.post("/predict", json={**PATIENT, "Cholesterol": 0})
    assert r.status_code == 200


def test_invalid_input_rejected(client):
    r = client.post("/predict", json={**PATIENT, "Sex": "X"})
    assert r.status_code == 422


def test_no_model_returns_503():
    with TestClient(api.app) as c:
        api.state.clear()
        assert c.post("/predict", json=PATIENT).status_code == 503
