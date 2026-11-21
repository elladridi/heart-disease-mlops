import numpy as np
import pandas as pd
import pytest

from model_pipeline import (evaluate_model, load_model, prepare_data,
                            save_model, train_model)


@pytest.fixture
def csv_path(tmp_path):
    """Small synthetic dataset with the same columns as heart.csv."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "Age": rng.integers(30, 80, n),
        "Sex": rng.choice(["M", "F"], n),
        "ChestPainType": rng.choice(["TA", "ATA", "NAP", "ASY"], n),
        "RestingBP": rng.integers(100, 180, n),
        "Cholesterol": rng.integers(150, 350, n),
        "FastingBS": rng.integers(0, 2, n),
        "RestingECG": rng.choice(["Normal", "ST", "LVH"], n),
        "MaxHR": rng.integers(80, 200, n),
        "ExerciseAngina": rng.choice(["Y", "N"], n),
        "Oldpeak": rng.random(n) * 3,
        "ST_Slope": rng.choice(["Up", "Flat", "Down"], n),
        "HeartDisease": rng.integers(0, 2, n),
    })
    df.loc[:9, "Cholesterol"] = 0  # simulate "not measured"
    path = tmp_path / "heart.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_prepare_data_shapes(csv_path):
    X_train, X_test, y_train, y_test = prepare_data(csv_path)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_train) + len(X_test) == 200
    assert "HeartDisease" not in X_train.columns


def test_zero_cholesterol_becomes_missing(csv_path):
    X_train, X_test, _, _ = prepare_data(csv_path)
    assert (X_train["Cholesterol"] == 0).sum() == 0
    assert (X_test["Cholesterol"] == 0).sum() == 0


def test_missing_target_raises(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        prepare_data(str(bad))


def test_train_and_evaluate(csv_path):
    X_train, X_test, y_train, y_test = prepare_data(csv_path)
    model = train_model(X_train, y_train, n_estimators=20)
    metrics = evaluate_model(model, X_test, y_test)
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert 0.0 <= metrics[key] <= 1.0
    assert len(metrics["confusion_matrix"]) == 2


def test_save_and_load_give_same_predictions(csv_path, tmp_path):
    X_train, X_test, y_train, _ = prepare_data(csv_path)
    model = train_model(X_train, y_train, n_estimators=20)
    path = str(tmp_path / "model.joblib")
    save_model(model, path)
    loaded = load_model(path)
    assert (model.predict(X_test) == loaded.predict(X_test)).all()
