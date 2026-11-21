"""Modular ML pipeline for heart disease prediction."""

import logging

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

TARGET = "HeartDisease"
ZERO_MEANS_MISSING = ["Cholesterol", "RestingBP"]


def prepare_data(path, test_size=0.2, random_state=42, zero_as_missing=True):
    """Load the CSV, clean it and split it into train/test sets.

    Args:
        path (str): Path to heart.csv.
        test_size (float): Fraction of data used for testing.
        random_state (int): Seed for a reproducible split.
        zero_as_missing (bool): If True, zeros in Cholesterol and RestingBP
            ("not measured") become NaN and are imputed in the pipeline.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Column '{TARGET}' not found in {path}")
    n_before = len(df)
    df = df.drop_duplicates()
    logger.info("Loaded %s: %d rows (%d after dropping duplicates)",
                path, n_before, len(df))

    if zero_as_missing:
        for col in ZERO_MEANS_MISSING:
            if col in df.columns:
                n_zero = int((df[col] == 0).sum())
                df[col] = df[col].replace(0, float("nan"))
                logger.info("%s: %d zeros converted to missing", col, n_zero)

    X = df.drop(columns=TARGET)
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size,
                            stratify=y, random_state=random_state)


def train_model(X_train, y_train, n_estimators=200, random_state=42):
    """Build and train a preprocessing + Random Forest pipeline.

    The imputer, encoder and scaler live inside the pipeline, so they are
    saved together with the model.

    Returns:
        Pipeline: the fitted model.
    """
    cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()
    num_cols = X_train.select_dtypes(include="number").columns.tolist()

    preprocessing = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), num_cols),
    ])
    model = Pipeline([
        ("prep", preprocessing),
        ("clf", RandomForestClassifier(n_estimators=n_estimators,
                                       random_state=random_state)),
    ])
    logger.info("Training on %d rows (%d trees)", len(X_train), n_estimators)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model on the test set.

    Returns:
        dict: accuracy, precision, recall, f1, roc_auc, confusion_matrix.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def save_model(model, path="model.joblib"):
    """Save the trained model to disk with joblib."""
    joblib.dump(model, path)
    logger.info("Model saved to %s", path)


def load_model(path="model.joblib"):
    """Load a saved model from disk."""
    logger.info("Loading model from %s", path)
    return joblib.load(path)


def cross_validate_model(X, y, n_splits=5, n_estimators=200, random_state=42):
    """Stratified k-fold cross-validation using the same pipeline.

    Returns:
        dict: {"mean": {metric: value}, "std": {metric: value}}
    """
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=random_state)
    rows = []
    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        model = train_model(X.iloc[tr], y.iloc[tr],
                            n_estimators=n_estimators,
                            random_state=random_state)
        metrics = evaluate_model(model, X.iloc[va], y.iloc[va])
        metrics.pop("confusion_matrix")
        rows.append(metrics)
        logger.info("Fold %d/%d: recall=%.3f f1=%.3f",
                    fold, n_splits, metrics["recall"], metrics["f1"])
    df = pd.DataFrame(rows)
    return {"mean": df.mean().to_dict(), "std": df.std().to_dict()}
