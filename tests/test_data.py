"""Data validation: checks on the real heart.csv before any training."""

import pandas as pd
import pytest

EXPECTED_COLUMNS = [
    "Age",
    "Sex",
    "ChestPainType",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "RestingECG",
    "MaxHR",
    "ExerciseAngina",
    "Oldpeak",
    "ST_Slope",
    "HeartDisease",
]


@pytest.fixture(scope="module")
def df():
    return pd.read_csv("heart.csv")


def test_expected_columns(df):
    assert list(df.columns) == EXPECTED_COLUMNS


def test_no_missing_values(df):
    assert not df.isna().any().any()


def test_target_is_binary(df):
    assert set(df["HeartDisease"].unique()) == {0, 1}


def test_categorical_values(df):
    assert set(df["Sex"]) <= {"M", "F"}
    assert set(df["ChestPainType"]) <= {"TA", "ATA", "NAP", "ASY"}
    assert set(df["RestingECG"]) <= {"Normal", "ST", "LVH"}
    assert set(df["ExerciseAngina"]) <= {"Y", "N"}
    assert set(df["ST_Slope"]) <= {"Up", "Flat", "Down"}


def test_plausible_ranges(df):
    assert df["Age"].between(18, 100).all()
    assert df["MaxHR"].between(40, 250).all()
    assert (df["Cholesterol"] >= 0).all()
