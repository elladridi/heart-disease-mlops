# Heart Disease Prediction: MLOps Pipeline

![CI](https://github.com/elladridi/heart-disease-mlops/actions/workflows/ci.yml/badge.svg)

Educational project for the MLOps course (Workshop 2: code modularization).
It is not a medical tool.

## 1. Business understanding

**Problem:** predict whether a patient has heart disease from 11 clinical
measurements (age, chest pain type, blood pressure, cholesterol, ECG, etc.).

**Target:** `HeartDisease` (1 = disease, 0 = no disease).

**Key metric: recall.** A missed sick patient (false negative) is far more
costly than a false alarm, so recall is the priority. F1 and ROC-AUC are
used as supporting metrics.

## 2. Data

Heart Failure Prediction Dataset (fedesoriano, Kaggle, CC0): 918 patients
after duplicate removal, 11 features.
https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

Data quality: 172 rows have `Cholesterol = 0` and 1 row has `RestingBP = 0`.
These values are impossible, so they are treated as missing and imputed
with the median (learned on the training set only).

## 3. Project structure

| File | Role |
|---|---|
| `model_pipeline.py` | `prepare_data`, `train_model`, `evaluate_model`, `save_model`, `load_model` |
| `main.py` | CLI to run each step |
| `tests/test_pipeline.py` | pytest unit tests (synthetic data) |
| `Makefile` | shortcuts: `make train`, `make evaluate`, `make all`, `make test` |
| `.github/workflows/ci.yml` | CI: install, test, train, upload model |

## 4. How to run

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python main.py --step prepare
python main.py --step train
python main.py --step evaluate
python main.py --step all      # train + evaluate
make test                      # run the unit tests
```

Options: `--data path/to/heart.csv`, `--model-path path/to/model.joblib`.

## 5. Model

Scikit-learn `Pipeline`: median imputation + standard scaling for numeric
columns, one-hot encoding for categorical columns, then a Random Forest
(200 trees, seed 42). The whole pipeline is saved with joblib, so predictions
on new data use the same preprocessing.

## 6. Results (test set: 184 patients, stratified 80/20 split)

| Metric | v1: raw data | v2: zeros imputed |
|---|---|---|
| Accuracy | 0.902 | 0.897 |
| Precision | 0.896 | 0.888 |
| Recall | 0.931 | 0.931 |
| F1 | 0.913 | 0.909 |
| ROC-AUC | 0.936 | 0.928 |

Confusion matrix (v2): 70 true negatives, 12 false positives,
7 false negatives, 95 true positives.

Cleaning the zero values did not significantly change performance (the
differences equal about one patient on a small test set), but the pipeline
no longer trains on impossible values.

## 7. MLOps maturity

- **Level 0:** manual notebook.
- **Level 1:** modular code, CLI, Makefile, tests, Git, reproducible seed.
- **Level 2 (in progress):** GitHub Actions runs tests and training on
  every push and stores the model as an artifact.

## 8. Limitations

Small dataset, single train/test split (cross-validation would give a more
reliable estimate), no probability calibration, and the model is not
clinically validated.

## 9. Experiment tracking (MLflow)

MLflow is optional (`pip install -r requirements-mlflow.txt`).

```bash
make track-v1   # raw data, zeros kept
make track      # zeros imputed
make ui         # open http://localhost:5000
```

Each run records parameters (`n_estimators`, `zero_as_missing`), metrics
(accuracy, precision, recall, F1, ROC-AUC, confusion matrix) and the model
file, so experiments can be compared side by side.

## 10. Docker

```bash
docker build -t heart-disease-mlops .
docker run --rm heart-disease-mlops   # train + evaluate inside the container
```

The image uses Python 3.12 and runs `python main.py --step all` by default.
(From WSL this requires Docker Desktop's WSL integration; otherwise run
the commands from PowerShell.)
