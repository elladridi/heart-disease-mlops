# Heart Disease Prediction: MLOps Pipeline

![CI](https://github.com/elladridi/heart-disease-mlops/actions/workflows/ci.yml/badge.svg)

This project applies the MLOps course's code modularization workshop to a heart disease prediction task. It turns a machine learning workflow into reusable Python modules, a command-line interface, automated tests, experiment tracking, a Docker image, a CI pipeline and a prediction API. It is an educational project, not a medical tool.

## 1. Business understanding

**Problem:** predict whether a patient has heart disease from 11 clinical measurements (age, chest pain type, blood pressure, cholesterol, ECG, etc.).

**Target:** `HeartDisease` (1 = disease, 0 = no disease).

**Key metric: recall.** A missed sick patient (false negative) is far more costly than a false alarm, so recall is the priority. F1 and ROC-AUC are used as supporting metrics.

## 2. Data

Heart Failure Prediction Dataset (fedesoriano, Kaggle, CC0): 918 patients after duplicate removal, 11 features.
https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

**Data quality:** 172 rows have `Cholesterol = 0` and 1 row has `RestingBP = 0`. These values are impossible (they mean "not measured"), so they are treated as missing and imputed with the median, which is learned on the training set only.

## 3. Project structure

| File | Role |
|---|---|
| `model_pipeline.py` | `prepare_data`, `train_model`, `evaluate_model`, `save_model`, `load_model`, `cross_validate_model` |
| `main.py` | CLI to run each step (`prepare`, `train`, `evaluate`, `cv`, `all`) |
| `api.py` | FastAPI service: `GET /health`, `POST /predict` |
| `tests/test_pipeline.py`, `tests/test_api.py` | 12 pytest tests (synthetic data, no CSV needed) |
| `Makefile` | shortcuts for every step |
| `Dockerfile` | container image running the training pipeline |
| `.github/workflows/ci.yml` | CI: install, test, train, upload model |
| `requirements.txt`, `requirements-mlflow.txt` | pinned dependencies (MLflow is optional) |

## 4. How to run

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python main.py --step prepare
python main.py --step train
python main.py --step evaluate
python main.py --step all      # train + evaluate
python main.py --step cv       # 5-fold cross-validation

make test                      # run the unit tests
make api                       # serve the model on http://localhost:8000
```

Options: `--data path/to/heart.csv`, `--model-path path/to/model.joblib`, `--keep-zeros`, `--track` (MLflow).

## 5. Model

Scikit-learn `Pipeline`: median imputation + standard scaling for numeric columns, one-hot encoding for categorical columns, then a Random Forest (200 trees, seed 42). The whole pipeline is saved with joblib, so predictions on new data use the same preprocessing.

## 6. Results

**Held-out test set** (184 patients, stratified 80/20 split):

| Metric | v1: raw data | v2: zeros imputed |
|---|---|---|
| Accuracy | 0.902 | 0.897 |
| Precision | 0.896 | 0.888 |
| Recall | 0.931 | 0.931 |
| F1 | 0.913 | 0.909 |
| ROC-AUC | 0.936 | 0.928 |

Confusion matrix (v2): 70 true negatives, 12 false positives, 7 false negatives, 95 true positives.

**5-fold stratified cross-validation** (training set only, v2 pipeline):

| Metric | Mean ± std |
|---|---|
| Accuracy | 0.853 ± 0.039 |
| Precision | 0.856 ± 0.058 |
| Recall | 0.889 ± 0.047 |
| F1 | 0.870 ± 0.030 |
| ROC-AUC | 0.927 ± 0.030 |

**Interpretation.** The test-set scores are somewhat higher than the cross-validation means (about one standard deviation), so the cross-validation figures are the more conservative estimate of real performance. Recall varies between 0.83 and 0.94 across folds. The v1 vs v2 differences (about 0.005) are far smaller than this fold-to-fold variation, so cleaning the zero values did not measurably change performance. It does mean the pipeline no longer trains on impossible values.

## 7. API (FastAPI)

```bash
make train   # creates model.joblib
make api     # serves on http://localhost:8000
```

Interactive docs at `/docs`. `GET /health` reports whether the model is loaded. `POST /predict` takes one patient (11 clinical fields) and returns the predicted class and probability. Inputs are validated (for example `Sex` must be `M` or `F`), and zeros in Cholesterol/RestingBP are treated as missing, exactly as in training.

Example: a high-risk profile (asymptomatic chest pain, exercise angina, flat ST slope) scores 0.985, and a low-risk profile scores 0.005. These are model scores, not calibrated probabilities.

## 8. Experiment tracking (MLflow)

MLflow is optional (`pip install -r requirements-mlflow.txt`).

```bash
make track-v1   # raw data, zeros kept
make track      # zeros imputed
make track-cv   # cross-validation summary
make ui         # open http://localhost:5000
```

Each run records parameters (`n_estimators`, `zero_as_missing`), metrics and the model file, so experiments can be compared side by side.

## 9. Docker

```bash
docker build -t heart-disease-mlops .
docker run --rm heart-disease-mlops   # train + evaluate inside the container
```

The image uses Python 3.12 and runs `python main.py --step all` by default. (From WSL this requires Docker Desktop's WSL integration; otherwise run the commands from PowerShell.)

## 10. MLOps maturity

- **Level 0 (manual):** notebook-style experiment run by hand.
- **Level 1 (partial automation):** modular code, CLI, Makefile, logging, unit tests, Git, reproducible seeds, MLflow tracking, Docker.
- **Level 2 (automation):** GitHub Actions runs the tests and trains the model on every push and stores it as an artifact; the model is served through an API. Automatic retraining and monitoring are not implemented.

## 11. Limitations

- Small dataset (918 patients), merged from five hospital sources, with far more men than women, so performance may differ between groups.
- Probabilities are not calibrated, and the 0.5 decision threshold is not tuned for recall.
- Cross-validation shows noticeable variance between folds, so small metric differences should not be over-interpreted.
- The model is not clinically validated.
