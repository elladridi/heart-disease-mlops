# Heart Disease Prediction MLOps Project

This project demonstrates a simple MLOps workflow for predicting heart disease. It covers business understanding, data collection, data preparation, model training, evaluation, and saving the model.

## Pipeline steps

1. **Business understanding:** Predict the `HeartDisease` target from 11 clinical measurements. Recall is the main metric because missed positive cases matter.
2. **Data collection:** The project uses `heart.csv`, based on the Heart Failure Prediction Dataset (fedesoriano, Kaggle, CC0).
3. **Data preparation:** Remove duplicate rows, treat zero Cholesterol and RestingBP values as missing, split into training and test sets, and preprocess features.
4. **Model training:** Train a Random Forest in a scikit-learn pipeline.
5. **Evaluation:** Report accuracy, precision, recall, F1, ROC-AUC, and the confusion matrix on held-out data.
6. **Save the model:** Save the trained preprocessing and model pipeline with joblib.

## Run locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

python main.py --step prepare
python main.py --step train
python main.py --step evaluate
python main.py --step all
pytest -v
```

Use `--data path/to/file.csv` to select a data file and `--model-path path/to/model.joblib` to choose where the model is saved or loaded.

## Automation

GitHub Actions runs the tests and the prepare/train/evaluate/save workflow on each push to `main` and each pull request. It saves the trained model as a workflow artifact.

## MLOps levels

- **Level 0 - Manual:** Run each pipeline step from the command line.
- **Level 1 - Partial automation:** Reuse modular pipeline functions, the CLI, pinned dependencies, and tests.
- **Level 2 - Automatic execution:** GitHub Actions runs the tests and complete pipeline on pushes to `main` and pull requests, then saves the trained model artifact.

Level 2 automates the six-step workflow in this repository. It does not include new-data triggers, scheduled retraining, deployment, or production monitoring.

## Report

See [the project report](docs/Heart_Disease_Report.pdf) for the workflow, course levels, reproduction steps, and scope limitations.

This is a course demonstration, not a medical diagnostic tool.
