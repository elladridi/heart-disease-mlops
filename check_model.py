"""Quality gate: fail if the trained model is below the minimum performance."""

import sys

from model_pipeline import evaluate_model, load_model, prepare_data

MIN_RECALL = 0.85
MIN_ROC_AUC = 0.85


def main():
    _, X_test, _, y_test = prepare_data("heart.csv")
    metrics = evaluate_model(load_model("model.joblib"), X_test, y_test)
    print(f"recall={metrics['recall']:.3f} roc_auc={metrics['roc_auc']:.3f}")
    if metrics["recall"] < MIN_RECALL or metrics["roc_auc"] < MIN_ROC_AUC:
        print("Quality gate FAILED")
        sys.exit(1)
    print("Quality gate passed")


if __name__ == "__main__":
    main()
