"""CLI entry point: run the pipeline steps from the terminal."""

import argparse
import logging

from model_pipeline import (
    cross_validate_model,
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)

logger = logging.getLogger("main")
N_ESTIMATORS = 200


def log_metrics(metrics):
    for name, value in metrics.items():
        logger.info("%s: %s", name, value)


def track_with_mlflow(args, metrics):
    """Record parameters, metrics and the model file in MLflow."""
    import mlflow  # imported here so MLflow stays optional

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("heart-disease")
    tn, fp = metrics["confusion_matrix"][0]
    fn, tp = metrics["confusion_matrix"][1]
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(
            {
                "model": "RandomForest",
                "n_estimators": N_ESTIMATORS,
                "zero_as_missing": not args.keep_zeros,
            }
        )
        mlflow.log_metrics(
            {
                **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )
        mlflow.log_artifact(args.model_path)
    logger.info("Run '%s' recorded in MLflow", args.run_name)


def track_cv_with_mlflow(args, summary):
    """Record cross-validation mean/std in MLflow."""
    import mlflow

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("heart-disease")
    with mlflow.start_run(run_name=args.run_name or "cv-5fold"):
        mlflow.log_params(
            {
                "model": "RandomForest",
                "n_estimators": N_ESTIMATORS,
                "zero_as_missing": not args.keep_zeros,
                "cv_folds": 5,
            }
        )
        for name, value in summary["mean"].items():
            mlflow.log_metric(f"cv_mean_{name}", value)
        for name, value in summary["std"].items():
            mlflow.log_metric(f"cv_std_{name}", value)
    logger.info("Cross-validation recorded in MLflow")


def main():
    parser = argparse.ArgumentParser(description="Heart disease ML pipeline")
    parser.add_argument(
        "--step",
        required=True,
        choices=["prepare", "train", "evaluate", "cv", "all"],
        help="Pipeline step to run",
    )
    parser.add_argument("--data", default="heart.csv", help="Path to the CSV")
    parser.add_argument("--model-path", default="model.joblib", help="Where to save/load the model")
    parser.add_argument(
        "--keep-zeros", action="store_true", help="Do NOT treat zero cholesterol/BP as missing (v1)"
    )
    parser.add_argument(
        "--track", action="store_true", help="Record the run in MLflow (with --step all or cv)"
    )
    parser.add_argument("--run-name", default=None, help="MLflow run name")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    # The split is seeded, so every run gets the same train/test sets.
    X_train, X_test, y_train, y_test = prepare_data(args.data, zero_as_missing=not args.keep_zeros)

    if args.step == "prepare":
        logger.info("Train: %s, Test: %s", X_train.shape, X_test.shape)

    elif args.step == "train":
        model = train_model(X_train, y_train, n_estimators=N_ESTIMATORS)
        save_model(model, args.model_path)

    elif args.step == "evaluate":
        model = load_model(args.model_path)
        log_metrics(evaluate_model(model, X_test, y_test))

    elif args.step == "cv":
        # Cross-validation uses the training part only; the test set stays untouched.
        summary = cross_validate_model(X_train, y_train, n_estimators=N_ESTIMATORS)
        for name, mean in summary["mean"].items():
            logger.info("CV %s: %.3f +/- %.3f", name, mean, summary["std"][name])
        if args.track:
            track_cv_with_mlflow(args, summary)

    elif args.step == "all":
        model = train_model(X_train, y_train, n_estimators=N_ESTIMATORS)
        save_model(model, args.model_path)
        metrics = evaluate_model(model, X_test, y_test)
        log_metrics(metrics)
        if args.track:
            track_with_mlflow(args, metrics)


if __name__ == "__main__":
    main()
