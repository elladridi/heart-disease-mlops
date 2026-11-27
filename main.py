"""Command-line entry point for the heart disease ML pipeline."""

import argparse
import logging

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Heart disease ML pipeline")
    parser.add_argument(
        "--step",
        required=True,
        choices=["prepare", "train", "evaluate", "all"],
        help="Pipeline step to run",
    )
    parser.add_argument("--data", default="heart.csv", help="Path to the CSV")
    parser.add_argument("--model-path", default="model.joblib", help="Model file path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    X_train, X_test, y_train, y_test = prepare_data(args.data)

    if args.step == "prepare":
        logger.info("Prepared data: train=%s, test=%s", X_train.shape, X_test.shape)
    elif args.step == "train":
        save_model(train_model(X_train, y_train), args.model_path)
    elif args.step == "evaluate":
        metrics = evaluate_model(load_model(args.model_path), X_test, y_test)
        for name, value in metrics.items():
            logger.info("%s: %s", name, value)
    else:
        model = train_model(X_train, y_train)
        save_model(model, args.model_path)
        metrics = evaluate_model(model, X_test, y_test)
        for name, value in metrics.items():
            logger.info("%s: %s", name, value)


if __name__ == "__main__":
    main()
