"""CLI entry point: run the pipeline steps from the terminal."""

import argparse

from model_pipeline import (evaluate_model, load_model, prepare_data,
                            save_model, train_model)


def print_metrics(metrics):
    for name, value in metrics.items():
        print(f"{name}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Heart disease ML pipeline")
    parser.add_argument("--step", required=True,
                        choices=["prepare", "train", "evaluate", "all"],
                        help="Pipeline step to run")
    parser.add_argument("--data", default="heart.csv", help="Path to the CSV")
    parser.add_argument("--model-path", default="model.joblib",
                        help="Where to save/load the model")
    args = parser.parse_args()

    # The split is seeded, so every run gets the same train/test sets.
    X_train, X_test, y_train, y_test = prepare_data(args.data)

    if args.step == "prepare":
        print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    elif args.step == "train":
        model = train_model(X_train, y_train)
        save_model(model, args.model_path)

    elif args.step == "evaluate":
        model = load_model(args.model_path)
        print_metrics(evaluate_model(model, X_test, y_test))

    elif args.step == "all":
        model = train_model(X_train, y_train)
        save_model(model, args.model_path)
        print_metrics(evaluate_model(model, X_test, y_test))


if __name__ == "__main__":
    main()
