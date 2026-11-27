import argparse
import logging
import subprocess
import sys

from prefect import flow, task

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Keep code checks focused on these project files.
CODE_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


@task
def install_dependencies():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            "requirements.txt",
            "-r",
            "requirements-dev.txt",
        ],
        check=True,
    )


@task
def format_code_task():
    subprocess.run(
        [sys.executable, "-m", "black", *CODE_FILES],
        check=True,
    )


@task
def quality_check_task():
    subprocess.run(
        [sys.executable, "-m", "ruff", "check", *CODE_FILES],
        check=True,
    )


@task
def security_check_task():
    # The pipeline uses subprocess to run its checks; skip those two
    # Bandit rules while scanning the rest of these files.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "bandit",
            "-q",
            "-s",
            "B404,B603",
            *CODE_FILES,
        ],
        check=True,
    )


@task
def run_tests_task():
    subprocess.run(
        [sys.executable, "-m", "pytest"],
        check=True,
    )


@task
def prepare_data_task():
    return prepare_data("heart.csv")


@task
def train_model_task(data):
    X_train, _, y_train, _ = data
    return train_model(X_train, y_train)


@task
def save_model_task(model):
    save_model(model, "model.joblib")


@task
def load_model_task():
    return load_model("model.joblib")


@task
def evaluate_model_task(model, data):
    _, X_test, _, y_test = data
    metrics = evaluate_model(model, X_test, y_test)
    logger.info("Evaluation metrics: %s", metrics)


@flow(name="code")
def code_flow():
    format_code_task()
    quality_check_task()
    security_check_task()
    run_tests_task()


@flow(name="all")
def all_flow():
    install_dependencies()
    code_flow()
    data = prepare_data_task()
    model = train_model_task(data)
    save_model_task(model)
    evaluate_model_task(model, data)


@flow(name="train")
def train_flow():
    install_dependencies()
    data = prepare_data_task()
    model = train_model_task(data)
    save_model_task(model)


@flow(name="evaluate")
def evaluate_flow():
    data = prepare_data_task()
    model = load_model_task()
    evaluate_model_task(model, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flow",
        choices=["all", "train", "evaluate", "code"],
        default="all",
    )
    selected_flow = parser.parse_args().flow

    if selected_flow == "all":
        all_flow()
    elif selected_flow == "train":
        train_flow()
    elif selected_flow == "evaluate":
        evaluate_flow()
    else:
        code_flow()
