from datetime import timedelta

from prefect import serve

from pipeline_prefect import all_flow, evaluate_flow, train_flow


if __name__ == "__main__":
    serve(
        all_flow.to_deployment(
            name="ml-pipeline-all",
            interval=timedelta(days=1),
        ),
        train_flow.to_deployment(name="ml-pipeline-train"),
        evaluate_flow.to_deployment(name="ml-pipeline-evaluate"),
    )
