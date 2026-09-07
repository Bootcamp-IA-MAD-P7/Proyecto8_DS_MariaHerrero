from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from configs.settings import MLFLOW_TRACKING_URI


MODEL_SELECTION_EXPERIMENT = (
    "stroke-risk-model-selection"
)
CALIBRATION_THRESHOLD_EXPERIMENT = (
    "stroke-risk-calibration-threshold"
)
FINAL_MODEL_EXPERIMENT = (
    "stroke-risk-final-model"
)
REGISTERED_MODEL_NAME = (
    "stroke-risk-screening-model"
)


def configure_tracking(tracking_uri=None):
    uri = tracking_uri or MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(uri)

    return uri


def select_experiment(
    experiment_name,
    artifact_location=None,
):
    client = MlflowClient(
        tracking_uri=mlflow.get_tracking_uri()
    )
    experiment = client.get_experiment_by_name(
        experiment_name
    )

    if experiment is None:
        experiment_id = client.create_experiment(
            experiment_name,
            artifact_location=artifact_location,
        )
        experiment = client.get_experiment(
            experiment_id
        )

    mlflow.set_experiment(
        experiment_id=experiment.experiment_id
    )

    return experiment


def start_run(run_name=None, tags=None):
    return mlflow.start_run(
        run_name=run_name,
        tags=tags,
    )


def log_params(params):
    mlflow.log_params(params)


def log_metrics(metrics):
    mlflow.log_metrics(metrics)


def log_tags(tags):
    mlflow.set_tags(tags)


def log_existing_artifacts(
    paths,
    artifact_path=None,
):
    logged_paths = []

    for path in paths:
        path = Path(path)

        if path.is_file():
            mlflow.log_artifact(
                str(path),
                artifact_path=artifact_path,
            )
            logged_paths.append(path)

        elif path.is_dir():
            mlflow.log_artifacts(
                str(path),
                artifact_path=artifact_path,
            )
            logged_paths.append(path)

    return logged_paths
