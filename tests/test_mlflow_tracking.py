import mlflow
import mlflow.sklearn
import pytest
from mlflow.tracking import MlflowClient
from sklearn.dummy import DummyClassifier

from src.tracking.mlflow_tracking import (
    CALIBRATION_THRESHOLD_EXPERIMENT,
    FINAL_MODEL_EXPERIMENT,
    MODEL_SELECTION_EXPERIMENT,
    REGISTERED_MODEL_NAME,
    configure_tracking,
    log_existing_artifacts,
    log_metrics,
    log_params,
    log_tags,
    select_experiment,
    start_run,
)


@pytest.fixture
def isolated_mlflow(tmp_path):
    original_tracking_uri = mlflow.get_tracking_uri()
    tracking_database = tmp_path / "tracking.db"
    tracking_uri = (
        f"sqlite:///{tracking_database.as_posix()}"
    )
    artifact_directory = tmp_path / "mlflow-artifacts"

    configure_tracking(tracking_uri)

    try:
        yield tracking_uri, artifact_directory
    finally:
        if mlflow.active_run() is not None:
            mlflow.end_run(status="KILLED")

        mlflow.set_tracking_uri(
            original_tracking_uri
        )


def test_expected_experiment_names_are_defined():
    assert MODEL_SELECTION_EXPERIMENT == (
        "stroke-risk-model-selection"
    )
    assert CALIBRATION_THRESHOLD_EXPERIMENT == (
        "stroke-risk-calibration-threshold"
    )
    assert FINAL_MODEL_EXPERIMENT == (
        "stroke-risk-final-model"
    )
    assert REGISTERED_MODEL_NAME == (
        "stroke-risk-screening-model"
    )


def test_sqlite_registry_links_model_version_to_run(
    isolated_mlflow,
):
    tracking_uri, artifact_directory = isolated_mlflow
    experiment = select_experiment(
        "temporary-registry-experiment",
        artifact_location=artifact_directory.as_uri(),
    )
    model = DummyClassifier(strategy="prior")
    model.fit([[0], [1]], [0, 1])

    with start_run(
        run_name="temporary-registry-run",
        tags={"model_version": "test_v1"},
    ) as active_run:
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=(
                REGISTERED_MODEL_NAME
            ),
        )
        run_id = active_run.info.run_id

    client = MlflowClient(tracking_uri=tracking_uri)
    versions = client.search_model_versions(
        f"name = '{REGISTERED_MODEL_NAME}'"
    )

    assert len(versions) == 1
    assert versions[0].run_id == run_id
    assert versions[0].name == REGISTERED_MODEL_NAME
    assert client.get_run(run_id).info.experiment_id == (
        experiment.experiment_id
    )


def test_tracking_records_and_recovers_run_metadata(
    isolated_mlflow,
    tmp_path,
):
    tracking_uri, artifact_directory = isolated_mlflow
    artifact = tmp_path / "metrics-summary.txt"
    artifact.write_text(
        "temporary test artifact",
        encoding="utf-8",
    )
    experiment = select_experiment(
        "temporary-test-experiment",
        artifact_location=(
            artifact_directory.as_uri()
        ),
    )
    selected_again = select_experiment(
        "temporary-test-experiment"
    )

    with start_run(
        run_name="temporary-test-run"
    ) as active_run:
        log_params(
            {
                "algorithm": "logistic_regression",
                "random_seed": 42,
            }
        )
        log_metrics(
            {
                "recall": 0.8,
                "pr_auc": 0.2,
            }
        )
        log_tags(
            {
                "stage": "test",
                "experiment_type": "infrastructure",
            }
        )
        logged = log_existing_artifacts(
            [
                artifact,
                tmp_path / "missing.txt",
            ],
            artifact_path="reports",
        )
        run_id = active_run.info.run_id

    client = MlflowClient(
        tracking_uri=tracking_uri
    )
    stored_experiment = (
        client.get_experiment_by_name(
            "temporary-test-experiment"
        )
    )
    stored_run = client.get_run(run_id)
    artifacts = client.list_artifacts(
        run_id,
        path="reports",
    )

    assert mlflow.get_tracking_uri() == tracking_uri
    assert stored_experiment.experiment_id == (
        experiment.experiment_id
    )
    assert selected_again.experiment_id == (
        experiment.experiment_id
    )
    assert stored_run.info.experiment_id == (
        experiment.experiment_id
    )
    assert stored_run.data.params["algorithm"] == (
        "logistic_regression"
    )
    assert stored_run.data.params["random_seed"] == "42"
    assert stored_run.data.metrics["recall"] == 0.8
    assert stored_run.data.metrics["pr_auc"] == 0.2
    assert stored_run.data.tags["stage"] == "test"
    assert stored_run.data.tags["experiment_type"] == (
        "infrastructure"
    )
    assert logged == [artifact]
    assert [item.path for item in artifacts] == [
        "reports/metrics-summary.txt"
    ]
