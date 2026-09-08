import inspect

import numpy as np
import pandas as pd
import pytest


pytest.importorskip("tensorflow")

from src.models import tabular_neural_network as tabular_nn  # noqa: E402


class SerializablePreprocessor:
    pass


def feature_dataframe(rows=8):
    return pd.DataFrame(
        {
            "gender": ["Female", "Male"] * (rows // 2),
            "age": np.linspace(30, 70, rows),
            "hypertension": [0, 1] * (rows // 2),
            "heart_disease": [0, 0, 1, 0] * (rows // 4),
            "ever_married": ["No", "Yes"] * (rows // 2),
            "work_type": ["Private", "Govt_job"] * (rows // 2),
            "Residence_type": ["Urban", "Rural"] * (rows // 2),
            "avg_glucose_level": np.linspace(80, 160, rows),
            "bmi": np.linspace(20, 35, rows),
            "smoking_status": [
                "never smoked",
                "formerly smoked",
            ]
            * (rows // 2),
        }
    )


def test_model_output_shape_and_probability_range():
    model = tabular_nn.build_model(input_dim=7)
    probabilities = model(np.zeros((3, 7), dtype=np.float32)).numpy()

    assert probabilities.shape == (3, 1)
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


def test_input_dimension_is_obtained_from_preprocessing():
    train = feature_dataframe()
    validation = feature_dataframe(4)
    _, transformed_train, transformed_validation = (
        tabular_nn.prepare_development_data(train, validation)
    )
    model = tabular_nn.build_model(transformed_train.shape[1])

    assert model.input_shape[-1] == transformed_train.shape[1]
    assert transformed_validation.shape[1] == transformed_train.shape[1]


def test_preprocessor_is_fitted_only_on_train(monkeypatch):
    train = feature_dataframe()
    validation = feature_dataframe(4)

    class SpyPreprocessor:
        def __init__(self):
            self.fitted_index = None

        def fit_transform(self, dataframe):
            self.fitted_index = dataframe.index.tolist()
            return np.zeros((len(dataframe), 2))

        def transform(self, dataframe):
            return np.zeros((len(dataframe), 2))

    spy = SpyPreprocessor()
    monkeypatch.setattr(tabular_nn, "create_preprocessor", lambda: spy)

    _, transformed_train, transformed_validation = (
        tabular_nn.prepare_development_data(train, validation)
    )

    assert spy.fitted_index == train.index.tolist()
    assert len(transformed_train) == len(train)
    assert len(transformed_validation) == len(validation)


def test_class_weights_are_calculated_from_supplied_train_target():
    y_train = pd.Series([0, 0, 0, 1])
    weights = tabular_nn.calculate_class_weights(y_train)

    assert weights == {0: pytest.approx(2 / 3), 1: pytest.approx(2.0)}


def test_validation_and_test_are_transformed_without_resampling():
    train = feature_dataframe()
    validation = feature_dataframe(4)
    test = feature_dataframe(4)
    preprocessor, _, transformed_validation = (
        tabular_nn.prepare_development_data(train, validation)
    )
    transformed_test = tabular_nn.transform_test_data(
        preprocessor,
        test,
    )

    assert len(transformed_validation) == len(validation)
    assert len(transformed_test) == len(test)


def test_threshold_selection_has_no_test_input():
    parameters = inspect.signature(tabular_nn.select_threshold).parameters
    y_validation = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.7, 0.9])
    threshold, results = tabular_nn.select_threshold(
        y_validation,
        probabilities,
    )

    assert "test" not in " ".join(parameters).lower()
    assert 0.05 <= threshold <= 0.50
    assert len(results) == 46


def test_early_stopping_restores_best_weights():
    callback = tabular_nn.create_early_stopping()

    assert callback.monitor == "val_loss"
    assert callback.mode == "min"
    assert callback.patience == 10
    assert callback.min_delta == pytest.approx(1e-4)
    assert callback.restore_best_weights is True


def test_calculate_metrics_returns_complete_contract():
    metrics = tabular_nn.calculate_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.4, 0.6, 0.9]),
        threshold=0.5,
    )

    assert set(metrics) == {
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "false_negatives",
        "false_positives",
        "true_positives",
        "true_negatives",
    }


def test_minimal_synthetic_training():
    tabular_nn.set_reproducibility()
    features = np.vstack(
        [
            np.zeros((8, 3), dtype=np.float32),
            np.ones((8, 3), dtype=np.float32),
        ]
    )
    target = np.array([0] * 8 + [1] * 8)
    model = tabular_nn.build_model(
        input_dim=3,
        hidden_units=(4, 2),
        dropout_rates=(0.0, 0.0),
    )
    history = model.fit(
        features,
        target,
        validation_data=(features, target),
        epochs=2,
        batch_size=4,
        verbose=0,
    )

    assert len(history.history["loss"]) == 2
    assert np.isfinite(history.history["val_loss"]).all()


def test_reports_and_artifacts_use_supplied_temporary_paths(tmp_path):
    features = np.zeros((4, 2), dtype=np.float32)
    target = np.array([0, 0, 1, 1])
    model = tabular_nn.build_model(
        input_dim=2,
        hidden_units=(4, 2),
        dropout_rates=(0.0, 0.0),
    )
    history = model.fit(
        features,
        target,
        validation_data=(features, target),
        epochs=1,
        verbose=0,
    )
    paths = tabular_nn.OutputPaths(
        history=tmp_path / "reports/history.csv",
        curves=tmp_path / "reports/curves.png",
        metrics=tmp_path / "reports/metrics.csv",
        confusion_matrix=tmp_path / "reports/confusion.png",
        model=tmp_path / "artifacts/model.keras",
        preprocessor=tmp_path / "artifacts/preprocessor.joblib",
    )
    history_dataframe = tabular_nn.save_training_history(
        history,
        paths.history,
    )
    tabular_nn.save_learning_curves(history_dataframe, paths.curves)
    metrics = tabular_nn.calculate_metrics(
        target,
        np.array([0.1, 0.2, 0.8, 0.9]),
        0.5,
    )
    tabular_nn.save_metrics_report(
        {"train": metrics, "validation": metrics, "test": metrics},
        paths.metrics,
    )
    tabular_nn.save_confusion_matrix(
        target,
        np.array([0.1, 0.2, 0.8, 0.9]),
        0.5,
        paths.confusion_matrix,
    )

    tabular_nn.save_model_artifacts(
        model,
        SerializablePreprocessor(),
        paths,
    )

    assert all(
        path.is_file()
        for path in (
            paths.history,
            paths.curves,
            paths.metrics,
            paths.confusion_matrix,
            paths.model,
            paths.preprocessor,
        )
    )


def test_neural_network_uses_separate_mlflow_experiment():
    assert tabular_nn.EXPERIMENT_NAME == (
        "stroke-risk-tabular-neural-network"
    )
    assert tabular_nn.REGISTERED_MODEL_NAME == (
        "stroke-risk-tabular-neural-network"
    )
    assert tabular_nn.REGISTERED_MODEL_NAME != (
        "stroke-risk-screening-model"
    )


def test_exactly_four_candidates_are_predefined():
    assert len(tabular_nn.CANDIDATE_CONFIGS) == 4
    assert {
        config["hidden_units"]
        for config in tabular_nn.CANDIDATE_CONFIGS
    } == {(32, 16), (16, 8)}
    assert {
        config["learning_rate"]
        for config in tabular_nn.CANDIDATE_CONFIGS
    } == {1e-3, 3e-4}
