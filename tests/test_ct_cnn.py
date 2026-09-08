from pathlib import Path

import numpy as np
import pandas as pd
import pytest


tf = pytest.importorskip("tensorflow")
pytest.importorskip("keras")

from src.models import ct_cnn  # noqa: E402


def make_manifest(tmp_path, split="train"):
    image_path = tmp_path / f"{split}.jpg"
    image = tf.cast(
        tf.reshape(tf.range(64 * 64), (64, 64, 1)) % 256,
        tf.uint8,
    )
    tf.io.write_file(str(image_path), tf.io.encode_jpeg(image))
    return pd.DataFrame(
        {
            "relative_path": [str(image_path)],
            "class_name": ["Normal"],
            "label": [0],
            "group_id": ["Normal_1"],
            "split": [split],
        }
    )


def test_decode_has_expected_shape_type_and_range(tmp_path):
    manifest = make_manifest(tmp_path)
    image, label = ct_cnn.decode_and_resize(
        manifest.loc[0, "relative_path"],
        manifest.loc[0, "label"],
    )

    assert image.shape == (224, 224, 1)
    assert image.dtype == tf.float32
    assert 0.0 <= float(tf.reduce_min(image))
    assert float(tf.reduce_max(image)) <= 1.0
    assert float(label) == 0.0


def test_manifest_preserves_label_mapping(tmp_path):
    path = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "relative_path": ["normal.jpg", "stroke.jpg"],
            "class_name": ["Normal", "Stroke"],
            "label": [0, 1],
            "group_id": ["Normal_1", "Stroke_1"],
            "split": ["train", "train"],
        }
    ).to_csv(path, index=False)

    manifest = ct_cnn.load_manifest(path, "train")
    assert dict(zip(manifest["class_name"], manifest["label"])) == {
        "Normal": 0,
        "Stroke": 1,
    }


def test_augmentation_is_active_only_during_training():
    augmentation = ct_cnn.build_augmentation()
    images = tf.reshape(
        tf.linspace(0.0, 1.0, 224 * 224),
        (1, 224, 224, 1),
    )

    inference = augmentation(images, training=False)
    training = augmentation(images, training=True)

    np.testing.assert_allclose(inference.numpy(), images.numpy())
    assert not np.allclose(training.numpy(), images.numpy())


def test_augmentation_contains_no_flips():
    layer_names = {
        type(layer).__name__
        for layer in ct_cnn.build_augmentation().layers
    }

    assert layer_names == {
        "RandomRotation",
        "RandomTranslation",
        "RandomZoom",
    }
    assert "RandomFlip" not in layer_names


def test_model_architecture_and_regularization():
    model = ct_cnn.build_model()
    layer_types = [type(layer).__name__ for layer in model.layers]

    assert model.input_shape == (None, 224, 224, 1)
    assert model.output_shape == (None, 1)
    assert model.layers[-1].activation.__name__ == "sigmoid"
    assert layer_types.count("BatchNormalization") == 3
    assert "GlobalAveragePooling2D" in layer_types
    assert "Dropout" in layer_types
    assert "Flatten" not in layer_types
    assert model.get_layer(index=-2).rate == pytest.approx(0.40)
    regularized = [
        layer
        for layer in model.layers
        if hasattr(layer, "kernel_regularizer")
        and layer.kernel_regularizer is not None
    ]
    assert len(regularized) == 4


def test_class_weights_use_supplied_train_labels():
    weights = ct_cnn.calculate_class_weights([0, 0, 0, 1])

    assert weights[0] == pytest.approx(2 / 3)
    assert weights[1] == pytest.approx(2.0)


def threshold_metrics(recall, fn, fp, f1):
    return {
        "precision": 0.0,
        "recall": recall,
        "f1": f1,
        "roc_auc": 0.5,
        "pr_auc": 0.5,
        "specificity": 1.0,
        "balanced_accuracy": 0.5,
        "fn": fn,
        "fp": fp,
        "tp": 0,
        "tn": 1,
    }


def test_threshold_prioritizes_recall_then_fp_and_f1(monkeypatch):
    rows = {
        0.05: threshold_metrics(0.9, 1, 5, 0.4),
        0.06: threshold_metrics(0.8, 2, 3, 0.5),
        0.07: threshold_metrics(0.8, 2, 3, 0.6),
    }
    monkeypatch.setattr(
        ct_cnn,
        "calculate_metrics",
        lambda _y, _p, threshold: rows.get(
            round(threshold, 2),
            threshold_metrics(0.0, 10, 0, 0.0),
        ),
    )

    threshold, _ = ct_cnn.select_threshold([0, 1], [0.1, 0.9])
    assert threshold == 0.07


def test_threshold_fallback_uses_approved_order(monkeypatch):
    rows = {
        0.05: threshold_metrics(0.7, 3, 8, 0.4),
        0.06: threshold_metrics(0.7, 2, 7, 0.4),
        0.07: threshold_metrics(0.7, 2, 6, 0.3),
        0.08: threshold_metrics(0.7, 2, 6, 0.5),
    }
    monkeypatch.setattr(
        ct_cnn,
        "calculate_metrics",
        lambda _y, _p, threshold: rows.get(
            round(threshold, 2),
            threshold_metrics(0.0, 10, 0, 0.0),
        ),
    )

    threshold, _ = ct_cnn.select_threshold(
        [0, 1], [0.1, 0.9], minimum_recall=0.95
    )
    assert threshold == 0.08


def test_metrics_with_synthetic_probabilities():
    metrics = ct_cnn.calculate_metrics(
        [0, 0, 1, 1],
        [0.1, 0.8, 0.4, 0.9],
        threshold=0.5,
    )

    for name in (
        "precision",
        "recall",
        "f1",
        "specificity",
        "balanced_accuracy",
    ):
        assert metrics[name] == 0.5
    assert (metrics["fn"], metrics["fp"], metrics["tp"], metrics["tn"]) == (
        1,
        1,
        1,
        1,
    )


def test_pipeline_exposes_test_once_after_threshold(monkeypatch):
    events = []
    train = pd.DataFrame({"label": [0, 1], "group_id": ["N1", "S1"]})
    validation = pd.DataFrame(
        {"label": [0, 1], "group_id": ["N2", "S2"]}
    )
    test = pd.DataFrame({"label": [0, 1], "group_id": ["N3", "S3"]})

    monkeypatch.setattr(ct_cnn, "set_reproducibility", lambda: None)
    monkeypatch.setattr(
        ct_cnn,
        "load_development_manifests",
        lambda: (events.append("development") or (train, validation)),
    )
    monkeypatch.setattr(
        ct_cnn,
        "build_development_datasets",
        lambda *_: (
            "train_dataset",
            "train_evaluation_dataset",
            "validation_dataset",
        ),
    )
    monkeypatch.setattr(ct_cnn, "calculate_class_weights", lambda _: {})
    monkeypatch.setattr(ct_cnn, "build_model", object)
    monkeypatch.setattr(ct_cnn, "train_model", lambda *_: object())
    prediction_calls = []

    def fake_predict(_model, dataset):
        prediction_calls.append(dataset)
        return np.array([0.1, 0.9])

    monkeypatch.setattr(ct_cnn, "predict_probabilities", fake_predict)
    monkeypatch.setattr(
        ct_cnn,
        "select_threshold",
        lambda *_: (events.append("threshold") or (0.5, pd.DataFrame())),
    )
    monkeypatch.setattr(
        ct_cnn,
        "load_test_manifest",
        lambda: (events.append("test") or test),
    )
    monkeypatch.setattr(
        ct_cnn, "build_test_dataset", lambda _: "test_dataset"
    )

    ct_cnn.run_pipeline()

    assert events == ["development", "threshold", "test"]
    assert prediction_calls.count("test_dataset") == 1


def test_mlflow_and_output_names_are_separate_from_tabular_pipeline():
    paths = ct_cnn.OutputPaths()

    assert ct_cnn.MLFLOW_EXPERIMENT_NAME == "stroke-risk-ct-cnn"
    assert ct_cnn.MLFLOW_REGISTERED_MODEL_NAME == "stroke-risk-ct-cnn"
    assert ct_cnn.MLFLOW_RUN_NAME == "ct-cnn-v1"
    assert "final_model" not in str(paths.model)
    assert "tabular_neural_network" not in str(paths.model)
    assert Path(paths.model).parts[:2] == ("artifacts", "ct_cnn")
