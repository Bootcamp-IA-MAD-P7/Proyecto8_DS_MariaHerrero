"""Experimental CNN prototype for grouped brain CT image manifests.

Test is deliberately loaded only after training and validation-only threshold
selection. Visible groups are series proxies, not confirmed patients, and
image slices are not independent samples.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight

from src.tracking.mlflow_tracking import (
    configure_tracking,
    log_existing_artifacts,
    log_metrics,
    log_params,
    log_tags,
    select_experiment,
    start_run,
)


MODEL_VERSION = "ct_cnn_v1"
MLFLOW_EXPERIMENT_NAME = "stroke-risk-ct-cnn"
MLFLOW_RUN_NAME = "ct-cnn-v1"
MLFLOW_REGISTERED_MODEL_NAME = "stroke-risk-ct-cnn"
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
CHANNELS = 1
BATCH_SIZE = 16
SEED = 42
MAX_EPOCHS = 100
LEARNING_RATE = 3e-4
L2_FACTOR = 1e-4
DROPOUT_RATE = 0.40
MINIMUM_RECALL = 0.80
THRESHOLD_START = 0.05
THRESHOLD_STOP = 0.50
THRESHOLD_STEP = 0.01

TRAIN_MANIFEST_PATH = Path("data/processed/brain_ct_train.csv")
VALIDATION_MANIFEST_PATH = Path("data/processed/brain_ct_validation.csv")
TEST_MANIFEST_PATH = Path("data/processed/brain_ct_test.csv")
REQUIRED_MANIFEST_COLUMNS = {
    "relative_path", "class_name", "label", "group_id", "split"
}
CLASS_LABELS = {"Normal": 0, "Stroke": 1}


@dataclass(frozen=True)
class OutputPaths:
    metrics: Path = Path("reports/ct_cnn_metrics.csv")
    history: Path = Path("reports/ct_cnn_training_history.csv")
    curves: Path = Path("reports/ct_cnn_learning_curves.png")
    confusion_matrix: Path = Path("reports/ct_cnn_confusion_matrix.png")
    analysis: Path = Path("reports/ct_cnn_analysis.md")
    model: Path = Path("artifacts/ct_cnn/ct_cnn_v1.keras")
    threshold: Path = Path("artifacts/ct_cnn/threshold_ct_cnn_v1.json")


def set_reproducibility(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def load_manifest(path, expected_split):
    manifest = pd.read_csv(path)
    missing = REQUIRED_MANIFEST_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if set(manifest["split"]) != {expected_split}:
        raise ValueError(f"Manifest is not exclusively {expected_split}.")
    if not set(manifest["label"]).issubset({0, 1}):
        raise ValueError("Manifest labels must be 0 or 1.")
    expected_labels = manifest["class_name"].map(CLASS_LABELS)
    if expected_labels.isna().any() or not manifest["label"].equals(
        expected_labels
    ):
        raise ValueError("Manifest class names and labels are inconsistent.")
    return manifest


def load_development_manifests(
    train_path=TRAIN_MANIFEST_PATH,
    validation_path=VALIDATION_MANIFEST_PATH,
):
    return (
        load_manifest(train_path, "train"),
        load_manifest(validation_path, "validation"),
    )


def load_test_manifest(test_path=TEST_MANIFEST_PATH):
    """Load test separately; call only after fixing the threshold."""
    return load_manifest(test_path, "test")


def decode_and_resize(relative_path, label):
    contents = tf.io.read_file(relative_path)
    image = tf.io.decode_jpeg(contents, channels=CHANNELS)
    image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.ensure_shape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS])
    return image, tf.cast(label, tf.float32)


def build_dataset(manifest, training, batch_size=BATCH_SIZE):
    paths = manifest["relative_path"].astype(str).to_numpy()
    labels = manifest["label"].astype(np.float32).to_numpy()
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(manifest),
            seed=SEED,
            reshuffle_each_iteration=True,
        )
    dataset = dataset.map(
        decode_and_resize,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    options = tf.data.Options()
    options.experimental_deterministic = True
    return (
        dataset.with_options(options)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def build_development_datasets(train_manifest, validation_manifest):
    return (
        build_dataset(train_manifest, training=True),
        build_dataset(train_manifest, training=False),
        build_dataset(validation_manifest, training=False),
    )


def build_test_dataset(test_manifest):
    return build_dataset(test_manifest, training=False)


def build_augmentation(seed=SEED):
    return keras.Sequential(
        [
            keras.layers.RandomRotation(
                factor=0.015, fill_mode="reflect", seed=seed
            ),
            keras.layers.RandomTranslation(
                height_factor=0.03,
                width_factor=0.03,
                fill_mode="reflect",
                seed=seed,
            ),
            keras.layers.RandomZoom(
                height_factor=(-0.05, 0.05),
                width_factor=(-0.05, 0.05),
                fill_mode="reflect",
                seed=seed,
            ),
        ],
        name="conservative_ct_augmentation",
    )


def _convolution_block(inputs, filters):
    x = keras.layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        use_bias=False,
        kernel_regularizer=keras.regularizers.L2(L2_FACTOR),
    )(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    return keras.layers.MaxPooling2D()(x)


def build_model():
    inputs = keras.Input(
        shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS), name="brain_ct"
    )
    x = build_augmentation()(inputs)
    x = _convolution_block(x, 16)
    x = _convolution_block(x, 32)
    x = _convolution_block(x, 64)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=keras.regularizers.L2(L2_FACTOR),
    )(x)
    x = keras.layers.Dropout(DROPOUT_RATE)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name="brain_ct_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="roc_auc", curve="ROC"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model


def calculate_class_weights(train_labels):
    labels = np.asarray(train_labels, dtype=int)
    classes = np.sort(np.unique(labels))
    if not np.array_equal(classes, np.array([0, 1])):
        raise ValueError("Train labels must contain both classes 0 and 1.")
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=labels
    )
    return {
        int(label): float(weight)
        for label, weight in zip(classes, weights)
    }


def create_callbacks():
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=10,
            min_delta=1e-4,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
        ),
    ]


def train_model(model, train_dataset, validation_dataset, class_weights):
    return model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=MAX_EPOCHS,
        class_weight=class_weights,
        callbacks=create_callbacks(),
        verbose=1,
    )


def predict_probabilities(model, dataset):
    return model.predict(dataset, verbose=0).reshape(-1)


def calculate_metrics(y_true, probabilities, threshold):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "specificity": float(specificity),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, predictions
        ),
        "fn": int(fn),
        "fp": int(fp),
        "tp": int(tp),
        "tn": int(tn),
    }


def select_threshold(
    y_validation,
    validation_probabilities,
    minimum_recall=MINIMUM_RECALL,
):
    rows = []
    thresholds = np.arange(
        THRESHOLD_START,
        THRESHOLD_STOP + THRESHOLD_STEP / 2,
        THRESHOLD_STEP,
    )
    for threshold in thresholds:
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                **calculate_metrics(
                    y_validation,
                    validation_probabilities,
                    float(threshold),
                ),
            }
        )
    results = pd.DataFrame(rows)
    valid = results[results["recall"] >= minimum_recall]
    if valid.empty:
        selected = results.sort_values(
            by=["recall", "fn", "fp", "f1"],
            ascending=[False, True, True, False],
            kind="stable",
        ).iloc[0]
    else:
        selected = valid.sort_values(
            by=["fp", "f1"],
            ascending=[True, False],
            kind="stable",
        ).iloc[0]
    return float(selected["threshold"]), results


def history_summary(history):
    frame = pd.DataFrame(history.history)
    frame.insert(0, "epoch", np.arange(1, len(frame) + 1))
    best_index = int(frame["val_loss"].idxmin())
    return frame, {
        "best_epoch": int(frame.loc[best_index, "epoch"]),
        "epochs_executed": len(frame),
        "best_val_loss": float(frame.loc[best_index, "val_loss"]),
    }


def run_pipeline():
    """Train, fix threshold, then expose test exactly once."""
    set_reproducibility()
    train_manifest, validation_manifest = load_development_manifests()
    (
        train_dataset,
        train_evaluation_dataset,
        validation_dataset,
    ) = build_development_datasets(train_manifest, validation_manifest)
    class_weights = calculate_class_weights(train_manifest["label"])
    model = build_model()
    history = train_model(
        model, train_dataset, validation_dataset, class_weights
    )
    train_probabilities = predict_probabilities(
        model, train_evaluation_dataset
    )
    validation_probabilities = predict_probabilities(
        model, validation_dataset
    )
    threshold, threshold_results = select_threshold(
        validation_manifest["label"], validation_probabilities
    )

    # Architecture, checkpoint and threshold are fixed before this first read.
    test_manifest = load_test_manifest()
    test_dataset = build_test_dataset(test_manifest)
    test_probabilities = predict_probabilities(model, test_dataset)

    metrics = {
        "train": calculate_metrics(
            train_manifest["label"], train_probabilities, threshold
        ),
        "validation": calculate_metrics(
            validation_manifest["label"],
            validation_probabilities,
            threshold,
        ),
        "test": calculate_metrics(
            test_manifest["label"], test_probabilities, threshold
        ),
    }
    return {
        "model": model,
        "history": history,
        "threshold": threshold,
        "threshold_results": threshold_results,
        "metrics": metrics,
        "class_weights": class_weights,
        "manifests": {
            "train": train_manifest,
            "validation": validation_manifest,
            "test": test_manifest,
        },
        "test_probabilities": test_probabilities,
    }


def save_outputs(result, paths=OutputPaths()):
    for path in vars(paths).values():
        path.parent.mkdir(parents=True, exist_ok=True)
    history_frame, summary = history_summary(result["history"])
    history_frame.to_csv(paths.history, index=False)
    rows = [
        {
            "split": split_name,
            **metrics,
            "threshold": result["threshold"],
            **summary,
        }
        for split_name, metrics in result["metrics"].items()
    ]
    pd.DataFrame(rows).to_csv(paths.metrics, index=False)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, metric in zip(axes, ("loss", "recall", "pr_auc")):
        axis.plot(history_frame["epoch"], history_frame[metric], label="train")
        axis.plot(
            history_frame["epoch"],
            history_frame[f"val_{metric}"],
            label="validation",
        )
        axis.set_title(metric)
        axis.set_xlabel("Epoch")
        axis.legend()
    figure.tight_layout()
    figure.savefig(paths.curves)
    plt.close(figure)

    labels = result["manifests"]["test"]["label"].to_numpy()
    predictions = (
        result["test_probabilities"] >= result["threshold"]
    ).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(5, 5))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, matrix[row, column], ha="center")
    axis.set_xticks([0, 1], labels=["Normal", "Stroke"])
    axis.set_yticks([0, 1], labels=["Normal", "Stroke"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(paths.confusion_matrix)
    plt.close(figure)

    result["model"].save(paths.model)
    paths.threshold.write_text(
        json.dumps(
            {
                "model_version": MODEL_VERSION,
                "threshold": result["threshold"],
                "selected_on": "validation",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths.analysis.write_text(
        "# CT CNN prototype\n\n"
        "Evaluation is image-level. Slices are correlated; group_id is a "
        "series/group proxy, not a confirmed patient identifier. This "
        "prototype is not a diagnostic system.\n",
        encoding="utf-8",
    )
    return history_frame, summary


def log_result(result, history_frame, summary, paths=OutputPaths()):
    configure_tracking()
    select_experiment(MLFLOW_EXPERIMENT_NAME)
    with start_run(run_name=MLFLOW_RUN_NAME):
        manifests = result["manifests"]
        log_params(
            {
                "model_version": MODEL_VERSION,
                "image_height": IMAGE_HEIGHT,
                "image_width": IMAGE_WIDTH,
                "channels": CHANNELS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "l2_factor": L2_FACTOR,
                "dropout_rate": DROPOUT_RATE,
                "max_epochs": MAX_EPOCHS,
                "seed": SEED,
                "class_weights": json.dumps(result["class_weights"]),
                "threshold": result["threshold"],
                "threshold_selected_on": "validation",
                "augmentation": "rotation_translation_zoom",
                "horizontal_flip": False,
                "train_images": len(manifests["train"]),
                "validation_images": len(manifests["validation"]),
                "test_images": len(manifests["test"]),
                "train_groups": manifests["train"]["group_id"].nunique(),
                "validation_groups": manifests["validation"][
                    "group_id"
                ].nunique(),
                "test_groups": manifests["test"]["group_id"].nunique(),
            }
        )
        metrics = dict(summary)
        for split_name, split_metrics in result["metrics"].items():
            metrics.update(
                {
                    f"{split_name}_{name}": value
                    for name, value in split_metrics.items()
                }
            )
        log_metrics(metrics)
        log_tags(
            {
                "project": "stroke-risk-ai",
                "experiment_type": "ct_cnn",
                "modality": "brain_ct_jpeg",
                "stage": "prototype",
                "deployed": "false",
                "group_split": "true",
                "group_unit": "series_proxy",
                "patient_independence_verified": "false",
                "target": "stroke",
                "model_version": MODEL_VERSION,
            }
        )
        for step, row in history_frame.iterrows():
            for name in (
                "loss",
                "val_loss",
                "recall",
                "val_recall",
                "roc_auc",
                "val_roc_auc",
                "pr_auc",
                "val_pr_auc",
            ):
                mlflow.log_metric(name, float(row[name]), step=int(step))
        log_existing_artifacts(
            [
                paths.metrics,
                paths.history,
                paths.curves,
                paths.confusion_matrix,
                paths.analysis,
                paths.threshold,
            ],
            artifact_path="project-artifacts",
        )
        mlflow.keras.log_model(
            model=result["model"],
            name="model",
            registered_model_name=MLFLOW_REGISTERED_MODEL_NAME,
            metadata={
                "model_version": MODEL_VERSION,
                "threshold": result["threshold"],
                "deployed": False,
            },
        )


def main():
    result = run_pipeline()
    history_frame, summary = save_outputs(result)
    log_result(result, history_frame, summary)


if __name__ == "__main__":
    main()
