from dataclasses import dataclass
import json
from pathlib import Path
import random

import joblib
import keras
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight

from src.preprocessing.pipeline import create_preprocessor
from src.tracking.mlflow_tracking import (
    configure_tracking,
    log_existing_artifacts,
    log_metrics,
    log_params,
    log_tags,
    select_experiment,
    start_run,
)


TRAIN_PATH = Path("data/processed/train.csv")
VALIDATION_PATH = Path("data/processed/validation.csv")
TEST_PATH = Path("data/processed/test.csv")

TARGET = "stroke"
RANDOM_SEED = 42

EXPERIMENT_NAME = "stroke-risk-tabular-neural-network"
RUN_NAME = "tabular-neural-network-v1"
REGISTERED_MODEL_NAME = "stroke-risk-tabular-neural-network"
MODEL_VERSION = "tabular_nn_v1"

MAX_EPOCHS = 100
PATIENCE = 10
MIN_DELTA = 1e-4
BATCH_SIZE = 32
L2_VALUE = 1e-4
MIN_RECALL = 0.80
CALIBRATION_METHOD = "none"
PRACTICAL_TIE_TOLERANCE = 1e-4

CANDIDATE_CONFIGS = (
    {
        "name": "candidate-1",
        "hidden_units": (32, 16),
        "dropout_rates": (0.30, 0.20),
        "learning_rate": 1e-3,
    },
    {
        "name": "candidate-2",
        "hidden_units": (32, 16),
        "dropout_rates": (0.20, 0.20),
        "learning_rate": 3e-4,
    },
    {
        "name": "candidate-3",
        "hidden_units": (16, 8),
        "dropout_rates": (0.30, 0.20),
        "learning_rate": 1e-3,
    },
    {
        "name": "candidate-4",
        "hidden_units": (16, 8),
        "dropout_rates": (0.20, 0.20),
        "learning_rate": 3e-4,
    },
)


@dataclass(frozen=True)
class OutputPaths:
    history: Path = Path(
        "reports/tabular_nn_training_history.csv"
    )
    curves: Path = Path(
        "reports/tabular_nn_learning_curves.png"
    )
    metrics: Path = Path(
        "reports/tabular_nn_metrics.csv"
    )
    confusion_matrix: Path = Path(
        "reports/tabular_nn_confusion_matrix.png"
    )
    model: Path = Path(
        "artifacts/tabular_neural_network/"
        "tabular_nn_v1.keras"
    )
    preprocessor: Path = Path(
        "artifacts/tabular_neural_network/"
        "preprocessor_tabular_nn_v1.joblib"
    )


def set_reproducibility(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def load_development_splits(
    train_path=TRAIN_PATH,
    validation_path=VALIDATION_PATH,
):
    return (
        pd.read_csv(train_path),
        pd.read_csv(validation_path),
    )


def load_test_split(test_path=TEST_PATH):
    return pd.read_csv(test_path)


def split_features_target(dataset):
    return (
        dataset.drop(columns=[TARGET]),
        dataset[TARGET].astype(int),
    )


def prepare_development_data(X_train, X_validation):
    preprocessor = create_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_validation_transformed = preprocessor.transform(
        X_validation
    )

    return (
        preprocessor,
        np.asarray(X_train_transformed, dtype=np.float32),
        np.asarray(X_validation_transformed, dtype=np.float32),
    )


def transform_test_data(preprocessor, X_test):
    transformed = preprocessor.transform(X_test)
    return np.asarray(transformed, dtype=np.float32)


def calculate_class_weights(y_train):
    classes = np.sort(np.unique(y_train))
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=np.asarray(y_train),
    )

    return {
        int(label): float(weight)
        for label, weight in zip(classes, weights)
    }


def build_model(
    input_dim,
    hidden_units=(32, 16),
    dropout_rates=(0.30, 0.20),
    learning_rate=1e-3,
    l2_value=L2_VALUE,
):
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(
                hidden_units[0],
                activation="relu",
                kernel_regularizer=keras.regularizers.L2(
                    l2_value
                ),
            ),
            keras.layers.Dropout(dropout_rates[0]),
            keras.layers.Dense(
                hidden_units[1],
                activation="relu",
                kernel_regularizer=keras.regularizers.L2(
                    l2_value
                ),
            ),
            keras.layers.Dropout(dropout_rates[1]),
            keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="tabular_neural_network",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="roc_auc", curve="ROC"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )

    return model


def create_early_stopping():
    return keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        restore_best_weights=True,
    )


def predict_probabilities(model, features):
    return model.predict(features, verbose=0).reshape(-1)


def calculate_metrics(y_true, probabilities, threshold):
    probabilities = np.asarray(probabilities)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "true_positives": int(tp),
        "true_negatives": int(tn),
    }


def select_threshold(
    y_validation,
    validation_probabilities,
    minimum_recall=MIN_RECALL,
):
    rows = []

    for threshold in np.arange(0.05, 0.51, 0.01):
        metrics = calculate_metrics(
            y_validation,
            validation_probabilities,
            float(threshold),
        )
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                **metrics,
            }
        )

    results = pd.DataFrame(rows)
    valid = results[results["recall"] >= minimum_recall]

    if valid.empty:
        selected = results.sort_values(
            by=[
                "recall",
                "false_negatives",
                "false_positives",
            ],
            ascending=[False, True, True],
        ).iloc[0]
    else:
        selected = valid.sort_values(
            by=["false_positives", "f1"],
            ascending=[True, False],
        ).iloc[0]

    return float(selected["threshold"]), results


def history_summary(history):
    history_data = history.history
    best_index = int(np.argmin(history_data["val_loss"]))

    return {
        "best_epoch": best_index + 1,
        "epochs_executed": len(history_data["loss"]),
        "best_val_loss": float(
            history_data["val_loss"][best_index]
        ),
    }


def train_candidate(
    config,
    X_train,
    y_train,
    X_validation,
    y_validation,
    class_weights,
):
    set_reproducibility()
    model = build_model(
        input_dim=X_train.shape[1],
        hidden_units=config["hidden_units"],
        dropout_rates=config["dropout_rates"],
        learning_rate=config["learning_rate"],
    )
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[create_early_stopping()],
        verbose=0,
    )
    train_probabilities = predict_probabilities(model, X_train)
    validation_probabilities = predict_probabilities(
        model,
        X_validation,
    )
    train_metrics = calculate_metrics(
        y_train,
        train_probabilities,
        threshold=0.5,
    )
    validation_metrics = calculate_metrics(
        y_validation,
        validation_probabilities,
        threshold=0.5,
    )
    summary = history_summary(history)

    return {
        "config": config,
        "model": model,
        "history": history,
        "train_probabilities": train_probabilities,
        "validation_probabilities": validation_probabilities,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "train_validation_f1_gap": abs(
            train_metrics["f1"] - validation_metrics["f1"]
        ),
        "train_validation_pr_auc_gap": abs(
            train_metrics["pr_auc"]
            - validation_metrics["pr_auc"]
        ),
        **summary,
    }


def _candidate_is_better(candidate, best, tolerance):
    candidate_metrics = candidate["validation_metrics"]
    best_metrics = best["validation_metrics"]
    pr_difference = (
        candidate_metrics["pr_auc"] - best_metrics["pr_auc"]
    )

    if pr_difference > tolerance:
        return True
    if abs(pr_difference) > tolerance:
        return False

    recall_difference = (
        candidate_metrics["recall"] - best_metrics["recall"]
    )
    if recall_difference > tolerance:
        return True
    if abs(recall_difference) > tolerance:
        return False

    candidate_gap = candidate["train_validation_pr_auc_gap"]
    best_gap = best["train_validation_pr_auc_gap"]
    if candidate_gap < best_gap - tolerance:
        return True
    if abs(candidate_gap - best_gap) > tolerance:
        return False

    return sum(candidate["config"]["hidden_units"]) < sum(
        best["config"]["hidden_units"]
    )


def select_best_candidate(
    candidates,
    tolerance=PRACTICAL_TIE_TOLERANCE,
):
    if not candidates:
        raise ValueError("No hay candidatos para seleccionar.")

    best = candidates[0]
    for candidate in candidates[1:]:
        if _candidate_is_better(candidate, best, tolerance):
            best = candidate

    return best


def trainable_parameter_count(model):
    return int(
        sum(np.prod(variable.shape) for variable in model.trainable_weights)
    )


def log_history_to_mlflow(history):
    for step in range(len(history.history["loss"])):
        for name, values in history.history.items():
            mlflow.log_metric(name, float(values[step]), step=step)


def log_candidate(candidate, class_weights):
    config = candidate["config"]
    with start_run(run_name=f"tabular-nn-{config['name']}"):
        log_params(
            {
                "architecture": "Dense-Dropout-Dense-Dropout-Sigmoid",
                "hidden_units": json.dumps(config["hidden_units"]),
                "dropout_rates": json.dumps(config["dropout_rates"]),
                "learning_rate": config["learning_rate"],
                "l2": L2_VALUE,
                "batch_size": BATCH_SIZE,
                "class_weights": json.dumps(class_weights),
                "random_seed": RANDOM_SEED,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "min_delta": MIN_DELTA,
                "best_epoch": candidate["best_epoch"],
                "preprocessing": "create_preprocessor",
            }
        )
        log_metrics(
            {
                **{
                    f"validation_{name}": float(value)
                    for name, value in candidate[
                        "validation_metrics"
                    ].items()
                },
                "train_validation_f1_gap": candidate[
                    "train_validation_f1_gap"
                ],
                "train_validation_pr_auc_gap": candidate[
                    "train_validation_pr_auc_gap"
                ],
                "epochs_executed": candidate["epochs_executed"],
                "best_val_loss": candidate["best_val_loss"],
            }
        )
        log_tags(
            {
                "project": "stroke-risk-ai",
                "experiment_type": "tabular_neural_network_candidate",
                "stage": "model_selection",
                "algorithm": "KerasDenseNeuralNetwork",
                "data_split": "train_validation",
                "target": TARGET,
                "deployed": "false",
            }
        )
        log_history_to_mlflow(candidate["history"])


def save_training_history(history, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(history.history)
    dataframe.index = dataframe.index + 1
    dataframe.index.name = "epoch"
    dataframe.to_csv(path)
    return dataframe


def save_learning_curves(history_dataframe, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    pairs = (
        ("loss", "val_loss", "Loss"),
        ("precision", "val_precision", "Precision"),
        ("recall", "val_recall", "Recall"),
        ("pr_auc", "val_pr_auc", "PR-AUC"),
    )

    for axis, (train_name, validation_name, title) in zip(
        axes.flat,
        pairs,
    ):
        axis.plot(
            history_dataframe.index,
            history_dataframe[train_name],
            label="Train",
        )
        axis.plot(
            history_dataframe.index,
            history_dataframe[validation_name],
            label="Validation",
        )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.legend()

    fig.suptitle("Tabular neural network learning curves")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_metrics_report(metrics_by_split, path, summary=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = summary or {}
    dataframe = pd.DataFrame(
        [
            {"dataset": split_name, **metrics, **summary}
            for split_name, metrics in metrics_by_split.items()
        ]
    )
    dataframe.to_csv(path, index=False)
    return dataframe


def save_confusion_matrix(y_test, probabilities, threshold, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["No stroke", "Stroke"],
        values_format="d",
    )
    display.ax_.set_title("Tabular neural network — test")
    display.figure_.tight_layout()
    display.figure_.savefig(path)
    plt.close(display.figure_)


def save_model_artifacts(model, preprocessor, paths):
    paths.model.parent.mkdir(parents=True, exist_ok=True)
    paths.preprocessor.parent.mkdir(parents=True, exist_ok=True)
    model.save(paths.model)
    joblib.dump(preprocessor, paths.preprocessor)


def log_final_run(
    selected,
    class_weights,
    threshold,
    metrics_by_split,
    paths,
):
    config = selected["config"]
    train_metrics = metrics_by_split["train"]
    validation_metrics = metrics_by_split["validation"]

    with start_run(run_name=RUN_NAME):
        log_params(
            {
                "model_version": MODEL_VERSION,
                "architecture": "Dense-Dropout-Dense-Dropout-Sigmoid",
                "hidden_units": json.dumps(config["hidden_units"]),
                "dropout_rates": json.dumps(config["dropout_rates"]),
                "learning_rate": config["learning_rate"],
                "l2": L2_VALUE,
                "batch_size": BATCH_SIZE,
                "class_weights": json.dumps(class_weights),
                "random_seed": RANDOM_SEED,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "min_delta": MIN_DELTA,
                "best_epoch": selected["best_epoch"],
                "threshold": threshold,
                "threshold_selected_on": "validation",
                "calibration_method": CALIBRATION_METHOD,
                "preprocessing": "create_preprocessor",
            }
        )
        flattened_metrics = {
            f"{split}_{name}": float(value)
            for split, metrics in metrics_by_split.items()
            for name, value in metrics.items()
        }
        flattened_metrics.update(
            {
                "train_validation_f1_gap": abs(
                    train_metrics["f1"] - validation_metrics["f1"]
                ),
                "train_validation_pr_auc_gap": abs(
                    train_metrics["pr_auc"]
                    - validation_metrics["pr_auc"]
                ),
                "best_epoch": selected["best_epoch"],
                "epochs_executed": selected["epochs_executed"],
                "best_val_loss": selected["best_val_loss"],
                "trainable_parameters": trainable_parameter_count(
                    selected["model"]
                ),
                "selected_threshold": threshold,
            }
        )
        log_metrics(flattened_metrics)
        log_tags(
            {
                "project": "stroke-risk-ai",
                "experiment_type": "tabular_neural_network",
                "stage": "comparison",
                "algorithm": "KerasDenseNeuralNetwork",
                "data_split": "train_validation_test",
                "target": TARGET,
                "deployed": "false",
            }
        )
        log_history_to_mlflow(selected["history"])
        log_existing_artifacts(
            [
                paths.history,
                paths.curves,
                paths.metrics,
                paths.confusion_matrix,
                paths.preprocessor,
            ],
            artifact_path="project-artifacts",
        )
        mlflow.keras.log_model(
            model=selected["model"],
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            metadata={
                "model_version": MODEL_VERSION,
                "threshold": threshold,
                "calibration_method": CALIBRATION_METHOD,
            },
        )


def main():
    set_reproducibility()
    train, validation = load_development_splits()
    X_train, y_train = split_features_target(train)
    X_validation, y_validation = split_features_target(validation)
    (
        preprocessor,
        X_train_transformed,
        X_validation_transformed,
    ) = prepare_development_data(X_train, X_validation)
    class_weights = calculate_class_weights(y_train)

    configure_tracking()
    select_experiment(EXPERIMENT_NAME)

    candidates = []
    for config in CANDIDATE_CONFIGS:
        candidate = train_candidate(
            config,
            X_train_transformed,
            y_train,
            X_validation_transformed,
            y_validation,
            class_weights,
        )
        log_candidate(candidate, class_weights)
        candidates.append(candidate)

    selected = select_best_candidate(candidates)
    threshold, _threshold_results = select_threshold(
        y_validation,
        selected["validation_probabilities"],
    )

    # Test remains unread until model configuration, checkpoint and
    # operating threshold have all been selected on train/validation.
    test = load_test_split()
    X_test, y_test = split_features_target(test)
    X_test_transformed = transform_test_data(preprocessor, X_test)
    test_probabilities = predict_probabilities(
        selected["model"],
        X_test_transformed,
    )

    metrics_by_split = {
        "train": calculate_metrics(
            y_train,
            selected["train_probabilities"],
            threshold,
        ),
        "validation": calculate_metrics(
            y_validation,
            selected["validation_probabilities"],
            threshold,
        ),
        "test": calculate_metrics(
            y_test,
            test_probabilities,
            threshold,
        ),
    }

    paths = OutputPaths()
    overfitting_summary = {
        "train_validation_f1_gap": abs(
            metrics_by_split["train"]["f1"]
            - metrics_by_split["validation"]["f1"]
        ),
        "train_validation_pr_auc_gap": abs(
            metrics_by_split["train"]["pr_auc"]
            - metrics_by_split["validation"]["pr_auc"]
        ),
        "best_epoch": selected["best_epoch"],
        "epochs_executed": selected["epochs_executed"],
        "best_val_loss": selected["best_val_loss"],
        "trainable_parameters": trainable_parameter_count(
            selected["model"]
        ),
        "selected_threshold": threshold,
        "calibration_method": CALIBRATION_METHOD,
    }
    history_dataframe = save_training_history(
        selected["history"],
        paths.history,
    )
    save_learning_curves(history_dataframe, paths.curves)
    save_metrics_report(
        metrics_by_split,
        paths.metrics,
        overfitting_summary,
    )
    save_confusion_matrix(
        y_test,
        test_probabilities,
        threshold,
        paths.confusion_matrix,
    )
    save_model_artifacts(selected["model"], preprocessor, paths)
    log_final_run(
        selected,
        class_weights,
        threshold,
        metrics_by_split,
        paths,
    )

    print("\n=== TABULAR NEURAL NETWORK ===")
    print(f"Selected candidate: {selected['config']['name']}")
    print(f"Input dimension: {X_train_transformed.shape[1]}")
    print(f"Best epoch: {selected['best_epoch']}")
    print(f"Epochs executed: {selected['epochs_executed']}")
    print(f"Selected threshold: {threshold:.2f}")
    print(f"Calibration: {CALIBRATION_METHOD}")
    print("\n=== FINAL METRICS ===")
    print(pd.DataFrame(metrics_by_split).T.to_string())


if __name__ == "__main__":
    main()
