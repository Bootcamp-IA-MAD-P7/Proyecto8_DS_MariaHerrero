from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from src.preprocessing.pipeline import create_preprocessor
from src.tracking.mlflow_tracking import (
    CALIBRATION_THRESHOLD_EXPERIMENT,
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

REPORT_PATH = Path(
    "reports/calibrated_threshold_comparison.csv"
)
SELECTED_THRESHOLD_PATH = Path(
    "reports/calibrated_selected_threshold.csv"
)

TARGET = "stroke"
RANDOM_SEED = 42

MODEL_VERSION = "logreg_v1"
CALIBRATION_METHOD = "sigmoid"

BEST_C = 0.001486
BEST_SOLVER = "liblinear"
BEST_MAX_ITER = 500

MIN_RECALL = 0.80


def build_calibrated_model():
    base_model = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "model",
                LogisticRegression(
                    C=BEST_C,
                    solver=BEST_SOLVER,
                    max_iter=BEST_MAX_ITER,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    return CalibratedClassifierCV(
        estimator=base_model,
        method=CALIBRATION_METHOD,
        cv=5,
    )


def evaluate_thresholds(
    y_true,
    probabilities,
):
    thresholds = np.arange(
        0.05,
        0.51,
        0.01,
    )

    results = []

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predictions,
        ).ravel()

        results.append(
            {
                "threshold": round(
                    float(threshold),
                    2,
                ),
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
                "false_negatives": int(fn),
                "false_positives": int(fp),
                "true_positives": int(tp),
                "true_negatives": int(tn),
            }
        )

    return pd.DataFrame(results)


def select_threshold(results):
    valid = results[
        results["recall"] >= MIN_RECALL
    ].copy()

    if valid.empty:
        return results.sort_values(
            by=[
                "recall",
                "false_negatives",
                "false_positives",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        ).iloc[0]

    return valid.sort_values(
        by=[
            "false_positives",
            "f1",
        ],
        ascending=[
            True,
            False,
        ],
    ).iloc[0]


def main():
    train = pd.read_csv(
        TRAIN_PATH
    )

    validation = pd.read_csv(
        VALIDATION_PATH
    )

    X_train = train.drop(
        columns=[TARGET]
    )
    y_train = train[TARGET]

    X_val = validation.drop(
        columns=[TARGET]
    )
    y_val = validation[TARGET]

    model = build_calibrated_model()

    model.fit(
        X_train,
        y_train,
    )

    probabilities = model.predict_proba(
        X_val
    )[:, 1]

    results = evaluate_thresholds(
        y_val,
        probabilities,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        REPORT_PATH,
        index=False,
    )

    selected = select_threshold(
        results
    )

    selected_result = pd.DataFrame(
        [
            {
                "model_version": MODEL_VERSION,
                "calibration_method": (
                    CALIBRATION_METHOD
                ),
                "threshold": selected[
                    "threshold"
                ],
                "precision": selected[
                    "precision"
                ],
                "recall": selected[
                    "recall"
                ],
                "f1": selected["f1"],
                "false_negatives": selected[
                    "false_negatives"
                ],
                "false_positives": selected[
                    "false_positives"
                ],
            }
        ]
    )

    selected_result.to_csv(
        SELECTED_THRESHOLD_PATH,
        index=False,
    )

    configure_tracking()
    select_experiment(
        CALIBRATION_THRESHOLD_EXPERIMENT
    )

    with start_run(
        run_name=(
            "calibrated-threshold-logistic-regression"
        )
    ):
        log_params(
            {
                "algorithm": "LogisticRegression",
                "model_version": MODEL_VERSION,
                "C": BEST_C,
                "solver": BEST_SOLVER,
                "max_iter": BEST_MAX_ITER,
                "class_weight": "balanced",
                "random_state": RANDOM_SEED,
                "preprocessing": "create_preprocessor",
                "calibration_method": CALIBRATION_METHOD,
                "calibration_cv": 5,
                "threshold_search_start": 0.05,
                "threshold_search_stop": 0.50,
                "threshold_search_step": 0.01,
                "minimum_recall": MIN_RECALL,
                "selected_threshold": float(
                    selected["threshold"]
                ),
                "threshold_selected_on": "validation",
            }
        )
        log_metrics(
            {
                "precision": float(
                    selected["precision"]
                ),
                "recall": float(selected["recall"]),
                "f1": float(selected["f1"]),
                "false_negatives": float(
                    selected["false_negatives"]
                ),
                "false_positives": float(
                    selected["false_positives"]
                ),
            }
        )
        log_tags(
            {
                "project": "stroke-risk-ai",
                "experiment_type": "calibrated_threshold",
                "stage": "calibration_threshold",
                "algorithm": "LogisticRegression",
                "target": TARGET,
                "data_split": "validation",
                "tracking_source": "native_run",
                "status": "selected",
                "deployed": "true",
            }
        )
        log_existing_artifacts(
            [
                REPORT_PATH,
                SELECTED_THRESHOLD_PATH,
            ],
            artifact_path="reports",
        )

    print(
        "\n=== CALIBRATED THRESHOLD ==="
    )

    print(
        selected_result.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
