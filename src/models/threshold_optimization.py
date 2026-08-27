from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from src.preprocessing.pipeline import create_preprocessor


TRAIN_PATH = Path("data/processed/train.csv")
VALIDATION_PATH = Path("data/processed/validation.csv")

REPORT_PATH = Path("reports/threshold_comparison.csv")
CURVE_PATH = Path("reports/precision_recall_curve.png")
THRESHOLD_PATH = Path("reports/selected_threshold.csv")

TARGET = "stroke"

RANDOM_SEED = 42

MODEL_VERSION = "logreg_v1"

BEST_C = 0.001486
BEST_SOLVER = "liblinear"
BEST_MAX_ITER = 500


def build_model():
    return Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
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


def evaluate_thresholds(y_true, probabilities):
    thresholds = np.arange(
        0.10,
        0.91,
        0.05,
    )

    results = []

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).ravel()

        results.append(
            {
                "threshold": round(float(threshold), 2),
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
                "false_negatives": fn,
                "false_positives": fp,
                "true_positives": tp,
                "true_negatives": tn,
            }
        )

    return pd.DataFrame(results)


def select_threshold(results_df):
    acceptable = results_df[
        results_df["recall"] >= 0.80
    ].copy()

    if acceptable.empty:
        return results_df.sort_values(
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

    return acceptable.sort_values(
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
    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)

    X_train = train.drop(columns=[TARGET])
    y_train = train[TARGET]

    X_val = validation.drop(columns=[TARGET])
    y_val = validation[TARGET]

    model = build_model()

    model.fit(
        X_train,
        y_train,
    )

    probabilities = model.predict_proba(
        X_val
    )[:, 1]

    results_df = evaluate_thresholds(
        y_val,
        probabilities,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        REPORT_PATH,
        index=False,
    )

    precision, recall, thresholds = (
        precision_recall_curve(
            y_val,
            probabilities,
        )
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(
        thresholds,
        precision[:-1],
        label="Precision",
    )

    ax.plot(
        thresholds,
        recall[:-1],
        label="Recall",
    )

    ax.set_title(
        "Precision-Recall según threshold"
    )

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.legend()

    plt.tight_layout()
    plt.savefig(CURVE_PATH)
    plt.close()

    selected = select_threshold(
        results_df
    )

    selected_result = pd.DataFrame(
        [
            {
                "model_version": MODEL_VERSION,
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
        THRESHOLD_PATH,
        index=False,
    )

    print(
        "\n=== THRESHOLD COMPARISON ==="
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        "\n=== SELECTED THRESHOLD ==="
    )

    print(
        selected_result.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()