from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.preprocessing.pipeline import create_preprocessor


TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")

REPORT_PATH = Path("reports/final_model_metrics.csv")
RANKING_PATH = Path("reports/final_model_ranking.csv")
CONFUSION_MATRIX_PATH = Path(
    "reports/final_confusion_matrix.png"
)

ARTIFACTS_DIR = Path("artifacts/final_model")

MODEL_PATH = (
    ARTIFACTS_DIR
    / "stroke_model_logreg_v1.joblib"
)

PREPROCESSOR_PATH = (
    ARTIFACTS_DIR
    / "preprocessor_logreg_v1.joblib"
)

THRESHOLD_PATH = (
    ARTIFACTS_DIR
    / "threshold_logreg_v1.json"
)

TARGET = "stroke"
RANDOM_SEED = 42

MODEL_VERSION = "logreg_v1"
CALIBRATION_METHOD = "sigmoid"

# Threshold seleccionado en validation
# DESPUÉS de aplicar calibración sigmoid.
SELECTED_THRESHOLD = 0.05

BEST_C = 0.001486
BEST_SOLVER = "liblinear"
BEST_MAX_ITER = 500


def build_base_model():
    return Pipeline(
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


def build_final_model():
    return CalibratedClassifierCV(
        estimator=build_base_model(),
        method=CALIBRATION_METHOD,
        cv=5,
    )


def calculate_metrics(
    y_true,
    probabilities,
    threshold,
):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
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
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "true_positives": int(tp),
        "true_negatives": int(tn),
    }


def calculate_gap(
    train_metrics,
    test_metrics,
):
    return abs(
        train_metrics["f1"]
        - test_metrics["f1"]
    )


def save_artifacts(
    model,
    X_train,
):
    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # El modelo calibrado completo contiene
    # internamente el preprocessing utilizado
    # durante el entrenamiento.
    joblib.dump(
        model,
        MODEL_PATH,
    )

    # Se guarda además una versión fitted
    # del preprocessing como artefacto
    # independiente y versionado.
    fitted_preprocessor = (
        create_preprocessor()
    )

    fitted_preprocessor.fit(
        X_train
    )

    joblib.dump(
        fitted_preprocessor,
        PREPROCESSOR_PATH,
    )

    threshold_metadata = {
        "model_version": MODEL_VERSION,
        "threshold": (
            SELECTED_THRESHOLD
        ),
        "calibration_method": (
            CALIBRATION_METHOD
        ),
        "threshold_selected_on": (
            "validation"
        ),
    }

    with open(
        THRESHOLD_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            threshold_metadata,
            file,
            indent=4,
        )


def save_confusion_matrix(
    y_true,
    predictions,
):
    cm = confusion_matrix(
        y_true,
        predictions,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "No stroke",
            "Stroke",
        ],
    )

    display.plot(
        values_format="d",
    )

    plt.title(
        "Final model confusion matrix"
    )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_PATH,
    )

    plt.close()


def create_ranking():
    ranking = pd.DataFrame(
        [
            {
                "rank": 1,
                "model": (
                    "Logistic Regression "
                    "calibrated"
                ),
                "selected": True,
                "reason": (
                    "Selected for recall, "
                    "stability, calibration "
                    "and interpretability"
                ),
            },
            {
                "rank": 2,
                "model": (
                    "Gradient Boosting"
                ),
                "selected": False,
                "reason": (
                    "Lower recall and less "
                    "stable in cross-validation"
                ),
            },
            {
                "rank": 3,
                "model": (
                    "Random Forest"
                ),
                "selected": False,
                "reason": (
                    "Strong overfitting "
                    "during validation"
                ),
            },
            {
                "rank": 4,
                "model": (
                    "Decision Tree"
                ),
                "selected": False,
                "reason": (
                    "Strong overfitting and "
                    "worse generalization"
                ),
            },
            {
                "rank": 5,
                "model": "SVM",
                "selected": False,
                "reason": (
                    "Lower recall and overall "
                    "performance"
                ),
            },
        ]
    )

    ranking.to_csv(
        RANKING_PATH,
        index=False,
    )


def main():
    train = pd.read_csv(
        TRAIN_PATH
    )

    test = pd.read_csv(
        TEST_PATH
    )

    X_train = train.drop(
        columns=[TARGET]
    )
    y_train = train[TARGET]

    X_test = test.drop(
        columns=[TARGET]
    )
    y_test = test[TARGET]

    model = build_final_model()

    model.fit(
        X_train,
        y_train,
    )

    train_probabilities = (
        model.predict_proba(
            X_train
        )[:, 1]
    )

    test_probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    train_metrics = calculate_metrics(
        y_train,
        train_probabilities,
        SELECTED_THRESHOLD,
    )

    test_metrics = calculate_metrics(
        y_test,
        test_probabilities,
        SELECTED_THRESHOLD,
    )

    gap = calculate_gap(
        train_metrics,
        test_metrics,
    )

    results = pd.DataFrame(
        [
            {
                "dataset": "train",
                **train_metrics,
            },
            {
                "dataset": "test",
                **test_metrics,
            },
        ]
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        REPORT_PATH,
        index=False,
    )

    create_ranking()

    test_predictions = (
        test_probabilities
        >= SELECTED_THRESHOLD
    ).astype(int)

    save_confusion_matrix(
        y_test,
        test_predictions,
    )

    save_artifacts(
        model,
        X_train,
    )

    print(
        "\n=== FINAL MODEL ==="
    )

    print(
        f"Model version: "
        f"{MODEL_VERSION}"
    )

    print(
        f"Calibration: "
        f"{CALIBRATION_METHOD}"
    )

    print(
        f"Threshold: "
        f"{SELECTED_THRESHOLD}"
    )

    print(
        "\n=== TRAIN METRICS ==="
    )

    for key, value in (
        train_metrics.items()
    ):
        print(
            f"{key}: {value}"
        )

    print(
        "\n=== TEST METRICS ==="
    )

    for key, value in (
        test_metrics.items()
    ):
        print(
            f"{key}: {value}"
        )

    print(
        "\n=== TRAIN / TEST GAP ==="
    )

    print(
        f"F1 gap: {gap:.4f}"
    )

    if gap <= 0.05:
        print(
            "Gap requirement satisfied."
        )
    else:
        print(
            "Gap exceeds 5 percentage "
            "points. Deviation must be "
            "documented."
        )


if __name__ == "__main__":
    main()