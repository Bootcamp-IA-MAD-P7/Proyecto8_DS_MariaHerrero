from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline

from src.preprocessing.pipeline import create_preprocessor


TRAIN_PATH = Path("data/processed/train.csv")
VALIDATION_PATH = Path("data/processed/validation.csv")

REPORT_PATH = Path("reports/calibration_comparison.csv")
CURVE_PATH = Path("reports/calibration_curve.png")

TARGET = "stroke"
RANDOM_SEED = 42

BEST_C = 0.001486
BEST_SOLVER = "liblinear"
BEST_MAX_ITER = 500


def build_base_pipeline():
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


def build_models():
    base = build_base_pipeline()

    sigmoid = CalibratedClassifierCV(
        estimator=build_base_pipeline(),
        method="sigmoid",
        cv=5,
    )

    isotonic = CalibratedClassifierCV(
        estimator=build_base_pipeline(),
        method="isotonic",
        cv=5,
    )

    return {
        "uncalibrated": base,
        "sigmoid": sigmoid,
        "isotonic": isotonic,
    }


def evaluate_calibration(name, model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)[:, 1]

    brier = brier_score_loss(
        y_val,
        probabilities,
    )

    prob_true, prob_pred = calibration_curve(
        y_val,
        probabilities,
        n_bins=10,
        strategy="quantile",
    )

    return {
        "name": name,
        "brier_score": brier,
        "prob_true": prob_true,
        "prob_pred": prob_pred,
    }


def main():
    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)

    X_train = train.drop(columns=[TARGET])
    y_train = train[TARGET]

    X_val = validation.drop(columns=[TARGET])
    y_val = validation[TARGET]

    results = []

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Calibración perfecta",
    )

    for name, model in build_models().items():
        result = evaluate_calibration(
            name,
            model,
            X_train,
            y_train,
            X_val,
            y_val,
        )

        results.append(
            {
                "model": name,
                "brier_score": result["brier_score"],
            }
        )

        ax.plot(
            result["prob_pred"],
            result["prob_true"],
            marker="o",
            label=name,
        )

    ax.set_title("Calibration curve")
    ax.set_xlabel("Probabilidad predicha")
    ax.set_ylabel("Frecuencia real observada")
    ax.legend()

    plt.tight_layout()
    plt.savefig(CURVE_PATH)
    plt.close()

    results_df = pd.DataFrame(results).sort_values(
        by="brier_score"
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        REPORT_PATH,
        index=False,
    )

    print("\n=== CALIBRATION COMPARISON ===")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()