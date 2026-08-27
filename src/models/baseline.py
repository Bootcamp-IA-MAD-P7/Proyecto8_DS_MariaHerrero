import matplotlib.pyplot as plt

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.preprocessing.pipeline import create_preprocessor


TRAIN_PATH = Path("data/processed/train.csv")
VALIDATION_PATH = Path("data/processed/validation.csv")
REPORT_PATH = Path("reports/baseline_metrics.csv")

TARGET = "stroke"
RANDOM_SEED = 42


def evaluate_model(name, model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)
    probabilities = model.predict_proba(X_val)[:, 1]

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "model": name,
        "precision": precision_score(
            y_val,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_val,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_val,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_val,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_val,
            probabilities,
        ),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }


def main():
    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)

    X_train = train.drop(columns=[TARGET])
    y_train = train[TARGET]

    X_val = validation.drop(columns=[TARGET])
    y_val = validation[TARGET]

    dummy_model = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "model",
                DummyClassifier(
                    strategy="most_frequent",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    logistic_model = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    results = [
        evaluate_model(
            "DummyClassifier",
            dummy_model,
            X_train,
            y_train,
            X_val,
            y_val,
        ),
        evaluate_model(
            "LogisticRegression",
            logistic_model,
            X_train,
            y_train,
            X_val,
            y_val,
        ),
    ]

    results_df = pd.DataFrame(results)

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        REPORT_PATH,
        index=False,
    )

    for result in results:
        matrix = [
            [
                result["true_negatives"],
                result["false_positives"],
            ],
            [
                result["false_negatives"],
                result["true_positives"],
            ],
        ]

        fig, ax = plt.subplots(figsize=(5, 4))

        image = ax.imshow(matrix)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        ax.set_xticklabels(["No stroke", "Stroke"])
        ax.set_yticklabels(["No stroke", "Stroke"])

        ax.set_xlabel("Predicción")
        ax.set_ylabel("Valor real")
        ax.set_title(
            f"Matriz de confusión — {result['model']}"
        )

        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    matrix[i][j],
                    ha="center",
                    va="center",
                )

        fig.colorbar(image, ax=ax)
        plt.tight_layout()

        output_path = (
            REPORT_PATH.parent
            / f"confusion_matrix_{result['model']}.png"
        )

        plt.savefig(output_path)
        plt.close()

    print("\n=== BASELINE RESULTS ===")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()