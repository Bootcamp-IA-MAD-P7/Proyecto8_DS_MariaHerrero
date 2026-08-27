from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.preprocessing.pipeline import create_preprocessor


TRAIN_PATH = Path("data/processed/train.csv")
REPORT_PATH = Path("reports/cross_validation_results.csv")

TARGET = "stroke"
RANDOM_SEED = 42
N_SPLITS = 5


def build_candidate_models():
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_SEED,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            random_state=RANDOM_SEED,
        ),
    }


def evaluate_cross_validation(name, estimator, X, y):
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    fold_results = []

    for fold, (train_index, val_index) in enumerate(
        cv.split(X, y),
        start=1,
    ):
        X_train_fold = X.iloc[train_index]
        X_val_fold = X.iloc[val_index]

        y_train_fold = y.iloc[train_index]
        y_val_fold = y.iloc[val_index]

        pipeline = Pipeline(
            steps=[
                ("preprocessor", create_preprocessor()),
                ("model", estimator),
            ]
        )

        pipeline.fit(X_train_fold, y_train_fold)

        predictions = pipeline.predict(X_val_fold)
        probabilities = pipeline.predict_proba(X_val_fold)[:, 1]

        fold_results.append(
            {
                "model": name,
                "fold": fold,
                "precision": precision_score(
                    y_val_fold,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_val_fold,
                    predictions,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y_val_fold,
                    predictions,
                    zero_division=0,
                ),
                "roc_auc": roc_auc_score(
                    y_val_fold,
                    probabilities,
                ),
                "pr_auc": average_precision_score(
                    y_val_fold,
                    probabilities,
                ),
            }
        )

    return pd.DataFrame(fold_results)


def summarize_results(results_df):
    metrics = [
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]

    summary = (
        results_df
        .groupby("model")[metrics]
        .agg(["mean", "std"])
    )

    return summary


def main():
    train = pd.read_csv(TRAIN_PATH)

    X = train.drop(columns=[TARGET])
    y = train[TARGET]

    all_results = []

    for name, estimator in build_candidate_models().items():
        model_results = evaluate_cross_validation(
            name,
            estimator,
            X,
            y,
        )

        all_results.append(model_results)

    results_df = pd.concat(
        all_results,
        ignore_index=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        REPORT_PATH,
        index=False,
    )

    summary = summarize_results(results_df)

    print("\n=== CROSS VALIDATION RESULTS ===")
    print(summary.round(4))


if __name__ == "__main__":
    main()