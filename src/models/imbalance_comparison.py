from pathlib import Path

import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.preprocessing.pipeline import create_preprocessor


TRAIN_PATH = Path("data/processed/train.csv")
VALIDATION_PATH = Path("data/processed/validation.csv")
REPORT_PATH = Path("reports/imbalance_comparison.csv")
CV_REPORT_PATH = Path("reports/imbalance_cross_validation.csv")

TARGET = "stroke"
RANDOM_SEED = 42
N_SPLITS = 5


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
        "strategy": name,
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
        "false_negatives": fn,
        "false_positives": fp,
        "true_positives": tp,
        "true_negatives": tn,
    }


def build_strategies():
    return {
        "baseline": Pipeline(
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
        ),
        "class_weight": Pipeline(
            steps=[
                ("preprocessor", create_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "random_oversampling": ImbPipeline(
            steps=[
                ("preprocessor", create_preprocessor()),
                (
                    "sampler",
                    RandomOverSampler(
                        random_state=RANDOM_SEED,
                    ),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "smote": ImbPipeline(
            steps=[
                ("preprocessor", create_preprocessor()),
                (
                    "sampler",
                    SMOTE(
                        random_state=RANDOM_SEED,
                    ),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
    }


def evaluate_cross_validation(strategies, X, y):
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    results = []

    for strategy_name, model in strategies.items():
        if strategy_name == "baseline":
            continue

        fold_metrics = []

        for fold, (train_index, val_index) in enumerate(
            cv.split(X, y),
            start=1,
        ):
            X_train_fold = X.iloc[train_index]
            X_val_fold = X.iloc[val_index]

            y_train_fold = y.iloc[train_index]
            y_val_fold = y.iloc[val_index]

            fold_model = clone(model)

            fold_model.fit(
                X_train_fold,
                y_train_fold,
            )

            predictions = fold_model.predict(
                X_val_fold
            )

            probabilities = fold_model.predict_proba(
                X_val_fold
            )[:, 1]

            tn, fp, fn, tp = confusion_matrix(
                y_val_fold,
                predictions,
                labels=[0, 1],
            ).ravel()

            fold_metrics.append(
                {
                    "fold": fold,
                    "recall": recall_score(
                        y_val_fold,
                        predictions,
                        zero_division=0,
                    ),
                    "precision": precision_score(
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
                    "false_negatives": fn,
                    "false_positives": fp,
                }
            )

        fold_df = pd.DataFrame(fold_metrics)

        results.append(
            {
                "strategy": strategy_name,
                "recall_mean": fold_df["recall"].mean(),
                "recall_std": fold_df["recall"].std(),
                "precision_mean": fold_df["precision"].mean(),
                "precision_std": fold_df["precision"].std(),
                "f1_mean": fold_df["f1"].mean(),
                "f1_std": fold_df["f1"].std(),
                "roc_auc_mean": fold_df["roc_auc"].mean(),
                "roc_auc_std": fold_df["roc_auc"].std(),
                "pr_auc_mean": fold_df["pr_auc"].mean(),
                "pr_auc_std": fold_df["pr_auc"].std(),
                "false_negatives_mean": fold_df["false_negatives"].mean(),
                "false_positives_mean": fold_df["false_positives"].mean(),
            }
        )

    return pd.DataFrame(results)


def main():
    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)

    X_train = train.drop(columns=[TARGET])
    y_train = train[TARGET]

    X_val = validation.drop(columns=[TARGET])
    y_val = validation[TARGET]

    results = []

    for name, model in build_strategies().items():
        result = evaluate_model(
            name,
            model,
            X_train,
            y_train,
            X_val,
            y_val,
        )
        results.append(result)

    results_df = pd.DataFrame(results)

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        REPORT_PATH,
        index=False,
    )

    print("\n=== IMBALANCE COMPARISON ===")
    print(
        results_df.sort_values(
            by=["recall", "false_negatives"],
            ascending=[False, True],
        ).to_string(index=False)
    )

    cv_results = evaluate_cross_validation(
        build_strategies(),
        X_train,
        y_train,
    )

    cv_results.to_csv(
        CV_REPORT_PATH,
        index=False,
    )

    print("\n=== IMBALANCE CROSS VALIDATION ===")
    print(
        cv_results.sort_values(
            by=["recall_mean", "false_negatives_mean"],
            ascending=[False, True],
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()