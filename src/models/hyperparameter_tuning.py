from pathlib import Path

import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.preprocessing.pipeline import create_preprocessor


TRAIN_PATH = Path("data/processed/train.csv")
REPORT_PATH = Path("reports/optuna_trials.csv")
BEST_PARAMS_PATH = Path("reports/best_hyperparameters.csv")

TARGET = "stroke"
RANDOM_SEED = 42
N_SPLITS = 5
N_TRIALS = 30


def objective(trial, X, y):
    c_value = trial.suggest_float(
        "C",
        1e-3,
        100,
        log=True,
    )

    solver = trial.suggest_categorical(
        "solver",
        ["liblinear", "lbfgs"],
    )

    max_iter = trial.suggest_categorical(
        "max_iter",
        [500, 1000, 2000],
    )

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    recall_scores = []
    roc_auc_scores = []
    pr_auc_scores = []

    for train_index, val_index in cv.split(X, y):
        X_train_fold = X.iloc[train_index]
        X_val_fold = X.iloc[val_index]

        y_train_fold = y.iloc[train_index]
        y_val_fold = y.iloc[val_index]

        model = Pipeline(
            steps=[
                ("preprocessor", create_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        C=c_value,
                        solver=solver,
                        max_iter=max_iter,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )

        model.fit(
            X_train_fold,
            y_train_fold,
        )

        predictions = model.predict(X_val_fold)
        probabilities = model.predict_proba(X_val_fold)[:, 1]

        recall_scores.append(
            recall_score(
                y_val_fold,
                predictions,
                zero_division=0,
            )
        )

        roc_auc_scores.append(
            roc_auc_score(
                y_val_fold,
                probabilities,
            )
        )

        pr_auc_scores.append(
            average_precision_score(
                y_val_fold,
                probabilities,
            )
        )

    trial.set_user_attr(
        "roc_auc_mean",
        sum(roc_auc_scores) / len(roc_auc_scores),
    )

    trial.set_user_attr(
        "pr_auc_mean",
        sum(pr_auc_scores) / len(pr_auc_scores),
    )

    return sum(recall_scores) / len(recall_scores)


def main():
    train = pd.read_csv(TRAIN_PATH)

    X = train.drop(columns=[TARGET])
    y = train[TARGET]

    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_SEED,
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="stroke_logistic_regression",
    )

    study.optimize(
        lambda trial: objective(
            trial,
            X,
            y,
        ),
        n_trials=N_TRIALS,
    )

    trials_df = study.trials_dataframe(
        attrs=(
            "number",
            "value",
            "params",
            "user_attrs",
            "state",
        )
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    trials_df.to_csv(
        REPORT_PATH,
        index=False,
    )

    best_result = pd.DataFrame(
        [
            {
                "best_recall_cv": study.best_value,
                **study.best_params,
                "roc_auc_mean": study.best_trial.user_attrs[
                    "roc_auc_mean"
                ],
                "pr_auc_mean": study.best_trial.user_attrs[
                    "pr_auc_mean"
                ],
            }
        ]
    )

    best_result.to_csv(
        BEST_PARAMS_PATH,
        index=False,
    )

    print("\n=== OPTUNA BEST RESULT ===")
    print(best_result.to_string(index=False))


if __name__ == "__main__":
    main()