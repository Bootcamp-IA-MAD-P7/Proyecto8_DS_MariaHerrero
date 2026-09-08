from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing.pipeline import create_preprocessor
from src.tracking.mlflow_tracking import (
    MODEL_SELECTION_EXPERIMENT,
    configure_tracking,
    log_existing_artifacts,
    log_metrics,
    log_params,
    log_tags,
    select_experiment,
    start_run,
)

from sklearn.calibration import CalibratedClassifierCV


TRAIN_PATH = Path("data/processed/train.csv")
VALIDATION_PATH = Path("data/processed/validation.csv")
REPORT_PATH = Path("reports/classic_models_comparison.csv")

TARGET = "stroke"
RANDOM_SEED = 42


def evaluate_model(name, model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    val_predictions = model.predict(X_val)

    train_probabilities = model.predict_proba(X_train)[:, 1]
    val_probabilities = model.predict_proba(X_val)[:, 1]

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        val_predictions,
        labels=[0, 1],
    ).ravel()

    train_f1 = f1_score(
        y_train,
        train_predictions,
        zero_division=0,
    )

    val_f1 = f1_score(
        y_val,
        val_predictions,
        zero_division=0,
    )

    return {
        "model": name,
        "precision": precision_score(
            y_val,
            val_predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_val,
            val_predictions,
            zero_division=0,
        ),
        "f1": val_f1,
        "roc_auc": roc_auc_score(
            y_val,
            val_probabilities,
        ),
        "pr_auc": average_precision_score(
            y_val,
            val_probabilities,
        ),
        "false_negatives": fn,
        "true_positives": tp,
        "train_f1": train_f1,
        "validation_f1": val_f1,
        "f1_gap": train_f1 - val_f1,
    }


def build_models():
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_SEED,
        ),
        "DecisionTree": DecisionTreeClassifier(
            random_state=RANDOM_SEED,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_SEED,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            random_state=RANDOM_SEED,
        ),
        "SVM": CalibratedClassifierCV(
            SVC(
            random_state=RANDOM_SEED,
            ),
            method="sigmoid",
            cv=5,
),
    }


def main():
    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)

    X_train = train.drop(columns=[TARGET])
    y_train = train[TARGET]

    X_val = validation.drop(columns=[TARGET])
    y_val = validation[TARGET]

    results = []
    models = build_models()

    for name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", create_preprocessor()),
                ("model", estimator),
            ]
        )

        result = evaluate_model(
            name,
            pipeline,
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

    configure_tracking()
    select_experiment(
        MODEL_SELECTION_EXPERIMENT
    )

    run_names = {
        "LogisticRegression": "classic-logistic-regression",
        "DecisionTree": "classic-decision-tree",
        "RandomForest": "classic-random-forest",
        "GradientBoosting": "classic-gradient-boosting",
        "SVM": "classic-calibrated-svm",
    }
    model_params = {
        "LogisticRegression": {
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": RANDOM_SEED,
        },
        "DecisionTree": {
            "criterion": "gini",
            "splitter": "best",
            "random_state": RANDOM_SEED,
        },
        "RandomForest": {
            "n_estimators": 200,
            "random_state": RANDOM_SEED,
        },
        "GradientBoosting": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "random_state": RANDOM_SEED,
        },
        "SVM": {
            "C": 1.0,
            "kernel": "rbf",
            "calibration_method": "sigmoid",
            "calibration_cv": 5,
            "random_state": RANDOM_SEED,
        },
    }

    for result in results:
        algorithm = result["model"]
        params = {
            "algorithm": algorithm,
            "preprocessing": "create_preprocessor",
            **model_params[algorithm],
        }

        with start_run(
            run_name=run_names[algorithm]
        ):
            log_params(params)
            log_metrics(
                {
                    key: float(value)
                    for key, value in result.items()
                    if key != "model"
                }
            )
            log_tags(
                {
                    "project": "stroke-risk-ai",
                    "experiment_type": "classic_models",
                    "stage": "model_selection",
                    "algorithm": algorithm,
                    "target": TARGET,
                    "data_split": "train_validation",
                    "tracking_source": "native_run",
                }
            )
            log_existing_artifacts(
                [REPORT_PATH],
                artifact_path="reports",
            )

    print("\n=== CLASSIC MODELS COMPARISON ===")
    print(
        results_df.sort_values(
            by="pr_auc",
            ascending=False,
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()
