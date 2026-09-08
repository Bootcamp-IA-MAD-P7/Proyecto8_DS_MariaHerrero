import pandas as pd

from src.models.compare_classic_models import (
    build_models,
    evaluate_model,
)
from src.preprocessing.pipeline import create_preprocessor
from sklearn.pipeline import Pipeline


def create_dataset():
    return pd.DataFrame(
        {
            "gender": ["Female", "Male"] * 50,
            "age": [20 + (i * 0.5) for i in range(100)],
            "hypertension": [0, 1] * 50,
            "heart_disease": [0, 0, 1, 0] * 25,
            "ever_married": ["Yes", "No"] * 50,
            "work_type": ["Private", "Self-employed"] * 50,
            "Residence_type": ["Urban", "Rural"] * 50,
            "avg_glucose_level": [80 + i for i in range(100)],
            "bmi": [20 + (i * 0.1) for i in range(100)],
            "smoking_status": ["never smoked", "formerly smoked"] * 50,
            "stroke": [0] * 90 + [1] * 10,
        }
    )


def test_all_required_models_exist():
    models = build_models()

    expected_models = {
        "LogisticRegression",
        "DecisionTree",
        "RandomForest",
        "GradientBoosting",
        "SVM",
    }

    assert set(models.keys()) == expected_models


def test_model_evaluation_returns_required_metrics():
    df = create_dataset()

    X = df.drop(columns=["stroke"])
    y = df["stroke"]

    estimator = build_models()["LogisticRegression"]

    pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("model", estimator),
        ]
    )

    result = evaluate_model(
        "LogisticRegression",
        pipeline,
        X,
        y,
        X,
        y,
    )

    required_metrics = {
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "false_negatives",
        "train_f1",
        "validation_f1",
        "f1_gap",
    }

    assert required_metrics.issubset(result.keys())


def test_f1_gap_is_calculated_correctly():
    df = create_dataset()

    X = df.drop(columns=["stroke"])
    y = df["stroke"]

    estimator = build_models()["DecisionTree"]

    pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("model", estimator),
        ]
    )

    result = evaluate_model(
        "DecisionTree",
        pipeline,
        X,
        y,
        X,
        y,
    )

    expected_gap = (
        result["train_f1"]
        - result["validation_f1"]
    )

    assert result["f1_gap"] == expected_gap