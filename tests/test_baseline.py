import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src.models.baseline import evaluate_model
from src.preprocessing.pipeline import create_preprocessor


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


def test_baseline_returns_required_metrics():
    df = create_dataset()

    X = df.drop(columns=["stroke"])
    y = df["stroke"]

    model = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]
    )

    result = evaluate_model(
        "DummyClassifier",
        model,
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
    }

    assert required_metrics.issubset(result.keys())


def test_false_negatives_are_recorded():
    df = create_dataset()

    X = df.drop(columns=["stroke"])
    y = df["stroke"]

    model = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]
    )

    result = evaluate_model(
        "DummyClassifier",
        model,
        X,
        y,
        X,
        y,
    )

    assert result["false_negatives"] == 10


def test_confusion_matrix_contains_all_samples():
    df = create_dataset()

    X = df.drop(columns=["stroke"])
    y = df["stroke"]

    model = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]
    )

    result = evaluate_model(
        "DummyClassifier",
        model,
        X,
        y,
        X,
        y,
    )

    total = (
        result["true_negatives"]
        + result["false_positives"]
        + result["false_negatives"]
        + result["true_positives"]
    )

    assert total == len(df)