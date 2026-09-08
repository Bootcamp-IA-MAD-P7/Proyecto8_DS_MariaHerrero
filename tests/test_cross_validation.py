import pandas as pd

from src.models.cross_validation import (
    N_SPLITS,
    build_candidate_models,
    evaluate_cross_validation,
    summarize_results,
)


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


def test_cross_validation_uses_five_folds():
    assert N_SPLITS == 5


def test_cross_validation_is_reproducible():
    df = create_dataset()

    X = df.drop(columns=["stroke"])
    y = df["stroke"]

    estimator = build_candidate_models()["LogisticRegression"]

    first = evaluate_cross_validation(
        "LogisticRegression",
        estimator,
        X,
        y,
    )

    second = evaluate_cross_validation(
        "LogisticRegression",
        estimator,
        X,
        y,
    )

    pd.testing.assert_frame_equal(first, second)


def test_cross_validation_returns_all_folds():
    df = create_dataset()

    X = df.drop(columns=["stroke"])
    y = df["stroke"]

    estimator = build_candidate_models()["LogisticRegression"]

    results = evaluate_cross_validation(
        "LogisticRegression",
        estimator,
        X,
        y,
    )

    assert len(results) == N_SPLITS
    assert set(results["fold"]) == {1, 2, 3, 4, 5}


def test_summary_contains_mean_and_std():
    df = create_dataset()

    X = df.drop(columns=["stroke"])
    y = df["stroke"]

    estimator = build_candidate_models()["LogisticRegression"]

    results = evaluate_cross_validation(
        "LogisticRegression",
        estimator,
        X,
        y,
    )

    summary = summarize_results(results)

    assert "mean" in summary.columns.get_level_values(1)
    assert "std" in summary.columns.get_level_values(1)