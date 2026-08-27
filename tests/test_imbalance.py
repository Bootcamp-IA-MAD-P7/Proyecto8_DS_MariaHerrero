import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline

from src.models.imbalance_comparison import (
    N_SPLITS,
    build_strategies,
    evaluate_cross_validation,
    evaluate_model,
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


def test_all_imbalance_strategies_exist():
    strategies = build_strategies()

    expected = {
        "baseline",
        "class_weight",
        "random_oversampling",
        "smote",
    }

    assert set(strategies.keys()) == expected


def test_resampling_strategies_use_imblearn_pipeline():
    strategies = build_strategies()

    assert isinstance(
        strategies["random_oversampling"],
        ImbPipeline,
    )

    assert isinstance(
        strategies["smote"],
        ImbPipeline,
    )

    assert isinstance(
        strategies["class_weight"],
        Pipeline,
    )


def test_evaluation_records_recall_and_false_negatives():
    df = create_dataset()

    X = df.drop(columns=["stroke"])
    y = df["stroke"]

    model = build_strategies()["class_weight"]

    result = evaluate_model(
        "class_weight",
        model,
        X,
        y,
        X,
        y,
    )

    assert "recall" in result
    assert "false_negatives" in result


def test_cross_validation_is_reproducible():
    df = create_dataset()

    X = df.drop(columns=["stroke"])
    y = df["stroke"]

    first = evaluate_cross_validation(
        build_strategies(),
        X,
        y,
    )

    second = evaluate_cross_validation(
        build_strategies(),
        X,
        y,
    )

    pd.testing.assert_frame_equal(first, second)


def test_cross_validation_uses_five_folds():
    assert N_SPLITS == 5