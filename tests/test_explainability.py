import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.clinical_safety import (
    EXPLAINABILITY_DISCLAIMER,
    EXPLAINABILITY_INTERPRETATION,
)
from src.models.explainability import (
    MODEL_VERSION,
    build_reference_values,
    calculate_global_importance,
    explain_individual,
    explanation_for_api,
)


def create_test_data():
    X = pd.DataFrame(
        {
            "age": [
                20, 25, 30, 35, 40,
                45, 50, 55, 60, 65,
                70, 75, 80, 85, 90,
                22, 32, 42, 52, 62,
            ],
            "glucose": [
                70, 75, 80, 85, 90,
                95, 100, 105, 110, 115,
                120, 125, 130, 135, 140,
                72, 82, 92, 102, 112,
            ],
        }
    )

    y = pd.Series(
        [
            0, 0, 0, 0, 0,
            0, 0, 0, 1, 1,
            1, 1, 1, 1, 1,
            0, 0, 0, 1, 1,
        ]
    )

    return X, y


class DeterministicInfluenceModel:
    def predict_proba(self, dataframe):
        probability = (
            0.5
            + 0.10 * dataframe["strong_increase"].to_numpy()
            + 0.02 * dataframe["weak_increase"].to_numpy()
            - 0.05 * dataframe["decrease"].to_numpy()
        )

        return np.column_stack(
            (1 - probability, probability)
        )


def test_reference_values_use_median_and_mode():
    data = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "gender": [
                "Female",
                "Female",
                "Male",
            ],
        }
    )

    reference = build_reference_values(
        data
    )

    assert reference["age"] == 30
    assert reference["gender"] == "Female"


def test_global_importance_returns_all_features():
    X, y = create_test_data()

    model = LogisticRegression(
        random_state=42
    )

    model.fit(
        X,
        y,
    )

    importance = calculate_global_importance(
        model,
        X,
        y,
        n_repeats=2,
    )

    assert len(importance) == len(
        X.columns
    )

    assert set(
        importance["feature"]
    ) == set(
        X.columns
    )

    assert "importance_mean" in (
        importance.columns
    )

    assert "importance_std" in (
        importance.columns
    )


def test_individual_explanation_contains_required_fields():
    X, y = create_test_data()

    model = LogisticRegression(
        random_state=42
    )

    model.fit(
        X,
        y,
    )

    reference = (
        build_reference_values(X)
    )

    explanation = explain_individual(
        model,
        X.iloc[0],
        reference,
    )

    assert explanation[
        "model_version"
    ] == MODEL_VERSION

    assert "score" in explanation
    assert "threshold" in explanation
    assert "prediction" in explanation

    assert (
        "factors_increasing_score"
        in explanation
    )

    assert (
        "factors_decreasing_score"
        in explanation
    )

    assert "interpretation" in explanation
    assert "disclaimer" in explanation


def test_individual_explanation_rejects_multiple_rows():
    instance = pd.DataFrame(
        {
            "feature": [1.0, 2.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="exactly one observation",
    ):
        explain_individual(
            DeterministicInfluenceModel(),
            instance,
            {"feature": 0.0},
        )


def test_individual_factors_are_split_and_ordered_by_influence():
    instance = pd.DataFrame(
        {
            "strong_increase": [2.0],
            "weak_increase": [1.0],
            "decrease": [2.0],
        }
    )
    reference = {
        "strong_increase": 0.0,
        "weak_increase": 0.0,
        "decrease": 0.0,
    }

    explanation = explain_individual(
        DeterministicInfluenceModel(),
        instance,
        reference,
    )

    increasing = explanation["factors_increasing_score"]
    decreasing = explanation["factors_decreasing_score"]

    assert [factor["feature"] for factor in increasing] == [
        "strong_increase",
        "weak_increase",
    ]
    assert [factor["feature"] for factor in decreasing] == [
        "decrease"
    ]
    assert all(factor["influence"] > 0 for factor in increasing)
    assert all(factor["influence"] < 0 for factor in decreasing)
    assert abs(increasing[0]["influence"]) > abs(
        increasing[1]["influence"]
    )


def test_explanation_uses_centralized_safety_language():
    X, y = create_test_data()

    model = LogisticRegression(
        random_state=42
    )

    model.fit(
        X,
        y,
    )

    reference = (
        build_reference_values(X)
    )

    explanation = explain_individual(
        model,
        X.iloc[0],
        reference,
    )

    assert explanation["interpretation"] == (
        EXPLAINABILITY_INTERPRETATION
    )
    assert explanation["disclaimer"] == (
        EXPLAINABILITY_DISCLAIMER
    )
    assert (
        "No implica causalidad médica"
        in explanation["disclaimer"]
    )
    assert (
        "no constituye un diagnóstico médico"
        in explanation["disclaimer"]
    )


def test_api_explanation_is_json_serializable():
    X, y = create_test_data()

    model = LogisticRegression(
        random_state=42
    )

    model.fit(
        X,
        y,
    )

    reference = (
        build_reference_values(X)
    )

    patient_data = (
        X.iloc[0].to_dict()
    )

    explanation = explanation_for_api(
        model,
        patient_data,
        reference,
    )

    serialized = json.dumps(
        explanation,
        ensure_ascii=False,
    )

    assert isinstance(
        explanation,
        dict,
    )

    assert isinstance(
        serialized,
        str,
    )
