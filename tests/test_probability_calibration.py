import numpy as np
import pandas as pd

from src.models.probability_calibration import (
    BEST_C,
    BEST_MAX_ITER,
    BEST_SOLVER,
    build_base_pipeline,
    build_models,
    evaluate_calibration,
)


def test_base_model_uses_optimized_hyperparameters():
    model = build_base_pipeline()

    classifier = model.named_steps["model"]

    assert classifier.C == BEST_C
    assert classifier.solver == BEST_SOLVER
    assert classifier.max_iter == BEST_MAX_ITER
    assert classifier.class_weight == "balanced"


def test_calibration_models_are_created():
    models = build_models()

    assert set(models.keys()) == {
        "uncalibrated",
        "sigmoid",
        "isotonic",
    }

    assert models["sigmoid"].method == "sigmoid"
    assert models["isotonic"].method == "isotonic"


def test_evaluate_calibration_returns_expected_results():
    X_train = pd.DataFrame(
        {
            "feature": np.arange(100),
        }
    )
    y_train = pd.Series(
        [0, 1] * 50
    )

    X_val = pd.DataFrame(
        {
            "feature": np.arange(20),
        }
    )
    y_val = pd.Series(
        [0, 1] * 10
    )

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(
        random_state=42,
    )

    result = evaluate_calibration(
        "test_model",
        model,
        X_train,
        y_train,
        X_val,
        y_val,
    )

    assert result["name"] == "test_model"
    assert 0 <= result["brier_score"] <= 1
    assert len(result["prob_true"]) > 0
    assert len(result["prob_pred"]) > 0


def test_brier_score_is_valid_probability_metric():
    X_train = pd.DataFrame(
        {
            "feature": np.arange(100),
        }
    )
    y_train = pd.Series(
        [0, 1] * 50
    )

    X_val = pd.DataFrame(
        {
            "feature": np.arange(20),
        }
    )
    y_val = pd.Series(
        [0, 1] * 10
    )

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(
        random_state=42,
    )

    result = evaluate_calibration(
        "test_model",
        model,
        X_train,
        y_train,
        X_val,
        y_val,
    )

    assert isinstance(result["brier_score"], float)
    assert result["brier_score"] >= 0