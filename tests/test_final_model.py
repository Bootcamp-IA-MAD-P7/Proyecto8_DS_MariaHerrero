
import numpy as np

from src.models.final_model import (
    BEST_C,
    BEST_MAX_ITER,
    BEST_SOLVER,
    CALIBRATION_METHOD,
    MODEL_VERSION,
    SELECTED_THRESHOLD,
    build_base_model,
    build_final_model,
    calculate_gap,
    calculate_metrics,
)


def test_final_model_configuration():
    model = build_base_model()
    classifier = model.named_steps["model"]

    assert classifier.C == BEST_C
    assert classifier.solver == BEST_SOLVER
    assert classifier.max_iter == BEST_MAX_ITER
    assert classifier.class_weight == "balanced"


def test_final_model_uses_sigmoid_calibration():
    model = build_final_model()

    assert model.method == "sigmoid"
    assert CALIBRATION_METHOD == "sigmoid"


def test_final_threshold_is_versioned():
    assert MODEL_VERSION == "logreg_v1"
    assert SELECTED_THRESHOLD == 0.05


def test_calculate_metrics_returns_expected_values():
    y_true = np.array(
        [0, 0, 1, 1]
    )

    probabilities = np.array(
        [0.10, 0.20, 0.80, 0.90]
    )

    metrics = calculate_metrics(
        y_true,
        probabilities,
        threshold=0.50,
    )

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0

    assert metrics["false_negatives"] == 0
    assert metrics["false_positives"] == 0
    assert metrics["true_positives"] == 2
    assert metrics["true_negatives"] == 2

    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["pr_auc"] <= 1


def test_train_test_gap_is_below_five_points():
    train_metrics = {
        "f1": 0.22415291051259775
    }

    test_metrics = {
        "f1": 0.23423423423423423
    }

    gap = calculate_gap(
        train_metrics,
        test_metrics,
    )

    assert gap < 0.05
    assert np.isclose(
        gap,
        0.01008132372163648,
    )