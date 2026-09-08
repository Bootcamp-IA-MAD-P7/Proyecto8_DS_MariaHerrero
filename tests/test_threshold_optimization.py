import numpy as np
import pandas as pd

from src.models.threshold_optimization import (
    MODEL_VERSION,
    evaluate_thresholds,
    select_threshold,
)


def test_thresholds_are_evaluated():
    y_true = pd.Series([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])

    results = evaluate_thresholds(
        y_true,
        probabilities,
    )

    assert not results.empty
    assert "threshold" in results.columns
    assert "precision" in results.columns
    assert "recall" in results.columns
    assert "f1" in results.columns
    assert "false_negatives" in results.columns
    assert "false_positives" in results.columns


def test_thresholds_include_expected_range():
    y_true = pd.Series([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])

    results = evaluate_thresholds(
        y_true,
        probabilities,
    )

    assert results["threshold"].min() == 0.10
    assert results["threshold"].max() == 0.90


def test_selected_threshold_respects_minimum_recall():
    results = pd.DataFrame(
        {
            "threshold": [0.3, 0.4, 0.5, 0.6],
            "precision": [0.10, 0.15, 0.20, 0.30],
            "recall": [0.95, 0.90, 0.825, 0.70],
            "f1": [0.18, 0.26, 0.32, 0.42],
            "false_negatives": [2, 4, 7, 12],
            "false_positives": [300, 250, 225, 150],
        }
    )

    selected = select_threshold(results)

    assert selected["recall"] >= 0.80
    assert selected["threshold"] == 0.5


def test_model_version_is_defined():
    assert MODEL_VERSION == "logreg_v1"