from pathlib import Path

import pandas as pd

from src.models.compare_neural_vs_classic import (
    CLASSIC_METRICS_PATH,
    METRICS,
    NEURAL_METRICS_PATH,
    MODEL_CLASSIC,
    MODEL_NEURAL,
    better_model,
    build_comparison,
    generate_reports,
    load_metrics,
)


def test_loads_existing_test_metrics():
    classic = load_metrics(CLASSIC_METRICS_PATH)
    neural = load_metrics(NEURAL_METRICS_PATH)

    assert classic["recall"] == 0.78
    assert neural["recall"] == 0.84
    assert classic["false_negatives"] == 11
    assert neural["false_negatives"] == 8


def test_calculates_differences_and_better_direction():
    classic = load_metrics(CLASSIC_METRICS_PATH)
    neural = load_metrics(NEURAL_METRICS_PATH)
    comparison = build_comparison(classic, neural).set_index("Metric")

    assert comparison.loc["Recall", "Difference"] == 0.84 - 0.78
    assert comparison.loc["Recall", "Better"] == MODEL_NEURAL
    assert comparison.loc["False negatives", "Difference"] == -3
    assert comparison.loc["False negatives", "Better"] == MODEL_NEURAL
    assert comparison.loc["False positives", "Difference"] == 18
    assert comparison.loc["False positives", "Better"] == MODEL_CLASSIC


def test_metric_directions_cover_counts_and_predictive_metrics():
    directions = {column: higher for _, column, higher in METRICS}

    for metric in ("precision", "recall", "f1", "roc_auc", "pr_auc"):
        assert directions[metric] is True
        assert better_model(0.4, 0.5, directions[metric]) == MODEL_NEURAL

    for metric in ("false_negatives", "false_positives"):
        assert directions[metric] is False
        assert better_model(5, 4, directions[metric]) == MODEL_NEURAL


def test_generate_reports_is_consistent_and_uses_only_reports(tmp_path):
    comparison_path = tmp_path / "comparison.csv"
    analysis_path = tmp_path / "analysis.md"

    generated = generate_reports(
        comparison_path=comparison_path,
        analysis_path=analysis_path,
    )
    stored = pd.read_csv(comparison_path)

    pd.testing.assert_frame_equal(generated, stored, check_dtype=False)
    assert len(stored) == len(METRICS)
    assert "red neuronal tabular es la ganadora experimental" in (
        analysis_path.read_text(encoding="utf-8")
    )


def test_comparison_module_has_no_model_or_dataset_dependency():
    module_path = Path("src/models/compare_neural_vs_classic.py")
    source = module_path.read_text(encoding="utf-8").lower()

    assert "data/" not in source
    assert "artifacts/" not in source
    assert "joblib" not in source
    assert "tensorflow" not in source
    assert "keras" not in source
    assert "predict(" not in source
    assert "predict_proba" not in source
