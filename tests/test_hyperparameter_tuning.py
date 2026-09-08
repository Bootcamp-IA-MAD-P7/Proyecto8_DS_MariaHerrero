import pandas as pd
import optuna

from src.models.hyperparameter_tuning import (
    N_SPLITS,
    N_TRIALS,
    RANDOM_SEED,
    objective,
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


def create_fixed_trial():
    return optuna.trial.FixedTrial(
        {
            "C": 0.01,
            "solver": "liblinear",
            "max_iter": 500,
        }
    )


def test_tuning_configuration_is_reproducible():
    assert RANDOM_SEED == 42
    assert N_SPLITS == 5
    assert N_TRIALS == 30


def test_objective_returns_recall_score():
    df = create_dataset()

    X = df.drop(columns=["stroke"])
    y = df["stroke"]

    trial = create_fixed_trial()

    score = objective(
        trial,
        X,
        y,
    )

    assert 0 <= score <= 1


def test_objective_records_secondary_metrics():
    df = create_dataset()

    X = df.drop(columns=["stroke"])
    y = df["stroke"]

    trial = create_fixed_trial()

    objective(
        trial,
        X,
        y,
    )

    assert "roc_auc_mean" in trial.user_attrs
    assert "pr_auc_mean" in trial.user_attrs