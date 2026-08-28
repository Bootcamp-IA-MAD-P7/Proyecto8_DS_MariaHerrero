from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.models.final_model import (
    MODEL_VERSION,
    SELECTED_THRESHOLD,
    build_final_model,
)


TRAIN_PATH = Path("data/processed/train.csv")
VALIDATION_PATH = Path(
    "data/processed/validation.csv"
)

GLOBAL_REPORT_PATH = Path(
    "reports/global_feature_importance.csv"
)

INDIVIDUAL_REPORT_PATH = Path(
    "reports/individual_explanation_example.json"
)

TARGET = "stroke"
RANDOM_SEED = 42


def calculate_global_importance(
    model,
    X,
    y,
    n_repeats=10,
):
    """
    Calculates global feature importance
    using permutation importance.

    Importance represents how much model
    performance decreases when a feature
    is randomly shuffled.
    """

    result = permutation_importance(
        model,
        X,
        y,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    importance = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": (
                result.importances_mean
            ),
            "importance_std": (
                result.importances_std
            ),
        }
    )

    importance = importance.sort_values(
        by="importance_mean",
        ascending=False,
    ).reset_index(
        drop=True
    )

    return importance


def build_reference_values(X_train):
    """
    Creates reference values for individual
    explanations.

    Numeric variables use the median.
    Categorical variables use the mode.
    """

    reference = {}

    for column in X_train.columns:
        if pd.api.types.is_numeric_dtype(
            X_train[column]
        ):
            reference[column] = (
                X_train[column].median()
            )
        else:
            reference[column] = (
                X_train[column]
                .mode(dropna=True)
                .iloc[0]
            )

    return reference


def explain_individual(
    model,
    instance,
    reference_values,
):
    """
    Explains one prediction by comparing the
    original score with the score obtained
    after replacing each feature individually
    with a reference value.

    Positive influence means that the current
    value increases the model score relative
    to the reference.

    Negative influence means that the current
    value decreases the model score relative
    to the reference.

    This describes model behaviour and must
    not be interpreted as medical causality.
    """

    if isinstance(
        instance,
        pd.Series,
    ):
        instance = (
            instance.to_frame().T
        )

    if len(instance) != 1:
        raise ValueError(
            "Individual explanation requires "
            "exactly one observation."
        )

    original_probability = float(
        model.predict_proba(
            instance
        )[0, 1]
    )

    factors = []

    for feature in instance.columns:
        modified = instance.copy()

        original_value = (
            instance.iloc[0][feature]
        )

        reference_value = (
            reference_values[feature]
        )

        modified.at[
            modified.index[0],
            feature,
        ] = reference_value

        modified_probability = float(
            model.predict_proba(
                modified
            )[0, 1]
        )

        influence = (
            original_probability
            - modified_probability
        )

        factors.append(
            {
                "feature": feature,
                "value": (
                    original_value.item()
                    if isinstance(
                        original_value,
                        np.generic,
                    )
                    else original_value
                ),
                "reference_value": (
                    reference_value.item()
                    if isinstance(
                        reference_value,
                        np.generic,
                    )
                    else reference_value
                ),
                "influence": float(
                    influence
                ),
            }
        )

    increasing = sorted(
        [
            factor
            for factor in factors
            if factor["influence"] > 0
        ],
        key=lambda item: abs(
            item["influence"]
        ),
        reverse=True,
    )

    decreasing = sorted(
        [
            factor
            for factor in factors
            if factor["influence"] < 0
        ],
        key=lambda item: abs(
            item["influence"]
        ),
        reverse=True,
    )

    prediction = int(
        original_probability
        >= SELECTED_THRESHOLD
    )

    return {
        "model_version": MODEL_VERSION,
        "score": original_probability,
        "threshold": SELECTED_THRESHOLD,
        "prediction": prediction,
        "factors_increasing_score": (
            increasing
        ),
        "factors_decreasing_score": (
            decreasing
        ),
        "interpretation": (
            "Las influencias describen cómo "
            "cada variable modifica el score "
            "generado por el modelo respecto "
            "a un valor de referencia."
        ),
        "disclaimer": (
            "La explicación describe el "
            "comportamiento del modelo y no "
            "implica causalidad médica ni "
            "constituye un diagnóstico."
        ),
    }


def explanation_for_api(
    model,
    patient_data,
    reference_values,
):
    """
    Returns an individual explanation in a
    JSON-compatible structure suitable for
    FastAPI.
    """

    if isinstance(
        patient_data,
        dict,
    ):
        patient_data = pd.DataFrame(
            [patient_data]
        )

    explanation = explain_individual(
        model,
        patient_data,
        reference_values,
    )

    return explanation


def main():
    train = pd.read_csv(
        TRAIN_PATH
    )

    validation = pd.read_csv(
        VALIDATION_PATH
    )

    X_train = train.drop(
        columns=[TARGET]
    )

    y_train = train[TARGET]

    X_val = validation.drop(
        columns=[TARGET]
    )

    y_val = validation[TARGET]

    model = build_final_model()

    model.fit(
        X_train,
        y_train,
    )

    global_importance = (
        calculate_global_importance(
            model,
            X_val,
            y_val,
        )
    )

    GLOBAL_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    global_importance.to_csv(
        GLOBAL_REPORT_PATH,
        index=False,
    )

    reference_values = (
        build_reference_values(
            X_train
        )
    )

    example_patient = (
        X_val.iloc[0]
    )

    explanation = explain_individual(
        model,
        example_patient,
        reference_values,
    )

    with open(
        INDIVIDUAL_REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            explanation,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(
        "\n=== GLOBAL FEATURE IMPORTANCE ==="
    )

    print(
        global_importance.head(
            10
        ).to_string(
            index=False
        )
    )

    print(
        "\n=== INDIVIDUAL EXPLANATION ==="
    )

    print(
        json.dumps(
            explanation,
            indent=4,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()