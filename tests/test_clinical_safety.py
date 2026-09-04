import pytest

from src.clinical_safety import (
    ABOVE_THRESHOLD_LABEL,
    BELOW_THRESHOLD_LABEL,
    CLINICAL_DISCLAIMER,
    EXPLAINABILITY_DISCLAIMER,
    EXPLAINABILITY_INTERPRETATION,
    SCORE_LABEL,
    safe_classification_label,
)


def test_clinical_disclaimer_uses_approved_language():
    assert "apoyo a la criba" in CLINICAL_DISCLAIMER
    assert (
        "No constituye un diagnóstico médico"
        in CLINICAL_DISCLAIMER
    )
    assert (
        "no sustituye la valoración de un "
        "profesional sanitario"
        in CLINICAL_DISCLAIMER
    )


@pytest.mark.parametrize(
    ("prediction", "expected"),
    [
        (0, BELOW_THRESHOLD_LABEL),
        (1, ABOVE_THRESHOLD_LABEL),
    ],
)
def test_classification_uses_threshold_language(
    prediction,
    expected,
):
    label = safe_classification_label(
        prediction
    )

    assert label == expected
    assert "umbral" in label.lower()


def test_classification_does_not_claim_future_outcome():
    labels = (
        safe_classification_label(0),
        safe_classification_label(1),
    )
    forbidden_claims = (
        "sufrirá un ictus",
        "no sufrirá un ictus",
        "tendrá un ictus",
        "no tendrá un ictus",
    )

    for label in labels:
        normalized = label.lower()

        assert all(
            claim not in normalized
            for claim in forbidden_claims
        )


def test_invalid_prediction_has_no_safety_label():
    with pytest.raises(
        ValueError,
        match="debe ser 0 o 1",
    ):
        safe_classification_label(2)


def test_explainability_uses_approved_language():
    assert (
        "comportamiento del modelo"
        in EXPLAINABILITY_DISCLAIMER
    )
    assert (
        "No implica causalidad médica"
        in EXPLAINABILITY_DISCLAIMER
    )
    assert (
        "no constituye un diagnóstico médico"
        in EXPLAINABILITY_DISCLAIMER
    )
    assert (
        "score generado por el modelo"
        in EXPLAINABILITY_INTERPRETATION
    )


def test_centralized_messages_do_not_recommend_treatment():
    messages = " ".join(
        (
            CLINICAL_DISCLAIMER,
            EXPLAINABILITY_DISCLAIMER,
            EXPLAINABILITY_INTERPRETATION,
            SCORE_LABEL,
            BELOW_THRESHOLD_LABEL,
            ABOVE_THRESHOLD_LABEL,
        )
    ).lower()
    treatment_language = (
        "medicación",
        "medicamento",
        "tratamiento",
        "dosis",
        "prescribir",
    )

    assert all(
        term not in messages
        for term in treatment_language
    )


def test_score_label_is_neutral():
    assert SCORE_LABEL == (
        "Score generado por el modelo"
    )
