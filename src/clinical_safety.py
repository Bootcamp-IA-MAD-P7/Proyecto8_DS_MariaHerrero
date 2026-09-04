CLINICAL_DISCLAIMER = (
    "Esta herramienta es un sistema de apoyo "
    "a la criba. No constituye un diagnóstico "
    "médico y no sustituye la valoración de "
    "un profesional sanitario."
)

EXPLAINABILITY_INTERPRETATION = (
    "Las influencias describen cómo cada "
    "variable modifica el score generado por "
    "el modelo respecto a un valor de referencia."
)

EXPLAINABILITY_DISCLAIMER = (
    "La explicación describe el comportamiento "
    "del modelo. No implica causalidad médica, "
    "no constituye un diagnóstico médico y no "
    "sustituye la valoración de un profesional "
    "sanitario."
)

SCORE_LABEL = "Score generado por el modelo"

BELOW_THRESHOLD_LABEL = "Por debajo del umbral"
ABOVE_THRESHOLD_LABEL = "Por encima del umbral"


def safe_classification_label(prediction):
    if prediction == 0:
        return BELOW_THRESHOLD_LABEL

    if prediction == 1:
        return ABOVE_THRESHOLD_LABEL

    raise ValueError(
        "La clasificación debe ser 0 o 1."
    )
