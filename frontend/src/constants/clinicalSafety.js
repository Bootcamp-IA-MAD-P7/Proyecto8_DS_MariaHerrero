export const CLINICAL_DISCLAIMER =
  "Esta herramienta es un sistema de apoyo a la criba. No constituye un diagnóstico médico y no sustituye la valoración de un profesional sanitario."

export const HISTORICAL_RESULT_DISCLAIMER =
  "Este resultado histórico procede de una herramienta de apoyo a la criba. No constituye un diagnóstico médico y no sustituye la valoración de un profesional sanitario."

export const BELOW_THRESHOLD_LABEL =
  "Por debajo del umbral"

export const ABOVE_THRESHOLD_LABEL =
  "Por encima del umbral"

export const NO_PREDICTION_LABEL =
  "Sin predicción disponible"


export function safeClassificationLabel(
  prediction,
) {
  if (prediction === 0) {
    return BELOW_THRESHOLD_LABEL
  }

  if (prediction === 1) {
    return ABOVE_THRESHOLD_LABEL
  }

  return NO_PREDICTION_LABEL
}
