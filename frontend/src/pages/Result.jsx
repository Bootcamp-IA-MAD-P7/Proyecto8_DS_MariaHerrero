import {
  Link,
  useLocation,
} from "react-router-dom"

import {
  CLINICAL_DISCLAIMER,
  safeClassificationLabel,
} from "../constants/clinicalSafety"


const FEATURE_LABELS = {
  age: "Edad",
  avg_glucose_level:
    "Nivel medio de glucosa",
  bmi: "BMI",
  hypertension:
    "Hipertensión",
  heart_disease:
    "Enfermedad cardíaca",
  smoking_status:
    "Estado respecto al tabaco",
  gender: "Sexo",
  ever_married:
    "Estado civil",
  work_type:
    "Tipo de trabajo",
  Residence_type:
    "Tipo de residencia",
}


function formatValue(
  value,
) {
  if (
    typeof value ===
    "number"
  ) {
    return Number.isInteger(
      value
    )
      ? value
      : value.toFixed(2)
  }

  return value
}


function FactorList({
  title,
  factors,
  emptyMessage,
}) {
  return (
    <div className="factor-group">
      <h3>
        {title}
      </h3>

      {
        factors.length === 0
          ? (
            <p className="muted-text">
              {emptyMessage}
            </p>
          )
          : (
            <div className="factor-list">
              {
                factors.map(
                  (
                    factor,
                    index,
                  ) => (
                    <div
                      className="factor-item"
                      key={
                        `${factor.feature}-${index}`
                      }
                    >
                      <div>
                        <strong>
                          {
                            FEATURE_LABELS[
                              factor.feature
                            ] ||
                            factor.feature
                          }
                        </strong>

                        <span>
                          Valor: {
                            formatValue(
                              factor.value
                            )
                          }
                        </span>

                        <span>
                          Referencia: {
                            formatValue(
                              factor.reference_value
                            )
                          }
                        </span>
                      </div>

                      <span className="influence-value">
                        {
                          Math.abs(
                            Number(
                              factor.influence
                            ),
                          ).toFixed(4)
                        }
                      </span>
                    </div>
                  ),
                )
              }
            </div>
          )
      }
    </div>
  )
}


function Result() {
  const location =
    useLocation()

  const result =
    location.state?.result

  if (!result) {
    return (
      <section className="page">
        <div className="page-header">
          <span className="eyebrow">
            Predicción
          </span>

          <h1>
            Resultado
          </h1>

          <p>
            No hay una evaluación disponible
            para mostrar.
          </p>
        </div>

        <div className="content-card">
          <div className="result-placeholder">
            <span className="result-icon">
              —
            </span>

            <h2>
              Sin evaluación disponible
            </h2>

            <p>
              Realiza una nueva evaluación para
              consultar el resultado.
            </p>

            <Link
              className="primary-button result-link"
              to="/assessment"
            >
              Nueva evaluación
            </Link>
          </div>

          <div className="safety-message">
            {CLINICAL_DISCLAIMER}
          </div>
        </div>
      </section>
    )
  }

  const explanation =
    result.explanation

  const scorePercentage =
    (
      Number(
        result.score
      ) * 100
    ).toFixed(1)

  const thresholdPercentage =
    (
      Number(
        result.threshold
      ) * 100
    ).toFixed(1)

  return (
    <section className="page">
      <div className="page-header">
        <span className="eyebrow">
          Predicción
        </span>

        <h1>
          Resultado de la evaluación
        </h1>

        <p>
          Estimación generada por el modelo
          como apoyo a la criba preventiva.
        </p>
      </div>

      <div className="content-card">
        <div className="risk-summary">
          <span className="risk-label">
            Resultado respecto al umbral
          </span>

          <h2>
            {
              safeClassificationLabel(
                result.prediction
              )
            }
          </h2>

          <p>
            El score generado por el modelo es
            de <strong>{scorePercentage}%</strong>.
            El umbral utilizado para esta
            clasificación es del{" "}
            <strong>
              {thresholdPercentage}%
            </strong>.
          </p>
        </div>

        <div className="result-metrics">
          <div className="metric-card">
            <span>
              Score
            </span>

            <strong>
              {
                Number(
                  result.score
                ).toFixed(4)
              }
            </strong>
          </div>

          <div className="metric-card">
            <span>
              Umbral
            </span>

            <strong>
              {
                Number(
                  result.threshold
                ).toFixed(4)
              }
            </strong>
          </div>

          <div className="metric-card">
            <span>
              Modelo
            </span>

            <strong>
              {
                result.model_version
              }
            </strong>
          </div>
        </div>

        <div className="explanation-section">
          <div className="section-heading">
            <span className="eyebrow">
              Explicabilidad
            </span>

            <h2>
              Factores influyentes
            </h2>

            <p>
              {
                explanation.interpretation
              }
            </p>
          </div>

          <div className="factor-columns">
            <FactorList
              title="Factores que aumentan el score"
              factors={
                explanation
                  .factors_increasing_score
              }
              emptyMessage={
                "No se identificaron factores principales que aumenten el score."
              }
            />

            <FactorList
              title="Factores que disminuyen el score"
              factors={
                explanation
                  .factors_decreasing_score
              }
              emptyMessage={
                "No se identificaron factores principales que disminuyan el score."
              }
            />
          </div>
        </div>

        <div className="safety-message">
          <strong>
            Importante:
          </strong>{" "}
          {
            explanation.disclaimer
          }
        </div>

        <div className="result-actions">
          <Link
            className="primary-button result-link"
            to="/assessment"
          >
            Nueva evaluación
          </Link>
        </div>
      </div>
    </section>
  )
}


export default Result
