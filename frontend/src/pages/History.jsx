import {
  useEffect,
  useState,
} from "react"

import {
  Link,
} from "react-router-dom"

import {
  getAssessmentHistory,
} from "../services/api"


function formatDate(value) {
  const date = new Date(value)

  if (Number.isNaN(date.getTime())) {
    return "Fecha no disponible"
  }

  return new Intl.DateTimeFormat(
    "es-ES",
    {
      dateStyle: "medium",
      timeStyle: "short",
    },
  ).format(date)
}


function formatScore(value) {
  if (value === null) {
    return "No disponible"
  }

  return Number(value).toFixed(4)
}


function classificationLabel(
  prediction,
) {
  if (prediction === null) {
    return "Sin predicción"
  }

  return prediction === 1
    ? "Por encima del umbral"
    : "Por debajo del umbral"
}


function History() {
  const [
    assessments,
    setAssessments,
  ] = useState([])
  const [
    status,
    setStatus,
  ] = useState("loading")
  const [
    error,
    setError,
  ] = useState("")

  useEffect(
    () => {
      let active = true

      async function loadHistory() {
        try {
          const data =
            await getAssessmentHistory()

          if (!active) {
            return
          }

          const ordered = [
            ...data,
          ].sort(
            (first, second) =>
              new Date(
                second.assessment_created_at
              ) -
              new Date(
                first.assessment_created_at
              ),
          )

          setAssessments(ordered)
          setStatus("success")
        } catch (requestError) {
          if (!active) {
            return
          }

          setError(requestError.message)
          setStatus("error")
        }
      }

      loadHistory()

      return () => {
        active = false
      }
    },
    [],
  )

  return (
    <section className="page">
      <div className="page-header">
        <span className="eyebrow">
          Seguimiento
        </span>

        <h1>
          Historial
        </h1>

        <p>
          Consulta las evaluaciones y predicciones
          registradas anteriormente.
        </p>
      </div>

      <div className="content-card">
        {
          status === "loading" &&
          (
            <div className="empty-state">
              <h2>
                Cargando historial
              </h2>

              <p>
                Consultando las evaluaciones
                almacenadas.
              </p>
            </div>
          )
        }

        {
          status === "error" &&
          (
            <div className="empty-state">
              <h2>
                No se pudo cargar el historial
              </h2>

              <p className="history-error">
                {error}
              </p>
            </div>
          )
        }

        {
          status === "success" &&
          assessments.length === 0 &&
          (
            <div className="empty-state">
              <h2>
                Sin evaluaciones registradas
              </h2>

              <p>
                Todavía no hay evaluaciones
                almacenadas en el historial.
              </p>
            </div>
          )
        }

        {
          status === "success" &&
          assessments.length > 0 &&
          (
            <div className="history-list">
              {
                assessments.map(
                  (assessment) => (
                    <article
                      className="history-item"
                      key={
                        assessment.assessment_id
                      }
                    >
                      <div className="history-item-main">
                        <span className="history-date">
                          {
                            formatDate(
                              assessment
                                .assessment_created_at
                            )
                          }
                        </span>

                        <h2>
                          {
                            classificationLabel(
                              assessment.prediction
                            )
                          }
                        </h2>
                      </div>

                      <dl className="history-summary">
                        <div>
                          <dt>Score</dt>
                          <dd>
                            {
                              formatScore(
                                assessment.score
                              )
                            }
                          </dd>
                        </div>

                        <div>
                          <dt>Modelo</dt>
                          <dd>
                            {
                              assessment.model_version ||
                              "No disponible"
                            }
                          </dd>
                        </div>
                      </dl>

                      <Link
                        className="history-detail-link"
                        to={
                          `/history/${assessment.assessment_id}`
                        }
                      >
                        Ver detalle
                      </Link>
                    </article>
                  ),
                )
              }
            </div>
          )
        }
      </div>
    </section>
  )
}

export default History
