import {
  useEffect,
  useState,
} from "react"

import {
  Link,
  useParams,
} from "react-router-dom"

import {
  getAssessmentDetail,
} from "../services/api"


const VALUE_LABELS = {
  Female: "Mujer",
  Male: "Hombre",
  Other: "Otro",
  Yes: "Sí",
  No: "No",
  Private: "Privado",
  "Self-employed": "Autónomo/a",
  Govt_job: "Empleo público",
  children: "Menor",
  Never_worked: "Nunca ha trabajado",
  Urban: "Urbana",
  Rural: "Rural",
  "formerly smoked": "Exfumador/a",
  "never smoked": "Nunca ha fumado",
  smokes: "Fumador/a",
  Unknown: "Desconocido",
}


function formatDate(value) {
  const date = new Date(value)

  if (Number.isNaN(date.getTime())) {
    return "Fecha no disponible"
  }

  return new Intl.DateTimeFormat(
    "es-ES",
    {
      dateStyle: "long",
      timeStyle: "short",
    },
  ).format(date)
}


function formatNumber(value) {
  if (value === null) {
    return "No disponible"
  }

  return Number(value).toFixed(4)
}


function labelValue(value) {
  return VALUE_LABELS[value] || value
}


function binaryLabel(value) {
  return value === 1 ? "Sí" : "No"
}


function AssessmentDetail() {
  const { assessmentId } = useParams()
  const [
    assessment,
    setAssessment,
  ] = useState(null)
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

      async function loadAssessment() {
        try {
          const data =
            await getAssessmentDetail(
              assessmentId
            )

          if (!active) {
            return
          }

          setAssessment(data)
          setStatus("success")
        } catch (requestError) {
          if (!active) {
            return
          }

          setError(requestError.message)
          setStatus(
            requestError.status === 404
              ? "not-found"
              : "error"
          )
        }
      }

      loadAssessment()

      return () => {
        active = false
      }
    },
    [assessmentId],
  )

  if (status !== "success") {
    const heading =
      status === "loading"
        ? "Cargando evaluación"
        : status === "not-found"
          ? "Evaluación no encontrada"
          : "No se pudo cargar la evaluación"

    const message =
      status === "loading"
        ? "Consultando los datos almacenados."
        : error

    return (
      <section className="page">
        <div className="page-header">
          <span className="eyebrow">
            Historial
          </span>

          <h1>
            Detalle de evaluación
          </h1>
        </div>

        <div className="content-card">
          <div className="empty-state">
            <h2>{heading}</h2>
            <p>{message}</p>

            <Link
              className="primary-button result-link"
              to="/history"
            >
              Volver al historial
            </Link>
          </div>
        </div>
      </section>
    )
  }

  const classification =
    assessment.prediction === null
      ? "Sin predicción disponible"
      : assessment.prediction === 1
        ? "Por encima del umbral"
        : "Por debajo del umbral"

  const clinicalData = [
    ["Sexo", assessment.gender],
    ["Edad", assessment.age],
    [
      "Hipertensión",
      binaryLabel(assessment.hypertension),
    ],
    [
      "Enfermedad cardíaca",
      binaryLabel(assessment.heart_disease),
    ],
    ["Ha estado casado/a", assessment.ever_married],
    ["Tipo de trabajo", assessment.work_type],
    ["Tipo de residencia", assessment.Residence_type],
    ["Nivel medio de glucosa", assessment.avg_glucose_level],
    ["BMI", assessment.bmi],
    ["Estado respecto al tabaco", assessment.smoking_status],
  ]

  return (
    <section className="page">
      <div className="page-header">
        <span className="eyebrow">
          Historial
        </span>

        <h1>
          Evaluación #{assessment.assessment_id}
        </h1>

        <p>
          Realizada el {
            formatDate(
              assessment.assessment_created_at
            )
          }.
        </p>
      </div>

      <div className="content-card">
        <div className="risk-summary">
          <span className="risk-label">
            Clasificación histórica
          </span>

          <h2>{classification}</h2>

          <p>
            Este resultado corresponde a la
            evaluación almacenada y al modelo
            utilizado en ese momento.
          </p>
        </div>

        <div className="result-metrics">
          <div className="metric-card">
            <span>Score</span>
            <strong>
              {formatNumber(assessment.score)}
            </strong>
          </div>

          <div className="metric-card">
            <span>Umbral</span>
            <strong>
              {formatNumber(assessment.threshold)}
            </strong>
          </div>

          <div className="metric-card">
            <span>Modelo</span>
            <strong>
              {
                assessment.model_version ||
                "No disponible"
              }
            </strong>
          </div>
        </div>

        <section className="detail-section">
          <h2>Datos de la evaluación</h2>

          <dl className="detail-grid">
            <div>
              <dt>Origen</dt>
              <dd>
                {
                  assessment.origin === "professional"
                    ? "Profesional"
                    : "Autodeclarada"
                }
              </dd>
            </div>

            {
              clinicalData.map(
                ([label, value]) => (
                  <div key={label}>
                    <dt>{label}</dt>
                    <dd>{labelValue(value)}</dd>
                  </div>
                ),
              )
            }
          </dl>
        </section>

        <div className="safety-message">
          Este resultado histórico sirve como
          apoyo a la criba preventiva y no
          constituye un diagnóstico médico.
        </div>

        <div className="result-actions">
          <Link
            className="primary-button result-link"
            to="/history"
          >
            Volver al historial
          </Link>
        </div>
      </div>
    </section>
  )
}


export default AssessmentDetail
