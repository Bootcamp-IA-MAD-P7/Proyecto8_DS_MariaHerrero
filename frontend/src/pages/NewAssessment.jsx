import {
  useState,
} from "react"

import {
  useNavigate,
} from "react-router-dom"

import {
  createPrediction,
} from "../services/api"


const INITIAL_FORM = {
  origin: "professional",
  gender: "",
  age: "",
  hypertension: "",
  heart_disease: "",
  ever_married: "",
  work_type: "",
  Residence_type: "",
  avg_glucose_level: "",
  bmi: "",
  smoking_status: "",
}


function validateForm(
  form,
) {
  const errors = {}

  if (!form.gender) {
    errors.gender =
      "Selecciona el sexo."
  }

  if (
    form.age === "" ||
    Number(form.age) < 0 ||
    Number(form.age) > 120
  ) {
    errors.age =
      "Introduce una edad válida entre 0 y 120 años."
  }

  if (
    form.hypertension === ""
  ) {
    errors.hypertension =
      "Indica si existe hipertensión."
  }

  if (
    form.heart_disease === ""
  ) {
    errors.heart_disease =
      "Indica si existe enfermedad cardíaca."
  }

  if (
    !form.ever_married
  ) {
    errors.ever_married =
      "Selecciona una opción."
  }

  if (
    !form.work_type
  ) {
    errors.work_type =
      "Selecciona el tipo de trabajo."
  }

  if (
    !form.Residence_type
  ) {
    errors.Residence_type =
      "Selecciona el tipo de residencia."
  }

  if (
    form.avg_glucose_level === "" ||
    Number(
      form.avg_glucose_level
    ) <= 0
  ) {
    errors.avg_glucose_level =
      "Introduce un nivel de glucosa válido."
  }

  if (
    form.bmi === "" ||
    Number(form.bmi) <= 0
  ) {
    errors.bmi =
      "Introduce un BMI válido."
  }

  if (
    !form.smoking_status
  ) {
    errors.smoking_status =
      "Selecciona el estado respecto al tabaco."
  }

  return errors
}


function NewAssessment() {
  const navigate =
    useNavigate()

  const [
    form,
    setForm,
  ] = useState(
    INITIAL_FORM
  )

  const [
    errors,
    setErrors,
  ] = useState({})

  const [
    status,
    setStatus,
  ] = useState("idle")

  const [
    apiError,
    setApiError,
  ] = useState("")


  function handleChange(
    event,
  ) {
    const {
      name,
      value,
    } = event.target

    setForm(
      (current) => ({
        ...current,
        [name]: value,
      }),
    )

    setErrors(
      (current) => ({
        ...current,
        [name]: undefined,
      }),
    )

    setApiError("")
  }


  async function handleSubmit(
    event,
  ) {
    event.preventDefault()

    const validationErrors =
      validateForm(form)

    if (
      Object.keys(
        validationErrors
      ).length > 0
    ) {
      setErrors(
        validationErrors
      )

      setStatus(
        "error"
      )

      return
    }

    const payload = {
      ...form,
      age: Number(
        form.age
      ),
      hypertension: Number(
        form.hypertension
      ),
      heart_disease: Number(
        form.heart_disease
      ),
      avg_glucose_level: Number(
        form.avg_glucose_level
      ),
      bmi: Number(
        form.bmi
      ),
    }

    try {
      setStatus(
        "loading"
      )

      setApiError("")

      const response =
        await createPrediction(
          payload
        )

      setStatus(
        "success"
      )

      navigate(
        "/result",
        {
          state: {
            result: response,
          },
        },
      )
    } catch (error) {
      setApiError(
        error.message
      )

      setStatus(
        "error"
      )
    }
  }


  return (
    <section className="page">
      <div className="page-header">
        <span className="eyebrow">
          Evaluación
        </span>

        <h1>
          Nueva evaluación
        </h1>

        <p>
          Introduce los datos clínicos necesarios
          para solicitar una estimación preventiva
          de riesgo.
        </p>
      </div>

      <div className="content-card">
        <h2>
          Datos del paciente
        </h2>

        <form
          className="assessment-form"
          onSubmit={
            handleSubmit
          }
          noValidate
        >
          <div className="form-grid">
            <div className="form-field">
              <label
                htmlFor="gender"
              >
                Sexo
              </label>

              <select
                id="gender"
                name="gender"
                value={
                  form.gender
                }
                onChange={
                  handleChange
                }
              >
                <option value="">
                  Selecciona una opción
                </option>

                <option value="Female">
                  Mujer
                </option>

                <option value="Male">
                  Hombre
                </option>

                <option value="Other">
                  Otro
                </option>
              </select>

              {
                errors.gender &&
                (
                  <span className="field-error">
                    {
                      errors.gender
                    }
                  </span>
                )
              }
            </div>

            <div className="form-field">
              <label
                htmlFor="age"
              >
                Edad
              </label>

              <input
                id="age"
                name="age"
                type="number"
                min="0"
                max="120"
                step="1"
                value={
                  form.age
                }
                onChange={
                  handleChange
                }
                placeholder="Ej. 47"
              />

              {
                errors.age &&
                (
                  <span className="field-error">
                    {
                      errors.age
                    }
                  </span>
                )
              }
            </div>

            <div className="form-field">
              <label
                htmlFor="hypertension"
              >
                Hipertensión
              </label>

              <select
                id="hypertension"
                name="hypertension"
                value={
                  form.hypertension
                }
                onChange={
                  handleChange
                }
              >
                <option value="">
                  Selecciona una opción
                </option>

                <option value="0">
                  No
                </option>

                <option value="1">
                  Sí
                </option>
              </select>

              {
                errors.hypertension &&
                (
                  <span className="field-error">
                    {
                      errors.hypertension
                    }
                  </span>
                )
              }
            </div>

            <div className="form-field">
              <label
                htmlFor="heart_disease"
              >
                Enfermedad cardíaca
              </label>

              <select
                id="heart_disease"
                name="heart_disease"
                value={
                  form.heart_disease
                }
                onChange={
                  handleChange
                }
              >
                <option value="">
                  Selecciona una opción
                </option>

                <option value="0">
                  No
                </option>

                <option value="1">
                  Sí
                </option>
              </select>

              {
                errors.heart_disease &&
                (
                  <span className="field-error">
                    {
                      errors.heart_disease
                    }
                  </span>
                )
              }
            </div>

            <div className="form-field">
              <label
                htmlFor="ever_married"
              >
                ¿Ha estado casado/a?
              </label>

              <select
                id="ever_married"
                name="ever_married"
                value={
                  form.ever_married
                }
                onChange={
                  handleChange
                }
              >
                <option value="">
                  Selecciona una opción
                </option>

                <option value="Yes">
                  Sí
                </option>

                <option value="No">
                  No
                </option>
              </select>

              {
                errors.ever_married &&
                (
                  <span className="field-error">
                    {
                      errors.ever_married
                    }
                  </span>
                )
              }
            </div>

            <div className="form-field">
              <label
                htmlFor="work_type"
              >
                Tipo de trabajo
              </label>

              <select
                id="work_type"
                name="work_type"
                value={
                  form.work_type
                }
                onChange={
                  handleChange
                }
              >
                <option value="">
                  Selecciona una opción
                </option>

                <option value="Private">
                  Privado
                </option>

                <option value="Self-employed">
                  Autónomo/a
                </option>

                <option value="Govt_job">
                  Empleo público
                </option>

                <option value="children">
                  Menor
                </option>

                <option value="Never_worked">
                  Nunca ha trabajado
                </option>
              </select>

              {
                errors.work_type &&
                (
                  <span className="field-error">
                    {
                      errors.work_type
                    }
                  </span>
                )
              }
            </div>

            <div className="form-field">
              <label
                htmlFor="Residence_type"
              >
                Tipo de residencia
              </label>

              <select
                id="Residence_type"
                name="Residence_type"
                value={
                  form.Residence_type
                }
                onChange={
                  handleChange
                }
              >
                <option value="">
                  Selecciona una opción
                </option>

                <option value="Urban">
                  Urbana
                </option>

                <option value="Rural">
                  Rural
                </option>
              </select>

              {
                errors.Residence_type &&
                (
                  <span className="field-error">
                    {
                      errors.Residence_type
                    }
                  </span>
                )
              }
            </div>

            <div className="form-field">
              <label
                htmlFor="avg_glucose_level"
              >
                Nivel medio de glucosa
              </label>

              <input
                id="avg_glucose_level"
                name="avg_glucose_level"
                type="number"
                min="0"
                step="0.01"
                value={
                  form.avg_glucose_level
                }
                onChange={
                  handleChange
                }
                placeholder="Ej. 99"
              />

              {
                errors.avg_glucose_level &&
                (
                  <span className="field-error">
                    {
                      errors.avg_glucose_level
                    }
                  </span>
                )
              }
            </div>

            <div className="form-field">
              <label
                htmlFor="bmi"
              >
                BMI
              </label>

              <input
                id="bmi"
                name="bmi"
                type="number"
                min="0"
                step="0.1"
                value={
                  form.bmi
                }
                onChange={
                  handleChange
                }
                placeholder="Ej. 30.1"
              />

              {
                errors.bmi &&
                (
                  <span className="field-error">
                    {
                      errors.bmi
                    }
                  </span>
                )
              }
            </div>

            <div className="form-field">
              <label
                htmlFor="smoking_status"
              >
                Estado respecto al tabaco
              </label>

              <select
                id="smoking_status"
                name="smoking_status"
                value={
                  form.smoking_status
                }
                onChange={
                  handleChange
                }
              >
                <option value="">
                  Selecciona una opción
                </option>

                <option value="formerly smoked">
                  Exfumador/a
                </option>

                <option value="never smoked">
                  Nunca ha fumado
                </option>

                <option value="smokes">
                  Fumador/a
                </option>

                <option value="Unknown">
                  Desconocido
                </option>
              </select>

              {
                errors.smoking_status &&
                (
                  <span className="field-error">
                    {
                      errors.smoking_status
                    }
                  </span>
                )
              }
            </div>
          </div>

          {
            apiError &&
            (
              <div className="form-message error-message">
                {
                  apiError
                }
              </div>
            )
          }

          <div className="form-actions">
            <button
              className="primary-button"
              type="submit"
              disabled={
                status ===
                "loading"
              }
            >
              {
                status ===
                  "loading"
                  ? "Enviando..."
                  : "Solicitar evaluación"
              }
            </button>
          </div>
        </form>

        <div className="safety-message">
          Esta estimación se utiliza como apoyo
          a la criba preventiva y no constituye
          un diagnóstico médico.
        </div>
      </div>
    </section>
  )
}


export default NewAssessment