const API_BASE_URL =
  import.meta.env.VITE_API_URL ||
  "http://localhost:8000/api/v1"


async function parseError(
  response,
) {
  try {
    const data =
      await response.json()

    if (
      typeof data.detail ===
      "string"
    ) {
      return data.detail
    }

    if (
      Array.isArray(
        data.detail
      )
    ) {
      return data.detail
        .map(
          (error) =>
            error.msg,
        )
        .join(" ")
    }
  } catch {
    return null
  }

  return null
}


export async function getHealth() {
  const response = await fetch(
    `${API_BASE_URL}/health`,
  )

  if (!response.ok) {
    const detail =
      await parseError(
        response
      )

    throw new Error(
      detail ||
      "No se pudo conectar con la API.",
    )
  }

  return response.json()
}


export async function createPrediction(
  payload,
) {
  let response

  try {
    response = await fetch(
      `${API_BASE_URL}/predictions`,
      {
        method: "POST",
        headers: {
          "Content-Type":
            "application/json",
        },
        body: JSON.stringify(
          payload
        ),
      },
    )
  } catch {
    throw new Error(
      "No se pudo conectar con el servidor. Comprueba que la API está iniciada.",
    )
  }

  if (!response.ok) {
    const detail =
      await parseError(
        response
      )

    throw new Error(
      detail ||
      "No se pudo generar la predicción.",
    )
  }

  return response.json()
}


async function getApiResource(
  path,
  fallbackMessage,
) {
  let response

  try {
    response = await fetch(
      `${API_BASE_URL}${path}`,
    )
  } catch {
    throw new Error(
      "No se pudo conectar con el servidor. Comprueba que la API está iniciada.",
    )
  }

  if (!response.ok) {
    const detail =
      await parseError(
        response
      )

    const error = new Error(
      detail || fallbackMessage,
    )

    error.status = response.status

    throw error
  }

  return response.json()
}


export function getAssessmentHistory() {
  return getApiResource(
    "/assessments",
    "No se pudo consultar el historial de evaluaciones.",
  )
}


export function getAssessmentDetail(
  assessmentId,
) {
  return getApiResource(
    `/assessments/${assessmentId}`,
    "No se pudo consultar el detalle de la evaluación.",
  )
}
