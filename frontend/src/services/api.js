const API_BASE_URL =
  import.meta.env.VITE_API_URL ||
  "http://localhost:8000/api/v1"

export async function getHealth() {
  const response = await fetch(
    `${API_BASE_URL}/health`,
  )

  if (!response.ok) {
    throw new Error(
      "No se pudo conectar con la API",
    )
  }

  return response.json()
}

export async function createPrediction(
  payload,
) {
  const response = await fetch(
    `${API_BASE_URL}/predictions`,
    {
      method: "POST",
      headers: {
        "Content-Type":
          "application/json",
      },
      body: JSON.stringify(payload),
    },
  )

  if (!response.ok) {
    throw new Error(
      "No se pudo generar la predicción",
    )
  }

  return response.json()
}