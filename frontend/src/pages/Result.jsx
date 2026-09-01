function Result() {
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
          Aquí se mostrará la estimación generada
          por el backend después de realizar una
          evaluación.
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
        </div>

        <div className="safety-message">
          Esta herramienta sirve como apoyo a la
          criba y no sustituye la valoración de
          un profesional sanitario.
        </div>
      </div>
    </section>
  )
}

export default Result