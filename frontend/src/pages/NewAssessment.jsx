function NewAssessment() {
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
          para realizar una estimación de riesgo.
        </p>
      </div>

      <div className="content-card">
        <h2>
          Datos del paciente
        </h2>

        <p>
          El formulario clínico se incorporará
          en la siguiente fase del proyecto.
        </p>

        <div className="placeholder-grid">
          <div className="placeholder-field">
            Edad
          </div>

          <div className="placeholder-field">
            Sexo
          </div>

          <div className="placeholder-field">
            Hipertensión
          </div>

          <div className="placeholder-field">
            Glucosa
          </div>

          <div className="placeholder-field">
            BMI
          </div>

          <div className="placeholder-field">
            Tabaquismo
          </div>
        </div>
      </div>
    </section>
  )
}

export default NewAssessment