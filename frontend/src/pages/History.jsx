function History() {
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
        <div className="empty-state">
          <h2>
            Historial de evaluaciones
          </h2>

          <p>
            Los registros almacenados por el
            backend se mostrarán aquí.
          </p>
        </div>
      </div>
    </section>
  )
}

export default History