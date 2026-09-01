import {
  NavLink,
} from "react-router-dom"

function Navbar() {
  return (
    <header className="navbar">
      <div className="navbar-brand">
        <span className="brand-mark">
          SR
        </span>

        <div>
          <strong>
            Stroke Risk AI
          </strong>

          <span>
            Clinical Support
          </span>
        </div>
      </div>

      <nav className="navbar-links">
        <NavLink
          to="/assessment"
          className={({ isActive }) =>
            isActive
              ? "nav-link active"
              : "nav-link"
          }
        >
          Nueva evaluación
        </NavLink>

        <NavLink
          to="/result"
          className={({ isActive }) =>
            isActive
              ? "nav-link active"
              : "nav-link"
          }
        >
          Resultado
        </NavLink>

        <NavLink
          to="/history"
          className={({ isActive }) =>
            isActive
              ? "nav-link active"
              : "nav-link"
          }
        >
          Historial
        </NavLink>
      </nav>
    </header>
  )
}

export default Navbar