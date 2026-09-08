import {
  NavLink,
} from "react-router-dom"

import CereviaLogo from "./CereviaLogo"

function Navbar() {
  return (
    <header className="navbar">
      <div className="navbar-inner">
        <CereviaLogo />

        <nav
          className="navbar-links"
          aria-label="Navegación principal"
        >
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
      </div>
    </header>
  )
}

export default Navbar
