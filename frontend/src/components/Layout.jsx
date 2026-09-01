import { Outlet } from "react-router-dom"

import Navbar from "./Navbar"

function Layout() {
  return (
    <div className="app-shell">
      <Navbar />

      <main className="main-content">
        <Outlet />
      </main>

      <footer className="app-footer">
        <p>
          Stroke Risk AI · Herramienta de
          apoyo a la criba
        </p>

        <p>
          No constituye un diagnóstico médico.
        </p>
      </footer>
    </div>
  )
}

export default Layout