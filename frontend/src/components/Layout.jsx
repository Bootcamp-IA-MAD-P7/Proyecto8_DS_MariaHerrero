import { Outlet } from "react-router-dom"

import Navbar from "./Navbar"
import CereviaLogo from "./CereviaLogo"
import {
  CLINICAL_DISCLAIMER,
} from "../constants/clinicalSafety"

function Layout() {
  return (
    <div className="app-shell">
      <Navbar />

      <main className="main-content">
        <Outlet />
      </main>

      <footer className="app-footer">
        <div className="footer-inner">
          <div className="footer-brand">
            <CereviaLogo compact />
            <p>Inteligencia · Prevención · Confianza</p>
          </div>

          <p className="footer-disclaimer">
            {CLINICAL_DISCLAIMER}
          </p>
        </div>
      </footer>
    </div>
  )
}

export default Layout
