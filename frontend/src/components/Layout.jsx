import { Outlet } from "react-router-dom"

import Navbar from "./Navbar"
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
        <p>
          Stroke Risk AI
        </p>

        <p>
          {CLINICAL_DISCLAIMER}
        </p>
      </footer>
    </div>
  )
}

export default Layout
