import {
  Navigate,
  Route,
  Routes,
} from "react-router-dom"

import Layout from "./components/Layout"
import History from "./pages/History"
import NewAssessment from "./pages/NewAssessment"
import Result from "./pages/Result"

function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route
          path="/"
          element={
            <Navigate
              to="/assessment"
              replace
            />
          }
        />

        <Route
          path="/assessment"
          element={<NewAssessment />}
        />

        <Route
          path="/result"
          element={<Result />}
        />

        <Route
          path="/history"
          element={<History />}
        />
      </Route>
    </Routes>
  )
}

export default App
