import { BrowserRouter, Route, Routes } from "react-router-dom";

import ComplaintDetail from "./pages/ComplaintDetail";
import Dashboard from "./pages/Dashboard";
import Intake from "./pages/Intake";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/intake" element={<Intake />} />
        <Route path="/complaints/:id" element={<ComplaintDetail />} />
      </Routes>
    </BrowserRouter>
  );
}
