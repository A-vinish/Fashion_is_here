import React from "react";
import ReactDOM from "react-dom/client";
import { Toaster } from "react-hot-toast";
import App from "./App.jsx";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <Toaster position="bottom-center" toastOptions={{
      style: { background: "#1F2937", color: "#fff", fontSize: "14px", borderRadius: "12px" },
    }} />
    <App />
  </React.StrictMode>
);
