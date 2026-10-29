import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router";
import "@fontsource-variable/inter";
import "@fontsource/barlow-condensed/200.css";
import "@fontsource/barlow-condensed/300.css";
import "@fontsource/barlow-condensed/500.css";
import "@fontsource/dm-mono/400.css";
import App from "./App";
import "./styles.css";
import "./styles-aether.css";
import { applyInitialUi } from "./theme";

applyInitialUi();

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>,
);
