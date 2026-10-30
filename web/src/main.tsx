import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router";
import "@fontsource-variable/inter";
import "@fontsource/barlow-condensed/200.css";
import "@fontsource/barlow-condensed/300.css";
import "@fontsource/barlow-condensed/500.css";
import "@fontsource/dm-mono/400.css";
import "@fontsource-variable/geist";
import "@fontsource/instrument-serif/400.css";
import "@fontsource/instrument-serif/400-italic.css";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
import App from "./App";
import "./styles.css";
import "./styles-aether.css";
import "./styles-aurora.css";
import { applyInitialUi } from "./theme";

applyInitialUi();

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>,
);
