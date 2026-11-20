// App shell: routing, header status and global state.
import { api } from "./api.js";
import { icon, LOGO } from "./icons.js";
import { renderLibrary } from "./library.js";
import { renderMethod } from "./method.js";
import { studio } from "./studio.js";
import { $, $$, esc, toast } from "./ui.js";

const state = { health: null, model: null, current: null };
const ROUTES = { studio: "Studio", library: "Library", method: "Method" };
const ICONS = { studio: "studio", library: "library", method: "book" };

function shell() {
  document.getElementById("topbar").innerHTML = `<div class="topbar-inner">
    <a class="brand" href="#/studio" aria-label="Reso-Noise home">${LOGO}<span class="brand-name">Reso-Noise</span></a>
    <nav class="nav" aria-label="Main">${Object.entries(ROUTES).map(([k, v]) => `<a href="#/${k}" data-route="${k}">${icon(ICONS[k])}<span>${v}</span></a>`).join("")}</nav>
    <div class="topbar-spacer"></div><div class="chip-model" id="model-chip"></div></div>`;
}
function modelChip() {
  const m = state.model;
  $("#model-chip").innerHTML = m ? `<span class="dot"></span>Classifier ready · ${(m.validation_accuracy * 100).toFixed(0)}% held-out${m.n_user_samples ? ` · ${m.n_user_samples} corrections` : ""}` : `<span class="dot off"></span>Classifier unavailable`;
}

async function route() {
  const key = (location.hash.replace(/^#\//, "") || "studio").split("/")[0];
  const name = ROUTES[key] ? key : "studio";
  $$("[data-route]").forEach((a) => (a.ariaCurrent = a.dataset.route === name ? "page" : null));
  if (state.current === "studio" && name !== "studio") studio.leave();
  const views = { studio: $("#view-studio"), library: $("#view-library"), method: $("#view-method") };
  Object.entries(views).forEach(([k, el]) => (el.hidden = k !== name));
  if (name === "library") await renderLibrary(views.library, {
    classes: state.model.classes,
    open: (id) => { location.hash = "#/studio"; studio.openRecording(id, 2); },
    openResult: (rid, pid) => { location.hash = "#/studio"; studio.openResult(rid, pid); },
  });
  if (name === "method") await renderMethod(views.method, { model: state.model, onModel: (m) => { state.model = m; studio.setModel(m); modelChip(); } });
  if (name === "studio" && state.current !== "studio") studio.init(views.studio, state);
  state.current = name;
}

async function boot() {
  shell();
  try {
    [state.health, state.model] = await Promise.all([api.health(), api.model()]);
  } catch (e) {
    $("#view-studio").innerHTML = `<div class="notice error">${icon("alert")}<span>${esc(e.message)}</span></div>`;
    return;
  }
  modelChip();
  window.addEventListener("hashchange", route);
  route();
}
boot();
