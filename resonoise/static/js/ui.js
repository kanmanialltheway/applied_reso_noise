// Shared helpers: formatting, escaping, toasts.
import { icon } from "./icons.js";

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

export function fmtTime(t) {
  if (!isFinite(t)) return "0:00";
  const m = Math.floor(t / 60), s = Math.floor(t % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}
export const fmtHz = (f) => (f >= 1000 ? `${(f / 1000).toFixed(f >= 10000 ? 1 : 2)} kHz` : `${f.toFixed(f < 100 ? 1 : 0)} Hz`);
export const fmtDb = (v, d = 1) => `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(d)} dB`;
export const fmtSize = (b) => (b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1e3))} kB`);
export const fmtDate = (ts) => new Date(ts * 1000).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });

export function toast(message, kind = "info", ms = 6000) {
  const box = document.getElementById("toasts");
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.setAttribute("role", kind === "error" ? "alert" : "status");
  el.innerHTML = `${icon(kind === "error" ? "alert" : kind === "good" ? "check" : "info")}<span>${esc(message)}</span><button aria-label="Dismiss">${icon("x")}</button>`;
  const close = () => el.remove();
  el.querySelector("button").onclick = close;
  box.appendChild(el);
  if (ms) setTimeout(close, ms);
}

export const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export function debounce(fn, ms) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

export function download(url) {
  const a = document.createElement("a");
  a.href = url; a.rel = "noopener";
  document.body.appendChild(a); a.click(); a.remove();
}
