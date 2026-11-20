// Canvas charts: no external libraries. All data comes from the server's analysis of the actual audio.
import { cssVar, fmtHz, fmtTime } from "./ui.js";

const colors = () => ({
  ink: cssVar("--ink"), ink2: cssVar("--ink-2"), ink3: cssVar("--ink-3"), line: cssVar("--line"), line2: cssVar("--line-2"),
  plum: cssVar("--plum"), plum2: cssVar("--plum-2"), lav: cssVar("--lav"), lav2: cssVar("--lav-2"), sage: cssVar("--sage"),
  sage2: cssVar("--sage-2"), bg: cssVar("--bg"), card: cssVar("--card"),
});
const FONT = '12px "Instrument Sans", system-ui, sans-serif';
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

class CanvasChart {
  constructor(box, { height = 260, pad = { l: 54, r: 14, t: 12, b: 30 } } = {}) {
    this.box = box; this.height = height; this.pad = pad;
    box.classList.add("chart");
    this.canvas = document.createElement("canvas");
    this.canvas.style.height = `${height}px`;
    this.tip = document.createElement("div");
    this.tip.className = "chart-tip"; this.tip.hidden = true;
    box.append(this.canvas, this.tip);
    this.ctx = this.canvas.getContext("2d");
    this.ready = false;                       // subclasses finish initialising before the first draw
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(box);
    this.resize();
    queueMicrotask(() => { this.ready = true; this.draw(); });
  }
  resize() {
    const w = Math.max(this.box.clientWidth, 120), dpr = window.devicePixelRatio || 1;
    if (w === this.w && dpr === this.dpr) return;
    this.w = w; this.dpr = dpr;
    this.canvas.width = Math.round(w * dpr); this.canvas.height = Math.round(this.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (this.ready) this.draw();
  }
  get plot() { const p = this.pad; return { x: p.l, y: p.t, w: this.w - p.l - p.r, h: this.height - p.t - p.b }; }
  mouse(e) { const r = this.canvas.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; }
  showTip(html, x, y) {
    this.tip.innerHTML = html; this.tip.hidden = false;
    const half = this.tip.offsetWidth / 2;
    this.tip.style.left = `${clamp(x, half + 4, this.w - half - 4)}px`;
    this.tip.style.top = `${Math.max(y - 10, this.tip.offsetHeight + 4)}px`;
  }
  hideTip() { this.tip.hidden = true; }
  destroy() { this.ro.disconnect(); this.box.innerHTML = ""; this.box.classList.remove("chart"); }
  clear() { this.ctx.clearRect(0, 0, this.w, this.height); }
  text(s, x, y, { align = "left", color, base = "alphabetic", font = FONT } = {}) {
    const c = this.ctx; c.font = font; c.fillStyle = color || colors().ink2; c.textAlign = align; c.textBaseline = base; c.fillText(s, x, y);
  }
}

function niceStep(range, target) {
  const raw = range / target, mag = 10 ** Math.floor(Math.log10(raw)), n = raw / mag;
  return (n < 1.5 ? 1 : n < 3.5 ? 2 : n < 7.5 ? 5 : 10) * mag;
}

/* ------------------------------------------------------------------ waveform */
export class WaveformChart extends CanvasChart {
  constructor(box, { height = 190, onSeek, onView } = {}) {
    super(box, { height, pad: { l: 46, r: 12, t: 10, b: 26 } });
    this.onSeek = onSeek; this.onView = onView;
    this.layers = []; this.duration = 0; this.v0 = 0; this.v1 = 1; this.playhead = 0; this.yMax = 1; this.token = 0;
    const c = this.canvas;
    c.addEventListener("mousedown", (e) => this.down(e));
    window.addEventListener("mousemove", this._mm = (e) => this.move(e));
    window.addEventListener("mouseup", this._mu = (e) => this.up(e));
    c.addEventListener("mouseleave", () => { if (!this.drag) this.hideTip(); });
    c.addEventListener("wheel", (e) => {
      if (!(e.ctrlKey || e.metaKey)) return;
      e.preventDefault();
      this.zoom(e.deltaY > 0 ? 1.25 : 0.8, this.xToT(this.mouse(e).x));
    }, { passive: false });
    c.addEventListener("dblclick", () => this.reset());
  }
  destroy() { window.removeEventListener("mousemove", this._mm); window.removeEventListener("mouseup", this._mu); super.destroy(); }

  /** layers: [{key, color, alpha, fetch(t0, t1, points) -> envelope}] */
  setData(duration, layers) {
    this.duration = duration; this.layers = layers.map((l) => ({ ...l, env: null }));
    this.v0 = 0; this.v1 = duration; this.refetch(0);
  }
  xToT(x) { const p = this.plot; return this.v0 + ((x - p.x) / p.w) * (this.v1 - this.v0); }
  tToX(t) { const p = this.plot; return p.x + ((t - this.v0) / (this.v1 - this.v0)) * p.w; }
  refetch(delay = 120) {
    clearTimeout(this._t);
    const run = async () => {
      const token = ++this.token, pts = Math.round(this.plot.w);
      try {
        const envs = await Promise.all(this.layers.map((l) => l.fetch(this.v0, this.v1, pts)));
        if (token !== this.token) return;
        envs.forEach((e, i) => (this.layers[i].env = e));
        this.yMax = Math.max(0.05, ...envs.map((e) => Math.max(...e.max.map(Math.abs), ...e.min.map(Math.abs)))) * 1.1;
        this.draw();
      } catch (e) { this.error = e.message; this.draw(); }
    };
    delay ? (this._t = setTimeout(run, delay)) : run();
  }
  setView(a, b) {
    const minSpan = Math.min(0.01, this.duration), span = clamp(b - a, minSpan, this.duration);
    this.v0 = clamp(a, 0, this.duration - span); this.v1 = this.v0 + span;
    this.draw(); this.refetch(); this.onView && this.onView(this.v0, this.v1);
  }
  zoom(f, centre) { const c = centre ?? (this.v0 + this.v1) / 2; this.setView(c - (c - this.v0) * f, c + (this.v1 - c) * f); }
  pan(dt) { this.setView(this.v0 + dt, this.v1 + dt); }
  reset() { this.setView(0, this.duration); }
  get zoomed() { return this.v1 - this.v0 < this.duration - 1e-6; }
  setPlayhead(t) { this.playhead = t; this.draw(); }

  down(e) { this.drag = { x: this.mouse(e).x, v0: this.v0, moved: false }; }
  move(e) {
    const m = this.mouse(e);
    if (this.drag) {
      const dx = m.x - this.drag.x;
      if (Math.abs(dx) > 4) this.drag.moved = true;
      if (this.drag.moved && this.zoomed) {
        const dt = (-dx / this.plot.w) * (this.v1 - this.v0);
        const span = this.v1 - this.v0, a = clamp(this.drag.v0 + dt, 0, this.duration - span);
        this.v0 = a; this.v1 = a + span; this.draw(); this.refetch();
      }
      return;
    }
    const p = this.plot;
    if (m.x < p.x || m.x > p.x + p.w || m.y < 0 || m.y > this.height) return this.hideTip();
    this.hover = m.x; this.draw();
    this.showTip(`${fmtTime(this.xToT(m.x))}.${String(Math.floor((this.xToT(m.x) % 1) * 100)).padStart(2, "0")}`, m.x, m.y);
  }
  up(e) {
    if (!this.drag) return;
    const d = this.drag; this.drag = null;
    const m = this.mouse(e), p = this.plot;
    if (!d.moved && m.x >= p.x && m.x <= p.x + p.w && m.y >= 0 && m.y <= this.height) {
      this.onSeek && this.onSeek(clamp(this.xToT(m.x), 0, this.duration));
    }
    if (d.moved) this.onView && this.onView(this.v0, this.v1);
  }

  draw() {
    if (!this.ctx) return;
    const c = this.ctx, k = colors(), p = this.plot;
    this.clear();
    // grid + amplitude labels
    c.lineWidth = 1; c.strokeStyle = k.line2;
    for (const f of [-1, -0.5, 0, 0.5, 1]) {
      const y = p.y + p.h / 2 - f * (p.h / 2);
      c.beginPath(); c.moveTo(p.x, y); c.lineTo(p.x + p.w, y); c.strokeStyle = f === 0 ? k.line : k.line2; c.stroke();
      if (f !== 0.5 && f !== -0.5) this.text((f * this.yMax).toFixed(2), p.x - 8, y + 4, { align: "right", color: k.ink3 });
    }
    // time ticks
    const span = this.v1 - this.v0, step = niceStep(span, Math.max(3, p.w / 90));
    for (let t = Math.ceil(this.v0 / step) * step; t <= this.v1 + 1e-9; t += step) {
      const x = this.tToX(t);
      c.beginPath(); c.moveTo(x, p.y + p.h); c.lineTo(x, p.y + p.h + 4); c.strokeStyle = k.line; c.stroke();
      this.text(step < 1 ? `${t.toFixed(2)}s` : fmtTime(t), x, p.y + p.h + 18, { align: "center", color: k.ink3 });
    }
    c.save(); c.beginPath(); c.rect(p.x, p.y, p.w, p.h); c.clip();
    for (const l of this.layers) if (l.env) this.drawEnv(l, p);
    if (this.hover != null) {
      c.beginPath(); c.moveTo(this.hover, p.y); c.lineTo(this.hover, p.y + p.h); c.strokeStyle = k.ink3; c.setLineDash([3, 3]); c.stroke(); c.setLineDash([]);
    }
    c.restore();
    if (this.playhead >= this.v0 && this.playhead <= this.v1) {
      const x = this.tToX(this.playhead);
      c.beginPath(); c.moveTo(x, p.y); c.lineTo(x, p.y + p.h); c.lineWidth = 2; c.strokeStyle = k.sage2; c.stroke(); c.lineWidth = 1;
      c.fillStyle = k.sage2; c.beginPath(); c.arc(x, p.y + 3, 4, 0, 7); c.fill();
    }
    if (this.error) this.text(this.error, p.x + p.w / 2, p.y + p.h / 2, { align: "center", color: k.ink2 });
  }
  drawEnv(l, p) {
    const c = this.ctx, e = l.env, n = e.min.length;
    const yy = (v) => p.y + p.h / 2 - (v / this.yMax) * (p.h / 2);
    const dt = (e.t1 - e.t0) / n;
    c.globalAlpha = l.alpha ?? 1; c.strokeStyle = l.color; c.fillStyle = l.color;
    if (e.samples && n > 1) {
      c.beginPath(); c.lineWidth = 1.5;
      for (let i = 0; i < n; i++) { const x = this.tToX(e.t0 + (i + 0.5) * dt), y = yy(e.max[i]); i ? c.lineTo(x, y) : c.moveTo(x, y); }
      c.stroke();
    } else {
      const bw = Math.max(1, (dt / (this.v1 - this.v0)) * p.w);
      c.lineWidth = Math.min(bw, 3);
      c.beginPath();
      for (let i = 0; i < n; i++) {
        const x = this.tToX(e.t0 + (i + 0.5) * dt);
        const top = yy(e.max[i]), bot = yy(e.min[i]);
        c.moveTo(x, top); c.lineTo(x, Math.max(bot, top + 1));
      }
      c.stroke();
    }
    c.globalAlpha = 1; c.lineWidth = 1;
  }
}

/* ------------------------------------------------------------------ spectrum */
export class SpectrumChart extends CanvasChart {
  constructor(box, { height = 290, yMin = null, yMax = null, yUnit = "dB", yRangeDb = 90, onXRange } = {}) {
    super(box, { height, pad: { l: 52, r: 14, t: 14, b: 30 } });
    Object.assign(this, { fixedMin: yMin, fixedMax: yMax, yUnit, yRangeDb, onXRange });
    this.series = []; this.markers = []; this.bands = []; this.scale = "log"; this.x0 = 20; this.x1 = 8000; this.fullX = [20, 8000];
    const c = this.canvas;
    c.addEventListener("mousedown", (e) => { this.sel = { a: this.mouse(e).x, b: this.mouse(e).x }; });
    window.addEventListener("mousemove", this._mm = (e) => this.move(e));
    window.addEventListener("mouseup", this._mu = (e) => this.up(e));
    c.addEventListener("mouseleave", () => { if (!this.sel) { this.hover = null; this.hideTip(); this.draw(); } });
    c.addEventListener("dblclick", () => this.setXRange(...this.fullX, true));
  }
  destroy() { window.removeEventListener("mousemove", this._mm); window.removeEventListener("mouseup", this._mu); super.destroy(); }

  setSeries(series) {
    this.series = series;
    const f = series[0]?.freq;
    if (f) { this.fullX = [Math.max(20, f[0]), f[f.length - 1]]; if (!this.keepRange) { this.x0 = this.fullX[0]; this.x1 = this.fullX[1]; } }
    this.draw();
  }
  setMarkers(m) { this.markers = m; this.draw(); }
  setBands(b) { this.bands = b; this.draw(); }
  setScale(s) { this.scale = s; this.draw(); }
  setXRange(a, b, notify = false) {
    this.x0 = clamp(a, this.fullX[0], this.fullX[1] * 0.99); this.x1 = clamp(b, this.x0 * 1.05, this.fullX[1]);
    this.keepRange = !(this.x0 === this.fullX[0] && this.x1 === this.fullX[1]);
    this.draw(); if (notify && this.onXRange) this.onXRange(this.x0, this.x1);
  }
  xOf(f) {
    const p = this.plot;
    return this.scale === "log"
      ? p.x + (Math.log(f / this.x0) / Math.log(this.x1 / this.x0)) * p.w
      : p.x + ((f - this.x0) / (this.x1 - this.x0)) * p.w;
  }
  fOf(x) {
    const p = this.plot, t = (x - p.x) / p.w;
    return this.scale === "log" ? this.x0 * (this.x1 / this.x0) ** t : this.x0 + t * (this.x1 - this.x0);
  }
  yRange() {
    if (this.fixedMin != null) return [this.fixedMin, this.fixedMax];
    let hi = -1e9;
    for (const s of this.series) for (let i = 0; i < s.freq.length; i++) if (s.freq[i] >= this.x0 && s.freq[i] <= this.x1 && s.values[i] > hi) hi = s.values[i];
    if (hi < -1e8) hi = 0;
    const top = Math.ceil((hi + 4) / 10) * 10;
    return [top - this.yRangeDb, top];
  }
  yOf(v, [lo, hi]) { const p = this.plot; return p.y + (1 - (v - lo) / (hi - lo)) * p.h; }

  move(e) {
    const m = this.mouse(e), p = this.plot;
    if (this.sel) { this.sel.b = clamp(m.x, p.x, p.x + p.w); this.draw(); return; }
    if (m.x < p.x || m.x > p.x + p.w || m.y < 0 || m.y > this.height || e.target !== this.canvas) return;
    this.hover = m.x; this.draw();
    const f = this.fOf(m.x), s0 = this.series[0];
    if (!s0) return;
    const idx = nearest(s0.freq, f);
    const rows = this.series.map((s) => `<span style="color:${s.color === colors().plum ? "#fff" : s.tipColor || "#fff"}">${s.name ? s.name + ": " : ""}${s.values[nearest(s.freq, f)].toFixed(1)} ${this.yUnit}</span>`);
    this.showTip(`<b>${fmtHz(s0.freq[idx])}</b><br>${rows.join("<br>")}`, m.x, m.y);
  }
  up() {
    if (!this.sel) return;
    const { a, b } = this.sel; this.sel = null;
    if (Math.abs(b - a) > 10) { const f1 = this.fOf(Math.min(a, b)), f2 = this.fOf(Math.max(a, b)); this.setXRange(f1, f2, true); } else this.draw();
  }

  draw() {
    if (!this.ctx) return;
    const c = this.ctx, k = colors(), p = this.plot, yr = this.yRange();
    this.clear();
    // y grid
    const ystep = niceStep(yr[1] - yr[0], 5);
    c.lineWidth = 1;
    for (let v = Math.ceil(yr[0] / ystep) * ystep; v <= yr[1]; v += ystep) {
      const y = this.yOf(v, yr);
      c.beginPath(); c.moveTo(p.x, y); c.lineTo(p.x + p.w, y); c.strokeStyle = k.line2; c.stroke();
      this.text(`${v}`, p.x - 8, y + 4, { align: "right", color: k.ink3 });
    }
    this.text(this.yUnit, 6, p.y + 8, { color: k.ink3 });
    // x grid
    let ticks;
    if (this.scale === "log") ticks = [20, 30, 50, 100, 200, 300, 500, 1000, 2000, 3000, 5000, 10000, 20000].filter((t) => t >= this.x0 && t <= this.x1);
    else { const st = niceStep(this.x1 - this.x0, 8); ticks = []; for (let t = Math.ceil(this.x0 / st) * st; t <= this.x1; t += st) ticks.push(t); }
    let lastLabel = -99;
    for (const t of ticks) {
      const x = this.xOf(t);
      c.beginPath(); c.moveTo(x, p.y); c.lineTo(x, p.y + p.h); c.strokeStyle = k.line2; c.stroke();
      if (x - lastLabel < 30) continue;                       // keep labels legible on narrow screens
      lastLabel = x;
      this.text(t >= 1000 ? `${t / 1000}k` : `${Math.round(t)}`, x, p.y + p.h + 18, { align: "center", color: k.ink3 });
    }
    this.text("Hz", p.x + p.w + 2, p.y + p.h + 18, { align: "right", color: k.ink3 });
    c.save(); c.beginPath(); c.rect(p.x, p.y - 2, p.w, p.h + 4); c.clip();
    for (const b of this.bands) {
      const xa = this.xOf(Math.max(b.lo, this.x0)), xb = this.xOf(Math.min(b.hi, this.x1));
      if (xb <= xa) continue;
      c.fillStyle = k.sage; c.globalAlpha = 0.28; c.fillRect(xa, p.y, Math.max(xb - xa, 2), p.h); c.globalAlpha = 1;
    }
    for (const s of this.series) this.drawSeries(s, yr, p);
    for (const m of this.markers) {
      if (m.freq < this.x0 || m.freq > this.x1) continue;
      const x = this.xOf(m.freq);
      c.beginPath(); c.moveTo(x, p.y); c.lineTo(x, p.y + p.h); c.strokeStyle = m.color || k.plum2; c.globalAlpha = 0.55; c.setLineDash([4, 4]); c.stroke(); c.setLineDash([]); c.globalAlpha = 1;
    }
    if (this.hover != null && !this.sel) { c.beginPath(); c.moveTo(this.hover, p.y); c.lineTo(this.hover, p.y + p.h); c.strokeStyle = k.ink3; c.stroke(); }
    if (this.sel) { c.fillStyle = k.plum; c.globalAlpha = 0.12; const a = Math.min(this.sel.a, this.sel.b); c.fillRect(a, p.y, Math.abs(this.sel.b - this.sel.a), p.h); c.globalAlpha = 1; }
    c.restore();
    let lastX = -99;
    for (const m of [...this.markers].sort((a, b) => a.freq - b.freq)) {
      if (!m.label || m.freq < this.x0 || m.freq > this.x1) continue;
      const x = this.xOf(m.freq);
      if (x - lastX < 38) continue;
      lastX = x;
      this.text(m.label, x, p.y + 10, { align: "center", color: m.color || k.plum2, font: '600 11.5px "Instrument Sans", sans-serif' });
    }
  }
  drawSeries(s, yr, p) {
    const c = this.ctx, f = s.freq, v = s.values;
    c.beginPath();
    let started = false, lastX = 0;
    for (let i = 0; i < f.length; i++) {
      if (f[i] < this.x0 * 0.98 || f[i] > this.x1 * 1.02) continue;
      const x = this.xOf(f[i]), y = clamp(this.yOf(v[i], yr), p.y - 4, p.y + p.h + 4);
      if (!started) { c.moveTo(x, y); started = true; } else c.lineTo(x, y);
      lastX = x;
    }
    if (!started) return;
    c.strokeStyle = s.color; c.lineWidth = s.width || 1.6; c.lineJoin = "round"; c.stroke();
    if (s.fill) {
      c.lineTo(lastX, p.y + p.h); c.lineTo(this.xOf(Math.max(f[0], this.x0)), p.y + p.h); c.closePath();
      c.fillStyle = s.color; c.globalAlpha = s.fillAlpha ?? 0.16; c.fill(); c.globalAlpha = 1;
    }
    c.lineWidth = 1;
  }
}
function nearest(arr, v) {
  let lo = 0, hi = arr.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; arr[m] < v ? (lo = m) : (hi = m); }
  return Math.abs(arr[lo] - v) < Math.abs(arr[hi] - v) ? lo : hi;
}

/* ------------------------------------------------------------------ spectrogram */
const STOPS = [[0, [250, 247, 242]], [0.28, [226, 214, 238]], [0.52, [183, 164, 204]], [0.76, [122, 74, 134]], [1, [42, 15, 51]]];
function colormap(t) {
  for (let i = 1; i < STOPS.length; i++) {
    if (t <= STOPS[i][0]) {
      const [t0, a] = STOPS[i - 1], [t1, b] = STOPS[i], u = (t - t0) / (t1 - t0);
      return a.map((v, j) => Math.round(v + (b[j] - v) * u));
    }
  }
  return STOPS[STOPS.length - 1][1];
}
export class SpectrogramChart extends CanvasChart {
  constructor(box, { height = 250 } = {}) {
    super(box, { height, pad: { l: 52, r: 12, t: 8, b: 28 } });
    this.canvas.addEventListener("mousemove", (e) => this.move(e));
    this.canvas.addEventListener("mouseleave", () => this.hideTip());
  }
  setImage(s) {
    this.spec = s;
    const bin = atob(s.data), off = document.createElement("canvas");
    off.width = s.cols; off.height = s.rows;
    const ctx = off.getContext("2d"), img = ctx.createImageData(s.cols, s.rows);
    this.raw = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) this.raw[i] = bin.charCodeAt(i);
    for (let r = 0; r < s.rows; r++) for (let col = 0; col < s.cols; col++) {
      const [R, G, B] = colormap(this.raw[r * s.cols + col] / 255), o = ((s.rows - 1 - r) * s.cols + col) * 4;
      img.data[o] = R; img.data[o + 1] = G; img.data[o + 2] = B; img.data[o + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    this.off = off; this.draw();
  }
  yOfFreq(f) { const p = this.plot, s = this.spec; return p.y + p.h - (Math.log(f / s.fmin) / Math.log(s.fmax / s.fmin)) * p.h; }
  draw() {
    if (!this.ctx || !this.spec) return;
    const c = this.ctx, k = colors(), p = this.plot, s = this.spec;
    this.clear();
    c.imageSmoothingEnabled = true; c.imageSmoothingQuality = "high";
    c.drawImage(this.off, p.x, p.y, p.w, p.h);
    for (const f of [50, 100, 200, 500, 1000, 2000, 4000, 8000, 16000].filter((v) => v >= s.fmin && v <= s.fmax * 1.01)) {
      const y = Math.max(this.yOfFreq(f), p.y);
      c.beginPath(); c.moveTo(p.x - 4, y); c.lineTo(p.x, y); c.strokeStyle = k.line; c.stroke();
      this.text(f >= 1000 ? `${f / 1000}k` : `${f}`, p.x - 8, y + 4, { align: "right", color: k.ink3 });
    }
    const step = niceStep(s.duration, Math.max(3, p.w / 90));
    for (let t = 0; t <= s.duration + 1e-9; t += step) {
      const x = p.x + (t / s.duration) * p.w;
      c.beginPath(); c.moveTo(x, p.y + p.h); c.lineTo(x, p.y + p.h + 4); c.strokeStyle = k.line; c.stroke();
      this.text(step < 1 ? `${t.toFixed(1)}s` : fmtTime(t), x, p.y + p.h + 18, { align: "center", color: k.ink3 });
    }
    if (this.playhead != null && this.playhead >= 0) {
      const x = p.x + (this.playhead / s.duration) * p.w;
      c.beginPath(); c.moveTo(x, p.y); c.lineTo(x, p.y + p.h); c.strokeStyle = k.sage; c.lineWidth = 2; c.stroke(); c.lineWidth = 1;
    }
  }
  setPlayhead(t) { this.playhead = t; this.draw(); }
  move(e) {
    if (!this.spec) return;
    const m = this.mouse(e), p = this.plot, s = this.spec;
    if (m.x < p.x || m.x > p.x + p.w || m.y < p.y || m.y > p.y + p.h) return this.hideTip();
    const t = ((m.x - p.x) / p.w) * s.duration, fr = (p.y + p.h - m.y) / p.h;
    const f = s.fmin * (s.fmax / s.fmin) ** fr;
    const col = clamp(Math.floor((t / s.duration) * s.cols), 0, s.cols - 1), row = clamp(Math.floor(fr * s.rows), 0, s.rows - 1);
    const dB = s.db_min + (this.raw[row * s.cols + col] / 255) * (s.db_max - s.db_min);
    this.showTip(`${fmtTime(t)} · ${fmtHz(f)}<br>${dB.toFixed(0)} dB`, m.x, m.y);
  }
}

/* ------------------------------------------------------------------ band comparison */
export class BandChart extends CanvasChart {
  constructor(box, { height = 230 } = {}) { super(box, { height, pad: { l: 46, r: 12, t: 12, b: 36 } }); this.bars = []; this.canvas.addEventListener("mousemove", (e) => this.move(e)); this.canvas.addEventListener("mouseleave", () => { this.hideTip(); this.hi = -1; this.draw(); }); }
  setBands(bands) {
    this.bars = bands.map((b) => ({ label: b.hi >= 1000 ? `${b.lo / 1000 >= 1 ? b.lo / 1000 : b.lo}${b.lo >= 1000 ? "k" : ""}–${b.hi / 1000}k`.replace("–8k", "–8k") : `${b.lo}–${b.hi}`, ...b }));
    this.bars.forEach((b) => { b.label = `${b.lo >= 1000 ? b.lo / 1000 + "k" : b.lo}–${b.hi >= 1000 ? b.hi / 1000 + "k" : b.hi}`; });
    this.draw();
  }
  range() {
    const all = this.bars.flatMap((b) => [b.original_db, b.processed_db]);
    const hi = Math.ceil((Math.max(...all) + 2) / 10) * 10, lo = Math.floor((Math.min(...all) - 2) / 10) * 10;
    return [lo, hi];
  }
  draw() {
    if (!this.ctx || !this.bars.length) return;
    const c = this.ctx, k = colors(), p = this.plot, [lo, hi] = this.range(), n = this.bars.length;
    this.clear();
    const y = (v) => p.y + (1 - (v - lo) / (hi - lo)) * p.h, step = niceStep(hi - lo, 5);
    for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) {
      c.beginPath(); c.moveTo(p.x, y(v)); c.lineTo(p.x + p.w, y(v)); c.strokeStyle = k.line2; c.stroke();
      this.text(`${v}`, p.x - 8, y(v) + 4, { align: "right", color: k.ink3 });
    }
    const gw = p.w / n, bw = Math.min(26, gw * 0.32);
    this.bars.forEach((b, i) => {
      const cx = p.x + gw * (i + 0.5);
      if (i === this.hi) { c.fillStyle = k.line2; c.fillRect(p.x + gw * i, p.y, gw, p.h); }
      c.fillStyle = k.lav; c.fillRect(cx - bw - 1, y(b.original_db), bw, p.y + p.h - y(b.original_db));
      c.fillStyle = k.plum; c.fillRect(cx + 1, y(b.processed_db), bw, p.y + p.h - y(b.processed_db));
      this.text(b.label, cx, p.y + p.h + 16, { align: "center", color: k.ink2 });
    });
    this.text("Hz band", p.x + p.w / 2, p.y + p.h + 31, { align: "center", color: k.ink3 });
    this.text("dB", 6, p.y + 8, { color: k.ink3 });
  }
  move(e) {
    const m = this.mouse(e), p = this.plot, n = this.bars.length, i = Math.floor(((m.x - p.x) / p.w) * n);
    if (i < 0 || i >= n || m.y > p.y + p.h) { this.hi = -1; this.hideTip(); return this.draw(); }
    this.hi = i; this.draw();
    const b = this.bars[i];
    this.showTip(`<b>${b.label} Hz</b><br>Before ${b.original_db.toFixed(1)} dB<br>After ${b.processed_db.toFixed(1)} dB<br>Change ${b.change_db > 0 ? "+" : ""}${b.change_db.toFixed(1)} dB`, p.x + (p.w / n) * (i + 0.5), m.y);
  }
}
