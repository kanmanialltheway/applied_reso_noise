// The Studio view: input -> spectrum -> source -> filter -> result.
import { api } from "./api.js";
import { BandChart, SpectrogramChart, SpectrumChart, WaveformChart } from "./charts.js";
import { icon } from "./icons.js";
import { Player } from "./player.js";
import { Recorder } from "./recorder.js";
import { $, $$, cssVar, debounce, download, esc, fmtDb, fmtHz, fmtSize, fmtTime, toast } from "./ui.js";

const STEPS = [
  { id: 1, title: "Input", sub: "Upload or record" },
  { id: 2, title: "Spectrum", sub: "Fourier analysis" },
  { id: 3, title: "Source", sub: "Classification" },
  { id: 4, title: "Filter", sub: "Shape H(f)" },
  { id: 5, title: "Result", sub: "Reconstruction" },
];

const S = { step: 1, rec: null, analysis: null, result: null, params: null, profileKey: null, extra: [], notchOn: null,
  charts: [], player: null, recorder: null, health: null, model: null, root: null, busy: null };

export const studio = {
  init(root, { health, model }) {
    S.root = root; S.health = health; S.model = model;
    root.innerHTML = `
      <div class="studio-head">
        <h1>Find the noise, then take it out</h1>
        <p>Reso-Noise reads the frequency content of a recording, works out which household or street source it comes from, and removes it with a filter shaped to that source.</p>
      </div>
      <ol class="stepper" id="stepper" aria-label="Workflow"></ol>
      <div id="current"></div>
      <div id="panel"></div>`;
    $("#stepper").addEventListener("click", (e) => { const b = e.target.closest("[data-step]"); if (b && !b.disabled) go(+b.dataset.step); });
    render();
  },
  setModel(m) { S.model = m; },
  async openRecording(id, step = 2) {
    try {
      S.rec = (await api.detail(id)).recording;
      S.analysis = null; S.result = null; S.params = null; S.extra = [];
      try { S.analysis = await api.analysis(id); } catch (e) { if (e.status !== 404) throw e; S.analysis = await api.analyze(id); }
      go(step);
    } catch (e) { toast(e.message, "error"); }
  },
  async openResult(rid, pid) {
    try {
      S.rec = (await api.detail(rid)).recording;
      S.analysis = await api.analysis(rid);
      S.result = await api.processed(pid);
      S.params = { ...S.result.params }; S.profileKey = S.params.profile; S.extra = []; S.notchOn = S.params.notch_freqs.length > 0;
      go(5);
    } catch (e) { toast(e.message, "error"); }
  },
  leave() { teardown(); },
};

/* ------------------------------------------------------------------ plumbing */
const unlocked = (n) => n === 1 || (n <= 4 && !!S.analysis) || (n === 5 && !!S.result);
function track(c) { S.charts.push(c); return c; }
function teardown() {
  S.charts.forEach((c) => c.destroy()); S.charts = [];
  if (S.player) { S.player.dispose(); S.player = null; }
  if (S.recorder) { S.recorder.cancel(); S.recorder = null; }
}
function go(n) { S.step = n; S.busy = null; render(); window.scrollTo({ top: 0, behavior: "smooth" }); }
function render() {
  teardown();
  $("#stepper").innerHTML = STEPS.map((s) => {
    const cls = s.id < S.step && unlocked(s.id) ? "done" : "";
    return `<li><button class="step-btn ${cls}" data-step="${s.id}" ${s.id === S.step ? 'aria-current="step"' : ""} ${unlocked(s.id) ? "" : "disabled"}>
      <span class="num">${cls ? icon("check", 15) : s.id}</span><span><span class="t">${s.title}</span><br><span class="s">${s.sub}</span></span></button></li>`;
  }).join("");
  const cur = $("#current");
  if (S.rec) {
    const r = S.rec;
    cur.innerHTML = `<div class="current-bar">${icon("file")}<span class="name">${esc(r.name)}</span>
      <span class="meta"><span>${fmtTime(r.duration)} long</span><span>${(r.sample_rate / 1000).toFixed(r.sample_rate % 1000 ? 1 : 0)} kHz</span><span>${r.channels === 1 ? "mono" : r.channels + " channels, mixed to mono"}</span>
      ${r.source === "sample" ? '<span class="badge sage">Sample scene</span>' : ""}${r.has_reference ? '<span class="badge plain">Clean reference stored</span>' : ""}</span>
      <span class="spacer"></span><button class="btn small" id="new-rec">${icon("plus")}New recording</button></div>`;
    $("#new-rec").onclick = () => { S.rec = null; S.analysis = null; S.result = null; S.params = null; go(1); };
  } else cur.innerHTML = "";
  const panel = $("#panel");
  panel.innerHTML = "";
  [null, input, spectrum, source, filter, result][S.step](panel);
}
function busyHtml(msg) { return `<div class="progress" role="status"><span class="spinner"></span>${esc(msg)}</div>`; }
function setBusy(msg) { const el = $("#busy"); if (el) el.innerHTML = msg ? busyHtml(msg) : ""; $$("button, input.file", S.root).forEach((b) => { if (b.id !== "busy") b.disabled = !!msg; }); }
const actions = (back, next) => `<div class="panel-actions"><div>${back || ""}</div><div>${next || ""}</div></div>`;

/* ------------------------------------------------------------------ step 1: input */
function input(panel) {
  const lim = S.health.limits;
  const canRec = Recorder.supported() && window.isSecureContext;
  panel.innerHTML = `
    <div id="busy" aria-live="polite"></div>
    <div class="grid-2">
      <section class="card">
        <div class="card-head"><div><h2>Upload a recording</h2><p>Reso-Noise reads the samples exactly as stored, mixes to mono and normalises the peak level.</p></div></div>
        <label class="dropzone" id="drop" for="file">${icon("upload")}<strong>Choose an audio file or drop it here</strong>
          <span>WAV, FLAC, OGG, MP3, M4A or AIFF, up to ${lim.max_upload_mb} MB and ${lim.max_duration_s} s</span></label>
        <input id="file" class="file" type="file" accept="audio/*,.wav,.flac,.ogg,.oga,.mp3,.m4a,.aac,.aif,.aiff,.opus,.webm" hidden />
      </section>
      <section class="card">
        <div class="card-head"><div><h2>Record with a microphone</h2><p>Browser noise suppression and gain control are switched off, so the signal you analyse is the raw one.</p></div></div>
        <div class="mic-panel" id="mic">
          ${canRec ? `<div class="rec-readout"><span class="rec-time" id="rec-time">0:00</span><div class="meter" id="meter"><i></i></div></div>
          <div class="row"><button class="btn primary" id="rec-start">${icon("mic")}Start recording</button>
          <button class="btn sage" id="rec-stop" hidden>${icon("stop")}Stop and analyse</button><button class="btn ghost" id="rec-cancel" hidden>Cancel</button></div>
          <p class="small muted" id="rec-note">Up to ${lim.max_duration_s} seconds. Your browser will ask for permission.</p>`
          : `<div class="notice warn">${icon("alert")}<span>Recording needs a browser with microphone support and a secure page (https:// or http://localhost). ${window.isSecureContext ? "This browser does not offer it." : "This page is not served securely."} You can still upload a file.</span></div>`}
        </div>
      </section>
    </div>
    <section class="card">
      <div class="card-head"><div><h3>No recording to hand? Try a sample scene</h3>
        <p>Each scene is generated from a physical model of the source. Scenes with a talker also keep the clean voice, so the result can be scored against ground truth.</p></div></div>
      <div class="sample-list">${S.model.samples.map((s) => `<button class="btn small" data-sample="${s.key}">${icon("sparkle")}${esc(s.label)}</button>`).join("")}</div>
    </section>`;

  const file = $("#file"), drop = $("#drop");
  file.onchange = () => file.files[0] && fromFile(file.files[0]);
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && fromFile(e.dataTransfer.files[0]));
  $$("[data-sample]").forEach((b) => (b.onclick = () => ingest(() => api.sample(b.dataset.sample), "Generating the scene")));
  if (canRec) wireRecorder(lim);

  function fromFile(f) {
    if (f.size > lim.max_upload_mb * 1024 * 1024) return toast(`${f.name} is ${fmtSize(f.size)}; the limit is ${lim.max_upload_mb} MB.`, "error");
    if (!f.size) return toast(`${f.name} is empty.`, "error");
    ingest(() => api.upload(f, f.name, "upload"), "Reading the audio");
  }
}

async function ingest(makeRecording, label) {
  setBusy(`${label}…`);
  try {
    const rec = await makeRecording();
    S.rec = rec; S.analysis = null; S.result = null; S.params = null; S.extra = []; S.notchOn = null;
    setBusy("Running the Fourier analysis and classifying the source…");
    S.analysis = await api.analyze(rec.id);
    go(2);
  } catch (e) {
    setBusy(null); toast(e.message, "error");
    if (S.rec && !S.analysis) { S.rec = null; render(); }
  }
}

function wireRecorder(lim) {
  const $t = $("#rec-time"), $m = $("#meter"), start = $("#rec-start"), stop = $("#rec-stop"), cancel = $("#rec-cancel"), note = $("#rec-note");
  const ui = (rec) => { start.hidden = rec; stop.hidden = !rec; cancel.hidden = !rec; };
  const finish = async () => {
    if (!S.recorder) return;
    const r = S.recorder; S.recorder = null;
    try {
      const { blob, seconds } = await r.stop();
      if (seconds < 0.5) { ui(false); return toast("That recording was too short. Record at least a second.", "error"); }
      ingest(() => api.upload(blob, "recording.wav", "recording"), "Uploading the recording");
    } catch (e) { toast(e.message, "error"); ui(false); }
  };
  start.onclick = async () => {
    S.recorder = new Recorder({
      maxSeconds: lim.max_duration_s,
      onLevel: (rms, peak) => { $m.firstElementChild.style.width = `${Math.min(100, rms * 280)}%`; $m.classList.toggle("hot", peak > 0.98); },
      onTick: (t) => ($t.textContent = fmtTime(t)),
      onAutoStop: () => { toast("Reached the maximum length, so recording stopped.", "info"); finish(); },
    });
    try { await S.recorder.start(); ui(true); note.innerHTML = `<span class="rec-dot" style="display:inline-block;margin-right:8px"></span>Recording. Aim for the noise you want to analyse, 5 to 30 seconds.`; }
    catch (e) { S.recorder = null; toast(e.message, "error"); }
  };
  stop.onclick = finish;
  cancel.onclick = () => { S.recorder && S.recorder.cancel(); S.recorder = null; ui(false); $t.textContent = "0:00"; note.textContent = "Recording discarded."; };
}

/* ------------------------------------------------------------------ shared pieces */
function waveCard(panel, { withProcessed = false, pid = null } = {}) {
  const id = "wave-" + Math.random().toString(36).slice(2, 7);
  const html = `<section class="card"><div class="card-head"><div><h2>Waveform</h2><p>Amplitude over time, drawn from the audio samples. Click to jump, drag to pan when zoomed.</p></div>
    <div class="chart-tools"><button class="btn small icon" data-z="in" aria-label="Zoom in" title="Zoom in">${icon("plus")}</button>
    <button class="btn small" data-z="out">Zoom out</button><button class="btn small" data-z="reset">Show all</button></div></div>
    <div id="${id}"></div>
    ${withProcessed ? `<div class="legend"><span><i style="background:var(--lav)"></i>Original</span><span><i style="background:var(--plum)"></i>Processed</span></div>` : ""}
    <p class="hint">Ctrl or ⌘ + scroll zooms around the pointer. Double-click shows the whole recording.</p>
    <div id="${id}-player" style="margin-top:14px"></div></section>`;
  panel.insertAdjacentHTML("beforeend", html);
  const box = $("#" + id), rec = S.rec;
  const layers = [{ key: "o", color: cssVar("--lav-2"), alpha: 0.95, fetch: (a, b, n) => api.waveform(rec.id, a, b, n) }];
  if (withProcessed) layers.push({ key: "p", color: cssVar("--plum"), alpha: 0.85, fetch: (a, b, n) => api.waveform(rec.id, a, b, n, pid) });
  const chart = track(new WaveformChart(box, { onSeek: (t) => S.player && S.player.seek(t) }));
  chart.setData(rec.duration, layers);
  panel.querySelectorAll(`[data-z]`).forEach((b) => (b.onclick = () => ({ in: () => chart.zoom(0.5), out: () => chart.zoom(2), reset: () => chart.reset() })[b.dataset.z]()));
  S.player = new Player($(`#${id}-player`), { onTime: (t) => { chart.setPlayhead(t); S.spectrograms && S.spectrograms.forEach((s) => s.setPlayhead(t)); } });
  S.player.load({ original: api.audioUrl(rec.id), processed: withProcessed ? api.processedUrl(pid) : null });
  return chart;
}

function spectrumCard(panel, { compare = false, notches = false, title, subtitle }) {
  const id = "spec-" + Math.random().toString(36).slice(2, 7);
  panel.insertAdjacentHTML("beforeend", `<section class="card"><div class="card-head"><div><h2>${title}</h2><p>${subtitle}</p></div>
    <div class="chart-tools"><div class="segmented" data-g="view"><button aria-pressed="false" data-v="fft" title="Magnitude of one Fourier transform over the whole recording">Full-signal DFT</button>
    <button aria-pressed="true" data-v="avg" title="Welch average of windowed transforms">Averaged</button></div>
    <div class="segmented" data-g="scale"><button aria-pressed="true" data-v="log">Log</button><button aria-pressed="false" data-v="lin">Linear</button></div></div></div>
    <div id="${id}"></div>
    ${notches ? `<div class="legend"><span><i style="background:var(--lav-2)"></i>Level of the original (dB re full scale)</span><span><i style="background:var(--sage)"></i>Bands the notches will cut</span></div>` : compare ? `<div class="legend"><span><i style="background:var(--lav)"></i>Original</span><span><i style="background:var(--plum)"></i>Processed</span><span><i style="background:var(--sage)"></i>Notched bands</span></div>`
      : `<div class="legend"><span><i style="background:var(--plum)"></i>Level (dB re full scale)</span><span><i style="background:var(--plum-2);opacity:.5"></i>Dashed lines mark tonal peaks</span></div>`}
    <p class="hint">Drag across the plot to zoom into a frequency range. Double-click to reset.</p></section>`);
  return { id, chart: track(new SpectrumChart($("#" + id), { height: 300 })), root: panel.lastElementChild };
}
function bindSpectrumControls(card, apply) {
  const st = { view: "avg", scale: "log" };
  $$(".segmented", card.root).forEach((seg) => seg.addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    $$("button", seg).forEach((x) => x.setAttribute("aria-pressed", x === b));
    st[seg.dataset.g] = b.dataset.v; apply(st);
  }));
  apply(st);
}

/* ------------------------------------------------------------------ step 2: spectrum */
function spectrum(panel) {
  const a = S.analysis, f = a.fourier;
  waveCard(panel);
  const card = spectrumCard(panel, { title: "Frequency spectrum", subtitle: "The Fourier transform X(f) of your recording. Narrow spikes are tones such as hum; broad humps are noise." });
  const peaks = [...a.tones].sort((x, y) => y.prominence_db - x.prominence_db).slice(0, 8);
  bindSpectrumControls(card, (st) => {
    const sp = f.spectrum;
    card.chart.setScale(st.scale);
    card.chart.setSeries([{ name: "", freq: sp.freq, values: st.view === "fft" ? sp.fft_db : sp.avg_db, color: cssVar("--plum"), fill: true }]);
    card.chart.setMarkers(peaks.map((p) => ({ freq: p.freq_hz, label: `${Math.round(p.freq_hz)}`, color: cssVar("--plum-2") })));
  });
  panel.insertAdjacentHTML("beforeend", `<section class="card"><div class="card-head"><div><h2>Spectrogram</h2><p>Frequency content as it changes over time. Steady horizontal lines are hum; vertical smears are impulsive sounds.</p></div></div><div id="spg"></div></section>`);
  const spg = track(new SpectrogramChart($("#spg"))); spg.setImage(a.spectrogram); S.spectrograms = [spg];

  const pe = f.parseval, h = f.harmonic, d = f.dominant;
  const dom = d[0] || { freq_hz: f.summary.peak_hz };
  panel.insertAdjacentHTML("beforeend", `<section class="card"><div class="card-head"><div><h2>What the transform found</h2><p>Numbers derived from the spectrum above. Parseval's identity confirms that no energy is lost between the time and frequency domains.</p></div></div>
    <div class="facts">
      <div class="fact"><div class="k">Strongest tone</div><div class="v">${fmtHz(dom.freq_hz)}</div><div class="n">${d.length ? `${d.length} tonal peaks found` : "no clear tonal peaks"}</div></div>
      <div class="fact"><div class="k">Spectral centroid</div><div class="v">${fmtHz(f.summary.centroid_hz)}</div><div class="n">centre of mass of the energy</div></div>
      <div class="fact"><div class="k">85% roll-off</div><div class="v">${fmtHz(f.summary.rolloff85_hz)}</div><div class="n">below this lies 85% of the power</div></div>
      <div class="fact"><div class="k">Harmonic series</div><div class="v">${h ? fmtHz(h.f0_hz) : "none"}</div><div class="n">${h ? `fundamental, ${h.harmonics.length} steady harmonics` : "no steady harmonic ladder"}</div></div>
      <div class="fact"><div class="k">Parseval check</div><div class="v">${pe.relative_error < 1e-9 ? "Exact" : pe.relative_error.toExponential(1)}</div><div class="n">time ${pe.time_energy.toPrecision(6)} vs frequency ${pe.frequency_energy.toPrecision(6)}</div></div>
      <div class="fact"><div class="k">Resolution</div><div class="v">${f.resolution_hz.toFixed(3)} Hz</div><div class="n">${f.n_fft.toLocaleString()}-point transform</div></div>
    </div></section>
    <div class="grid-2"><section class="card"><h3 style="margin-bottom:12px">Tonal peaks</h3>${a.tones.length ? `<div class="table-wrap"><table class="data"><thead><tr><th>Frequency</th><th class="num-r">Above floor</th><th>Behaviour</th></tr></thead><tbody>
      ${[...a.tones].sort((x, y) => x.freq_hz - y.freq_hz).map((t) => `<tr><td>${fmtHz(t.freq_hz)}</td><td class="num-r">${t.prominence_db.toFixed(1)} dB</td><td>${t.stationary ? '<span class="badge sage">Steady</span>' : '<span class="badge amber">Varies</span>'}</td></tr>`).join("")}</tbody></table></div>
      <p class="hint">Steady peaks are candidates for notch filtering. Peaks that change over time, like voice harmonics, are left alone.</p>` : '<p class="muted">No narrow peaks stand out, so this noise is broadband.</p>'}</section>
    <section class="card"><h3 style="margin-bottom:12px">Where the energy sits</h3><div class="table-wrap"><table class="data"><thead><tr><th>Band</th><th>Share</th><th class="num-r">%</th></tr></thead><tbody>
      ${f.bands.map((b) => `<tr><td>${b.lo >= 1000 ? b.lo / 1000 + "k" : b.lo}–${b.hi >= 1000 ? b.hi / 1000 + "k" : b.hi} Hz</td><td style="width:50%"><div class="prob-bar"><i style="width:${(b.share * 100).toFixed(1)}%"></i></div></td><td class="num-r">${(b.share * 100).toFixed(1)}</td></tr>`).join("")}</tbody></table></div></section></div>
    ${actions(`<button class="btn" id="back">${icon("chevronL")}Input</button>`, `<button class="btn primary" id="next">Identify the source${icon("chevron")}</button>`)}`);
  $("#back").onclick = () => go(1); $("#next").onclick = () => go(3);
}

/* ------------------------------------------------------------------ step 3: source */
function probRows(list, n = 5) {
  return list.slice(0, n).map((p, i) => `<div class="prob-row ${i === 0 ? "top" : ""}"><span>${esc(p.label)}</span><div class="prob-bar"><i style="width:${(p.prob * 100).toFixed(1)}%"></i></div><span class="pct">${(p.prob * 100).toFixed(0)}%</span></div>`).join("");
}
function source(panel) {
  const c = S.analysis.classification, o = c.overall, bg = c.background, rec = c.recommended;
  const info = S.model.classes.find((x) => x.key === rec.key);
  const m = S.model;
  panel.innerHTML = `
    <section class="card"><div class="verdict">
      <div class="verdict-main"><div class="group">${esc(info.group)}${rec.source === "user" ? " · set by you" : ""}</div>
        <div class="label">${esc(info.label)}</div>
        <p class="reason">${esc(rec.reason)}</p>
        <p class="small muted" style="margin-top:12px">${esc(info.description)}</p>
        ${o.uncertain && rec.source !== "user" ? `<div class="notice warn" style="margin-top:14px">${icon("alert")}<span>The classifier is not sure about the whole clip (top choice ${(o.confidence * 100).toFixed(0)}%). Check the filter settings before applying them.</span></div>` : ""}</div>
      <div><h3 style="margin-bottom:6px">Whole recording</h3>${probRows(o.probabilities)}
        ${bg ? `<h3 style="margin:18px 0 6px">Quiet moments only</h3>${bg.significant ? "" : '<p class="small muted">The quiet parts are far below the loud parts, so this background is probably not a problem.</p>'}${probRows(bg.probabilities, 3)}` : ""}
      </div></div></section>
    <div class="grid-2">
      <section class="card"><h3>What drove the decision</h3><p class="small muted" style="margin:4px 0 12px">Measured features that most favour “${esc(o.label)}” over “${esc(S.model.classes.find((x) => x.key === o.runner_up).label)}”.</p>
        ${o.drivers.length ? `<ul class="drivers">${o.drivers.map((d) => `<li><span style="flex:1">${esc(d.label)}</span><span class="prob-bar" style="width:90px;align-self:center"><i style="width:${Math.max(8, (d.contribution / o.drivers[0].contribution) * 100).toFixed(0)}%;background:var(--sage)"></i></span></li>`).join("")}</ul>` : '<p class="muted">No single feature stands out.</p>'}
        <p class="hint">The classifier is ${esc(m.algorithm)} on ${m.n_features} Fourier-derived features. Trained on ${m.n_synthetic_train} modelled clips${m.n_user_samples ? ` plus ${m.n_user_samples} you labelled` : ""}; held-out accuracy ${(m.validation_accuracy * 100).toFixed(0)}% on modelled sounds, which is easier than real rooms.</p></section>
      <section class="card"><h3>Not what you hear?</h3><p class="small muted" style="margin:4px 0 12px">Tell Reso-Noise which source you want to remove. The filter follows your choice now, and the correction is stored so you can retrain the classifier on the Method page.</p>
        <label class="field" for="fix">Noise source</label>
        <div class="inline-add"><select id="fix" style="max-width:240px">${S.model.classes.map((x) => `<option value="${x.key}" ${x.key === rec.key ? "selected" : ""}>${esc(x.label)}</option>`).join("")}</select>
        <button class="btn" id="save-label">Use this source</button></div></section></div>
    ${actions(`<button class="btn" id="back">${icon("chevronL")}Spectrum</button>`, `<button class="btn primary" id="next">Design the filter${icon("chevron")}</button>`)}`;
  $("#back").onclick = () => go(2);
  $("#next").onclick = () => { S.params = null; go(4); };
  $("#save-label").onclick = async () => {
    try { await api.label(S.rec.id, $("#fix").value); S.analysis = await api.analyze(S.rec.id); S.params = null; toast("Saved. The filter will target this source.", "good"); render(); }
    catch (e) { toast(e.message, "error"); }
  };
}

/* ------------------------------------------------------------------ step 4: filter */
const logToVal = (v, lo, hi) => lo * (hi / lo) ** (v / 1000);
const valToLog = (f, lo, hi) => Math.round((Math.log(f / lo) / Math.log(hi / lo)) * 1000);

function filter(panel) {
  const a = S.analysis, sr = S.rec.sample_rate, nyq = sr / 2, rec = a.classification.recommended;
  if (!S.params) { S.params = structuredClone(a.filter_defaults[rec.key]); S.profileKey = rec.key; S.extra = []; S.notchOn = S.params.notch_freqs.length > 0; }
  const P = S.params, hpMax = 1000, lpMin = 1000, lpMax = nyq * 0.98;
  const tones = [...a.tones].sort((x, y) => x.freq_hz - y.freq_hz);
  const near = (f) => P.notch_freqs.some((x) => Math.abs(x - f) < 0.6);
  const chips = [...tones.map((t) => ({ f: t.freq_hz, lvl: t.prominence_db, steady: t.stationary })), ...S.extra.map((f) => ({ f, lvl: null, steady: true, manual: true }))];
  const profiles = S.model.profiles;
  panel.innerHTML = `
    <div id="busy"></div>
    <div class="filter-layout">
      <section class="card">
        <div class="card-head"><div><h2>Filter settings</h2></div></div>
        <label class="field" for="profile">Start from</label>
        <select id="profile">${profiles.map((p) => `<option value="${p.key}" ${S.profileKey === p.key ? "selected" : ""}>${esc(p.label)}${p.key === rec.key ? " (recommended)" : ""}</option>`).join("")}<option value="custom" ${S.profileKey === "custom" ? "selected" : ""}>Custom</option></select>
        <p class="small muted" id="profile-note" style="margin-top:8px">${esc((profiles.find((p) => p.key === S.profileKey) || { summary: "Your own combination of the filters below." }).summary)}</p>
        <div class="ctl"><div class="ctl-head"><b>High-pass</b><span class="ctl-val" id="hp-val"></span><label class="switch"><input type="checkbox" id="hp-on" ${P.highpass_hz ? "checked" : ""} aria-label="High-pass on" /><i></i></label></div>
          <input type="range" id="hp" min="0" max="1000" value="${valToLog(P.highpass_hz || 80, 20, hpMax)}" aria-label="High-pass cutoff" /></div>
        <div class="ctl"><div class="ctl-head"><b>Low-pass</b><span class="ctl-val" id="lp-val"></span><label class="switch"><input type="checkbox" id="lp-on" ${P.lowpass_hz ? "checked" : ""} aria-label="Low-pass on" /><i></i></label></div>
          <input type="range" id="lp" min="0" max="1000" value="${valToLog(P.lowpass_hz || Math.min(8000, lpMax), lpMin, lpMax)}" aria-label="Low-pass cutoff" /></div>
        <div class="ctl"><div class="ctl-head"><b>Notches on tones</b><span class="ctl-val" id="nc-val"></span><label class="switch"><input type="checkbox" id="nc-on" ${S.notchOn ? "checked" : ""} aria-label="Notches on" /><i></i></label></div>
          <div class="tone-list" id="tones">${chips.length ? chips.map((c) => `<label class="tone ${c.steady ? "" : "varies"}"><input type="checkbox" data-f="${c.f}" ${near(c.f) ? "checked" : ""} /><span>${fmtHz(c.f)}<small>${c.manual ? "added by you" : `${c.lvl.toFixed(0)} dB · ${c.steady ? "steady" : "varies"}`}</small></span></label>`).join("") : '<span class="small muted">No tonal peaks were found.</span>'}</div>
          <div class="inline-add"><input type="number" id="add-hz" min="5" max="${Math.floor(nyq)}" step="0.1" placeholder="Hz" aria-label="Add notch frequency" /><button class="btn small" id="add-btn">${icon("plus")}Add</button></div>
          <label class="field" style="margin-top:12px">Depth <span class="ctl-val" id="depth-val"></span></label><input type="range" id="depth" min="6" max="60" step="1" value="${P.notch_depth_db}" />
          <label class="field" style="margin-top:8px">Width <span class="ctl-val" id="width-val"></span></label><input type="range" id="width" min="1" max="40" step="0.5" value="${P.notch_width_hz}" /></div>
        <div class="ctl"><div class="ctl-head"><b>Adaptive spectral gate</b><span class="ctl-val" id="gate-val"></span></div>
          <input type="range" id="gate" min="0" max="100" value="${Math.round(P.gate_strength * 100)}" aria-label="Gate strength" />
          <p class="hint">Learns the noise spectrum from the quietest moments and turns down time-frequency cells that match it.</p></div>
      </section>
      <div class="stack">
        <div id="chart-slot"></div>
        <section class="card"><div class="card-head"><div><h3>Filter response H(f)</h3><p>The gain applied to each frequency before the inverse transform. The spectral gate then adapts on top of this curve.</p></div></div><div id="resp"></div></section>
      </div>
    </div>
    ${actions(`<button class="btn" id="back">${icon("chevronL")}Source</button>`, `<button class="btn primary" id="apply">${icon("filter")}Apply and reconstruct</button>`)}`;

  const card = spectrumCard($("#chart-slot"), { notches: true, title: "Spectrum and notches", subtitle: "Shaded bands show where notches will cut." });
  const resp = track(new SpectrumChart($("#resp"), { height: 200, yMin: -66, yMax: 6, yUnit: "dB" }));
  let spectrumState = { view: "avg", scale: "log" };
  const link = (src, dst) => { src.onXRange = (a, b) => dst.setXRange(a, b); };
  link(card.chart, resp); link(resp, card.chart);
  bindSpectrumControls(card, (st) => {
    spectrumState = st;
    card.chart.setScale(st.scale); resp.setScale(st.scale);
    card.chart.setSeries([{ name: "", freq: a.fourier.spectrum.freq, values: st.view === "fft" ? a.fourier.spectrum.fft_db : a.fourier.spectrum.avg_db, color: cssVar("--lav-2"), fill: true }]);
  });

  const read = () => {
    const chosen = $$("#tones input:checked").map((i) => +i.dataset.f);
    const p = {
      profile: $("#profile").value,
      highpass_hz: $("#hp-on").checked ? +logToVal($("#hp").value, 20, hpMax).toFixed(1) : null,
      lowpass_hz: $("#lp-on").checked ? +logToVal($("#lp").value, lpMin, lpMax).toFixed(0) : null,
      notch_freqs: $("#nc-on").checked ? chosen.map((f) => +f.toFixed(2)) : [],
      notch_depth_db: +$("#depth").value, notch_width_hz: +$("#width").value, gate_strength: $("#gate").value / 100,
    };
    return p;
  };
  const labels = () => {
    const p = read();
    $("#hp-val").textContent = p.highpass_hz ? fmtHz(p.highpass_hz) : "Off";
    $("#lp-val").textContent = p.lowpass_hz ? fmtHz(p.lowpass_hz) : "Off";
    $("#nc-val").textContent = p.notch_freqs.length ? `${p.notch_freqs.length} active` : "Off";
    $("#depth-val").textContent = `${p.notch_depth_db} dB`; $("#width-val").textContent = `${p.notch_width_hz} Hz`;
    $("#gate-val").textContent = p.gate_strength ? `${Math.round(p.gate_strength * 100)}%` : "Off";
    return p;
  };
  const preview = debounce(async () => {
    const p = S.params;
    card.chart.setBands(p.notch_freqs.map((f) => ({ lo: f - p.notch_width_hz, hi: f + p.notch_width_hz })));
    try {
      const r = await api.preview(S.rec.id, p);
      resp.setSeries([{ name: "Gain", freq: r.freq, values: r.gain_db, color: cssVar("--sage-2"), fill: true, fillAlpha: 0.18, width: 2 }]);
    } catch (e) { toast(e.message, "error"); }
  }, 120);
  const changed = (fromUser = true) => {
    if (fromUser && $("#profile").value !== "custom") { $("#profile").value = "custom"; S.profileKey = "custom"; $("#profile-note").textContent = "Your own combination of the filters below."; }
    S.params = labels(); S.notchOn = $("#nc-on").checked; preview();
  };
  $$("input[type=range], input[type=checkbox]", panel).forEach((el) => el.addEventListener("input", () => changed(true)));
  $("#tones").addEventListener("change", () => { if (!$("#nc-on").checked && $$("#tones input:checked").length) $("#nc-on").checked = true; changed(true); });
  $("#profile").onchange = () => {
    const k = $("#profile").value;
    if (k !== "custom") { S.params = structuredClone(a.filter_defaults[k]); S.profileKey = k; S.extra = []; S.notchOn = S.params.notch_freqs.length > 0; filterKeepCharts(); }
  };
  const filterKeepCharts = () => { render(); };
  $("#add-btn").onclick = () => {
    const f = +$("#add-hz").value;
    if (!f || f < 5 || f >= nyq) return toast(`Enter a frequency between 5 Hz and ${Math.floor(nyq)} Hz.`, "error");
    S.params = read(); S.extra.push(f); S.params.notch_freqs.push(f); S.notchOn = true; S.params.profile = "custom"; S.profileKey = "custom"; render();
  };
  labels(); S.params.profile = $("#profile").value; changed(false);
  S.params = read();
  $("#back").onclick = () => go(3);
  $("#apply").onclick = async () => {
    const p = read();
    if (!p.highpass_hz && !p.lowpass_hz && !p.notch_freqs.length && !p.gate_strength) return toast("Switch on at least one filter first.", "error");
    S.params = p;
    setBusy("Applying H(f), then reconstructing with the inverse transform…");
    try { S.result = await api.process(S.rec.id, p); go(5); }
    catch (e) { setBusy(null); toast(e.message, "error"); }
  };
}

/* ------------------------------------------------------------------ step 5: result */
function result(panel) {
  const R = S.result, m = R.metrics, a = S.analysis, pid = R.processed_id;
  const floor = m.noise_floor_reduction_db, pct = m.energy_removed_pct;
  const head = floor >= 1
    ? `The noise floor dropped by ${floor.toFixed(1)} dB and ${pct.toFixed(0)}% of the signal energy was removed.`
    : `The overall level changed by ${fmtDb(m.level_change_db)} and ${pct.toFixed(0)}% of the signal energy was removed.`;
  panel.insertAdjacentHTML("beforeend", `<section class="card"><p class="headline">${head}</p>
    <p class="muted" style="margin-top:10px">${m.reference ? `Measured against the clean voice that was kept with this sample, the signal-to-distortion ratio improved by ${fmtDb(m.reference.improvement_db)}. ` : ""}Listen to both versions below and compare them on the waveform and spectrum.</p></section>`);
  waveCard(panel, { withProcessed: true, pid });

  const card = spectrumCard(panel, { compare: true, title: "Before and after", subtitle: "Spectrum of the original against the reconstructed signal." });
  const notches = R.params.notch_freqs;
  bindSpectrumControls(card, (st) => {
    const sp = R.spectrum, k = st.view === "fft" ? ["original_fft_db", "processed_fft_db"] : ["original_avg_db", "processed_avg_db"];
    card.chart.setScale(st.scale);
    card.chart.setSeries([
      { name: "Original", freq: sp.freq, values: sp[k[0]], color: cssVar("--lav"), fill: true, fillAlpha: 0.3, width: 1.5, tipColor: "#d9cde8" },
      { name: "Processed", freq: sp.freq, values: sp[k[1]], color: cssVar("--plum"), fill: true, fillAlpha: 0.14, width: 1.8 }]);
    card.chart.setBands(notches.map((f) => ({ lo: f - R.params.notch_width_hz / 2, hi: f + R.params.notch_width_hz / 2 })));
  });

  panel.insertAdjacentHTML("beforeend", `<section class="card"><div class="card-head"><div><h2>Spectrograms</h2><p>Same colour scale for both, so the difference you see is the real difference.</p></div></div>
    <div class="spec-pair"><div><h3>Original</h3><div id="sg-o"></div></div><div><h3>Processed</h3><div id="sg-p"></div></div></div></section>`);
  const so = track(new SpectrogramChart($("#sg-o"), { height: 220 })), sp = track(new SpectrogramChart($("#sg-p"), { height: 220 }));
  so.setImage(a.spectrogram); sp.setImage(R.spectrogram); S.spectrograms = [so, sp];

  const tile = (k, v, n, good) => `<div class="metric"><div class="k">${k}</div><div class="v ${good ? "good" : ""}">${v}</div><div class="n">${n}</div></div>`;
  const pe = R.parseval;
  panel.insertAdjacentHTML("beforeend", `<div class="metric-grid" style="margin-top:18px">
    ${tile("Noise floor", fmtDb(-floor), `${m.noise_floor_in_db.toFixed(1)} → ${m.noise_floor_out_db.toFixed(1)} dBFS in the quietest 10% of frames`, floor > 3)}
    ${tile("Overall level", fmtDb(m.level_change_db), `${m.rms_in_db.toFixed(1)} → ${m.rms_out_db.toFixed(1)} dBFS RMS`)}
    ${tile("Energy removed", `${pct.toFixed(1)}%`, `Parseval: output energy ${pe.relative_error < 1e-9 ? "matches" : "differs from"} its spectrum`)}
    ${tile("Waveform correlation", m.correlation.toFixed(3), "1.000 means unchanged shape")}
    ${tile("Spectral distance", `${m.log_spectral_distance_db.toFixed(1)} dB`, "RMS difference of the two spectra, 20 Hz to 8 kHz")}
    ${tile("Speech band 0.3–3.4 kHz", fmtDb(m.voice_band_change_db), "how much of the voice range was touched")}
    ${tile("Loud-to-floor range", `${m.signal_to_floor_out_db.toFixed(1)} dB`, `was ${m.signal_to_floor_in_db.toFixed(1)} dB (${fmtDb(m.signal_to_floor_gain_db)})`, m.signal_to_floor_gain_db > 1)}
    ${m.reference ? tile("SI-SDR vs clean voice", fmtDb(m.reference.improvement_db), `${m.reference.si_sdr_in_db.toFixed(1)} → ${m.reference.si_sdr_out_db.toFixed(1)} dB`, m.reference.improvement_db > 1) : ""}
  </div>`);

  panel.insertAdjacentHTML("beforeend", `<div class="grid-2" style="margin-top:18px"><section class="card"><div class="card-head"><div><h3>Level by frequency band</h3></div></div><div id="bands"></div>
    <div class="legend"><span><i style="background:var(--lav)"></i>Before</span><span><i style="background:var(--plum)"></i>After</span></div></section>
    <section class="card"><h3 style="margin-bottom:12px">Key tones</h3>${m.tracked_peaks.length ? `<div class="table-wrap"><table class="data"><thead><tr><th>Frequency</th><th class="num-r">Before</th><th class="num-r">After</th><th class="num-r">Change</th></tr></thead><tbody>
      ${m.tracked_peaks.map((t) => `<tr><td>${fmtHz(t.freq_hz)}</td><td class="num-r">${t.original_db.toFixed(1)}</td><td class="num-r">${t.processed_db.toFixed(1)}</td><td class="num-r"><b>${fmtDb(t.change_db)}</b></td></tr>`).join("")}</tbody></table></div>` : '<p class="muted">No tonal peaks to track.</p>'}
      <h3 style="margin:20px 0 10px">Filter applied</h3><ul class="drivers">
      <li><span>High-pass</span><b>${R.params.highpass_hz ? fmtHz(R.params.highpass_hz) : "off"}</b></li><li><span>Low-pass</span><b>${R.params.lowpass_hz ? fmtHz(R.params.lowpass_hz) : "off"}</b></li>
      <li><span>Notches</span><b>${notches.length ? `${notches.length} × ${R.params.notch_depth_db} dB` : "none"}</b></li><li><span>Spectral gate</span><b>${R.params.gate_strength ? Math.round(R.params.gate_strength * 100) + "%" : "off"}</b></li></ul>
      <p class="hint">Computed in ${R.elapsed_ms} ms${R.output_gain_db ? `; output scaled by ${fmtDb(R.output_gain_db)} to avoid clipping` : ""}.</p></section></div>`);
  track(new BandChart($("#bands"))).setBands(m.bands);

  panel.insertAdjacentHTML("beforeend", `<section class="card" style="margin-top:18px"><div class="card-head"><div><h2>Take it with you</h2><p>The processed file is the exact signal analysed above, saved as 16-bit WAV at ${S.rec.sample_rate} Hz.</p></div></div>
    <div class="row"><button class="btn primary" data-dl="${api.processedUrl(pid, true)}">${icon("download")}Processed audio (WAV)</button>
    <button class="btn" data-dl="${api.audioUrl(S.rec.id, true)}">${icon("download")}Original (WAV)</button>
    <button class="btn" data-dl="${api.reportUrl(pid)}">${icon("download")}Analysis report (JSON)</button></div></section>
    ${actions(`<button class="btn" id="back">${icon("sliders")}Adjust the filter</button>`, `<button class="btn" id="again">${icon("plus")}Another recording</button>`)}`);
  $$("[data-dl]").forEach((b) => (b.onclick = () => download(b.dataset.dl)));
  $("#back").onclick = () => go(4);
  $("#again").onclick = () => { S.rec = null; S.analysis = null; S.result = null; S.params = null; go(1); };
}
