// Method view: the mathematics, the pipeline and the classifier's model card.
import { api } from "./api.js";
import { icon } from "./icons.js";
import { $, esc, toast } from "./ui.js";

const m = (s) => `<math display="block"><mrow>${s}</mrow></math>`;
const EQ = [
  { t: "Fourier transform", math: m('<mi>X</mi><mo>(</mo><mi>f</mi><mo>)</mo><mo>=</mo><mo>∫</mo><mi>x</mi><mo>(</mo><mi>t</mi><mo>)</mo><msup><mi>e</mi><mrow><mo>−</mo><mi>j</mi><mn>2</mn><mi>π</mi><mi>f</mi><mi>t</mi></mrow></msup><mi>d</mi><mi>t</mi>'),
    p: "Turns the samples of the recording into a spectrum. Computed as a fast Fourier transform of the whole signal.", w: "fourier.forward_transform" },
  { t: "Magnitude spectrum", math: m('<mo>|</mo><mi>X</mi><mo>(</mo><mi>f</mi><mo>)</mo><mo>|</mo><mo>=</mo><msqrt><msup><mrow><mi>Re</mi><mo>(</mo><mi>X</mi><mo>)</mo></mrow><mn>2</mn></msup><mo>+</mo><msup><mrow><mi>Im</mi><mo>(</mo><mi>X</mi><mo>)</mo></mrow><mn>2</mn></msup></msqrt>'),
    p: "How strongly each frequency is present. This is what the spectrum chart draws and what peak detection reads.", w: "fourier.magnitude_spectrum" },
  { t: "Parseval's identity", math: m('<mo>∫</mo><msup><mrow><mo>|</mo><mi>x</mi><mo>(</mo><mi>t</mi><mo>)</mo><mo>|</mo></mrow><mn>2</mn></msup><mi>d</mi><mi>t</mi><mo>=</mo><mo>∫</mo><msup><mrow><mo>|</mo><mi>X</mi><mo>(</mo><mi>f</mi><mo>)</mo><mo>|</mo></mrow><mn>2</mn></msup><mi>d</mi><mi>f</mi>'),
    p: "Energy is the same in both domains. Reso-Noise checks this on every recording and uses it to report how much energy a filter removed.", w: "fourier.parseval_check" },
  { t: "Convolution theorem", math: m('<mi mathvariant="script">F</mi><mo>{</mo><mi>x</mi><mo>(</mo><mi>t</mi><mo>)</mo><mo>∗</mo><mi>h</mi><mo>(</mo><mi>t</mi><mo>)</mo><mo>}</mo><mo>=</mo><mi>X</mi><mo>(</mo><mi>f</mi><mo>)</mo><mi>H</mi><mo>(</mo><mi>f</mi><mo>)</mo>'),
    p: "Filtering in time is multiplication in frequency. The filter H(f) is built from high-pass, low-pass and notch shapes and multiplied into X(f).", w: "filtering.apply_frequency_filter" },
  { t: "Inverse Fourier transform", math: m('<mi>y</mi><mo>(</mo><mi>t</mi><mo>)</mo><mo>=</mo><mo>∫</mo><mi>Y</mi><mo>(</mo><mi>f</mi><mo>)</mo><msup><mi>e</mi><mrow><mi>j</mi><mn>2</mn><mi>π</mi><mi>f</mi><mi>t</mi></mrow></msup><mi>d</mi><mi>f</mi>'),
    p: "Rebuilds the cleaned waveform from the filtered spectrum. This is the audio you hear and download.", w: "fourier.inverse_transform" },
];
const PIPE = [
  ["Acquire and prepare", "Upload or record, convert to mono, remove DC offset, normalise the peak.", "audio_io.py"],
  ["Fourier analysis", "Spectrum, Welch averages, peaks, harmonic series, Parseval check, spectrogram.", "fourier.py, visuals.py"],
  ["Feature extraction", "25 level-independent features: band energies, tonality, harmonicity, modulation, pitch.", "features.py"],
  ["Classification", "Softmax regression picks fan, air conditioner, traffic, excavator or people talking.", "classifier.py"],
  ["Selective filtering", "Notches only on steady tones, band limits, then an adaptive spectral gate.", "filtering.py"],
  ["Inverse transform", "Reconstruct the audio, quantise to 16 bits, compare with the original.", "metrics.py, service.py"],
];

export async function renderMethod(root, { model, onModel }) {
  root.innerHTML = `<div class="studio-head"><h1>How it works</h1><p>Every number in the Studio is computed from your audio with the mathematics below. Nothing is simulated or pre-recorded.</p></div>
    <section class="card"><div class="card-head"><div><h2>Mathematical foundation</h2></div></div><div class="eq-list">
      ${EQ.map((e) => `<div class="eq"><h3>${e.t}</h3>${e.math}<p>${e.p}</p><div class="where">${e.w}</div></div>`).join("")}</div></section>
    <section class="card"><div class="card-head"><div><h2>The pipeline</h2></div></div><div class="pipeline">
      ${PIPE.map(([t, p, f]) => `<div class="pipe"><b>${t}</b><p>${p}</p><p class="small" style="color:var(--sage-2);margin-top:6px">${f}</p></div>`).join("")}</div></section>
    <section class="card" id="model-card"></section>
    <section class="card"><div class="card-head"><div><h2>What to keep in mind</h2></div></div><div class="prose">
      <p>The classifier learns from physical models of each source plus any corrections you save. It is accurate on those models, but a real room adds echo, other sources and microphone character, so treat its answer as a good starting point and use the correction control when it is wrong.</p>
      <p>Notch filters are placed only on tones that stay steady over time, so a voice is not notched. Broadband noise is reduced with the adaptive gate, which can thin out very quiet speech when set high.</p>
      <p>The processed file is a 16-bit WAV. Metrics are computed on exactly those quantised samples.</p></div></section>`;
  card(model);

  function card(mod) {
    const acc = mod.validation_accuracy, conf = mod.confusion, cls = mod.classes;
    $("#model-card").innerHTML = `<div class="card-head"><div><h2>The classifier</h2><p>${esc(mod.algorithm)} on ${mod.n_features} features. Trained ${esc(mod.trained_at || "")} on ${mod.n_synthetic_train} modelled clips${mod.n_user_samples ? ` and ${mod.n_user_samples} of your corrections` : ""}.</p></div>
      <button class="btn" id="retrain">${icon("brain")}Retrain with my corrections</button></div>
      <div class="facts" style="margin-bottom:16px"><div class="fact"><div class="k">Held-out accuracy</div><div class="v">${acc != null ? (acc * 100).toFixed(1) + "%" : "n/a"}</div><div class="n">${mod.validation_n} unseen modelled clips</div></div>
      <div class="fact"><div class="k">Your corrections</div><div class="v">${mod.n_user_samples}</div><div class="n">${mod.custom ? "included in the active model" : "none applied yet"}</div></div></div>
      ${conf ? `<div class="table-wrap"><table class="confusion"><thead><tr><th></th>${cls.map((c) => `<th>${esc(c.label)}</th>`).join("")}</tr></thead><tbody>
        ${conf.map((row, i) => { const tot = row.reduce((a, b) => a + b, 0) || 1; return `<tr><th style="text-align:right">${esc(cls[i].label)}</th>${row.map((v, j) => `<td style="background:rgba(75,31,82,${(0.08 + 0.7 * v / tot).toFixed(2)});color:${v / tot > 0.45 ? "#fff" : "var(--ink)"}">${v}</td>`).join("")}</tr>`; }).join("")}</tbody></table></div>
        <p class="hint">Rows are the true source, columns the prediction.</p>` : ""}`;
    $("#retrain").onclick = async (e) => {
      e.target.disabled = true;
      try { const info = await api.retrain(); const full = await api.model(); onModel(full); toast("The classifier was retrained with your corrections.", "good"); card(full); }
      catch (err) { toast(err.message, "error"); e.target.disabled = false; }
    };
  }
}
