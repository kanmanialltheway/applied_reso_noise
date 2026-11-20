# Reso-Noise

**Fourier-transform based analysis and reduction of residential noise.**

Reso-Noise is a complete audio-analysis application. You give it a recording (upload or microphone); it computes the
Fourier spectrum, works out which household or street source the noise comes from, removes that noise with filters shaped
to the source, rebuilds the audio with the inverse transform, and measures what changed. Everything is computed from your
actual samples. There is no simulated processing anywhere in the app.

It implements the system in the project presentation *"Fourier Transform-Based Analysis and Reduction of Residential Noise
Pollution"*: audio acquisition, Fourier analysis, feature extraction, noise-source classification, selective filtering,
inverse transform, evaluation, and a database holding original audio, features, processed audio and results.

---

## Quick start

Requires **Python 3.10 or newer**.

```bash
cd reso-noise
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

Open **http://127.0.0.1:5050**.

Options: `python run.py --host 0.0.0.0 --port 8080`. Microphone recording needs `http://localhost` or HTTPS (a browser rule).
If you expose the app on a network, put it behind HTTPS and authentication; it has none of its own.

### Audio formats

| Format | Needs |
|---|---|
| WAV (8/16/24/32-bit PCM, 32/64-bit float, any channel count) | nothing extra |
| FLAC, OGG, MP3, AIFF | `soundfile` (listed in requirements.txt) **or** `ffmpeg` on your PATH |
| M4A/AAC, WebM, Opus | `ffmpeg` on your PATH |

Limits (change with environment variables): 60 MB upload (`RESONOISE_MAX_UPLOAD_MB`), 120 s (`RESONOISE_MAX_DURATION_S`),
0.5 s minimum. Sample rates above 48 kHz are resampled to 48 kHz. Other variables: `RESONOISE_DATA` (where the database and
audio are stored, default `./data`).

Invalid input is reported clearly instead of crashing: non-audio files, empty or truncated files, silent recordings,
clips that are too short or too long, NaN samples, oversized uploads, bad filter settings and missing records each return a
specific message (HTTP 4xx) that the interface shows as a notification.

---

## Using the Studio

1. **Input.** Drop in a file, record with the microphone (browser noise suppression, echo cancellation and auto-gain are
   switched off so you analyse the raw signal), or load a *sample scene*: modelled recordings of a talker over a fan, an AC
   unit, a road or an excavator. Sample scenes keep the clean voice so the result can be scored against ground truth.
2. **Spectrum.** Zoomable waveform with playback, the magnitude spectrum (full-signal DFT or Welch average, log or linear
   axis, drag to zoom, hover for values), a spectrogram, tonal peaks marked *steady* or *varies*, harmonic series, spectral
   centroid and roll-off, band energy shares, and the Parseval check.
3. **Source.** The classifier's verdict for the whole clip and for its quiet moments, the features that drove the decision,
   and a control to correct it.
4. **Filter.** Start from the recommended profile or build your own: high-pass, low-pass, notch on any detected or typed
   frequency (depth and width adjustable), and an adaptive spectral gate. The filter response H(f) updates live.
5. **Result.** A/B playback of original vs processed, overlaid waveforms, before/after spectra and spectrograms, evaluation
   metrics, and downloads: processed WAV, original WAV, JSON report.

The **Library** lists everything stored in the database (open, re-download, delete). The **Method** page shows the equations,
the pipeline, the classifier's model card with its confusion matrix, and a **Retrain** button.

---

## How it works

### Mathematics (`resonoise/fourier.py`, `resonoise/filtering.py`)

| Slide formula | Implementation |
|---|---|
| Fourier transform `X(f) = ∫ x(t) e^{-j2πft} dt` | `forward_transform`: FFT of the whole (zero-padded) signal |
| Magnitude `‖X(f)‖ = √(Re² + Im²)` | `magnitude_spectrum` |
| Parseval `∫‖x‖² dt = ∫‖X‖² df` | `parseval_check`, run on every recording and every output |
| Convolution theorem `F{x∗h} = X(f)H(f)` | `apply_frequency_filter`: multiply X(f) by a designed H(f) |
| Inverse transform `y(t) = ∫ Y(f) e^{j2πft} df` | `inverse_transform` rebuilds the processed audio |

The filter H(f) is real and non-negative, so filtering is **zero-phase** (no delay, no phase smearing). It is the product of
4th-order Butterworth-magnitude high/low-pass shapes and smooth Gaussian notches. The signal is zero-padded by at least one
second so the circular transform behaves like linear filtering. The test suite verifies the result against an explicit
time-domain convolution with `h = F⁻¹{H}`.

### Classification (`features.py`, `classifier.py`, `synthetic.py`)

* **Features (25).** Band energies, spectral centroid/spread/roll-off/flatness, tonal-peak share, number of peaks, harmonic
  strength and fundamental, level fluctuation, dynamic range, 2-8 Hz syllable-rate modulation, crest factor, impulsiveness,
  zero-crossing rate, spectral flux, voiced-frame ratio and pitch variation. All are level independent and computed at 16 kHz.
* **Algorithm.** Multinomial logistic (softmax) regression, implemented from scratch in NumPy: standardised inputs,
  L2-regularised cross-entropy, analytic gradient (checked against finite differences in the tests), optimised with L-BFGS.
* **Classes** (from the project slides): fan, air conditioner, road traffic, excavator, people talking.
* **Background pass.** If the whole clip reads as speech, the quietest segments are classified separately. A talker over a
  fan therefore gets the *fan* filter, while the voice is preserved. The background counts only if it sits within 30 dB of
  the loud parts.
* **Training data.** No recordings are bundled, so the model is trained on physically-motivated synthetic sources
  (`synthetic.py`): blade-pass tones, mains-harmonic hum, road rumble with pass-bys, diesel pulse trains with clanks, and
  formant-filtered speech, with random microphone colouration and background mixing. `python scripts/train_model.py`
  rebuilds `resonoise/models/default_model.json` (about 40 s).
* **Learning from you.** On the Source step you can correct the verdict. The correction (features + label) is stored in the
  database; **Method → Retrain** merges your corrections (weighted) with the modelled data and saves a personal model to
  `data/model.json`.

**Honest limits.** Held-out accuracy (~96%) is measured on *modelled* clips, which is easier than a real room. Fan vs air
conditioner is the main confusion, as in life. Real recordings add echo, other sources and microphone character, so treat
the verdict as a strong starting point and correct it when it is wrong. Corrections make the model better for *your* sounds.

### Filtering (`filtering.py`)

1. **Stage 1, `Y(f) = X(f)H(f)`.** High-pass, low-pass and notches.
   Notches go only on spectral peaks that are **stationary**: the peak's level must vary by less than 5 dB over time. Hum,
   blade-pass tones and engine harmonics qualify; voice harmonics do not, so speech is not notched.
2. **Stage 2, adaptive spectral gate.** An STFT estimates the noise power per bin from the quietest 15% of frames and applies
   a smoothed Wiener-style gain, then the inverse STFT reconstructs. Strength 0-100% maps to 6-24 dB maximum attenuation.
3. Clip protection, then 16-bit quantisation. Metrics are computed on the quantised samples that you download.

Profile defaults (fan, AC, traffic, excavator, speech) were tuned by measuring SI-SDR against the clean voice of sample scenes
over several random seeds.

### Evaluation metrics (`metrics.py`)

All metrics come from the original and processed sample arrays:

* noise-floor change (10th percentile of 32 ms frame levels) and loud-to-floor range;
* overall RMS level change and energy removed (computed in the frequency domain via Parseval);
* waveform correlation, log-spectral distance (20 Hz-8 kHz) and speech-band (0.3-3.4 kHz) change;
* per-octave-band level before/after and per-tone attenuation;
* **SI-SDR against the clean reference**, available for sample scenes, which is true ground-truth scoring.

---

## Project layout

```
reso-noise/
├── run.py                     start the server
├── requirements.txt
├── scripts/train_model.py     rebuild the bundled classifier
├── resonoise/
│   ├── config.py errors.py utils.py
│   ├── audio_io.py            decode, validate, preprocess, WAV writing
│   ├── fourier.py             FFT/IFFT, Parseval, Welch, STFT, peaks, harmonics
│   ├── features.py            classifier features, background-segment finder
│   ├── synthetic.py           source models + demo scenes + training data
│   ├── classifier.py          softmax regression, training, model card
│   ├── filtering.py           profiles, H(f) design, stage-1 and stage-2 filters
│   ├── metrics.py             evaluation metrics
│   ├── visuals.py             waveform envelopes, spectra, spectrograms for the UI
│   ├── database.py            SQLite storage (recordings, analyses, processed, labels)
│   ├── service.py             orchestrates the whole pipeline
│   ├── server.py              Flask JSON API + static files
│   ├── models/                bundled model + cached training features
│   └── static/                index.html, css/, js/ (no build step, no frameworks)
└── tests/                     84 unit/API tests + optional browser test
```

Frontend modules (`static/js`): `studio.js` (workflow), `charts.js` (canvas charts), `player.js` (A/B transport),
`recorder.js` (microphone to WAV), `library.js`, `method.js`, `api.js`, `ui.js`, `icons.js`, `main.js` (routing).

## API

| Method and path | Purpose |
|---|---|
| `POST /api/recordings` | multipart `file` (+ `source`): decode, preprocess, store |
| `POST /api/samples` | `{kind}`: create a sample scene |
| `POST /api/recordings/<id>/analyze` | Fourier analysis, features, classification |
| `GET  /api/recordings/<id>/analysis` | stored analysis |
| `POST /api/recordings/<id>/filter-preview` | H(f) curve for given settings |
| `POST /api/recordings/<id>/process` | filter, reconstruct, evaluate, store |
| `GET  /api/recordings/<id>/waveform?t0&t1&points&pid` | min/max envelope at any zoom |
| `GET  /api/recordings/<id>/audio[?download=1]` | original WAV (supports Range requests) |
| `GET  /api/processed/<pid>/audio` · `/report` · `/` | processed WAV, JSON report, payload |
| `POST /api/recordings/<id>/label` · `POST /api/model/retrain` | teach and retrain |
| `GET/DELETE /api/recordings[/<id>]` · `DELETE /api/processed/<pid>` | library |

Filter settings (`process` body): `highpass_hz`, `lowpass_hz` (or null), `notch_freqs` (list, max 16), `notch_depth_db`
(3-60), `notch_width_hz` (1-200), `gate_strength` (0-1). Errors are `{"error": {"code", "message", "details"}}`.

## Testing

```bash
python -m unittest discover -s tests          # 84 tests, about 7 s, no extra packages
python tests/browser_e2e.py                   # optional: real browser run (pip install playwright; playwright install chromium)
```

What the tests prove, with independent references rather than the code's own output:

* FFT equals a direct DFT sum; inverse recovers the signal; Parseval holds to machine precision (even, odd, zero-padded);
  STFT reconstruction is exact; a sine of amplitude A reads A in both spectrum views.
* The frequency-domain filter equals explicit time-domain convolution; notch depth/width and Butterworth cutoffs match their
  formulas; filtering is zero-phase; a notch removes the tone and leaves neighbours alone.
* The classifier's gradient matches finite differences; it learns separable data, round-trips to disk, reaches ≥ 90% on fresh
  unseen clips; user corrections measurably change it.
* Metrics hit closed-form values (half amplitude = −6.02 dB, 75% energy removed, SI-SDR of a known mixture = 20 dB).
* End to end through the HTTP API: upload → analyse → classify → filter → reconstruct → download, then re-measure the
  *downloaded* file (the 120 Hz hum must drop by more than 20 dB and the server's metrics must match a recomputation).
* Bad input of every kind returns the right status and a readable message.
* The browser test drives the actual interface in Chromium: upload, error toast, charts drawn, hover and drag-zoom, playback,
  A/B switching, notch editing, file download, library re-open, sample scenes, **microphone capture via a simulated device**,
  and no overflow at phone width.

Not covered in this environment: the `soundfile` decoding path and the `waitress` server (neither package could be installed
here). The ffmpeg path and Flask server were exercised.

## Troubleshooting

* *"Decoding MP3 files needs either the 'soundfile' package or ffmpeg"*: `pip install soundfile`, or install ffmpeg.
* *Recording is greyed out*: open the app at `http://localhost:5050` (or over HTTPS) and allow the microphone.
* *Fonts look plain*: the interface loads Fraunces and Instrument Sans from Google Fonts; offline it falls back to system fonts.
* *Reset everything*: stop the server and delete the `data/` folder.
