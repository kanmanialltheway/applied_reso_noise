"""Feature extraction (stage 4 of the system architecture).

Turns a signal into a fixed-length vector describing its frequency distribution (Fourier
features), harmonic structure and temporal behaviour. The vector feeds the classifier.
All features are level independent. Computation always happens at 16 kHz.
"""
from __future__ import annotations

import numpy as np
from scipy import fft as sfft

from . import config
from .audio_io import resample
from .fourier import detect_harmonic_series, find_spectral_peaks, hann, welch_psd
from .utils import db

BAND_EDGES = [20.0, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]
BAND_NAMES = [f"band_{int(lo)}_{int(hi)}" for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:])]

FEATURE_NAMES = (
    ["spread_oct", "centroid_log2", "rolloff_log2", "flatness_db"]
    + BAND_NAMES
    + ["tonal_ratio", "n_peaks", "harmonic_strength", "f0_log2",
       "level_std_db", "dynamic_range_db", "mod_2_8hz", "crest_db", "kurtosis_log",
       "zcr", "flux", "voiced_ratio", "pitch_var_st"]
)

FEATURE_LABELS = {
    "spread_oct": "Spectral spread", "centroid_log2": "Spectral centroid", "rolloff_log2": "85% roll-off frequency",
    "flatness_db": "Spectral flatness", "tonal_ratio": "Share of energy in tonal peaks",
    "n_peaks": "Number of tonal peaks", "harmonic_strength": "Harmonic structure",
    "f0_log2": "Fundamental frequency", "level_std_db": "Level fluctuation",
    "dynamic_range_db": "Dynamic range", "mod_2_8hz": "Syllable-rate modulation (2-8 Hz)",
    "crest_db": "Crest factor", "kurtosis_log": "Impulsiveness", "zcr": "Zero-crossing rate",
    "flux": "Spectral flux", "voiced_ratio": "Voiced-frame ratio", "pitch_var_st": "Pitch variation",
}
for _n, (_lo, _hi) in zip(BAND_NAMES, zip(BAND_EDGES[:-1], BAND_EDGES[1:])):
    FEATURE_LABELS[_n] = f"Energy {int(_lo)}-{int(_hi)} Hz"

FRAME_MS = 32
HOP_MS = 10


def frame_rms_db(y: np.ndarray, sr: int, frame_ms: float = FRAME_MS, hop_ms: float = HOP_MS) -> np.ndarray:
    """Short-time RMS level in dB (full scale = 0 dB)."""
    n, hop = int(sr * frame_ms / 1000), int(sr * hop_ms / 1000)
    if y.size < n:
        y = np.pad(y, (0, n - y.size))
    frames = np.lib.stride_tricks.sliding_window_view(y, n)[::hop]
    return 20.0 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-9)


def _pitch_track(y: np.ndarray, sr: int, active_floor_db: float = -35.0):
    """Frame-wise autocorrelation pitch tracker. Returns (voiced_ratio, pitch_std_semitones)."""
    n, hop = int(0.04 * sr), int(0.02 * sr)
    if y.size < n + hop:
        return 0.0, 0.0
    frames = np.lib.stride_tricks.sliding_window_view(y, n)[::hop]
    energy = np.mean(frames ** 2, axis=1)
    e_db = 10 * np.log10(energy + 1e-12)
    active = e_db > (e_db.max() + active_floor_db)
    if active.sum() < 3:
        return 0.0, 0.0
    fr = frames[active] * hann(n)
    spec = sfft.rfft(fr, n=2 * n, axis=1)
    ac = sfft.irfft(spec.real ** 2 + spec.imag ** 2, axis=1)[:, :n]
    ac = ac / (ac[:, :1] + 1e-12)
    lo, hi = int(sr / 400), int(sr / 70)
    seg = ac[:, lo:hi]
    best = np.argmax(seg, axis=1)
    val = seg[np.arange(seg.shape[0]), best]
    voiced = val > 0.5
    ratio = float(voiced.mean())
    if voiced.sum() < 3:
        return ratio, 0.0
    f0 = sr / (best[voiced] + lo)
    st = 12.0 * np.log2(f0)
    mad = np.median(np.abs(st - np.median(st))) * 1.4826       # robust to octave slips
    return ratio, float(mad)


def extract_features(x: np.ndarray, sr: int) -> dict[str, float]:
    """Compute the classifier feature vector as a name -> value dict."""
    y = resample(np.asarray(x, dtype=np.float64), sr, config.CLASSIFIER_SAMPLE_RATE)
    fs = config.CLASSIFIER_SAMPLE_RATE
    if np.max(np.abs(y)) < 1e-9:
        raise ValueError("signal is silent")

    freqs, psd = welch_psd(y, fs, n_fft=8192)
    m = (freqs >= 20.0) & (freqs <= 8000.0)
    f, p = freqs[m], psd[m]
    total = float(p.sum()) + 1e-30

    logf = np.log2(f)
    mean_log = float(np.sum(p * logf) / total)
    spread = float(np.sqrt(np.sum(p * (logf - mean_log) ** 2) / total))
    centroid = float(np.sum(f * p) / total)
    rolloff = float(f[min(np.searchsorted(np.cumsum(p) / total, 0.85), len(f) - 1)])
    flatness = float(np.exp(np.mean(np.log(p + 1e-30))) / (np.mean(p) + 1e-30))

    feats: dict[str, float] = {
        "spread_oct": spread,
        "centroid_log2": float(np.log2(centroid)),
        "rolloff_log2": float(np.log2(rolloff)),
        "flatness_db": float(db(flatness)),
    }
    df = freqs[1] - freqs[0]
    for name, lo, hi in zip(BAND_NAMES, BAND_EDGES[:-1], BAND_EDGES[1:]):
        share = float(np.sum(psd[(freqs >= lo) & (freqs < hi)]) * df / (total * df))
        feats[name] = float(10 * np.log10(share + 1e-4))

    peaks = find_spectral_peaks(freqs, psd, min_freq=20.0, max_freq=8000.0, prominence_db=10.0, baseline_hz=150.0)
    tonal = 0.0
    for pk in peaks:
        b = pk["bin"]
        tonal += float(np.sum(psd[max(b - 2, 0):b + 3]))
    feats["tonal_ratio"] = min(tonal / total, 1.0)
    feats["n_peaks"] = float(min(len(peaks), 12))
    harm = detect_harmonic_series(peaks)
    feats["harmonic_strength"] = harm["strength"] if harm else 0.0
    feats["f0_log2"] = float(np.log2(harm["f0_hz"])) if harm else 0.0

    lv = frame_rms_db(y, fs)
    active = lv > lv.max() - 60.0
    feats["level_std_db"] = float(np.std(lv[active]))
    feats["dynamic_range_db"] = float(np.percentile(lv, 95) - np.percentile(lv, 10))
    env = 10 ** (lv / 20.0)
    env = env - env.mean()
    if env.size >= 64:
        espec = np.abs(sfft.rfft(env)) ** 2
        ef = np.fft.rfftfreq(env.size, HOP_MS / 1000.0)
        band = (ef >= 2) & (ef <= 8)
        ref = (ef >= 0.5) & (ef <= 50)
        feats["mod_2_8hz"] = float(espec[band].sum() / (espec[ref].sum() + 1e-30))
    else:
        feats["mod_2_8hz"] = 0.0
    rms_val = float(np.sqrt(np.mean(y ** 2)))
    feats["crest_db"] = float(20 * np.log10(np.max(np.abs(y)) / (rms_val + 1e-12)))
    c = y - y.mean()
    kurt = float(np.mean(c ** 4) / (np.mean(c ** 2) ** 2 + 1e-30) - 3.0)
    feats["kurtosis_log"] = float(np.log1p(max(kurt, 0.0)))
    feats["zcr"] = float(np.mean(np.abs(np.diff(np.signbit(y).astype(np.int8)))))

    n, hop = int(0.032 * fs), int(0.010 * fs)
    frames = np.lib.stride_tricks.sliding_window_view(y if y.size >= n else np.pad(y, (0, n - y.size)), n)[::hop]
    mag = np.abs(sfft.rfft(frames * hann(n), axis=1))
    if mag.shape[0] > 1:
        num = np.linalg.norm(np.diff(mag, axis=0), axis=1)
        den = np.linalg.norm(mag[1:], axis=1) + 1e-12
        feats["flux"] = float(np.mean(num / den))
    else:
        feats["flux"] = 0.0

    voiced, pvar = _pitch_track(y, fs)
    feats["voiced_ratio"] = voiced
    feats["pitch_var_st"] = pvar

    return {k: float(feats[k]) for k in FEATURE_NAMES}


def feature_vector(feats: dict[str, float]) -> np.ndarray:
    return np.array([feats[k] for k in FEATURE_NAMES], dtype=np.float64)


def quiet_segments(x: np.ndarray, sr: int, fraction: float = 0.25, seg_ms: float = 200.0):
    """Concatenate the quietest segments of a signal: an estimate of its background noise.

    Returns the background signal, or None when fewer than ~1.2 s of quiet material exist.
    """
    seg = int(sr * seg_ms / 1000)
    n_seg = x.size // seg
    if n_seg < 8:
        return None
    blocks = x[:n_seg * seg].reshape(n_seg, seg)
    level = np.sqrt(np.mean(blocks ** 2, axis=1))
    k = max(int(round(n_seg * fraction)), 1)
    chosen = np.sort(np.argsort(level)[:k])
    if k * seg_ms / 1000.0 < 1.2:
        return None
    fade = int(0.01 * sr)
    ramp = np.linspace(0, 1, fade)
    parts = []
    for i in chosen:
        b = blocks[i].copy()
        b[:fade] *= ramp
        b[-fade:] *= ramp[::-1]
        parts.append(b)
    out = np.concatenate(parts)
    return out if np.max(np.abs(out)) > 1e-9 else None
