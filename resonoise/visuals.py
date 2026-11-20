"""Turns real signals into chart-ready data: waveform envelopes, spectra and spectrograms."""
from __future__ import annotations

import base64

import numpy as np

from .fourier import (amplitude_spectrum, forward_transform, frequency_axis, hann, next_fast_length, stft,
                      welch_amplitude_spectrum, analysis_nfft)
from .utils import db


def waveform_envelope(x: np.ndarray, sr: int, t0: float = 0.0, t1: float | None = None, points: int = 1200) -> dict:
    """Min/max of the samples inside each of `points` equal time slices of [t0, t1]."""
    dur = x.size / sr
    t1 = dur if t1 is None else t1
    t0 = float(np.clip(t0, 0.0, dur))
    t1 = float(np.clip(t1, t0, dur))
    i0, i1 = int(t0 * sr), max(int(np.ceil(t1 * sr)), int(t0 * sr) + 1)
    seg = x[i0:i1]
    points = int(np.clip(points, 16, 6000))
    if seg.size <= points:                       # zoomed far enough to see individual samples
        return {"t0": i0 / sr, "t1": i0 / sr + seg.size / sr, "min": np.round(seg, 5), "max": np.round(seg, 5),
                "duration": dur, "samples": True}
    edges = np.linspace(0, seg.size, points + 1).astype(int)
    starts = edges[:-1]
    mins = np.minimum.reduceat(seg, starts)
    maxs = np.maximum.reduceat(seg, starts)
    return {"t0": i0 / sr, "t1": i1 / sr, "min": np.round(mins, 5), "max": np.round(maxs, 5),
            "duration": dur, "samples": False}


def log_bin_curve(freqs: np.ndarray, values: np.ndarray, n_points: int = 900, fmin: float = 20.0,
                  fmax: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Max-pool a dense spectrum into log-spaced bins so narrow peaks survive downsampling."""
    fmax = fmax or float(freqs[-1])
    edges = np.geomspace(fmin, fmax, n_points + 1)
    centres = np.sqrt(edges[:-1] * edges[1:])
    sel = (freqs >= edges[0]) & (freqs < edges[-1])
    f, v = freqs[sel], values[sel]
    out = np.full(n_points, np.nan)
    if f.size:
        idx = np.searchsorted(edges, f, side="right") - 1
        starts = np.flatnonzero(np.r_[True, idx[1:] != idx[:-1]])
        out[idx[starts]] = np.maximum.reduceat(v, starts)
    ok = ~np.isnan(out)
    if ok.sum() >= 2:
        out = np.interp(np.log(centres), np.log(centres[ok]), out[ok])
    elif ok.sum() == 1:
        out[:] = out[ok][0]
    else:
        out[:] = -160.0
    return centres, out


def spectrum_payload(x: np.ndarray, sr: int, n_points: int = 900) -> dict:
    """Two views of the frequency content: the single full-signal DFT magnitude and the Welch average."""
    n = x.size
    m = next_fast_length(n)
    X = forward_transform(x, n=m)
    amp = amplitude_spectrum(X, m)
    f_dense = frequency_axis(m, sr)
    fmax = min(sr / 2.0, 0.999 * sr / 2.0)
    fc, fft_db = log_bin_curve(f_dense, db(amp ** 2), n_points, 20.0, fmax)
    fw, wamp = welch_amplitude_spectrum(x, sr, analysis_nfft(sr))
    _, avg_db = log_bin_curve(fw, db(wamp ** 2), n_points, 20.0, fmax)
    # re-evaluate on the same centres so both curves share an x axis
    return {"freq": np.round(fc, 2), "fft_db": np.round(fft_db, 2), "avg_db": np.round(avg_db, 2),
            "n_fft": int(m), "resolution_hz": float(sr / m), "welch_resolution_hz": float(sr / analysis_nfft(sr))}


def spectrogram_image(x: np.ndarray, sr: int, n_cols: int = 640, n_rows: int = 160, fmin: float = 40.0,
                      db_range: tuple[float, float] | None = None, dynamic_db: float = 75.0) -> dict:
    """Log-frequency spectrogram quantised to uint8 (base64), ready for canvas drawing."""
    n_fft = int(2 ** np.ceil(np.log2(max(0.032 * sr, 256))))
    hop = max(n_fft // 2, int(np.ceil(x.size / 1600)))
    Z, _ = stft(x, n_fft, hop)
    n_edge = max(n_fft // hop, 1)
    if Z.shape[0] > 2 * n_edge + 2:
        Z = Z[n_edge:-n_edge]
    power = (Z.real ** 2 + Z.imag ** 2) / np.sum(hann(n_fft) ** 2)
    S = db(power)                                               # frames x bins
    if S.shape[0] > n_cols:
        g = int(np.ceil(S.shape[0] / n_cols))
        keep = (S.shape[0] // g) * g
        S = S[:keep].reshape(-1, g, S.shape[1]).mean(axis=1)
    freqs = frequency_axis(n_fft, sr)
    fmax = sr / 2.0 * 0.999
    edges = np.geomspace(fmin, fmax, n_rows + 1)
    bins = np.clip(np.round(edges / (sr / n_fft)).astype(int), 0, S.shape[1] - 1)
    img = np.empty((S.shape[0], n_rows))
    for r in range(n_rows):
        lo, hi = bins[r], bins[r + 1]
        if hi <= lo:                                            # sub-bin rows: interpolate
            c = np.sqrt(edges[r] * edges[r + 1]) / (sr / n_fft)
            i0 = int(np.clip(np.floor(c), 0, S.shape[1] - 2))
            w = c - i0
            img[:, r] = S[:, i0] * (1 - w) + S[:, i0 + 1] * w
        else:
            img[:, r] = S[:, lo:hi + 1].max(axis=1)
    top = float(np.max(img)) if db_range is None else db_range[1]
    lo_db = top - dynamic_db if db_range is None else db_range[0]
    q = np.clip((img - lo_db) / (top - lo_db), 0.0, 1.0)
    data = (q.T * 255).astype(np.uint8)                         # rows (low freq first) x cols
    return {"cols": int(data.shape[1]), "rows": int(data.shape[0]), "duration": x.size / sr,
            "fmin": float(fmin), "fmax": float(fmax), "db_min": float(lo_db), "db_max": float(top),
            "data": base64.b64encode(data.tobytes()).decode("ascii")}
