"""Fourier analysis engine (stage 3 of the system architecture).

Mathematical foundation implemented here (all one-sided, real-signal forms):

    X(f)   = integral x(t) e^{-j 2 pi f t} dt        -> forward_transform  (DFT via FFT)
    |X(f)| = sqrt(Re(X)^2 + Im(X)^2)                 -> magnitude_spectrum
    Parseval: sum |x|^2 = (1/N) sum |X_k|^2          -> parseval_check
    y(t)   = integral Y(f) e^{+j 2 pi f t} df        -> inverse_transform

Plus the supporting tools used by feature extraction and filtering: Welch-averaged power
spectral density, STFT / inverse STFT, spectral peak picking and harmonic-series detection.
"""
from __future__ import annotations

import numpy as np
from scipy import fft as sfft
from scipy import ndimage, signal

from .utils import db

# ----------------------------------------------------------------------------- core transform pair
def forward_transform(x: np.ndarray, n: int | None = None) -> np.ndarray:
    """One-sided discrete Fourier transform X_k of a real signal (optionally zero-padded to n)."""
    return sfft.rfft(np.asarray(x, dtype=np.float64), n=n)


def inverse_transform(X: np.ndarray, n: int) -> np.ndarray:
    """Inverse DFT: reconstruct the n-sample real signal from a one-sided spectrum."""
    return sfft.irfft(X, n=n)


def frequency_axis(n: int, sr: float) -> np.ndarray:
    """Frequency (Hz) of each one-sided DFT bin for an n-point transform."""
    return np.arange(n // 2 + 1) * (sr / n)


def magnitude_spectrum(X: np.ndarray) -> np.ndarray:
    """|X(f)| = sqrt(Re(X(f))^2 + Im(X(f))^2)."""
    return np.hypot(X.real, X.imag)


def amplitude_spectrum(X: np.ndarray, n: int) -> np.ndarray:
    """Magnitude scaled so a sine of amplitude A shows a peak of height A."""
    amp = magnitude_spectrum(X) * (2.0 / n)
    amp[0] *= 0.5
    if n % 2 == 0:
        amp[-1] *= 0.5
    return amp


def _bin_weights(n: int) -> np.ndarray:
    """Multiplicity of each one-sided bin in the full two-sided spectrum."""
    w = np.full(n // 2 + 1, 2.0)
    w[0] = 1.0
    if n % 2 == 0:
        w[-1] = 1.0
    return w


def parseval_check(x: np.ndarray, X: np.ndarray | None = None, n_fft: int | None = None) -> dict:
    """Compare time-domain energy sum|x|^2 with frequency-domain energy (1/N) sum|X_k|^2.

    `n_fft` is the transform length used to obtain X (default: len(x)); zero padding preserves energy.
    """
    x = np.asarray(x, dtype=np.float64)
    n = n_fft or x.size
    if X is None:
        X = forward_transform(x, n=n)
    time_energy = float(np.sum(x * x))
    freq_energy = float(np.sum(_bin_weights(n) * magnitude_spectrum(X) ** 2) / n)
    rel = abs(time_energy - freq_energy) / max(time_energy, 1e-300)
    return {"time_energy": time_energy, "frequency_energy": freq_energy, "relative_error": float(rel)}


def next_fast_length(n: int) -> int:
    return int(sfft.next_fast_len(int(n), real=True))


# ----------------------------------------------------------------------------- windowed analysis
def hann(n: int) -> np.ndarray:
    return signal.get_window("hann", n, fftbins=True)


def analysis_nfft(sr: float, resolution_hz: float = 3.0) -> int:
    """Smallest power of two giving a frequency resolution of at least `resolution_hz`."""
    return int(2 ** np.ceil(np.log2(max(sr / resolution_hz, 256))))


def _frames(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    view = np.lib.stride_tricks.sliding_window_view(x, n_fft)
    return view[::hop]


def welch_psd(x: np.ndarray, sr: float, n_fft: int | None = None, overlap: float = 0.5):
    """Welch-averaged one-sided power spectral density (power per Hz). Returns (freqs, psd)."""
    x = np.asarray(x, dtype=np.float64)
    n_fft = n_fft or analysis_nfft(sr)
    hop = max(1, int(n_fft * (1.0 - overlap)))
    win = hann(n_fft)
    acc = np.zeros(n_fft // 2 + 1)
    count = 0
    frames = _frames(x, n_fft, hop)
    for start in range(0, frames.shape[0], 256):          # blocks keep memory bounded
        block = frames[start:start + 256] * win
        spec = sfft.rfft(block, axis=1)
        acc += np.sum(spec.real ** 2 + spec.imag ** 2, axis=0)
        count += block.shape[0]
    scale = sr * np.sum(win ** 2)
    psd = acc / (count * scale)
    psd[1:-1] *= 2.0                                       # fold negative frequencies
    return frequency_axis(n_fft, sr), psd


def welch_amplitude_spectrum(x: np.ndarray, sr: float, n_fft: int | None = None, overlap: float = 0.5):
    """Welch-averaged amplitude spectrum: a sine of amplitude A reads A at its peak bin (Hann window)."""
    x = np.asarray(x, dtype=np.float64)
    n_fft = n_fft or analysis_nfft(sr)
    hop = max(1, int(n_fft * (1.0 - overlap)))
    win = hann(n_fft)
    acc = np.zeros(n_fft // 2 + 1)
    count = 0
    frames = _frames(x, n_fft, hop)
    for start in range(0, frames.shape[0], 256):
        spec = sfft.rfft(frames[start:start + 256] * win, axis=1)
        acc += np.sum(spec.real ** 2 + spec.imag ** 2, axis=0)
        count += min(256, frames.shape[0] - start)
    return frequency_axis(n_fft, sr), np.sqrt(acc / count) * (2.0 / np.sum(win))


def stft(x: np.ndarray, n_fft: int, hop: int):
    """Short-time Fourier transform. Returns (Z[frames, bins], pad_left) for istft."""
    x = np.asarray(x, dtype=np.float64)
    total = x.size + 2 * n_fft
    n_frames = int(np.ceil((total - n_fft) / hop)) + 1
    length = n_fft + (n_frames - 1) * hop
    xp = np.zeros(length)
    xp[n_fft:n_fft + x.size] = x
    frames = np.lib.stride_tricks.sliding_window_view(xp, n_fft)[::hop][:n_frames]
    return sfft.rfft(frames * hann(n_fft), axis=1), n_fft


def istft(Z: np.ndarray, n_fft: int, hop: int, length: int, pad_left: int | None = None) -> np.ndarray:
    """Inverse STFT by weighted overlap-add (exact reconstruction for unmodified Z)."""
    pad_left = n_fft if pad_left is None else pad_left
    win = hann(n_fft)
    frames = sfft.irfft(Z, n=n_fft, axis=1) * win
    n_frames = Z.shape[0]
    out = np.zeros(n_fft + (n_frames - 1) * hop)
    wsum = np.zeros_like(out)
    w2 = win ** 2
    for i in range(n_frames):
        s = i * hop
        out[s:s + n_fft] += frames[i]
        wsum[s:s + n_fft] += w2
    out = out / np.maximum(wsum, 1e-8)
    return out[pad_left:pad_left + length]


# ----------------------------------------------------------------------------- peaks and harmonics
def find_spectral_peaks(freqs: np.ndarray, psd: np.ndarray, *, min_freq: float = 20.0,
                        max_freq: float | None = None, prominence_db: float = 10.0,
                        baseline_hz: float = 150.0, max_peaks: int = 16) -> list[dict]:
    """Detect narrow spectral peaks that stand out above the local median level.

    Each peak is refined by parabolic interpolation of the dB spectrum. Returned dicts hold
    freq_hz, level_db (PSD level) and prominence_db (height above the local baseline).
    """
    psd_db = db(psd)
    df = float(freqs[1] - freqs[0])
    kernel = int(max(5, round(baseline_hz / df))) | 1
    baseline = ndimage.median_filter(psd_db, size=kernel, mode="nearest")
    cand, _ = signal.find_peaks(psd_db, height=baseline + prominence_db, prominence=6.0)
    peaks = []
    for i in cand:
        if i < 1 or i >= len(freqs) - 1:
            continue
        a, b, c = psd_db[i - 1], psd_db[i], psd_db[i + 1]
        denom = a - 2 * b + c
        delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
        f = float(freqs[i] + delta * df)
        if f < min_freq or (max_freq is not None and f > max_freq):
            continue
        peaks.append({"freq_hz": f, "level_db": float(b - 0.25 * (a - c) * delta),
                      "prominence_db": float(b - baseline[i]), "bin": int(i)})
    peaks.sort(key=lambda p: p["prominence_db"], reverse=True)
    return peaks[:max_peaks]


def detect_harmonic_series(peaks: list[dict], tolerance: float = 0.03, min_f0: float = 20.0) -> dict | None:
    """Find the fundamental whose integer multiples explain the most prominent peaks.

    Candidate fundamentals are every peak and its sub-multiples f/2, f/3, f/4. Returns None unless
    at least two peaks fall on a common harmonic ladder.
    """
    if len(peaks) < 2:
        return None
    total_prom = sum(p["prominence_db"] for p in peaks)
    candidates = {round(p["freq_hz"] / m, 2) for p in peaks for m in (1, 2, 3, 4) if p["freq_hz"] / m >= min_f0}
    scored = []
    for f0 in candidates:
        matched = []
        for p in peaks:
            k = int(round(p["freq_hz"] / f0))
            if k >= 1 and abs(p["freq_hz"] / (k * f0) - 1.0) <= tolerance:
                matched.append((k, p))
        if len({k for k, _ in matched}) < 2:
            continue
        scored.append((sum(p["prominence_db"] for _, p in matched), f0, matched))
    if not scored:
        return None
    top = max(s[0] for s in scored)
    # a fundamental and its sub-multiples explain the same peaks equally well; prefer the highest fundamental
    # among all candidates that explain (almost) as much as the best one
    best = max(((f0, m) for sc, f0, m in scored if sc >= 0.85 * top), key=lambda fm: fm[0])
    f0, matched = best
    matched.sort(key=lambda kp: kp[0])
    # refine the fundamental with a least-squares fit f = k * f0 through the matched peaks
    ks = np.array([k for k, _ in matched], dtype=float)
    fs = np.array([p["freq_hz"] for _, p in matched])
    f0_refined = float(np.sum(ks * fs) / np.sum(ks * ks))
    return {
        "f0_hz": f0_refined,
        "harmonics": [{"k": int(k), "freq_hz": float(p["freq_hz"]), "level_db": float(p["level_db"])}
                      for k, p in matched],
        "strength": float(sum(p["prominence_db"] for _, p in matched) / max(total_prom, 1e-9)),
    }


def peak_stationarity(x: np.ndarray, sr: float, peak_freqs: list[float], n_fft: int = 4096) -> list[float]:
    """Standard deviation (dB) of each peak's level over time. Low = steady hum / tone; high = speech-like."""
    if not peak_freqs:
        return []
    n_fft = min(n_fft, max(256, 2 ** int(np.floor(np.log2(max(x.size, 256))))))
    hop = n_fft // 2
    Z, _ = stft(x, n_fft, hop)
    power = Z.real ** 2 + Z.imag ** 2
    # discard the zero-padded edge frames
    n_edge = n_fft // hop
    power = power[n_edge:-n_edge] if power.shape[0] > 2 * n_edge + 2 else power
    df = sr / n_fft
    out = []
    for f in peak_freqs:
        b = int(round(f / df))
        lo, hi = max(b - 1, 0), min(b + 2, power.shape[1])
        level = db(power[:, lo:hi].max(axis=1))
        keep = level > (level.max() - 40.0)
        out.append(float(np.std(level[keep])) if keep.sum() > 1 else 0.0)
    return out


def band_energies(freqs: np.ndarray, psd: np.ndarray, edges: list[float]) -> list[dict]:
    """Power in each [lo, hi) band and its share of the total (Parseval energy distribution)."""
    df = float(freqs[1] - freqs[0])
    total = float(np.sum(psd[(freqs >= edges[0]) & (freqs < edges[-1])]) * df)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        p = float(np.sum(psd[(freqs >= lo) & (freqs < hi)]) * df)
        out.append({"lo": float(lo), "hi": float(hi), "power": p,
                    "share": p / total if total > 0 else 0.0, "level_db": float(db(p))})
    return out


def spectral_summary(freqs: np.ndarray, psd: np.ndarray, fmin: float = 20.0, fmax: float | None = None) -> dict:
    """Centroid, 85% roll-off, flatness and dominant bin of a PSD."""
    fmax = fmax or float(freqs[-1])
    m = (freqs >= fmin) & (freqs <= fmax)
    f, p = freqs[m], psd[m]
    total = float(p.sum()) + 1e-30
    centroid = float(np.sum(f * p) / total)
    cum = np.cumsum(p) / total
    rolloff = float(f[min(np.searchsorted(cum, 0.85), len(f) - 1)])
    flatness = float(np.exp(np.mean(np.log(p + 1e-30))) / (np.mean(p) + 1e-30))
    return {"centroid_hz": centroid, "rolloff85_hz": rolloff, "flatness_db": float(db(flatness)),
            "peak_hz": float(f[int(np.argmax(p))])}
