"""Selective filtering engine and reconstruction (stages 6 and 7 of the system architecture).

Stage 1 - frequency-domain filter H(f), applied through the Convolution Theorem:

        y1 = F^-1{ X(f) . H(f) }            ( = x * h in the time domain )

    H(f) is real and non-negative, so the filter is zero-phase (no delay or phase smearing).
    It is the product of
      * a Butterworth-magnitude high-pass and/or low-pass,
      * smooth Gaussian notches centred on stationary tonal peaks (hum, blade-pass, engine harmonics).

Stage 2 - adaptive spectral gate (optional): a short-time Fourier transform estimates the noise
    power per bin from the quietest frames, derives a Wiener-style gain per time-frequency cell
    and reconstructs with the inverse STFT.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import ndimage

from .errors import ValidationError
from .fourier import (find_spectral_peaks, forward_transform, inverse_transform, istft, next_fast_length,
                      peak_stationarity, stft, welch_psd)
from .utils import db

STATIONARY_STD_DB = 5.0       # peaks that vary less than this over time are treated as steady noise
MAX_NOTCHES = 16


@dataclass
class FilterProfile:
    key: str
    label: str
    summary: str
    highpass_hz: float | None
    lowpass_hz: float | None
    auto_notches: bool
    notch_depth_db: float
    notch_width_hz: float
    gate_strength: float


PROFILES: dict[str, FilterProfile] = {
    "fan": FilterProfile("fan", "Fan", "Notches the blade-pass tone and its harmonics, trims rumble below 60 Hz and gates the airflow hiss.",
                         60.0, None, True, 30.0, 5.0, 0.65),
    "ac": FilterProfile("ac", "Air conditioner", "Notches the compressor hum on its mains harmonics, trims rumble below 60 Hz and gates the airflow noise.",
                        60.0, None, True, 36.0, 3.0, 0.65),
    "traffic": FilterProfile("traffic", "Road traffic", "Removes road rumble below 100 Hz and gates the broadband tyre and engine noise.",
                             100.0, None, False, 20.0, 6.0, 0.65),
    "excavator": FilterProfile("excavator", "Excavator", "Cuts the engine fundamental region below 80 Hz, notches steady engine and hydraulic tones, gates clank residue.",
                               80.0, None, True, 26.0, 5.0, 0.65),
    "speech": FilterProfile("speech", "People talking", "Keeps the speech band (80 Hz-8 kHz) intact; only rumble and hiss outside it are reduced.",
                            80.0, 8000.0, False, 0.0, 4.0, 0.25),
}


@dataclass
class FilterParams:
    profile: str = "custom"
    highpass_hz: float | None = None
    lowpass_hz: float | None = None
    notch_freqs: list[float] = field(default_factory=list)
    notch_depth_db: float = 30.0
    notch_width_hz: float = 4.0
    gate_strength: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None, sr: float) -> "FilterParams":
        """Validate user input. Raises ValidationError with a specific message."""
        d = d or {}
        if not isinstance(d, dict):
            raise ValidationError("Filter settings must be an object.")
        nyq = sr / 2.0

        def num(name, lo, hi, default, allow_none=False):
            v = d.get(name, default)
            if v is None:
                if allow_none:
                    return None
                raise ValidationError(f"'{name}' is required.")
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
                raise ValidationError(f"'{name}' must be a number.")
            if not lo <= v <= hi:
                raise ValidationError(f"'{name}' must be between {lo:g} and {hi:g}.", details={"value": v})
            return float(v)

        profile = d.get("profile", "custom")
        if profile != "custom" and profile not in PROFILES:
            raise ValidationError(f"Unknown filter profile '{profile}'.")
        hp = num("highpass_hz", 5.0, nyq * 0.98, None, True)
        lp = num("lowpass_hz", 50.0, nyq * 0.999, None, True)
        if hp is not None and lp is not None and hp >= lp:
            raise ValidationError("The high-pass cutoff must be lower than the low-pass cutoff.")
        notches = d.get("notch_freqs", [])
        if not isinstance(notches, list) or len(notches) > MAX_NOTCHES:
            raise ValidationError(f"'notch_freqs' must be a list of at most {MAX_NOTCHES} frequencies.")
        clean = []
        for f in notches:
            if isinstance(f, bool) or not isinstance(f, (int, float)) or not np.isfinite(f):
                raise ValidationError("Every notch frequency must be a number.")
            if not 5.0 <= f < nyq:
                raise ValidationError(f"Notch frequency {f:g} Hz must be between 5 Hz and {nyq:g} Hz.")
            clean.append(float(f))
        return cls(profile=profile, highpass_hz=hp, lowpass_hz=lp, notch_freqs=sorted(set(clean)),
                   notch_depth_db=num("notch_depth_db", 3.0, 60.0, 30.0),
                   notch_width_hz=num("notch_width_hz", 1.0, 200.0, 4.0),
                   gate_strength=num("gate_strength", 0.0, 1.0, 0.0))


# ----------------------------------------------------------------------------- filter design
def highpass_gain(f: np.ndarray, fc: float, order: int = 4) -> np.ndarray:
    """Butterworth magnitude |H| = 1/sqrt(1 + (fc/f)^(2n))."""
    f = np.maximum(f, 1e-9)
    return 1.0 / np.sqrt(1.0 + (fc / f) ** (2 * order))


def lowpass_gain(f: np.ndarray, fc: float, order: int = 4) -> np.ndarray:
    return 1.0 / np.sqrt(1.0 + (f / fc) ** (2 * order))


def notch_gain(f: np.ndarray, f0: float, width_hz: float, depth_db: float) -> np.ndarray:
    """Smooth Gaussian notch: gain 10^(-depth/20) at f0, back to 1 away from it (width = FWHM)."""
    sigma = width_hz / 2.3548
    floor = 10.0 ** (-depth_db / 20.0)
    return 1.0 - (1.0 - floor) * np.exp(-0.5 * ((f - f0) / sigma) ** 2)


def design_response(freqs: np.ndarray, p: FilterParams) -> np.ndarray:
    """Combined real-valued frequency response H(f) of the stage-1 filter."""
    H = np.ones_like(freqs, dtype=np.float64)
    if p.highpass_hz:
        H *= highpass_gain(freqs, p.highpass_hz)
    if p.lowpass_hz:
        H *= lowpass_gain(freqs, p.lowpass_hz)
    for f0 in p.notch_freqs:
        H *= notch_gain(freqs, f0, p.notch_width_hz, p.notch_depth_db)
    return H


def response_curve(sr: float, p: FilterParams, n_points: int = 700, fmin: float = 20.0) -> dict:
    """H(f) in dB on a log-frequency grid, for plotting."""
    f = np.geomspace(fmin, sr / 2.0 * 0.999, n_points)
    return {"freq": f, "gain_db": 20.0 * np.log10(np.maximum(design_response(f, p), 1e-6))}


# ----------------------------------------------------------------------------- tonal analysis
def find_stationary_tones(x: np.ndarray, sr: float, freqs: np.ndarray, psd: np.ndarray, peaks: list[dict] | None = None):
    """Spectral peaks annotated with how steady they are over time. Returns the annotated list."""
    if peaks is None:
        peaks = find_spectral_peaks(freqs, psd, prominence_db=10.0, max_freq=sr * 0.45)
    std = peak_stationarity(x, sr, [p["freq_hz"] for p in peaks])
    out = []
    for p, s in zip(peaks, std):
        q = dict(p)
        q["stationarity_db"] = float(s)
        q["stationary"] = bool(s <= STATIONARY_STD_DB)
        out.append(q)
    return out


def default_notches(tones: list[dict]) -> list[float]:
    """Frequencies of the steady tonal peaks, strongest first, limited to MAX_NOTCHES."""
    sel = [t for t in tones if t["stationary"] and t["prominence_db"] >= 10.0]
    sel.sort(key=lambda t: t["prominence_db"], reverse=True)
    return sorted(round(t["freq_hz"], 2) for t in sel[:MAX_NOTCHES])


def params_for_profile(key: str, tones: list[dict], sr: float) -> FilterParams:
    """The parameter set a profile suggests for this particular recording."""
    pr = PROFILES[key]
    nyq = sr / 2.0
    lp = pr.lowpass_hz if (pr.lowpass_hz and pr.lowpass_hz < nyq * 0.95) else None
    return FilterParams(profile=key, highpass_hz=pr.highpass_hz, lowpass_hz=lp,
                        notch_freqs=default_notches(tones) if pr.auto_notches else [],
                        notch_depth_db=pr.notch_depth_db or 30.0, notch_width_hz=pr.notch_width_hz,
                        gate_strength=pr.gate_strength)


# ----------------------------------------------------------------------------- stage 1: convolution theorem
def apply_frequency_filter(x: np.ndarray, sr: float, p: FilterParams) -> np.ndarray:
    """y = F^-1{ X(f) H(f) } with zero padding so the circular transform equals linear filtering."""
    n = x.size
    pad = int(min(max(sr, 1.0), n))                       # >= 1 s guard band absorbs ringing and wrap-around
    m = next_fast_length(n + 2 * pad)
    xp = np.zeros(m)
    xp[pad:pad + n] = x
    X = forward_transform(xp)
    f = np.arange(X.size) * (sr / m)
    Y = X * design_response(f, p)
    return inverse_transform(Y, m)[pad:pad + n]


# ----------------------------------------------------------------------------- stage 2: adaptive spectral gate
def estimate_noise_psd(power: np.ndarray, quiet_fraction: float = 0.15, min_frames: int = 8) -> np.ndarray:
    """Per-bin mean power of the quietest STFT frames (noise-only estimate)."""
    frame_power = power.sum(axis=1)
    k = int(min(max(round(len(frame_power) * quiet_fraction), min_frames), len(frame_power)))
    idx = np.argsort(frame_power)[:k]
    return power[idx].mean(axis=0)


def spectral_gate(x: np.ndarray, sr: float, strength: float) -> np.ndarray:
    """Wiener-style gain on the STFT. strength 0..1 -> max attenuation 6..24 dB, over-subtraction 1..3.5."""
    if strength <= 0:
        return x
    n_fft = int(2 ** np.ceil(np.log2(max(0.04 * sr, 256))))
    hop = n_fft // 4
    Z, pad = stft(x, n_fft, hop)
    power = Z.real ** 2 + Z.imag ** 2
    noise = estimate_noise_psd(power[(n_fft // hop):-(n_fft // hop)] if power.shape[0] > 20 else power)
    alpha = 1.0 + 2.5 * strength
    g_min = 10.0 ** (-(6.0 + 18.0 * strength) / 20.0)
    gain = np.sqrt(np.maximum(1.0 - alpha * noise / (power + 1e-20), 0.0))
    gain = np.maximum(gain, g_min)
    gain = ndimage.uniform_filter(gain, size=(3, 3), mode="nearest")          # suppress musical noise
    return istft(Z * np.maximum(gain, g_min), n_fft, hop, x.size, pad)


# ----------------------------------------------------------------------------- full pipeline
@dataclass
class FilterResult:
    signal: np.ndarray
    params: FilterParams
    output_gain_db: float
    response: dict


def process(x: np.ndarray, sr: float, p: FilterParams) -> FilterResult:
    """Stage 1 (convolution theorem) + stage 2 (spectral gate) + clip protection."""
    y = apply_frequency_filter(x, sr, p)
    y = spectral_gate(y, sr, p.gate_strength)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    gain_db = 0.0
    if peak > 0.99:
        g = 0.99 / peak
        y = y * g
        gain_db = float(20 * np.log10(g))
    return FilterResult(y, p, gain_db, response_curve(sr, p))
