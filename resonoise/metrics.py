"""Evaluation metrics (stage 8): computed from the actual input and output samples."""
from __future__ import annotations

import numpy as np

from .features import BAND_EDGES, frame_rms_db
from .fourier import analysis_nfft, band_energies, forward_transform, parseval_check, welch_psd
from .utils import db, rms_db


def si_sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Scale-invariant signal-to-distortion ratio in dB (higher is better)."""
    ref = reference - reference.mean()
    est = estimate - estimate.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-20)
    target = alpha * ref
    noise = est - target
    return float(10 * np.log10((np.sum(target ** 2) + 1e-20) / (np.sum(noise ** 2) + 1e-20)))


def evaluate(x: np.ndarray, y: np.ndarray, sr: int, reference: np.ndarray | None = None,
             tracked_peaks: list[float] | None = None) -> dict:
    """Compare original x with processed y. All quantities derive from the two signals."""
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]

    pe_x, pe_y = parseval_check(x), parseval_check(y)
    energy_removed = 100.0 * (1.0 - pe_y["frequency_energy"] / max(pe_x["frequency_energy"], 1e-300))

    lv_x, lv_y = frame_rms_db(x, sr), frame_rms_db(y, sr)
    floor_x, floor_y = float(np.percentile(lv_x, 10)), float(np.percentile(lv_y, 10))
    top_x, top_y = float(np.percentile(lv_x, 95)), float(np.percentile(lv_y, 95))

    nfft = analysis_nfft(sr)
    fx, px = welch_psd(x, sr, nfft)
    _, py = welch_psd(y, sr, nfft)
    df = fx[1] - fx[0]

    edges = [e for e in BAND_EDGES if e < sr / 2.0]
    edges.append(min(BAND_EDGES[-1], sr / 2.0)) if edges[-1] < min(BAND_EDGES[-1], sr / 2.0) else None
    bx, by = band_energies(fx, px, edges), band_energies(fx, py, edges)
    bands = [{"lo": a["lo"], "hi": a["hi"], "original_db": a["level_db"], "processed_db": b["level_db"],
              "change_db": b["level_db"] - a["level_db"]} for a, b in zip(bx, by)]

    tracked = []
    for f0 in (tracked_peaks or []):
        k = int(round(f0 / df))
        sl = slice(max(k - 2, 0), k + 3)
        tracked.append({"freq_hz": float(f0), "original_db": float(db(px[sl].max())),
                        "processed_db": float(db(py[sl].max())),
                        "change_db": float(db(py[sl].max()) - db(px[sl].max()))})

    band = (fx >= 20) & (fx <= min(8000, sr / 2.0))
    lsd = float(np.sqrt(np.mean((db(px[band]) - db(py[band])) ** 2)))

    def centroid(p):
        return float(np.sum(fx[band] * p[band]) / (np.sum(p[band]) + 1e-30))

    def flatness(p):
        q = p[band] + 1e-30
        return float(db(np.exp(np.mean(np.log(q))) / np.mean(q)))

    voice = (fx >= 300) & (fx <= min(3400, sr / 2.0 - 1))
    voice_change = float(db(np.sum(py[voice])) - db(np.sum(px[voice])))

    xc, yc = x - x.mean(), y - y.mean()
    corr = float(np.dot(xc, yc) / (np.linalg.norm(xc) * np.linalg.norm(yc) + 1e-20))

    out = {
        "rms_in_db": rms_db(x), "rms_out_db": rms_db(y), "level_change_db": rms_db(y) - rms_db(x),
        "peak_in": float(np.max(np.abs(x))), "peak_out": float(np.max(np.abs(y))),
        "energy_in": pe_x["time_energy"], "energy_out": pe_y["time_energy"],
        "energy_in_freq": pe_x["frequency_energy"], "energy_out_freq": pe_y["frequency_energy"],
        "energy_removed_pct": float(energy_removed),
        "noise_floor_in_db": floor_x, "noise_floor_out_db": floor_y, "noise_floor_reduction_db": floor_x - floor_y,
        "signal_to_floor_in_db": top_x - floor_x, "signal_to_floor_out_db": top_y - floor_y,
        "signal_to_floor_gain_db": (top_y - floor_y) - (top_x - floor_x),
        "correlation": corr, "log_spectral_distance_db": lsd,
        "centroid_in_hz": centroid(px), "centroid_out_hz": centroid(py),
        "flatness_in_db": flatness(px), "flatness_out_db": flatness(py),
        "voice_band_change_db": voice_change,
        "bands": bands, "tracked_peaks": tracked,
        "reference": None,
    }
    if reference is not None and reference.size >= n:
        ref = reference[:n]
        before, after = si_sdr(x, ref), si_sdr(y, ref)
        out["reference"] = {"si_sdr_in_db": before, "si_sdr_out_db": after, "improvement_db": after - before}
    return out
