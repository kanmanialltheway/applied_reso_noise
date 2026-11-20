"""Shared test utilities."""
import io
import os
import sys

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def wav16(x, sr):
    buf = io.BytesIO()
    wavfile.write(buf, sr, np.clip(np.round(np.asarray(x) * 32768), -32768, 32767).astype(np.int16))
    return buf.getvalue()


def sine(freq, dur, sr, amp=1.0, phase=0.0):
    t = np.arange(int(dur * sr)) / sr
    return amp * np.sin(2 * np.pi * freq * t + phase)


def level_db_at(x, sr, freq, half_width_hz=3.0):
    """Peak amplitude (dB) of the whole-signal DFT magnitude near `freq`."""
    n = len(x)
    spec = np.abs(np.fft.rfft(x * np.hanning(n))) * 2 / np.sum(np.hanning(n))
    f = np.fft.rfftfreq(n, 1 / sr)
    m = (f > freq - half_width_hz) & (f < freq + half_width_hz)
    return 20 * np.log10(spec[m].max() + 1e-12)
