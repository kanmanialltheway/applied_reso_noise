"""Physically-motivated synthetic residential noise, used to train the classifier and to build demo clips.

No third-party recordings are bundled, so the generators model how each source sounds:

* fan        - blade-pass tone and harmonics (drifting) over turbulent broadband noise
* ac         - compressor / transformer hum on mains harmonics (100/120 Hz) over airflow noise
* traffic    - low-frequency road rumble, tyre band noise and pass-by events with engine harmonics
* excavator  - diesel firing pulse train (low fundamental, rich harmonics), hydraulic whine, impulsive clanks
* speech     - glottal source filtered by formant resonators, syllabic envelope, pauses, fricatives
"""
from __future__ import annotations

import numpy as np
from scipy import fft as sfft
from scipy import signal

SR = 16000
CLASS_KEYS = ["fan", "ac", "traffic", "excavator", "speech"]


def _normalise(x: np.ndarray) -> np.ndarray:
    return x / (np.std(x) + 1e-12)


def colored_noise(n: int, rng, exponent: float = 1.0, sr: int = SR, fmin: float = 20.0, fmax: float | None = None):
    """Noise with power ~ 1/f^exponent, high-passed at fmin and low-passed at fmax."""
    X = sfft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / sr)
    shape = np.zeros_like(f)
    shape[1:] = f[1:] ** (-exponent / 2.0)
    shape *= 1.0 / np.sqrt(1.0 + (fmin / np.maximum(f, 1e-9)) ** 4)
    if fmax:
        shape *= 1.0 / np.sqrt(1.0 + (f / fmax) ** 8)
    return _normalise(sfft.irfft(X * shape, n))


def _bandpass(x, lo, hi, sr=SR, order=4):
    hi = min(hi, 0.45 * sr)
    sos = signal.butter(order, [lo, hi], btype="band", fs=sr, output="sos")
    return signal.sosfilt(sos, x)


def _slow_drift(n, rng, sr, rate_hz=0.5):
    """Smooth random curve in [-1, 1]."""
    m = max(int(n / sr * rate_hz * 4), 4)
    pts = rng.standard_normal(m)
    curve = np.interp(np.linspace(0, m - 1, n), np.arange(m), pts)
    return curve / (np.max(np.abs(curve)) + 1e-12)


def _harmonics(n, rng, f0, n_harm, decay, sr=SR, f_drift=0.0, rate_hz=0.5):
    inst = f0 * (1.0 + f_drift * _slow_drift(n, rng, sr, rate_hz))
    phase = 2 * np.pi * np.cumsum(inst) / sr
    out = np.zeros(n)
    for k in range(1, n_harm + 1):
        if k * f0 > 0.45 * sr:
            break
        out += (k ** -decay) * rng.uniform(0.6, 1.0) * np.sin(k * phase + rng.uniform(0, 2 * np.pi))
    return _normalise(out)


def gen_fan(n, rng, sr=SR):
    f0 = rng.uniform(45, 240)
    tonal = _harmonics(n, rng, f0, int(rng.integers(3, 9)), rng.uniform(0.5, 1.4), sr, rng.uniform(0.001, 0.01))
    broad = colored_noise(n, rng, rng.uniform(0.3, 1.2), sr, fmax=rng.uniform(2000, 6000))
    y = broad + rng.uniform(0.35, 2.0) * tonal
    return _normalise(y * (1.0 + 0.05 * _slow_drift(n, rng, sr, 0.3)))


def gen_ac(n, rng, sr=SR):
    mains = float(rng.choice([50.0, 60.0]))
    f0 = 2 * mains * (1.0 + rng.uniform(-0.001, 0.001))
    hum = _harmonics(n, rng, f0, int(rng.integers(4, 13)), rng.uniform(0.3, 1.0), sr, 0.0005)
    if rng.random() < 0.6:
        hum += rng.uniform(0.2, 0.6) * np.sin(2 * np.pi * mains * np.arange(n) / sr + rng.uniform(0, 6.28))
    air = colored_noise(n, rng, rng.uniform(0.8, 1.8), sr, fmax=rng.uniform(1000, 3500))
    air += rng.uniform(0.2, 0.7) * _bandpass(rng.standard_normal(n), 300, 2500, sr)
    y = _normalise(air) + rng.uniform(0.5, 2.5) * _normalise(hum)
    return _normalise(y * (1.0 + 0.03 * _slow_drift(n, rng, sr, 0.2)))


def _bump(n, sr, centre_s, width_s):
    t = np.arange(n) / sr
    return np.exp(-0.5 * ((t - centre_s) / width_s) ** 2)


def gen_traffic(n, rng, sr=SR):
    dur = n / sr
    rumble = colored_noise(n, rng, rng.uniform(1.4, 2.2), sr, fmin=25, fmax=rng.uniform(700, 2200))
    tyre = _bandpass(rng.standard_normal(n), 500, 1800, sr)
    env = np.full(n, rng.uniform(0.25, 0.6))
    engine = np.zeros(n)
    for _ in range(int(rng.integers(1, 5))):
        bump = _bump(n, sr, rng.uniform(0.1, 0.9) * dur, rng.uniform(0.5, 1.8))
        env += rng.uniform(0.8, 2.5) * bump
        eng = _harmonics(n, rng, rng.uniform(28, 90), int(rng.integers(3, 8)), rng.uniform(0.8, 1.6), sr, 0.05, 1.0)
        engine += eng * bump * rng.uniform(0.3, 0.8)
    y = _normalise(rumble) * env + 0.35 * _normalise(tyre) * env + engine
    return _normalise(y)


def gen_excavator(n, rng, sr=SR):
    dur = n / sr
    fire = rng.uniform(18, 46)
    phase = 2 * np.pi * np.cumsum(fire * (1 + 0.02 * _slow_drift(n, rng, sr, 0.4))) / sr
    engine = np.zeros(n)
    for k in range(1, 31):
        if k * fire > 0.45 * sr:
            break
        engine += (k ** -rng.uniform(0.6, 1.1)) * np.sin(k * phase + rng.uniform(0, 6.28))
    engine = _normalise(engine)
    engine *= 1.0 + 0.35 * np.sin(2 * np.pi * rng.uniform(0.3, 1.2) * np.arange(n) / sr)
    whine_f = rng.uniform(250, 900)
    whine = _harmonics(n, rng, whine_f, 3, 1.0, sr, 0.03, 0.8)
    gate = np.clip(0.5 + 0.9 * _slow_drift(n, rng, sr, 0.35), 0, 1)
    clanks = np.zeros(n)
    for _ in range(int(rng.integers(int(2 * dur / 2) + 2, int(9 * dur / 2) + 4))):
        start = int(rng.uniform(0, 0.97) * n)
        length = min(int(sr * rng.uniform(0.04, 0.2)), n - start)
        env = np.exp(-np.arange(length) / (sr * rng.uniform(0.01, 0.05)))
        burst = _bandpass(rng.standard_normal(length), rng.uniform(600, 1500), rng.uniform(2000, 4500), sr)
        clanks[start:start + length] += burst * env * rng.uniform(1.0, 3.0)
    broad = colored_noise(n, rng, 1.0, sr, fmax=4000)
    y = engine * rng.uniform(1.0, 2.0) + 0.35 * whine * gate + 0.5 * _normalise(clanks) + 0.3 * broad
    return _normalise(y)


_VOWELS = [(730, 1090, 2440), (270, 2290, 3010), (300, 870, 2240), (570, 840, 2410), (530, 1840, 2480), (660, 1720, 2410)]


def _resonate(x, freq, bw, sr):
    r = np.exp(-np.pi * bw / sr)
    a = [1.0, -2 * r * np.cos(2 * np.pi * freq / sr), r * r]
    return signal.lfilter([1.0 - r], a, x)


def gen_speaker(n, rng, sr=SR, continuous=False):
    """One talker: syllables with changing vowels, pitch contour, pauses, occasional fricatives."""
    out = np.zeros(n)
    f0_base = rng.uniform(95, 250)
    scale = rng.uniform(0.88, 1.18) * (1.15 if f0_base > 170 else 1.0)
    t = int(rng.uniform(0, 0.4) * sr)
    contour = f0_base * 2 ** (3.0 * _slow_drift(n, rng, sr, 1.5) / 12.0)
    while t < n:
        syl = int(sr * rng.uniform(0.11, 0.26))
        end = min(t + syl, n)
        seg = end - t
        if seg < 32:
            break
        inst = contour[t:end] * (1 + 0.02 * np.sin(2 * np.pi * 5.5 * np.arange(seg) / sr))
        ph = np.cumsum(inst) / sr
        src = 2.0 * (ph - np.floor(ph)) - 1.0
        src = signal.lfilter([1.0], [1.0, -0.95], src)
        f1, f2, f3 = _VOWELS[int(rng.integers(len(_VOWELS)))]
        v = _resonate(src, f1 * scale, 80, sr)
        v = _resonate(v, f2 * scale, 110, sr) + 0.0
        v = _resonate(v, f3 * scale, 160, sr)
        v = _normalise(v) * np.hanning(seg) ** 0.7 * rng.uniform(0.6, 1.0)
        out[t:end] += v
        if rng.random() < 0.35:                                   # fricative onset
            fl = min(int(sr * 0.06), t)
            if fl > 16:
                fr = _bandpass(rng.standard_normal(fl), rng.uniform(2500, 4000), 7000, sr)
                out[t - fl:t] += 0.25 * _normalise(fr) * np.hanning(fl)
        t = end
        if not continuous and rng.random() < 0.18:                # phrase pause
            t += int(sr * rng.uniform(0.25, 0.9))
        else:
            t += int(sr * rng.uniform(0.0, 0.05))
    return _normalise(out) if np.std(out) > 0 else out


def gen_speech(n, rng, sr=SR):
    people = int(rng.integers(1, 4))
    y = np.zeros(n)
    for _ in range(people):
        y += gen_speaker(n, rng, sr, continuous=people > 1) * rng.uniform(0.6, 1.0)
    y = _normalise(y) + 0.03 * colored_noise(n, rng, 1.5, sr, fmax=3000)
    return _normalise(y)


GENERATORS = {"fan": gen_fan, "ac": gen_ac, "traffic": gen_traffic, "excavator": gen_excavator, "speech": gen_speech}


def _random_mic_colouration(x, rng, sr=SR):
    """Random tilt + high-pass, emulating different microphones and rooms."""
    X = sfft.rfft(x)
    f = np.fft.rfftfreq(x.size, 1 / sr)
    tilt = (np.maximum(f, 50.0) / 1000.0) ** rng.uniform(-0.25, 0.25)
    hp = 1.0 / np.sqrt(1.0 + (rng.uniform(15, 90) / np.maximum(f, 1e-9)) ** 4)
    return _normalise(sfft.irfft(X * tilt * hp, x.size))


def make_training_clip(label: str, rng, duration: float = 6.0, sr=SR) -> np.ndarray:
    """A randomised clip of one class, with microphone colouration and optional background."""
    n = int(duration * sr)
    y = GENERATORS[label](n, rng, sr)
    y = _random_mic_colouration(y, rng, sr)
    if rng.random() < 0.5:                                        # unrelated background source
        other = str(rng.choice([c for c in CLASS_KEYS if c != label]))
        bg = GENERATORS[other](n, rng, sr)
        y = y + bg * 10 ** (-rng.uniform(14, 30) / 20.0)
    y = y + rng.standard_normal(n) * 10 ** (-rng.uniform(32, 55) / 20.0)
    return y / np.max(np.abs(y)) * 0.9


def noise_scene(kind: str, rng, duration: float = 8.0, snr_db: float = 3.0, sr=SR):
    """Demo clip. kind is a class key (pure noise) or 'speech_<noise>' (a talker over that noise).

    Returns dict(mix, clean, noise, sr); `clean` is the speech component when applicable.
    """
    n = int(duration * sr)
    if kind in GENERATORS and kind != "speech":
        noise = _random_mic_colouration(GENERATORS[kind](n, rng, sr), rng, sr)
        return {"mix": noise / np.max(np.abs(noise)) * 0.9, "clean": None, "noise": noise, "sr": sr}
    if kind == "speech":
        sp = gen_speaker(n, rng, sr)
        room = 0.01 * colored_noise(n, rng, 1.5, sr, fmax=3000)
        mix = sp + room
        return {"mix": mix / np.max(np.abs(mix)) * 0.9, "clean": sp, "noise": room, "sr": sr}
    if kind.startswith("speech_") and kind[7:] in GENERATORS:
        sp = gen_speaker(n, rng, sr)
        noise = GENERATORS[kind[7:]](n, rng, sr)
        noise = noise * (np.std(sp) / np.std(noise)) * 10 ** (-snr_db / 20.0)
        mix = sp + noise
        g = 0.9 / np.max(np.abs(mix))
        return {"mix": mix * g, "clean": sp * g, "noise": noise * g, "sr": sr}
    raise ValueError(f"unknown scene '{kind}'")


def build_feature_dataset(per_class_train: int = 64, per_class_val: int = 16, seed: int = 7, progress=None):
    """Generate clips and extract their features. Returns (X, y, split) with split 0=train, 1=validation."""
    from .features import FEATURE_NAMES, extract_features

    rng = np.random.default_rng(seed)
    rows, labels, split = [], [], []
    total = len(CLASS_KEYS) * (per_class_train + per_class_val)
    done = 0
    for ci, key in enumerate(CLASS_KEYS):
        for i in range(per_class_train + per_class_val):
            clip = make_training_clip(key, rng, duration=float(rng.uniform(4.0, 7.0)))
            f = extract_features(clip, SR)
            rows.append([f[k] for k in FEATURE_NAMES])
            labels.append(ci)
            split.append(0 if i < per_class_train else 1)
            done += 1
            if progress:
                progress(done, total)
    return np.array(rows), np.array(labels), np.array(split)
