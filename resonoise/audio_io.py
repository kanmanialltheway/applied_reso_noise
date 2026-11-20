"""Audio acquisition and preprocessing (stages A.1 and A.2 of the system architecture).

* decode_audio      - bytes -> float samples, from WAV natively, other formats via soundfile or ffmpeg
* preprocess        - mono conversion, DC removal, validation, peak normalisation
* wav_bytes         - float samples -> 16-bit PCM WAV bytes
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from math import gcd

import numpy as np
from scipy import signal
from scipy.io import wavfile

from . import config
from .errors import InvalidAudioError, UnsupportedFormatError

try:  # optional: adds FLAC / OGG / MP3 / AIFF without needing ffmpeg
    import soundfile as _sf
except Exception:  # pragma: no cover - depends on the environment
    _sf = None


@dataclass
class DecodedAudio:
    samples: np.ndarray        # float64, shape (n,) mono, nominal range [-1, 1]
    sample_rate: int
    channels: int
    container: str


@dataclass
class PreparedAudio:
    samples: np.ndarray        # mono, DC-free, peak-normalised float64
    sample_rate: int
    original_sample_rate: int
    channels: int
    duration: float
    peak_before: float
    container: str


# ----------------------------------------------------------------------------- format sniffing
def sniff_container(data: bytes) -> str:
    """Identify the container from its magic bytes; returns 'unknown' when not audio."""
    h = data[:16]
    if h[:4] == b"RIFF" and h[8:12] == b"WAVE":
        return "wav"
    if h[:4] == b"fLaC":
        return "flac"
    if h[:4] == b"OggS":
        return "ogg"
    if h[:3] == b"ID3" or (len(h) > 1 and h[0] == 0xFF and (h[1] & 0xE0) == 0xE0):
        return "mp3"
    if h[4:8] == b"ftyp":
        return "m4a"
    if h[:4] == b"FORM" and h[8:12] in (b"AIFF", b"AIFC"):
        return "aiff"
    if h[:4] == b"\x1a\x45\xdf\xa3":
        return "webm"
    return "unknown"


# ----------------------------------------------------------------------------- decoders
def _pcm_to_float(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return (arr.astype(np.float64) - 128.0) / 128.0
    if arr.dtype == np.int16:
        return arr.astype(np.float64) / 32768.0
    if arr.dtype == np.int32:
        return arr.astype(np.float64) / 2147483648.0
    if arr.dtype in (np.float32, np.float64):
        return arr.astype(np.float64)
    raise ValueError(f"unsupported sample type {arr.dtype}")


def _decode_wav(data: bytes) -> DecodedAudio:
    rate, arr = wavfile.read(io.BytesIO(data))
    x = _pcm_to_float(np.asarray(arr))
    channels = 1 if x.ndim == 1 else x.shape[1]
    if x.ndim == 2:
        x = x.mean(axis=1)
    return DecodedAudio(x, int(rate), channels, "wav")


def _decode_soundfile(data: bytes, container: str) -> DecodedAudio:
    arr, rate = _sf.read(io.BytesIO(data), dtype="float64", always_2d=True)
    return DecodedAudio(arr.mean(axis=1), int(rate), arr.shape[1], container)


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _decode_ffmpeg(data: bytes, container: str) -> DecodedAudio:
    """Decode through the ffmpeg binary (temporary files keep the WAV header sizes valid)."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise UnsupportedFormatError(
            f"Decoding {container.upper()} files needs either the 'soundfile' package or ffmpeg. "
            "Install one of them or convert the file to WAV."
        )
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, "in.bin"), os.path.join(tmp, "out.wav")
        with open(src, "wb") as fh:
            fh.write(data)
        cmd = [exe, "-v", "error", "-nostdin", "-y", "-i", src, "-vn", "-acodec", "pcm_f32le", dst]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=120)
        except subprocess.TimeoutExpired as exc:
            raise InvalidAudioError("Decoding the file took too long and was cancelled.") from exc
        if proc.returncode != 0 or not os.path.exists(dst):
            msg = proc.stderr.decode("utf-8", "replace").strip().splitlines()
            raise InvalidAudioError(
                "The file could not be decoded as audio; it may be corrupt or truncated.",
                details={"decoder": (msg[-1] if msg else "ffmpeg failed")},
            )
        with open(dst, "rb") as fh:
            decoded = _decode_wav(fh.read())
    decoded.container = container
    return decoded


def decode_audio(data: bytes, filename: str = "") -> DecodedAudio:
    """Decode an uploaded or recorded file into mono float samples."""
    if not data:
        raise InvalidAudioError("The file is empty.")
    container = sniff_container(data)
    ext = os.path.splitext(filename or "")[1].lower()
    if container == "unknown":
        raise UnsupportedFormatError(
            "This file is not a recognised audio format. Use WAV, FLAC, OGG, MP3, M4A or AIFF.",
            details={"extension": ext or None},
        )

    errors: list[str] = []
    if container == "wav":
        try:
            return _decode_wav(data)
        except Exception as exc:  # unsupported WAV encodings (ADPCM, u-law ...) or damage
            errors.append(str(exc))
    if _sf is not None:
        try:
            return _decode_soundfile(data, container)
        except Exception as exc:
            errors.append(str(exc))
    if ffmpeg_available():
        return _decode_ffmpeg(data, container)
    if container == "wav":
        raise InvalidAudioError(
            "This WAV file uses an encoding that could not be read, or the file is damaged.",
            details={"reason": errors[-1] if errors else None},
        )
    raise UnsupportedFormatError(
        f"Decoding {container.upper()} files needs either the 'soundfile' package or ffmpeg. "
        "Install one of them or convert the file to WAV."
    )


# ----------------------------------------------------------------------------- DSP helpers
def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """Polyphase resampling (anti-aliased)."""
    if sr_from == sr_to:
        return np.asarray(x, dtype=np.float64)
    g = gcd(int(sr_from), int(sr_to))
    return signal.resample_poly(np.asarray(x, dtype=np.float64), int(sr_to) // g, int(sr_from) // g)


def preprocess(decoded: DecodedAudio) -> PreparedAudio:
    """Validate, convert to mono, remove DC offset, limit sample rate and peak-normalise."""
    x, sr = decoded.samples, decoded.sample_rate
    if not config.MIN_SAMPLE_RATE <= sr <= config.MAX_NATIVE_SAMPLE_RATE:
        raise InvalidAudioError(
            f"Sample rate {sr} Hz is outside the supported range "
            f"({config.MIN_SAMPLE_RATE}-{config.MAX_NATIVE_SAMPLE_RATE} Hz)."
        )
    if x.size == 0:
        raise InvalidAudioError("The file contains no audio samples.")
    if not np.all(np.isfinite(x)):
        raise InvalidAudioError("The file contains invalid (NaN or infinite) sample values.")
    duration = x.size / sr
    if duration < config.MIN_DURATION_S:
        raise InvalidAudioError(
            f"The recording is {duration:.2f} s long; at least {config.MIN_DURATION_S:.1f} s is needed."
        )
    if duration > config.MAX_DURATION_S:
        raise InvalidAudioError(
            f"The recording is {duration:.0f} s long; the limit is {config.MAX_DURATION_S:.0f} s."
        )

    x = x - np.mean(x)                                   # DC removal
    peak_before = float(np.max(np.abs(x)))
    if peak_before < config.SILENCE_PEAK:
        raise InvalidAudioError("The recording is silent (no signal above the noise of the converter).")

    if sr > config.WORK_MAX_SAMPLE_RATE:
        x = resample(x, sr, config.WORK_MAX_SAMPLE_RATE)
        out_sr = config.WORK_MAX_SAMPLE_RATE
    else:
        out_sr = sr
    x = x * (config.PEAK_TARGET / max(float(np.max(np.abs(x))), 1e-12))   # peak normalisation
    return PreparedAudio(
        samples=x, sample_rate=int(out_sr), original_sample_rate=int(sr), channels=decoded.channels,
        duration=x.size / out_sr, peak_before=peak_before, container=decoded.container,
    )


def quantize16(x: np.ndarray) -> np.ndarray:
    """Return the signal exactly as a 16-bit file would store it (still as floats)."""
    return np.clip(np.round(np.asarray(x) * 32768.0), -32768, 32767) / 32768.0


def wav_bytes(x: np.ndarray, sr: int) -> bytes:
    """Encode float samples as 16-bit PCM mono WAV."""
    pcm = np.clip(np.round(np.asarray(x) * 32768.0), -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, int(sr), pcm)
    return buf.getvalue()


def read_wav_file(path) -> tuple[np.ndarray, int]:
    """Read a WAV written by this application back into float samples."""
    with open(path, "rb") as fh:
        dec = _decode_wav(fh.read())
    return dec.samples, dec.sample_rate
