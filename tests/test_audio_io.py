import io
import shutil
import struct
import unittest

import numpy as np
from scipy.io import wavfile

from tests.helpers import sine, wav16
from resonoise import audio_io, config
from resonoise.errors import InvalidAudioError, UnsupportedFormatError


class DecodeTests(unittest.TestCase):
    def test_wav_16bit_roundtrip(self):
        x = sine(440, 1.0, 16000, 0.5)
        d = audio_io.decode_audio(wav16(x, 16000), "a.wav")
        self.assertEqual(d.sample_rate, 16000)
        self.assertEqual(d.channels, 1)
        self.assertLess(np.max(np.abs(d.samples - x)), 1e-4)

    def test_wav_float32_and_int32_and_uint8(self):
        x = sine(300, 0.6, 8000, 0.4)
        for dtype, data in (("f32", x.astype(np.float32)), ("i32", (x * 2**31 * 0.999).astype(np.int32)),
                            ("u8", (x * 127 + 128).astype(np.uint8))):
            buf = io.BytesIO()
            wavfile.write(buf, 8000, data)
            d = audio_io.decode_audio(buf.getvalue(), "x.wav")
            tol = 0.02 if dtype == "u8" else 1e-3
            self.assertLess(np.max(np.abs(d.samples - x)), tol, dtype)

    def test_wav_24bit(self):
        x = sine(440, 0.6, 8000, 0.5)
        pcm = (x * 8388607).astype(np.int32)
        body = b"".join(int(v).to_bytes(4, "little", signed=True)[:3] for v in pcm)
        hdr = (b"RIFF" + struct.pack("<I", 36 + len(body)) + b"WAVEfmt " +
               struct.pack("<IHHIIHH", 16, 1, 1, 8000, 8000 * 3, 3, 24) + b"data" + struct.pack("<I", len(body)))
        d = audio_io.decode_audio(hdr + body, "x.wav")
        self.assertLess(np.max(np.abs(d.samples - x)), 1e-5)

    def test_stereo_is_downmixed(self):
        l, r = sine(200, 1, 8000, 0.5), sine(200, 1, 8000, 0.5)
        buf = io.BytesIO()
        wavfile.write(buf, 8000, np.stack([l, r], axis=1).astype(np.float32))
        d = audio_io.decode_audio(buf.getvalue(), "s.wav")
        self.assertEqual(d.channels, 2)
        self.assertEqual(d.samples.ndim, 1)
        self.assertLess(np.max(np.abs(d.samples - l)), 1e-6)

    def test_rejects_non_audio(self):
        with self.assertRaises(UnsupportedFormatError):
            audio_io.decode_audio(b"%PDF-1.4 not audio at all", "doc.pdf")
        with self.assertRaises(UnsupportedFormatError):
            audio_io.decode_audio(b"\x89PNG\r\n\x1a\n" + b"0" * 50, "pic.wav")

    def test_rejects_empty_and_truncated(self):
        with self.assertRaises(InvalidAudioError):
            audio_io.decode_audio(b"", "e.wav")
        good = wav16(sine(440, 1, 8000, 0.5), 8000)
        with self.assertRaises((InvalidAudioError, UnsupportedFormatError)):
            audio_io.decode_audio(good[:30], "cut.wav")

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
    def test_non_wav_formats_via_ffmpeg(self):
        import subprocess, tempfile, os
        x = sine(1000, 1.5, 22050, 0.5)
        with tempfile.TemporaryDirectory() as tmp:
            wav = os.path.join(tmp, "a.wav")
            open(wav, "wb").write(wav16(x, 22050))
            for ext in ("flac", "mp3", "ogg", "m4a"):
                out = os.path.join(tmp, "a." + ext)
                r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", wav, out], capture_output=True)
                if r.returncode != 0:
                    continue                                   # encoder not compiled in
                d = audio_io.decode_audio(open(out, "rb").read(), "a." + ext)
                self.assertEqual(d.samples.ndim, 1, ext)
                spec = np.abs(np.fft.rfft(d.samples))
                peak = np.fft.rfftfreq(len(d.samples), 1 / d.sample_rate)[np.argmax(spec)]
                self.assertAlmostEqual(peak, 1000, delta=15, msg=ext)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
    def test_corrupt_flac_is_reported(self):
        with self.assertRaises((InvalidAudioError, UnsupportedFormatError)):
            audio_io.decode_audio(b"fLaC" + b"\x00" * 200, "bad.flac")


class PreprocessTests(unittest.TestCase):
    def dec(self, x, sr=16000, ch=1):
        return audio_io.DecodedAudio(np.asarray(x, dtype=float), sr, ch, "wav")

    def test_normalises_and_removes_dc(self):
        x = 0.1 * sine(440, 1, 16000) + 0.3
        p = audio_io.preprocess(self.dec(x))
        self.assertAlmostEqual(np.max(np.abs(p.samples)), config.PEAK_TARGET, places=6)
        self.assertLess(abs(np.mean(p.samples)), 1e-3)

    def test_rejects_silence_short_long_nan_and_bad_rate(self):
        with self.assertRaises(InvalidAudioError):
            audio_io.preprocess(self.dec(np.zeros(16000)))
        with self.assertRaises(InvalidAudioError):
            audio_io.preprocess(self.dec(sine(440, 0.2, 16000)))
        with self.assertRaises(InvalidAudioError):
            audio_io.preprocess(self.dec(np.r_[np.sin(np.arange(16000)), np.nan]))
        with self.assertRaises(InvalidAudioError):
            audio_io.preprocess(self.dec(sine(100, 1, 1000), sr=1000))
        old = config.MAX_DURATION_S
        config.MAX_DURATION_S = 2.0
        try:
            with self.assertRaises(InvalidAudioError):
                audio_io.preprocess(self.dec(sine(440, 3, 16000)))
        finally:
            config.MAX_DURATION_S = old

    def test_high_sample_rate_is_resampled(self):
        p = audio_io.preprocess(self.dec(sine(1000, 1.0, 96000, 0.5), sr=96000))
        self.assertEqual(p.sample_rate, 48000)
        self.assertEqual(p.original_sample_rate, 96000)
        self.assertAlmostEqual(p.duration, 1.0, places=3)
        f = np.fft.rfftfreq(len(p.samples), 1 / 48000)
        self.assertAlmostEqual(f[np.argmax(np.abs(np.fft.rfft(p.samples)))], 1000, delta=2)

    def test_resample_preserves_tone_frequency(self):
        y = audio_io.resample(sine(500, 1.0, 44100, 0.5), 44100, 16000)
        f = np.fft.rfftfreq(len(y), 1 / 16000)
        self.assertAlmostEqual(f[np.argmax(np.abs(np.fft.rfft(y)))], 500, delta=2)

    def test_wav_bytes_and_quantize(self):
        x = sine(440, 0.5, 8000, 0.7)
        back = audio_io.decode_audio(audio_io.wav_bytes(x, 8000), "a.wav").samples
        self.assertLessEqual(np.max(np.abs(back - x)), 0.5 / 32768 + 1e-12)
        np.testing.assert_allclose(audio_io.quantize16(x), back, atol=1e-12)
        self.assertEqual(audio_io.quantize16(np.array([2.0, -2.0])).tolist(), [32767 / 32768, -1.0])


if __name__ == "__main__":
    unittest.main()
