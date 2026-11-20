import unittest

import numpy as np

from tests.helpers import sine
from resonoise import fourier as F


def naive_dft(x):
    n = len(x)
    k = np.arange(n // 2 + 1)[:, None]
    return (x[None, :] * np.exp(-2j * np.pi * k * np.arange(n)[None, :] / n)).sum(axis=1)


class TransformTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_matches_definition_of_the_dft(self):
        for n in (64, 255, 1000):
            x = self.rng.standard_normal(n)
            np.testing.assert_allclose(F.forward_transform(x), naive_dft(x), atol=1e-9)

    def test_inverse_reconstructs_signal(self):
        for n in (500, 501):
            x = self.rng.standard_normal(n)
            y = F.inverse_transform(F.forward_transform(x), n)
            np.testing.assert_allclose(y, x, atol=1e-12)

    def test_magnitude_is_sqrt_of_re2_plus_im2(self):
        X = F.forward_transform(self.rng.standard_normal(256))
        np.testing.assert_allclose(F.magnitude_spectrum(X), np.sqrt(X.real ** 2 + X.imag ** 2))

    def test_amplitude_spectrum_of_a_sine(self):
        sr, n = 8000, 8000
        amp = F.amplitude_spectrum(F.forward_transform(sine(1000, 1.0, sr, 0.37)), n)
        f = F.frequency_axis(n, sr)
        self.assertAlmostEqual(f[np.argmax(amp)], 1000.0)
        self.assertAlmostEqual(amp.max(), 0.37, places=6)

    def test_parseval_even_odd_and_padded(self):
        for n in (1000, 1001):
            x = self.rng.standard_normal(n)
            self.assertLess(F.parseval_check(x)["relative_error"], 1e-12)
        x = self.rng.standard_normal(1001)
        m = F.next_fast_length(1001)
        self.assertLess(F.parseval_check(x, F.forward_transform(x, m), n_fft=m)["relative_error"], 1e-12)

    def test_frequency_axis(self):
        f = F.frequency_axis(1000, 8000)
        self.assertEqual(len(f), 501)
        self.assertEqual(f[1], 8.0)
        self.assertEqual(f[-1], 4000.0)


class WindowedTests(unittest.TestCase):
    def test_stft_istft_perfect_reconstruction(self):
        x = np.random.default_rng(1).standard_normal(5000)
        for n_fft, hop in ((256, 64), (512, 128), (1024, 256)):
            Z, pad = F.stft(x, n_fft, hop)
            np.testing.assert_allclose(F.istft(Z, n_fft, hop, x.size, pad), x, atol=1e-10)

    def test_welch_psd_integrates_to_variance(self):
        x = np.random.default_rng(2).standard_normal(64000) * 0.5
        f, p = F.welch_psd(x, 16000, 1024)
        self.assertAlmostEqual(np.sum(p) * (f[1] - f[0]), np.var(x), delta=0.02 * np.var(x))

    def test_welch_amplitude_reads_sine_amplitude(self):
        f, a = F.welch_amplitude_spectrum(sine(1000.0, 4.0, 16000, 0.5), 16000, 4096)
        self.assertAlmostEqual(a.max(), 0.5, delta=0.02)
        self.assertAlmostEqual(f[np.argmax(a)], 1000.0, delta=4)


class PeakTests(unittest.TestCase):
    def setUp(self):
        self.sr = 16000
        rng = np.random.default_rng(3)
        t = np.arange(self.sr * 4) / self.sr
        self.hum = sum(np.sin(2 * np.pi * 120 * k * t) / k for k in (1, 2, 3, 4)) + 0.3 * rng.standard_normal(t.size)

    def test_finds_hum_harmonics_with_accurate_frequencies(self):
        f, p = F.welch_psd(self.hum, self.sr, F.analysis_nfft(self.sr))
        peaks = F.find_spectral_peaks(f, p)
        found = sorted(round(pk["freq_hz"]) for pk in peaks[:4])
        self.assertEqual(found, [120, 240, 360, 480])
        for pk in peaks[:4]:
            self.assertGreater(pk["prominence_db"], 15)

    def test_harmonic_series_fundamental(self):
        f, p = F.welch_psd(self.hum, self.sr, F.analysis_nfft(self.sr))
        h = F.detect_harmonic_series(F.find_spectral_peaks(f, p))
        self.assertAlmostEqual(h["f0_hz"], 120.0, delta=1.0)
        self.assertGreaterEqual(len(h["harmonics"]), 3)

    def test_no_harmonic_series_in_white_noise(self):
        x = np.random.default_rng(4).standard_normal(self.sr * 3)
        f, p = F.welch_psd(x, self.sr)
        self.assertIsNone(F.detect_harmonic_series(F.find_spectral_peaks(f, p)))

    def test_stationarity_separates_hum_from_bursty_tone(self):
        t = np.arange(self.sr * 6) / self.sr
        bursty = np.sin(2 * np.pi * 200 * t) * (np.sin(2 * np.pi * 0.7 * t) > 0.2)
        bursty += 0.01 * np.random.default_rng(5).standard_normal(t.size)
        steady, burst = F.peak_stationarity(self.hum, self.sr, [120.0])[0], F.peak_stationarity(bursty, self.sr, [200.0])[0]
        self.assertLess(steady, 2.0)
        self.assertGreater(burst, 5.0)

    def test_band_energies_sum_to_total_and_locate_tone(self):
        x = sine(300, 2, 16000, 1.0)
        f, p = F.welch_psd(x, 16000)
        bands = F.band_energies(f, p, [20, 250, 500, 1000, 8000])
        self.assertAlmostEqual(sum(b["share"] for b in bands), 1.0, places=6)
        self.assertGreater(bands[1]["share"], 0.95)

    def test_spectral_summary(self):
        f, p = F.welch_psd(sine(1000, 2, 16000), 16000)
        s = F.spectral_summary(f, p)
        self.assertAlmostEqual(s["peak_hz"], 1000, delta=4)
        self.assertAlmostEqual(s["centroid_hz"], 1000, delta=30)


if __name__ == "__main__":
    unittest.main()
