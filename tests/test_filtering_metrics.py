import unittest

import numpy as np
from scipy import signal

from tests.helpers import level_db_at, sine
from resonoise import filtering as Fl
from resonoise import fourier, metrics, synthetic
from resonoise.errors import ValidationError

SR = 16000


class ResponseTests(unittest.TestCase):
    def test_butterworth_cutoffs_are_minus_3db(self):
        self.assertAlmostEqual(20 * np.log10(Fl.highpass_gain(np.array([100.0]), 100.0)[0]), -3.01, places=1)
        self.assertAlmostEqual(20 * np.log10(Fl.lowpass_gain(np.array([5000.0]), 5000.0)[0]), -3.01, places=1)
        self.assertGreater(Fl.highpass_gain(np.array([1000.0]), 100.0)[0], 0.999)
        self.assertLess(Fl.highpass_gain(np.array([10.0]), 100.0)[0], 1e-3)

    def test_notch_depth_and_width(self):
        f = np.linspace(80, 160, 8001)
        g = Fl.notch_gain(f, 120.0, 4.0, 30.0)
        self.assertAlmostEqual(20 * np.log10(g[np.argmin(np.abs(f - 120))]), -30.0, places=2)
        half = f[g < (1 + 10 ** (-30 / 20)) / 2]
        self.assertAlmostEqual(half.max() - half.min(), 4.0, delta=0.05)         # FWHM
        self.assertGreater(g[np.argmin(np.abs(f - 130))], 0.999)

    def test_design_response_is_the_product_of_parts(self):
        p = Fl.FilterParams(highpass_hz=100, lowpass_hz=5000, notch_freqs=[300.0], notch_depth_db=20, notch_width_hz=5)
        f = np.array([50.0, 300.0, 1000.0, 7000.0])
        expected = Fl.highpass_gain(f, 100) * Fl.lowpass_gain(f, 5000) * Fl.notch_gain(f, 300, 5, 20)
        np.testing.assert_allclose(Fl.design_response(f, p), expected)


class FilterTests(unittest.TestCase):
    def test_identity_filter_returns_the_input(self):
        x = np.random.default_rng(0).standard_normal(SR * 2)
        np.testing.assert_allclose(Fl.apply_frequency_filter(x, SR, Fl.FilterParams()), x, atol=1e-10)

    def test_notch_removes_tone_and_keeps_neighbours(self):
        x = sine(120, 4, SR, 0.4) + sine(1000, 4, SR, 0.4)
        y = Fl.apply_frequency_filter(x, SR, Fl.FilterParams(notch_freqs=[120.0], notch_depth_db=40, notch_width_hz=4))
        self.assertLess(level_db_at(y, SR, 120) - level_db_at(x, SR, 120), -35)
        self.assertAlmostEqual(level_db_at(y, SR, 1000), level_db_at(x, SR, 1000), delta=0.2)

    def test_highpass_and_lowpass_act_on_the_right_side(self):
        x = sine(50, 3, SR) + sine(500, 3, SR) + sine(6000, 3, SR)
        hp = Fl.apply_frequency_filter(x, SR, Fl.FilterParams(highpass_hz=200))
        lp = Fl.apply_frequency_filter(x, SR, Fl.FilterParams(lowpass_hz=2000))
        self.assertLess(level_db_at(hp, SR, 50) - level_db_at(x, SR, 50), -40)
        self.assertAlmostEqual(level_db_at(hp, SR, 500), level_db_at(x, SR, 500), delta=0.5)
        expected = 20 * np.log10(Fl.lowpass_gain(np.array([6000.0]), 2000.0)[0])          # -38.2 dB for a 4th-order response
        self.assertAlmostEqual(level_db_at(lp, SR, 6000) - level_db_at(x, SR, 6000), expected, delta=0.3)
        self.assertAlmostEqual(level_db_at(lp, SR, 500), level_db_at(x, SR, 500), delta=0.1)

    def test_filter_is_zero_phase(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(SR)
        y = Fl.apply_frequency_filter(x, SR, Fl.FilterParams(lowpass_hz=3000))
        xc = signal.correlate(y, x, mode="full")
        self.assertEqual(int(np.argmax(xc)) - (len(x) - 1), 0)

    def test_convolution_theorem_matches_time_domain_convolution(self):
        """Y = X.H in frequency equals x * h in time, with h = F^-1{H}."""
        rng = np.random.default_rng(2)
        x = rng.standard_normal(6000)
        p = Fl.FilterParams(highpass_hz=300, notch_freqs=[1500.0], notch_depth_db=30, notch_width_hz=20)
        y_freq = Fl.apply_frequency_filter(x, SR, p)
        m = 1 << 17                                              # fine grid -> accurate impulse response
        H = Fl.design_response(np.arange(m // 2 + 1) * SR / m, p)
        h = np.fft.fftshift(np.fft.irfft(H, m))                  # zero-phase kernel centred on index m/2
        y_time = signal.fftconvolve(x, h, mode="full")[m // 2: m // 2 + x.size]
        self.assertLess(np.max(np.abs(y_freq - y_time)), 1e-3 * np.max(np.abs(y_freq)))

    def test_spectral_gate_lowers_stationary_noise_but_keeps_loud_tone_bursts(self):
        rng = np.random.default_rng(3)
        n = SR * 6
        noise = 0.05 * rng.standard_normal(n)
        burst = np.zeros(n)
        burst[SR * 2:SR * 4] = sine(1000, 2, SR, 0.5)
        x = noise + burst
        y = Fl.spectral_gate(x, SR, 0.8)
        quiet = slice(0, SR)
        self.assertLess(np.std(y[quiet]), 0.5 * np.std(x[quiet]))
        self.assertAlmostEqual(level_db_at(y[SR * 2:SR * 4], SR, 1000), level_db_at(x[SR * 2:SR * 4], SR, 1000), delta=1.5)
        np.testing.assert_array_equal(Fl.spectral_gate(x, SR, 0.0), x)

    def test_clip_protection(self):
        x = sine(100, 2, SR, 0.95)
        res = Fl.process(x * 1.0, SR, Fl.FilterParams(notch_freqs=[3000.0]))
        self.assertLessEqual(np.max(np.abs(res.signal)), 0.99 + 1e-9)


class ParameterTests(unittest.TestCase):
    def check(self, d):
        with self.assertRaises(ValidationError):
            Fl.FilterParams.from_dict(d, SR)

    def test_validation(self):
        self.check({"highpass_hz": -5})
        self.check({"highpass_hz": 9000})
        self.check({"lowpass_hz": "high"})
        self.check({"highpass_hz": 4000, "lowpass_hz": 1000})
        self.check({"notch_freqs": [100, "x"]})
        self.check({"notch_freqs": [9000]})
        self.check({"notch_freqs": list(range(100, 100 + 17 * 10, 10))})
        self.check({"notch_depth_db": 90})
        self.check({"gate_strength": 2})
        self.check({"profile": "toaster"})
        self.check({"gate_strength": True})
        self.check({"gate_strength": float("nan")})
        self.check([1, 2])

    def test_valid_params_are_normalised(self):
        p = Fl.FilterParams.from_dict({"notch_freqs": [200, 100, 100], "gate_strength": 0.5, "highpass_hz": 60}, SR)
        self.assertEqual(p.notch_freqs, [100.0, 200.0])
        self.assertEqual(Fl.FilterParams.from_dict(None, SR).gate_strength, 0.0)

    def test_profile_defaults_use_detected_tones_and_respect_nyquist(self):
        tones = [{"freq_hz": 120.0, "prominence_db": 25, "stationary": True},
                 {"freq_hz": 240.0, "prominence_db": 18, "stationary": True},
                 {"freq_hz": 150.0, "prominence_db": 30, "stationary": False}]
        p = Fl.params_for_profile("ac", tones, SR)
        self.assertEqual(p.notch_freqs, [120.0, 240.0])                    # non-stationary peak is never notched
        self.assertEqual(Fl.params_for_profile("traffic", tones, SR).notch_freqs, [])
        self.assertIsNone(Fl.params_for_profile("speech", tones, 12000).lowpass_hz)   # 8 kHz is above 0.95 x Nyquist
        self.assertEqual(Fl.params_for_profile("speech", tones, SR).lowpass_hz, 8000.0 if 8000 < SR / 2 * 0.95 else None)


class EndToEndFilterQualityTests(unittest.TestCase):
    """Uses the known clean speech to prove the filtering really improves the signal."""

    def test_si_sdr_improves_for_each_noise_type(self):
        for kind in ("speech_fan", "speech_ac", "speech_excavator", "speech_traffic"):
            sc = synthetic.noise_scene(kind, np.random.default_rng(11), 8.0)
            f, p = fourier.welch_psd(sc["mix"], SR)
            tones = Fl.find_stationary_tones(sc["mix"], SR, f, p)
            res = Fl.process(sc["mix"], SR, Fl.params_for_profile(kind.split("_")[1], tones, SR))
            gain = metrics.si_sdr(res.signal, sc["clean"]) - metrics.si_sdr(sc["mix"], sc["clean"])
            self.assertGreater(gain, 4.0, kind)


class MetricsTests(unittest.TestCase):
    def test_identical_signals(self):
        x = np.random.default_rng(0).standard_normal(SR * 2) * 0.1
        m = metrics.evaluate(x, x.copy(), SR)
        self.assertAlmostEqual(m["level_change_db"], 0, places=9)
        self.assertAlmostEqual(m["energy_removed_pct"], 0, places=9)
        self.assertAlmostEqual(m["correlation"], 1.0, places=9)
        self.assertAlmostEqual(m["log_spectral_distance_db"], 0, places=9)
        self.assertAlmostEqual(m["noise_floor_reduction_db"], 0, places=9)

    def test_half_amplitude_is_minus_6db_and_75_percent_energy_removed(self):
        x = np.random.default_rng(1).standard_normal(SR * 2) * 0.1
        m = metrics.evaluate(x, 0.5 * x, SR)
        self.assertAlmostEqual(m["level_change_db"], -6.0206, places=3)
        self.assertAlmostEqual(m["energy_removed_pct"], 75.0, places=6)
        self.assertAlmostEqual(m["noise_floor_reduction_db"], 6.0206, places=3)
        self.assertAlmostEqual(m["correlation"], 1.0, places=9)
        self.assertAlmostEqual(m["log_spectral_distance_db"], 6.0206, places=3)
        self.assertTrue(all(abs(b["change_db"] + 6.0206) < 1e-6 for b in m["bands"]))

    def test_si_sdr_known_values(self):
        ref = np.random.default_rng(2).standard_normal(20000)
        noise = np.random.default_rng(3).standard_normal(20000)
        noise -= np.dot(noise, ref) / np.dot(ref, ref) * ref           # orthogonal to ref
        est = ref + 0.1 * noise * np.std(ref) / np.std(noise)
        self.assertAlmostEqual(metrics.si_sdr(est, ref), 20.0, delta=0.05)
        self.assertGreater(metrics.si_sdr(3.0 * ref, ref), 100)           # perfect up to scale

    def test_removed_tone_is_reported_per_peak_and_per_band(self):
        t = np.random.default_rng(4).standard_normal(SR * 4) * 0.02
        x = t + sine(120, 4, SR, 0.3)
        y = Fl.apply_frequency_filter(x, SR, Fl.FilterParams(notch_freqs=[120.0], notch_depth_db=40, notch_width_hz=4))
        m = metrics.evaluate(x, y, SR, tracked_peaks=[120.0])
        self.assertLess(m["tracked_peaks"][0]["change_db"], -30)
        low_band = next(b for b in m["bands"] if b["lo"] == 63.0)
        self.assertLess(low_band["change_db"], -20)
        self.assertGreater(m["energy_removed_pct"], 90)

    def test_reference_metrics_only_when_reference_given(self):
        sc = synthetic.noise_scene("speech_fan", np.random.default_rng(5), 5.0)
        self.assertIsNone(metrics.evaluate(sc["mix"], sc["mix"], SR)["reference"])
        m = metrics.evaluate(sc["mix"], sc["clean"], SR, reference=sc["clean"])
        self.assertGreater(m["reference"]["improvement_db"], 30)


if __name__ == "__main__":
    unittest.main()
