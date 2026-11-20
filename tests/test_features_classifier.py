import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.optimize import check_grad

from tests.helpers import sine
from resonoise import config, features, synthetic
from resonoise.classifier import (NOISE_CLASSES, NoiseClassifier, cross_entropy_objective, softmax, train_classifier)

SR = 16000


class FeatureTests(unittest.TestCase):
    def test_vector_is_complete_and_finite(self):
        clip = synthetic.make_training_clip("fan", np.random.default_rng(0), 4.0)
        f = features.extract_features(clip, SR)
        self.assertEqual(list(f), features.FEATURE_NAMES)
        self.assertTrue(all(np.isfinite(v) for v in f.values()))
        self.assertEqual(features.feature_vector(f).shape, (len(features.FEATURE_NAMES),))

    def test_features_are_level_independent(self):
        clip = synthetic.make_training_clip("traffic", np.random.default_rng(1), 4.0)
        a, b = features.extract_features(clip, SR), features.extract_features(clip * 0.05, SR)
        for k in a:
            self.assertAlmostEqual(a[k], b[k], delta=1e-6 + 1e-3 * abs(a[k]), msg=k)

    def test_other_sample_rates_are_handled(self):
        clip = synthetic.make_training_clip("ac", np.random.default_rng(2), 3.0)
        from resonoise.audio_io import resample
        f44 = features.extract_features(resample(clip, 16000, 44100), 44100)
        f16 = features.extract_features(clip, 16000)
        self.assertAlmostEqual(f44["centroid_log2"], f16["centroid_log2"], delta=0.15)

    def test_hum_is_tonal_and_noise_is_not(self):
        hum = features.extract_features(sine(120, 4, SR) + sine(240, 4, SR) * 0.5 + 0.05 * np.random.default_rng(3).standard_normal(4 * SR), SR)
        noise = features.extract_features(np.random.default_rng(4).standard_normal(4 * SR), SR)
        self.assertGreater(hum["tonal_ratio"], 0.7)
        self.assertLess(noise["tonal_ratio"], 0.2)
        self.assertGreater(hum["harmonic_strength"], 0.5)
        self.assertGreater(noise["flatness_db"], hum["flatness_db"])

    def test_speech_has_syllabic_modulation_and_pitch_movement(self):
        sp = features.extract_features(synthetic.gen_speaker(8 * SR, np.random.default_rng(5)), SR)
        fan = features.extract_features(synthetic.gen_fan(8 * SR, np.random.default_rng(5)), SR)
        self.assertGreater(sp["mod_2_8hz"], fan["mod_2_8hz"])
        self.assertGreater(sp["pitch_var_st"], fan["pitch_var_st"])

    def test_silence_is_rejected(self):
        with self.assertRaises(ValueError):
            features.extract_features(np.zeros(SR), SR)

    def test_quiet_segments_selects_the_pauses(self):
        x = np.random.default_rng(6).standard_normal(SR * 10) * 0.01
        x[:SR * 3] *= 50                                           # loud first 3 s
        bg = features.quiet_segments(x, SR)
        self.assertIsNotNone(bg)
        self.assertLess(np.std(bg), 0.02)
        self.assertIsNone(features.quiet_segments(x[:SR], SR))     # too short to estimate a background


class ClassifierTests(unittest.TestCase):
    def test_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(0)
        Z, k = rng.standard_normal((40, 6)), 3
        Y = np.eye(k)[rng.integers(0, k, 40)]
        w = rng.uniform(0.5, 2, 40)
        theta = rng.standard_normal(6 * k + k) * 0.3
        f = lambda t: cross_entropy_objective(t, Z, Y, w, 0.1)[0]
        g = lambda t: cross_entropy_objective(t, Z, Y, w, 0.1)[1]
        self.assertLess(check_grad(f, g, theta), 1e-5)

    def test_softmax_rows_sum_to_one(self):
        p = softmax(np.array([[1000.0, 1001.0, 999.0], [0, 0, 0]]))
        np.testing.assert_allclose(p.sum(axis=1), 1.0)
        self.assertTrue(np.all(np.isfinite(p)))

    def test_learns_separable_problem_and_roundtrips(self):
        rng = np.random.default_rng(1)
        n = len(features.FEATURE_NAMES)
        centres = rng.standard_normal((5, n)) * 3
        X = np.vstack([c + rng.standard_normal((60, n)) for c in centres])
        y = np.repeat(np.arange(5), 60)
        clf = NoiseClassifier().fit(X, y)
        self.assertGreater(np.mean(clf.predict(X) == y), 0.98)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.json"
            clf.save(path)
            again = NoiseClassifier.load(path)
        np.testing.assert_allclose(again.predict_proba(X[:10]), clf.predict_proba(X[:10]))

    def test_rejects_bad_training_input(self):
        with self.assertRaises(ValueError):
            NoiseClassifier().fit(np.zeros((10, 3)), np.zeros(10))
        with self.assertRaises(ValueError):
            NoiseClassifier().fit(np.zeros((10, len(features.FEATURE_NAMES))), np.zeros(10))

    def test_sample_weights_shift_the_decision(self):
        n = len(features.FEATURE_NAMES)
        rng = np.random.default_rng(2)
        X = np.vstack([rng.standard_normal((30, n)), rng.standard_normal((30, n)) + 0.5])
        y = np.r_[np.zeros(30, int), np.ones(30, int)]
        w_a = np.r_[np.full(30, 20.0), np.ones(30)]
        pa = NoiseClassifier().fit(X, y, sample_weight=w_a).predict_proba(X).mean(axis=0)
        pb = NoiseClassifier().fit(X, y, sample_weight=w_a[::-1]).predict_proba(X).mean(axis=0)
        self.assertGreater(pa[0], pb[0])


class BundledModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clf = NoiseClassifier.load(config.DEFAULT_MODEL_PATH)

    def test_metadata_documents_validation(self):
        self.assertGreater(self.clf.meta["validation_accuracy"], 0.9)
        self.assertEqual(self.clf.classes, list(NOISE_CLASSES))

    def test_classifies_fresh_unseen_clips(self):
        rng = np.random.default_rng(31337)               # seed unrelated to training
        hits = total = 0
        for key in synthetic.CLASS_KEYS:
            for _ in range(8):
                r = self.clf.classify(features.extract_features(synthetic.make_training_clip(key, rng, 5.0), SR))
                hits += r["top"] == key
                total += 1
                self.assertAlmostEqual(sum(p["prob"] for p in r["probabilities"]), 1.0, places=6)
        self.assertGreaterEqual(hits / total, 0.9)

    def test_result_structure_and_drivers(self):
        r = self.clf.classify(features.extract_features(synthetic.gen_excavator(5 * SR, np.random.default_rng(3)), SR))
        self.assertEqual(r["top"], "excavator")
        self.assertEqual(r["probabilities"][0]["key"], r["top"])
        self.assertTrue(r["drivers"])
        self.assertTrue(all(d["contribution"] > 0 for d in r["drivers"]))

    def test_evaluate_reports_confusion_matrix(self):
        d = np.load(config.SYNTH_DATASET_PATH)
        ev = self.clf.evaluate(d["X"][d["split"] == 1], d["y"][d["split"] == 1])
        self.assertGreater(ev["accuracy"], 0.9)
        self.assertEqual(np.array(ev["confusion"]).sum(), ev["n"])

    def test_user_labels_change_the_model(self):
        d = np.load(config.SYNTH_DATASET_PATH)
        # teach it that an AC-like feature vector is a fan, with heavy weight
        ac_rows = d["X"][(d["y"] == 1) & (d["split"] == 0)][:6]
        clf = train_classifier(d["X"], d["y"], d["split"], ac_rows, np.zeros(6, int), extra_weight=30.0)
        self.assertEqual(clf.meta["n_user_samples"], 6)
        self.assertGreater(clf.predict_proba(ac_rows)[:, 0].mean(), self.clf.predict_proba(ac_rows)[:, 0].mean())


class SyntheticTests(unittest.TestCase):
    def test_generators_are_finite_and_have_requested_length(self):
        rng = np.random.default_rng(0)
        for key, gen in synthetic.GENERATORS.items():
            y = gen(SR * 2, rng)
            self.assertEqual(len(y), SR * 2, key)
            self.assertTrue(np.all(np.isfinite(y)), key)
            self.assertGreater(np.std(y), 0.1, key)

    def test_scene_snr_is_as_requested(self):
        sc = synthetic.noise_scene("speech_fan", np.random.default_rng(1), 6.0, snr_db=3.0)
        snr = 10 * np.log10(np.sum(sc["clean"] ** 2) / np.sum(sc["noise"] ** 2))
        self.assertAlmostEqual(snr, 3.0, delta=0.05)
        np.testing.assert_allclose(sc["clean"] + sc["noise"], sc["mix"], atol=1e-9)

    def test_unknown_scene(self):
        with self.assertRaises(ValueError):
            synthetic.noise_scene("dishwasher", np.random.default_rng(0))


if __name__ == "__main__":
    unittest.main()
