"""Complete workflow through the HTTP API, verified against the downloaded audio files."""
import io
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
from scipy.io import wavfile

from tests.helpers import level_db_at, sine, wav16
from resonoise import audio_io, metrics, synthetic
from resonoise.server import create_app

SR = 16000


def upload(client, data, name="clip.wav", **form):
    return client.post("/api/recordings", data={"file": (io.BytesIO(data), name), **form}, content_type="multipart/form-data")


def read_wav(resp):
    return audio_io.decode_audio(resp.data, "x.wav")


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.app = create_app(cls.tmp.name)
        cls.c = cls.app.test_client()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def noisy_speech_with_hum(self, seed=1):
        sc = synthetic.noise_scene("speech", np.random.default_rng(seed), 8.0)
        t = np.arange(len(sc["mix"])) / SR
        hum = 0.25 * (np.sin(2 * np.pi * 120 * t) + 0.6 * np.sin(2 * np.pi * 240 * t) + 0.4 * np.sin(2 * np.pi * 360 * t))
        mix = sc["mix"] * 0.5 + hum + 0.01 * np.random.default_rng(seed).standard_normal(t.size)
        return mix / np.max(np.abs(mix)) * 0.9, sc["clean"]

    def test_full_workflow_hum_removal(self):
        mix, _ = self.noisy_speech_with_hum()
        r = upload(self.c, wav16(mix, SR), "kitchen.wav")
        self.assertEqual(r.status_code, 201)
        rec = r.json
        rid = rec["id"]
        self.assertEqual((rec["sample_rate"], rec["name"], rec["channels"]), (SR, "kitchen", 1))

        # analysis: genuine Fourier results
        a = self.c.post(f"/api/recordings/{rid}/analyze").json
        self.assertLess(a["fourier"]["parseval"]["relative_error"], 1e-10)
        freqs = sorted(round(t["freq_hz"]) for t in a["fourier"]["dominant"])
        self.assertTrue({120, 240, 360} <= set(freqs), freqs)
        self.assertAlmostEqual(a["fourier"]["harmonic"]["f0_hz"], 120, delta=1.5)
        self.assertEqual(len(a["fourier"]["spectrum"]["freq"]), 900)
        sp = a["fourier"]["spectrum"]
        peak_f = sp["freq"][int(np.argmax(sp["fft_db"]))]
        self.assertAlmostEqual(peak_f, 120, delta=8)
        self.assertGreater(a["spectrogram"]["cols"], 50)
        self.assertEqual(self.c.get(f"/api/recordings/{rid}/analysis").json["classification"], a["classification"])
        cls = a["classification"]
        self.assertAlmostEqual(sum(p["prob"] for p in cls["overall"]["probabilities"]), 1.0, places=6)
        self.assertIn(cls["recommended"]["key"], ("fan", "ac", "speech", "excavator", "traffic"))

        # filter: notch the detected steady tones + gate
        notches = [t["freq_hz"] for t in a["tones"] if t["stationary"]][:6]
        self.assertTrue(any(abs(f - 120) < 2 for f in notches))
        params = {"profile": "custom", "highpass_hz": 60, "notch_freqs": notches, "notch_depth_db": 40,
                  "notch_width_hz": 4, "gate_strength": 0.5}
        prev = self.c.post(f"/api/recordings/{rid}/filter-preview", json=params).json
        self.assertLess(min(prev["gain_db"]), -30)
        p = self.c.post(f"/api/recordings/{rid}/process", json=params)
        self.assertEqual(p.status_code, 201)
        pid = p.json["processed_id"]

        # download both files and check everything independently of the server's own numbers
        orig = read_wav(self.c.get(f"/api/recordings/{rid}/audio"))
        proc = read_wav(self.c.get(f"/api/processed/{pid}/audio?download=1"))
        self.assertEqual(len(orig.samples), len(proc.samples))
        self.assertEqual(proc.sample_rate, SR)
        self.assertFalse(np.allclose(orig.samples, proc.samples))
        for f in (120, 240, 360):
            drop = level_db_at(proc.samples, SR, f) - level_db_at(orig.samples, SR, f)
            self.assertLess(drop, -20, f"{f} Hz only dropped {drop:.1f} dB")
        recomputed = metrics.evaluate(orig.samples, proc.samples, SR)
        m = p.json["metrics"]
        for key in ("level_change_db", "energy_removed_pct", "noise_floor_reduction_db", "correlation"):
            self.assertAlmostEqual(m[key], recomputed[key], places=3, msg=key)
        tones = [t["freq_hz"] for t in m["tracked_peaks"]]
        self.assertTrue(all(abs(a - b) > 2 for i, a in enumerate(tones) for b in tones[i + 1:]), tones)   # no duplicate rows
        self.assertLess(m["level_change_db"], -3)
        self.assertGreater(m["energy_removed_pct"], 50)
        self.assertGreater(m["correlation"], 0.1)

        # the processed spectrum served to the UI matches the file
        sp_p = p.json["spectrum"]
        self.assertLess(max(sp_p["processed_fft_db"][i] for i, f in enumerate(sp_p["freq"]) if 115 < f < 125),
                        max(sp_p["original_fft_db"][i] for i, f in enumerate(sp_p["freq"]) if 115 < f < 125) - 20)
        self.assertEqual(self.c.get(f"/api/processed/{pid}").json["metrics"]["energy_removed_pct"], m["energy_removed_pct"])

        # waveforms, report, history
        w = self.c.get(f"/api/recordings/{rid}/waveform?points=500&pid={pid}").json
        self.assertEqual(len(w["min"]), 500)
        self.assertAlmostEqual(max(w["max"]), float(np.max(proc.samples)), delta=1e-4)
        rep = self.c.get(f"/api/processed/{pid}/report")
        self.assertIn("attachment", rep.headers["Content-Disposition"])
        self.assertEqual(rep.json["filter"]["notch_freqs"], p.json["params"]["notch_freqs"])
        detail = self.c.get(f"/api/recordings/{rid}").json
        self.assertEqual(len(detail["processed"]), 1)
        lib = self.c.get("/api/recordings").json
        self.assertIn(rid, [r["id"] for r in lib["recordings"]])
        self.assertEqual(self.c.delete(f"/api/processed/{pid}").status_code, 200)
        self.assertEqual(self.c.get(f"/api/processed/{pid}/audio").status_code, 404)
        self.assertEqual(self.c.delete(f"/api/recordings/{rid}").status_code, 200)
        self.assertEqual(self.c.get(f"/api/recordings/{rid}/audio").status_code, 404)

    def test_sample_scene_improves_against_clean_reference(self):
        rec = self.c.post("/api/samples", json={"kind": "speech_fan", "seed": 101}).json
        self.assertTrue(rec["has_reference"])
        a = self.c.post(f"/api/recordings/{rec['id']}/analyze").json
        key = a["classification"]["recommended"]["key"]
        self.assertIn(key, ("fan", "ac"))
        p = self.c.post(f"/api/recordings/{rec['id']}/process", json=a["filter_defaults"][key]).json
        self.assertGreater(p["metrics"]["reference"]["improvement_db"], 5.0)

    def test_range_requests_for_seeking(self):
        rec = self.c.post("/api/samples", json={"kind": "traffic", "seed": 1}).json
        r = self.c.get(f"/api/recordings/{rec['id']}/audio", headers={"Range": "bytes=0-99"})
        self.assertEqual(r.status_code, 206)
        self.assertEqual(len(r.data), 100)

    def test_other_formats_rates_and_channels(self):
        mix, _ = self.noisy_speech_with_hum(2)
        buf = io.BytesIO()
        from resonoise.audio_io import resample
        wavfile.write(buf, 44100, np.stack([resample(mix, SR, 44100)] * 2, axis=1).astype(np.float32))
        rec = upload(self.c, buf.getvalue(), "stereo.wav").json
        self.assertEqual((rec["sample_rate"], rec["channels"], rec["original_sample_rate"]), (44100, 2, 44100))
        a = self.c.post(f"/api/recordings/{rec['id']}/analyze").json
        self.assertAlmostEqual(a["fourier"]["harmonic"]["f0_hz"], 120, delta=2)
        hi = io.BytesIO()
        wavfile.write(hi, 96000, resample(mix, SR, 96000).astype(np.float32))
        rec = upload(self.c, hi.getvalue(), "hi.wav").json
        self.assertEqual((rec["sample_rate"], rec["original_sample_rate"]), (48000, 96000))
        p = self.c.post(f"/api/recordings/{rec['id']}/process", json={"highpass_hz": 100, "gate_strength": 0.3})
        self.assertEqual(p.status_code, 201)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
    def test_mp3_upload_end_to_end(self):
        mix, _ = self.noisy_speech_with_hum(3)
        with tempfile.TemporaryDirectory() as tmp:
            open(f"{tmp}/a.wav", "wb").write(wav16(mix, SR))
            r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{tmp}/a.wav", f"{tmp}/a.mp3"], capture_output=True)
            if r.returncode != 0:
                self.skipTest("mp3 encoder unavailable")
            data = open(f"{tmp}/a.mp3", "rb").read()
        rec = upload(self.c, data, "voice.mp3")
        self.assertEqual(rec.status_code, 201)
        a = self.c.post(f"/api/recordings/{rec.json['id']}/analyze").json
        self.assertAlmostEqual(a["fourier"]["harmonic"]["f0_hz"], 120, delta=2)

    def test_labelling_and_retraining_changes_the_model(self):
        rec = self.c.post("/api/samples", json={"kind": "speech_ac", "seed": 9}).json
        self.assertEqual(self.c.post("/api/model/retrain").status_code, 400)           # nothing labelled yet
        self.assertEqual(self.c.post(f"/api/recordings/{rec['id']}/label", json={"label": "fan"}).status_code, 400)  # not analysed
        a = self.c.post(f"/api/recordings/{rec['id']}/analyze").json
        self.assertEqual(self.c.post(f"/api/recordings/{rec['id']}/label", json={"label": "dishwasher"}).status_code, 400)
        self.assertEqual(self.c.post(f"/api/recordings/{rec['id']}/label", json={"label": "traffic"}).status_code, 200)
        a2 = self.c.post(f"/api/recordings/{rec['id']}/analyze").json
        self.assertEqual(a2["classification"]["recommended"]["key"], "traffic")
        self.assertEqual(a2["classification"]["recommended"]["source"], "user")
        info = self.c.post("/api/model/retrain").json
        self.assertEqual(info["n_user_samples"], 2 if a["classification"]["background"] else 1)
        self.assertTrue(info["custom"])
        self.assertGreater(info["validation_accuracy"], 0.85)
        self.assertEqual(self.c.get("/api/model").json["n_user_samples"], info["n_user_samples"])

    # ------------------------------------------------------------------ error handling
    def test_invalid_uploads_return_clear_errors(self):
        cases = [
            (b"just some text", "notes.txt", 415, "unsupported_format"),
            (b"", "empty.wav", 422, "invalid_audio"),
            (wav16(np.zeros(SR), SR), "silence.wav", 422, "invalid_audio"),
            (wav16(sine(440, 0.1, SR, 0.5), SR), "short.wav", 422, "invalid_audio"),
            (wav16(sine(440, 1, SR, 0.5), SR)[:40], "truncated.wav", None, None),
        ]
        for data, name, status, code in cases:
            r = upload(self.c, data, name)
            self.assertGreaterEqual(r.status_code, 400, name)
            self.assertLess(r.status_code, 500, name)
            self.assertIn("message", r.json["error"])
            if status:
                self.assertEqual((r.status_code, r.json["error"]["code"]), (status, code), name)
        self.assertEqual(self.c.post("/api/recordings").status_code, 400)
        self.assertEqual(upload(self.c, wav16(sine(440, 1, SR, 0.5), SR), source="bogus").status_code, 400)

    def test_oversize_upload_is_rejected(self):
        self.app.config["MAX_CONTENT_LENGTH"] = 1000
        try:
            r = upload(self.c, wav16(sine(440, 1, SR, 0.5), SR))
            self.assertEqual(r.status_code, 413)
            self.assertEqual(r.json["error"]["code"], "file_too_large")
        finally:
            self.app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024

    def test_bad_processing_requests(self):
        rec = self.c.post("/api/samples", json={"kind": "traffic", "seed": 2}).json
        rid = rec["id"]
        url = f"/api/recordings/{rid}/process"
        self.assertEqual(self.c.post(url, json={}).status_code, 400)                                  # nothing to apply
        self.assertEqual(self.c.post(url, json={"highpass_hz": 99999}).status_code, 400)
        self.assertEqual(self.c.post(url, json={"gate_strength": "lots"}).status_code, 400)
        self.assertEqual(self.c.post(url, data="not json", content_type="application/json").status_code, 400)
        self.assertEqual(self.c.post(url, json=[1]).status_code, 400)
        self.assertEqual(self.c.post("/api/recordings/missing/process", json={"highpass_hz": 80}).status_code, 404)
        self.assertEqual(self.c.post("/api/recordings/missing/analyze").status_code, 404)
        self.assertEqual(self.c.get(f"/api/recordings/{rid}/analysis").status_code, 404)             # not analysed yet
        self.assertEqual(self.c.get(f"/api/recordings/{rid}/waveform?t0=abc").status_code, 400)
        self.assertEqual(self.c.get("/api/processed/missing").status_code, 404)
        self.assertEqual(self.c.post("/api/samples", json={"kind": "dishwasher"}).status_code, 400)

    def test_processing_runs_analysis_automatically(self):
        rec = self.c.post("/api/samples", json={"kind": "speech_traffic", "seed": 3}).json
        r = self.c.post(f"/api/recordings/{rec['id']}/process", json={"highpass_hz": 100, "gate_strength": 0.5})
        self.assertEqual(r.status_code, 201)
        self.assertEqual(self.c.get(f"/api/recordings/{rec['id']}/analysis").status_code, 200)

    def test_waveform_zoom_returns_raw_samples(self):
        rec = self.c.post("/api/samples", json={"kind": "traffic", "seed": 5}).json
        w = self.c.get(f"/api/recordings/{rec['id']}/waveform?t0=1&t1=1.01&points=1200").json
        self.assertTrue(w["samples"])
        self.assertEqual(len(w["min"]), 160)

    def test_frontend_and_health_are_served(self):
        self.assertEqual(self.c.get("/api/health").json["status"], "ok")
        r = self.c.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"Reso-Noise", r.data)


if __name__ == "__main__":
    unittest.main()
