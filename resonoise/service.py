"""Application service: runs the complete pipeline and persists every stage.

    audio input -> preprocessing -> Fourier analysis -> features -> classification
    -> selective filtering (convolution theorem) -> inverse transform -> evaluation -> database
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from . import config, filtering, metrics, synthetic
from .audio_io import (DecodedAudio, decode_audio, preprocess, quantize16, read_wav_file, wav_bytes)
from .classifier import NOISE_CLASSES, NoiseClassifier, train_classifier
from .database import Database
from .errors import NotFoundError, ProcessingError, ValidationError
from .features import BAND_EDGES, FEATURE_LABELS, extract_features, frame_rms_db, quiet_segments
from .fourier import (band_energies, detect_harmonic_series, find_spectral_peaks, forward_transform,
                      analysis_nfft, next_fast_length, parseval_check, spectral_summary, welch_psd)
from .utils import db as to_db
from .utils import rms_db, to_jsonable
from .visuals import spectrogram_image, spectrum_payload, waveform_envelope

SAMPLE_KINDS = {
    "speech_fan": "A talker over a ceiling fan",
    "speech_ac": "A talker over an air-conditioner hum",
    "speech_traffic": "A talker beside a busy road",
    "speech_excavator": "A talker near an excavator",
    "traffic": "Road traffic only",
    "speech": "People talking only",
}
BACKGROUND_SIGNIFICANT_GAP_DB = 30.0       # noise floor within this many dB of the loud parts counts as noise


class ModelStore:
    """Loads the bundled classifier, or the user-retrained one stored next to the database."""

    def __init__(self, data_dir: Path):
        self.custom_path = Path(data_dir) / "model.json"
        self._lock = threading.RLock()
        self.clf = self._load()

    def _load(self) -> NoiseClassifier:
        if self.custom_path.exists():
            try:
                return NoiseClassifier.load(self.custom_path)
            except Exception:
                pass
        if not config.DEFAULT_MODEL_PATH.exists():
            raise ProcessingError("The bundled classifier is missing. Run: python scripts/train_model.py")
        return NoiseClassifier.load(config.DEFAULT_MODEL_PATH)

    def info(self) -> dict:
        m = self.clf.meta
        return {"classes": [{"key": k, **NOISE_CLASSES[k]} for k in self.clf.classes],
                "algorithm": m.get("algorithm"), "validation_accuracy": m.get("validation_accuracy"),
                "validation_n": m.get("validation_n"), "n_user_samples": m.get("n_user_samples", 0),
                "n_synthetic_train": m.get("n_synthetic_train"), "trained_at": m.get("trained_at"),
                "custom": self.custom_path.exists() and self.clf.meta.get("n_user_samples", 0) > 0,
                "confusion": m.get("confusion"), "n_features": len(self.clf.feature_names)}


class Workbench:
    def __init__(self, data_dir: Path | None = None):
        self.data_dir = Path(data_dir or config.DATA_DIR)
        self.db = Database(self.data_dir)
        self.models = ModelStore(self.data_dir)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def public_recording(rec: dict) -> dict:
        return {"id": rec["id"], "name": rec["name"], "source": rec["source"],
                "original_filename": rec["original_filename"], "created_at": rec["created_at"],
                "sample_rate": rec["sample_rate"], "original_sample_rate": rec["original_sample_rate"],
                "channels": rec["channels"], "duration": rec["duration"], "size_bytes": rec["size_bytes"],
                "container": rec["container"], "has_reference": bool(rec["reference_path"]),
                "source_class": rec.get("source_class"), "confidence": rec.get("confidence"),
                "n_processed": rec.get("n_processed")}

    def _load_audio(self, rid: str):
        rec = self.db.get_recording(rid)
        path = Path(rec["audio_path"])
        if not path.exists():
            raise NotFoundError("The audio file for this recording is missing from storage.")
        x, sr = read_wav_file(path)
        return rec, x, sr

    # ------------------------------------------------------------------ ingest
    def ingest(self, data: bytes, filename: str = "", source: str = "upload", name: str | None = None) -> dict:
        decoded = decode_audio(data, filename)
        return self._store(decoded, filename, source, name, len(data))

    def _store(self, decoded: DecodedAudio, filename: str, source: str, name: str | None, size_bytes: int,
               reference: np.ndarray | None = None) -> dict:
        prepared = preprocess(decoded)
        ref_bytes = None
        if reference is not None:
            gain = prepared.samples.std() / max(decoded.samples.std(), 1e-12)
            ref_bytes = wav_bytes(np.clip(reference * gain, -1, 1), prepared.sample_rate)
        stem = Path(filename).stem if filename else ""
        label = name or stem or ("Microphone recording" if source == "recording" else "Untitled clip")
        rec = self.db.add_recording(
            name=label[:120], source=source, original_filename=filename or None, sample_rate=prepared.sample_rate,
            original_sample_rate=prepared.original_sample_rate, channels=prepared.channels,
            duration=prepared.duration, size_bytes=size_bytes, audio_bytes=wav_bytes(prepared.samples, prepared.sample_rate),
            container=prepared.container, reference_bytes=ref_bytes)
        return self.public_recording(rec)

    def create_sample(self, kind: str, seed: int | None = None) -> dict:
        if kind not in SAMPLE_KINDS:
            raise ValidationError(f"Unknown sample '{kind}'.", details={"available": list(SAMPLE_KINDS)})
        rng = np.random.default_rng(seed if seed is not None else int(time.time() * 1000) % (2 ** 32))
        scene = synthetic.noise_scene(kind, rng, duration=8.0, snr_db=3.0)
        decoded = DecodedAudio(scene["mix"], scene["sr"], 1, "synthetic")
        ref = scene["clean"]
        return self._store(decoded, "", "sample", SAMPLE_KINDS[kind], scene["mix"].size * 2, reference=ref)

    # ------------------------------------------------------------------ analysis
    def analyze(self, rid: str) -> dict:
        rec, x, sr = self._load_audio(rid)
        try:
            t0 = time.time()
            n = x.size
            m = next_fast_length(n)
            X = forward_transform(x, n=m)                           # X(f) of the whole recording
            parseval = parseval_check(x, X, n_fft=m)
            spectrum = spectrum_payload(x, sr)

            nfft = analysis_nfft(sr)
            freqs, psd = welch_psd(x, sr, nfft)
            raw_peaks = find_spectral_peaks(freqs, psd, prominence_db=10.0, max_freq=sr * 0.45)
            tones = filtering.find_stationary_tones(x, sr, freqs, psd, raw_peaks)
            dominant = sorted(tones, key=lambda t: t["level_db"], reverse=True)[:5]
            harmonic = detect_harmonic_series([q for q in tones if q["stationary"]])
            summary = spectral_summary(freqs, psd, 20.0, min(8000.0, sr / 2.0))
            edges = [e for e in BAND_EDGES if e < sr / 2.0] + [min(BAND_EDGES[-1], sr / 2.0)]
            bands = band_energies(freqs, psd, sorted(set(edges)))

            lv = frame_rms_db(x, sr)
            signal_stats = {"n_samples": int(n), "sample_rate": sr, "duration": n / sr, "rms_db": rms_db(x),
                            "peak": float(np.max(np.abs(x))),
                            "crest_db": float(20 * np.log10(np.max(np.abs(x)) / max(np.sqrt(np.mean(x ** 2)), 1e-12))),
                            "noise_floor_db": float(np.percentile(lv, 10)), "loud_level_db": float(np.percentile(lv, 95))}

            feats = extract_features(x, sr)
            overall = self.models.clf.classify(feats)
            bg_sig = quiet_segments(x, sr)
            background, bg_feats = None, None
            gap = signal_stats["loud_level_db"] - signal_stats["noise_floor_db"]
            if bg_sig is not None:
                bg_feats = extract_features(bg_sig, sr)
                background = self.models.clf.classify(bg_feats)
                background["gap_db"] = float(gap)
                background["significant"] = bool(gap < BACKGROUND_SIGNIFICANT_GAP_DB)
            recommended = self._recommend(rid, overall, background)

            payload = {
                "recording": self.public_recording(rec), "signal": signal_stats,
                "fourier": {"parseval": parseval, "n_fft": m, "resolution_hz": sr / m, "spectrum": spectrum,
                            "dominant": dominant, "harmonic": harmonic, "summary": summary, "bands": bands,
                            "analysis_resolution_hz": sr / nfft},
                "spectrogram": spectrogram_image(x, sr),
                "tones": tones,
                "features": [{"name": k, "label": FEATURE_LABELS.get(k, k), "value": v} for k, v in feats.items()],
                "classification": {"overall": overall, "background": background, "recommended": recommended},
                "filter_defaults": {k: filtering.params_for_profile(k, tones, sr).to_dict() for k in filtering.PROFILES},
                "elapsed_ms": int((time.time() - t0) * 1000),
            }
            payload = to_jsonable(payload)
            self.db.save_analysis(rid, {"overall": feats, "background": bg_feats}, payload, recommended["key"],
                                  recommended["confidence"])
            return payload
        except (NotFoundError, ValidationError):
            raise
        except Exception as exc:
            raise ProcessingError(f"Analysis failed: {exc}") from exc

    def _recommend(self, rid: str, overall: dict, background: dict | None) -> dict:
        taught = self.db.get_label(rid)
        if taught:
            return {"key": taught, "label": NOISE_CLASSES[taught]["label"], "confidence": 1.0, "source": "user",
                    "reason": f"You labelled this recording as {NOISE_CLASSES[taught]['label'].lower()}."}
        top = overall["top"]
        if top != "speech":
            return {"key": top, "label": overall["label"], "confidence": overall["confidence"], "source": "overall",
                    "reason": f"The whole recording is dominated by {overall['label'].lower()} "
                              f"({overall['confidence'] * 100:.0f}% probability)."}
        if (background and background["significant"] and background["top"] != "speech"
                and background["confidence"] >= 0.5):
            return {"key": background["top"], "label": background["label"], "confidence": background["confidence"],
                    "source": "background",
                    "reason": f"People are talking, but the quiet moments between speech sound like "
                              f"{background['label'].lower()} ({background['confidence'] * 100:.0f}%). "
                              "That background is what gets reduced; the speech is preserved."}
        return {"key": "speech", "label": overall["label"], "confidence": overall["confidence"], "source": "overall",
                "reason": "Speech dominates and no strong background noise was found, so only rumble and hiss "
                          "outside the speech band are reduced."}

    def get_analysis(self, rid: str) -> dict:
        self.db.get_recording(rid)
        a = self.db.get_analysis(rid)
        if a is None:
            raise NotFoundError("This recording has not been analysed yet.")
        return a["payload"]

    # ------------------------------------------------------------------ filtering
    def preview_filter(self, rid: str, params: dict) -> dict:
        rec = self.db.get_recording(rid)
        p = filtering.FilterParams.from_dict(params, rec["sample_rate"])
        c = filtering.response_curve(rec["sample_rate"], p)
        return to_jsonable({"freq": np.round(c["freq"], 2), "gain_db": np.round(c["gain_db"], 2)})

    def process(self, rid: str, params: dict) -> dict:
        rec, x, sr = self._load_audio(rid)
        p = filtering.FilterParams.from_dict(params, sr)
        if not (p.highpass_hz or p.lowpass_hz or p.notch_freqs or p.gate_strength > 0):
            raise ValidationError("Nothing to apply. Switch on at least one filter before reconstructing the audio.")
        analysis = self.db.get_analysis(rid)
        if analysis is None:
            self.analyze(rid)
            analysis = self.db.get_analysis(rid)
        base = analysis["payload"]
        try:
            t0 = time.time()
            result = filtering.process(x, sr, p)
            y = quantize16(result.signal)                             # exactly what the WAV file will contain
            ref = None
            if rec["reference_path"] and Path(rec["reference_path"]).exists():
                ref, _ = read_wav_file(rec["reference_path"])
            tracked = []
            for f0 in [t["freq_hz"] for t in base["fourier"]["dominant"][:3]] + list(p.notch_freqs[:4]):
                if all(abs(f0 - q) > 2.0 for q in tracked):          # one row per distinct tone
                    tracked.append(f0)
            met = metrics.evaluate(x, y, sr, ref, tracked_peaks=tracked)
            spec_o, spec_p = base["fourier"]["spectrum"], spectrum_payload(y, sr)
            spg = base["spectrogram"]
            payload = to_jsonable({
                "recording_id": rid, "params": p.to_dict(), "profile": p.profile, "metrics": met,
                "output_gain_db": result.output_gain_db,
                "spectrum": {"freq": spec_p["freq"], "original_fft_db": spec_o["fft_db"], "processed_fft_db": spec_p["fft_db"],
                             "original_avg_db": spec_o["avg_db"], "processed_avg_db": spec_p["avg_db"]},
                "response": {"freq": np.round(result.response["freq"], 2), "gain_db": np.round(result.response["gain_db"], 2)},
                "spectrogram": spectrogram_image(y, sr, db_range=(spg["db_min"], spg["db_max"])),
                "parseval": parseval_check(y), "elapsed_ms": int((time.time() - t0) * 1000),
                "created_at": time.time(),
            })
            pid = self.db.add_processed(rid, profile=p.profile, params=p.to_dict(), metrics=to_jsonable(met),
                                        payload=payload, audio_bytes=wav_bytes(y, sr))
            payload["processed_id"] = pid
            return payload
        except (NotFoundError, ValidationError):
            raise
        except Exception as exc:
            raise ProcessingError(f"Processing failed: {exc}") from exc

    def get_processed_payload(self, pid: str) -> dict:
        return self.db.get_processed(pid)["payload"]

    # ------------------------------------------------------------------ waveform and files
    def waveform(self, rid: str, t0: float, t1: float | None, points: int, pid: str | None = None) -> dict:
        if pid:
            rec = self.db.get_processed(pid)
            if rec["recording_id"] != rid:
                raise NotFoundError("That processed result belongs to a different recording.")
            x, sr = read_wav_file(rec["audio_path"])
        else:
            _, x, sr = self._load_audio(rid)
        return to_jsonable(waveform_envelope(x, sr, t0, t1, points))

    def audio_path(self, rid: str) -> tuple[Path, str]:
        rec = self.db.get_recording(rid)
        return Path(rec["audio_path"]), rec["name"]

    def processed_audio_path(self, pid: str) -> tuple[Path, str]:
        rec = self.db.get_processed(pid)
        parent = self.db.get_recording(rec["recording_id"])
        return Path(rec["audio_path"]), parent["name"]

    def report(self, pid: str) -> dict:
        proc = self.db.get_processed(pid)
        rec = self.db.get_recording(proc["recording_id"])
        analysis = self.db.get_analysis(rec["id"])
        a = analysis["payload"] if analysis else {}
        return to_jsonable({
            "generated_by": "Reso-Noise", "recording": self.public_recording(rec),
            "signal": a.get("signal"), "classification": a.get("classification"),
            "fourier": {k: a.get("fourier", {}).get(k) for k in ("parseval", "dominant", "harmonic", "summary", "bands")},
            "filter": proc["params"], "metrics": proc["metrics"], "processed_id": pid, "created_at": proc["created_at"]})

    # ------------------------------------------------------------------ library, labels, model
    def library(self) -> dict:
        return {"recordings": [self.public_recording(r) for r in self.db.list_recordings()], "stats": self.db.stats()}

    def recording_detail(self, rid: str) -> dict:
        rec = self.db.get_recording(rid)
        out = {"recording": self.public_recording(rec), "processed": self.db.list_processed(rid),
               "analysed": self.db.get_analysis(rid) is not None, "label": self.db.get_label(rid)}
        return to_jsonable(out)

    def delete_recording(self, rid: str) -> None:
        self.db.delete_recording(rid)

    def delete_processed(self, pid: str) -> None:
        self.db.delete_processed(pid)

    def set_label(self, rid: str, label: str) -> dict:
        if label not in NOISE_CLASSES:
            raise ValidationError(f"Unknown noise class '{label}'.", details={"classes": list(NOISE_CLASSES)})
        a = self.db.get_analysis(rid)
        if a is None:
            raise ValidationError("Analyse the recording before labelling it.")
        self.db.set_label(rid, label, a["features"]["overall"], "overall")
        if a["features"].get("background"):
            self.db.set_label(rid, label, a["features"]["background"], "background")
        return {"label": label, "labels_stored": len(self.db.list_labels())}

    def retrain(self) -> dict:
        labels = self.db.list_labels()
        if not labels:
            raise ValidationError("There are no labelled recordings yet. Correct a classification first, then retrain.")
        if config.SYNTH_DATASET_PATH.exists():
            d = np.load(config.SYNTH_DATASET_PATH)
            X, y, split = d["X"], d["y"], d["split"]
        else:
            X, y, split = synthetic.build_feature_dataset()
        keys = self.models.clf.classes
        names = self.models.clf.feature_names
        ex = np.array([[l["features"][n] for n in names] for l in labels])
        ey = np.array([keys.index(l["label"]) for l in labels])
        clf = train_classifier(X, y, split, ex, ey)
        with self.models._lock:
            clf.save(self.models.custom_path)
            self.models.clf = clf
        return self.models.info()
