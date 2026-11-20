"""Flask application: JSON API + static single-page frontend."""
from __future__ import annotations

import logging
import re

from flask import Flask, jsonify, request, send_file, send_from_directory
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from . import __version__, config, filtering
from .audio_io import ffmpeg_available
from .classifier import NOISE_CLASSES
from .errors import NotFoundError, ResoNoiseError, ValidationError
from .service import SAMPLE_KINDS, Workbench

log = logging.getLogger("resonoise")


def _num(name: str, default: float | None, lo: float, hi: float) -> float | None:
    raw = request.args.get(name)
    if raw is None or raw == "":
        return default
    try:
        v = float(raw)
    except ValueError:
        raise ValidationError(f"Query parameter '{name}' must be a number.") from None
    if not lo <= v <= hi:
        raise ValidationError(f"Query parameter '{name}' must be between {lo:g} and {hi:g}.")
    return v


def _json_body() -> dict:
    body = request.get_json(silent=True)
    if body is None:
        raise ValidationError("The request body must be valid JSON.")
    if not isinstance(body, dict):
        raise ValidationError("The request body must be a JSON object.")
    return body


def _safe_filename(name: str, suffix: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "recording"
    return f"{stem[:80]}{suffix}"


def create_app(data_dir=None) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_BYTES
    bench = Workbench(data_dir)
    app.extensions["workbench"] = bench

    # ------------------------------------------------------------------ errors
    @app.errorhandler(ResoNoiseError)
    def on_app_error(e: ResoNoiseError):
        return jsonify(e.to_dict()), e.status

    @app.errorhandler(RequestEntityTooLarge)
    def on_too_large(_e):
        mb = config.MAX_UPLOAD_BYTES // (1024 * 1024)
        return jsonify({"error": {"code": "file_too_large", "message": f"The file is larger than the {mb} MB limit.",
                                  "details": {}}}), 413

    @app.errorhandler(HTTPException)
    def on_http_error(e: HTTPException):
        return jsonify({"error": {"code": e.name.lower().replace(" ", "_"), "message": e.description,
                                  "details": {}}}), e.code

    @app.errorhandler(Exception)
    def on_unexpected(e: Exception):
        log.exception("Unhandled error")
        return jsonify({"error": {"code": "internal_error", "message": "Something went wrong on the server.",
                                  "details": {"type": type(e).__name__}}}), 500

    # ------------------------------------------------------------------ frontend
    @app.get("/")
    def index():
        return _static("index.html")

    @app.get("/static/<path:filename>")
    def static_files(filename):
        return _static(filename)

    def _static(filename):
        resp = send_from_directory(config.STATIC_DIR, filename)
        resp.headers["Cache-Control"] = "no-cache"
        return resp

    # ------------------------------------------------------------------ meta
    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "version": __version__, "ffmpeg": ffmpeg_available(),
                        "limits": {"max_upload_mb": config.MAX_UPLOAD_BYTES // (1024 * 1024),
                                   "max_duration_s": config.MAX_DURATION_S, "min_duration_s": config.MIN_DURATION_S},
                        "formats": list(config.SUPPORTED_EXTENSIONS), "stats": bench.db.stats()})

    @app.get("/api/model")
    def model():
        info = bench.models.info()
        info["profiles"] = [{"key": k, "label": v.label, "summary": v.summary} for k, v in filtering.PROFILES.items()]
        info["samples"] = [{"key": k, "label": v} for k, v in SAMPLE_KINDS.items()]
        return jsonify(info)

    @app.post("/api/model/retrain")
    def retrain():
        return jsonify(bench.retrain())

    # ------------------------------------------------------------------ recordings
    @app.post("/api/recordings")
    def upload():
        f = request.files.get("file")
        if f is None or f.filename is None:
            raise ValidationError("No file was received. Send the audio as multipart field 'file'.")
        source = request.form.get("source", "upload")
        if source not in ("upload", "recording"):
            raise ValidationError("'source' must be 'upload' or 'recording'.")
        rec = bench.ingest(f.read(), f.filename, source, request.form.get("name") or None)
        return jsonify(rec), 201

    @app.post("/api/samples")
    def sample():
        body = _json_body()
        return jsonify(bench.create_sample(str(body.get("kind", "")), body.get("seed"))), 201

    @app.get("/api/recordings")
    def library():
        return jsonify(bench.library())

    @app.get("/api/recordings/<rid>")
    def detail(rid):
        return jsonify(bench.recording_detail(rid))

    @app.delete("/api/recordings/<rid>")
    def delete(rid):
        bench.delete_recording(rid)
        return jsonify({"deleted": rid})

    @app.post("/api/recordings/<rid>/analyze")
    def analyze(rid):
        return jsonify(bench.analyze(rid))

    @app.get("/api/recordings/<rid>/analysis")
    def analysis(rid):
        return jsonify(bench.get_analysis(rid))

    @app.post("/api/recordings/<rid>/filter-preview")
    def preview(rid):
        return jsonify(bench.preview_filter(rid, _json_body()))

    @app.post("/api/recordings/<rid>/process")
    def process(rid):
        return jsonify(bench.process(rid, _json_body())), 201

    @app.post("/api/recordings/<rid>/label")
    def label(rid):
        body = _json_body()
        return jsonify(bench.set_label(rid, str(body.get("label", ""))))

    @app.get("/api/recordings/<rid>/waveform")
    def waveform(rid):
        t0 = _num("t0", 0.0, 0.0, 1e6)
        t1 = _num("t1", None, 0.0, 1e6)
        points = int(_num("points", 1200, 16, 6000))
        return jsonify(bench.waveform(rid, t0, t1, points, request.args.get("pid")))

    @app.get("/api/recordings/<rid>/audio")
    def audio(rid):
        path, name = bench.audio_path(rid)
        if not path.exists():
            raise NotFoundError("The audio file is missing from storage.")
        dl = request.args.get("download") == "1"
        return send_file(path, mimetype="audio/wav", conditional=True, as_attachment=dl,
                         download_name=_safe_filename(name, "_original.wav"))

    # ------------------------------------------------------------------ processed results
    @app.get("/api/processed/<pid>")
    def processed(pid):
        return jsonify(bench.get_processed_payload(pid))

    @app.delete("/api/processed/<pid>")
    def delete_processed(pid):
        bench.delete_processed(pid)
        return jsonify({"deleted": pid})

    @app.get("/api/processed/<pid>/audio")
    def processed_audio(pid):
        path, name = bench.processed_audio_path(pid)
        if not path.exists():
            raise NotFoundError("The processed audio file is missing from storage.")
        dl = request.args.get("download") == "1"
        return send_file(path, mimetype="audio/wav", conditional=True, as_attachment=dl,
                         download_name=_safe_filename(name, "_denoised.wav"))

    @app.get("/api/processed/<pid>/report")
    def report(pid):
        resp = jsonify(bench.report(pid))
        resp.headers["Content-Disposition"] = f'attachment; filename="reso-noise-report-{pid}.json"'
        return resp

    return app
