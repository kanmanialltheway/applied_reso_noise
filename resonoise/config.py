"""Central configuration. Values can be overridden with environment variables."""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
BASE_DIR = PACKAGE_DIR.parent
STATIC_DIR = PACKAGE_DIR / "static"
MODEL_DIR = PACKAGE_DIR / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "default_model.json"
SYNTH_DATASET_PATH = MODEL_DIR / "synthetic_features.npz"

DATA_DIR = Path(os.environ.get("RESONOISE_DATA", BASE_DIR / "data"))

MAX_UPLOAD_BYTES = int(os.environ.get("RESONOISE_MAX_UPLOAD_MB", "60")) * 1024 * 1024
MAX_DURATION_S = float(os.environ.get("RESONOISE_MAX_DURATION_S", "120"))
MIN_DURATION_S = 0.5
MIN_SAMPLE_RATE = 4000
MAX_NATIVE_SAMPLE_RATE = 192000
WORK_MAX_SAMPLE_RATE = 48000      # anything above is resampled down to this rate
CLASSIFIER_SAMPLE_RATE = 16000    # classifier features are always computed at this rate
PEAK_TARGET = 0.9                 # peak level after normalisation
SILENCE_PEAK = 1e-4               # below this the file is considered silent

SUPPORTED_EXTENSIONS = (".wav", ".flac", ".ogg", ".oga", ".mp3", ".m4a", ".aac", ".aif", ".aiff", ".webm", ".opus")
