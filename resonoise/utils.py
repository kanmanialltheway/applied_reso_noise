"""Small helpers shared across modules."""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy types to JSON-safe Python values (NaN/Inf become null)."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def db(power, floor: float = 1e-20):
    """Power ratio to decibels."""
    return 10.0 * np.log10(np.maximum(power, floor))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def rms_db(x: np.ndarray) -> float:
    return float(20.0 * np.log10(max(rms(x), 1e-12)))
