"""Noise-source classification (stage 5 of the system architecture).

The classifier is multinomial logistic (softmax) regression on standardised Fourier-derived
features, implemented from scratch with NumPy. Training minimises the L2-regularised
cross-entropy with L-BFGS using the analytic gradient (verified against finite differences in
the tests). It is trained on synthetic residential noise plus any clips the user has labelled.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from .features import FEATURE_LABELS, FEATURE_NAMES, feature_vector
from .synthetic import CLASS_KEYS

NOISE_CLASSES = {
    "fan": {"label": "Fan", "group": "Household appliance",
            "description": "A steady blade-pass tone with harmonics over turbulent broadband air noise."},
    "ac": {"label": "Air conditioner", "group": "Household appliance",
           "description": "Compressor hum on mains harmonics (100 or 120 Hz) with soft airflow noise."},
    "traffic": {"label": "Road traffic", "group": "Traffic (cars)",
                "description": "Low-frequency rumble and tyre noise that swells as vehicles pass."},
    "excavator": {"label": "Excavator", "group": "Construction",
                  "description": "A diesel engine pulse train with strong low harmonics, whine and impulsive clanks."},
    "speech": {"label": "People talking", "group": "Human activity",
               "description": "Voiced harmonics with formants, syllable-rate modulation and pauses."},
}
UNCERTAIN_BELOW = 0.5


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def cross_entropy_objective(theta, Z, Y, w, l2):
    """Weighted L2-regularised cross-entropy and its gradient. theta packs (W, b)."""
    d, k = Z.shape[1], Y.shape[1]
    W, b = theta[: d * k].reshape(d, k), theta[d * k:]
    logits = Z @ W + b
    logits -= logits.max(axis=1, keepdims=True)
    logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    sw = w.sum()
    loss = -np.sum(w * np.sum(Y * logp, axis=1)) / sw + 0.5 * l2 * np.sum(W * W)
    P = np.exp(logp)
    G = w[:, None] * (P - Y) / sw
    grad = np.concatenate([(Z.T @ G + l2 * W).ravel(), G.sum(axis=0)])
    return loss, grad


class NoiseClassifier:
    def __init__(self, classes=None, feature_names=None):
        self.classes = list(classes or CLASS_KEYS)
        self.feature_names = list(feature_names or FEATURE_NAMES)
        d, k = len(self.feature_names), len(self.classes)
        self.mean = np.zeros(d)
        self.scale = np.ones(d)
        self.W = np.zeros((d, k))
        self.b = np.zeros(k)
        self.meta: dict = {}

    # ------------------------------------------------------------------ training
    def fit(self, X, y, *, l2: float = 0.02, sample_weight=None, max_iter: int = 800):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError("feature matrix has the wrong shape")
        if len(np.unique(y)) < 2:
            raise ValueError("at least two classes are needed to train")
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        self.mean = np.average(X, axis=0, weights=w)
        self.scale = np.sqrt(np.average((X - self.mean) ** 2, axis=0, weights=w)) + 1e-9
        Z = np.clip((X - self.mean) / self.scale, -6, 6)
        k = len(self.classes)
        Y = np.zeros((len(y), k))
        Y[np.arange(len(y)), y] = 1.0
        theta0 = np.zeros(Z.shape[1] * k + k)
        res = minimize(cross_entropy_objective, theta0, args=(Z, Y, w, l2), jac=True,
                       method="L-BFGS-B", options={"maxiter": max_iter})
        d = Z.shape[1]
        self.W = res.x[: d * k].reshape(d, k)
        self.b = res.x[d * k:]
        return self

    # ------------------------------------------------------------------ inference
    def _standardise(self, X):
        return np.clip((np.atleast_2d(X) - self.mean) / self.scale, -6, 6)

    def predict_proba(self, X) -> np.ndarray:
        return softmax(self._standardise(X) @ self.W + self.b)

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def classify(self, feats: dict[str, float]) -> dict:
        """Classify one feature dict. Returns ranked probabilities and the features that drove the decision."""
        x = feature_vector(feats)
        z = self._standardise(x)[0]
        p = softmax(z @ self.W + self.b)
        order = np.argsort(p)[::-1]
        top, second = int(order[0]), int(order[1])
        contrib = z * (self.W[:, top] - self.W[:, second])
        drivers = []
        for j in np.argsort(contrib)[::-1][:4]:
            if contrib[j] <= 0:
                break
            name = self.feature_names[j]
            drivers.append({"feature": name, "label": FEATURE_LABELS.get(name, name),
                            "value": float(feats[name]), "contribution": float(contrib[j])})
        key = self.classes[top]
        return {
            "top": key, "label": NOISE_CLASSES[key]["label"], "confidence": float(p[top]),
            "uncertain": bool(p[top] < UNCERTAIN_BELOW),
            "probabilities": [{"key": self.classes[i], "label": NOISE_CLASSES[self.classes[i]]["label"],
                               "prob": float(p[i])} for i in order],
            "runner_up": self.classes[second], "drivers": drivers,
        }

    def evaluate(self, X, y) -> dict:
        pred = self.predict(X)
        k = len(self.classes)
        cm = np.zeros((k, k), dtype=int)
        for t, p in zip(y, pred):
            cm[int(t), int(p)] += 1
        return {"accuracy": float(np.mean(pred == np.asarray(y))), "confusion": cm.tolist(), "n": int(len(y))}

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {"classes": self.classes, "feature_names": self.feature_names, "mean": self.mean.tolist(),
                "scale": self.scale.tolist(), "W": self.W.tolist(), "b": self.b.tolist(), "meta": self.meta}

    @classmethod
    def from_dict(cls, d: dict) -> "NoiseClassifier":
        obj = cls(d["classes"], d["feature_names"])
        obj.mean, obj.scale = np.array(d["mean"]), np.array(d["scale"])
        obj.W, obj.b = np.array(d["W"]), np.array(d["b"])
        obj.meta = d.get("meta", {})
        return obj

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(path) + ".tmp")
        tmp.write_text(json.dumps(self.to_dict()))
        tmp.replace(path)

    @classmethod
    def load(cls, path) -> "NoiseClassifier":
        return cls.from_dict(json.loads(Path(path).read_text()))


def train_classifier(X, y, split, extra_X=None, extra_y=None, extra_weight: float = 4.0) -> NoiseClassifier:
    """Train on the training split (plus user-labelled samples) and validate on the held-out split."""
    X, y, split = np.asarray(X), np.asarray(y), np.asarray(split)
    tr, va = split == 0, split == 1
    Xt, yt = X[tr], y[tr]
    w = np.ones(len(yt))
    n_user = 0
    if extra_X is not None and len(extra_X):
        Xt = np.vstack([Xt, extra_X])
        yt = np.concatenate([yt, extra_y])
        w = np.concatenate([w, np.full(len(extra_y), extra_weight)])
        n_user = int(len(extra_y))
    clf = NoiseClassifier().fit(Xt, yt, sample_weight=w)
    ev = clf.evaluate(X[va], y[va]) if va.any() else {"accuracy": None, "confusion": None, "n": 0}
    clf.meta = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"), "n_synthetic_train": int(tr.sum()), "n_user_samples": n_user,
        "validation_accuracy": ev["accuracy"], "validation_n": ev["n"], "confusion": ev["confusion"],
        "algorithm": "softmax regression (L-BFGS, L2)", "n_features": len(FEATURE_NAMES),
    }
    return clf
