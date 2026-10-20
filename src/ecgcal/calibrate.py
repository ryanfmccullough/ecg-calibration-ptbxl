"""Calibration metrics and temperature scaling.

All functions operate on NumPy arrays so they can be tested without a dataset.
Multi-label outputs are handled one class at a time: ``probs`` and ``labels``
have shape (n_samples, n_classes) or (n_samples,).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReliabilityBins:
    """Per-bin statistics behind a reliability diagram."""

    edges: np.ndarray  # (n_bins + 1,)
    confidence: np.ndarray  # mean predicted probability per bin, NaN where empty
    accuracy: np.ndarray  # observed positive rate per bin, NaN where empty
    count: np.ndarray  # samples per bin


def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> ReliabilityBins:
    """Bin predicted probabilities and compute observed frequency per bin."""
    probs = np.asarray(probs, dtype=float).ravel()
    labels = np.asarray(labels, dtype=float).ravel()
    if probs.shape != labels.shape:
        raise ValueError("probs and labels must have the same shape")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, n_bins - 1)
    count = np.bincount(idx, minlength=n_bins).astype(float)
    conf_sum = np.bincount(idx, weights=probs, minlength=n_bins)
    acc_sum = np.bincount(idx, weights=labels, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        confidence = np.where(count > 0, conf_sum / count, np.nan)
        accuracy = np.where(count > 0, acc_sum / count, np.nan)
    return ReliabilityBins(edges=edges, confidence=confidence, accuracy=accuracy, count=count)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """ECE: count-weighted mean |confidence - accuracy| across bins."""
    rb = reliability_bins(probs, labels, n_bins)
    mask = rb.count > 0
    gap = np.abs(rb.confidence[mask] - rb.accuracy[mask])
    return float(np.sum(gap * rb.count[mask]) / np.sum(rb.count[mask]))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _bce(logits: np.ndarray, labels: np.ndarray) -> float:
    p = _sigmoid(logits)
    eps = 1e-7
    return float(-np.mean(labels * np.log(p + eps) + (1 - labels) * np.log(1 - p + eps)))


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    grid: tuple[float, float] = (0.05, 10.0),
    n_grid: int = 400,
) -> float:
    """Fit a single temperature T minimising BCE of sigmoid(logits / T).

    Grid search on a log scale followed by a local refinement; one scalar per
    class is enough for this study and avoids an optimizer dependency.
    """
    logits = np.asarray(logits, dtype=float).ravel()
    labels = np.asarray(labels, dtype=float).ravel()
    temps = np.exp(np.linspace(np.log(grid[0]), np.log(grid[1]), n_grid))
    losses = np.array([_bce(logits / t, labels) for t in temps])
    best = temps[int(np.argmin(losses))]
    fine = np.linspace(best * 0.8, best * 1.25, 200)
    losses = np.array([_bce(logits / t, labels) for t in fine])
    return float(fine[int(np.argmin(losses))])


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Return calibrated probabilities sigmoid(logits / T)."""
    return _sigmoid(np.asarray(logits, dtype=float) / temperature)


def fit_temperature_per_class(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """One temperature per column of a (n, k) logit matrix."""
    logits = np.atleast_2d(np.asarray(logits, dtype=float))
    labels = np.atleast_2d(np.asarray(labels, dtype=float))
    if logits.shape[0] == 1 and labels.shape[0] != 1:
        logits, labels = logits.T, labels.T
    return np.array([fit_temperature(logits[:, j], labels[:, j]) for j in range(logits.shape[1])])


__all__ = [
    "ReliabilityBins",
    "reliability_bins",
    "expected_calibration_error",
    "fit_temperature",
    "fit_temperature_per_class",
    "apply_temperature",
]
