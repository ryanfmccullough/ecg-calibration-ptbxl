import numpy as np

from ecgcal.calibrate import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    fit_temperature_per_class,
    reliability_bins,
)


def _synthetic(n=20000, temperature=2.5, seed=0):
    """Labels drawn from well-calibrated probabilities; logits then distorted by a known temperature."""
    rng = np.random.default_rng(seed)
    true_logits = rng.normal(0, 2, n)
    p_true = 1 / (1 + np.exp(-true_logits))
    labels = (rng.random(n) < p_true).astype(float)
    return true_logits * temperature, labels, temperature


def test_reliability_bins_shapes():
    rb = reliability_bins(np.linspace(0, 1, 50), np.zeros(50), n_bins=10)
    assert rb.confidence.shape == (10,)
    assert rb.count.sum() == 50


def test_ece_is_zero_for_perfect_calibration():
    probs = np.repeat(np.linspace(0.05, 0.95, 10), 1000)
    rng = np.random.default_rng(1)
    labels = (rng.random(probs.size) < probs).astype(float)
    assert expected_calibration_error(probs, labels) < 0.02


def test_ece_detects_overconfidence():
    probs = np.full(1000, 0.95)
    labels = np.array([1.0] * 500 + [0.0] * 500)
    assert abs(expected_calibration_error(probs, labels) - 0.45) < 1e-9


def test_fit_temperature_recovers_known_value():
    logits, labels, true_t = _synthetic()
    t = fit_temperature(logits, labels)
    assert abs(t - true_t) / true_t < 0.1
    assert expected_calibration_error(
        apply_temperature(logits, t), labels
    ) < expected_calibration_error(1 / (1 + np.exp(-logits)), labels)


def test_fit_temperature_per_class_handles_matrix():
    l1, y1, _ = _synthetic(temperature=2.0, seed=2)
    l2, y2, _ = _synthetic(temperature=0.5, seed=3)
    temps = fit_temperature_per_class(np.column_stack([l1, l2]), np.column_stack([y1, y2]))
    assert temps.shape == (2,)
    assert abs(temps[0] - 2.0) / 2.0 < 0.1
    assert abs(temps[1] - 0.5) / 0.5 < 0.1
