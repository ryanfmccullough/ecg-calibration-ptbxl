import numpy as np
import pytest

from ecgcal.thresholds import (
    operating_point_table,
    threshold_at_alert_budget,
    threshold_at_sensitivity,
)


def _scores():
    rng = np.random.default_rng(0)
    labels = (rng.random(5000) < 0.2).astype(int)
    probs = np.clip(rng.normal(0.3 + 0.4 * labels, 0.2), 0, 1)
    return probs, labels


def test_table_monotone_in_threshold():
    probs, labels = _scores()
    tab = operating_point_table(probs, labels)
    assert (np.diff(tab["alerts_per_100"]) <= 1e-9).all()
    assert (np.diff(tab["sensitivity"]) <= 1e-9).all()
    assert tab.iloc[0]["alerts_per_100"] == 100.0
    assert tab.iloc[0]["sensitivity"] == 1.0


def test_threshold_at_sensitivity_meets_floor_with_fewest_alerts():
    probs, labels = _scores()
    tab = operating_point_table(probs, labels)
    row = threshold_at_sensitivity(tab, 0.9)
    assert row["sensitivity"] >= 0.9
    higher = tab[tab["threshold"] > row["threshold"]]
    assert (higher["sensitivity"] < 0.9).all()


def test_threshold_at_alert_budget():
    probs, labels = _scores()
    tab = operating_point_table(probs, labels)
    row = threshold_at_alert_budget(tab, 25.0)
    assert row["alerts_per_100"] <= 25.0
    lower = tab[tab["threshold"] < row["threshold"]]
    assert (lower["alerts_per_100"] > 25.0).all()


def test_unreachable_sensitivity_raises():
    tab = operating_point_table(np.array([0.1, 0.2]), np.array([0, 0]))
    with pytest.raises(ValueError):
        threshold_at_sensitivity(tab, 0.5)
