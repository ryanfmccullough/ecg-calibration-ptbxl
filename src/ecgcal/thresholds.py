"""Operating-point analysis: what each threshold costs and buys.

For a binary score the table reports, per threshold, sensitivity, specificity,
positive predictive value, and alerts per 100 recordings (the fraction of
recordings flagged, scaled to 100). The last column is the one a unit feels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def operating_point_table(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """One row per threshold. Thresholds default to 0.00 .. 1.00 in steps of 0.01."""
    probs = np.asarray(probs, dtype=float).ravel()
    labels = np.asarray(labels, dtype=int).ravel()
    if probs.shape != labels.shape:
        raise ValueError("probs and labels must have the same shape")
    if thresholds is None:
        thresholds = np.round(np.linspace(0.0, 1.0, 101), 2)
    n = probs.size
    n_pos = int(labels.sum())
    n_neg = n - n_pos
    rows = []
    for t in thresholds:
        flagged = probs >= t
        tp = int(np.sum(flagged & (labels == 1)))
        fp = int(np.sum(flagged & (labels == 0)))
        fn = n_pos - tp
        tn = n_neg - fp
        sens = tp / n_pos if n_pos else np.nan
        spec = tn / n_neg if n_neg else np.nan
        ppv = tp / (tp + fp) if (tp + fp) else np.nan
        rows.append(
            {
                "threshold": float(t),
                "sensitivity": sens,
                "specificity": spec,
                "ppv": ppv,
                "alerts_per_100": 100.0 * (tp + fp) / n if n else np.nan,
                "false_alerts_per_100": 100.0 * fp / n if n else np.nan,
                "missed": fn,
            }
        )
    return pd.DataFrame(rows)


def threshold_at_sensitivity(table: pd.DataFrame, target: float = 0.90) -> pd.Series:
    """Highest threshold whose sensitivity is at least ``target``.

    Picking the highest such threshold minimises alerts while meeting the
    sensitivity floor, which is how a unit would set it.
    """
    ok = table[table["sensitivity"] >= target]
    if ok.empty:
        raise ValueError(f"no threshold reaches sensitivity {target}")
    return ok.loc[ok["threshold"].idxmax()]


def threshold_at_alert_budget(table: pd.DataFrame, alerts_per_100: float) -> pd.Series:
    """Lowest threshold whose alert rate fits the budget; reports the sensitivity it buys."""
    ok = table[table["alerts_per_100"] <= alerts_per_100]
    if ok.empty:
        raise ValueError(f"no threshold fits an alert budget of {alerts_per_100} per 100")
    return ok.loc[ok["threshold"].idxmin()]


__all__ = ["operating_point_table", "threshold_at_alert_budget", "threshold_at_sensitivity"]
