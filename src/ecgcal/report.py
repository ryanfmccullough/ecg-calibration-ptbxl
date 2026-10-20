"""Evaluate a run: discrimination, calibration before/after temperature scaling, operating points."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .calibrate import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature_per_class,
    reliability_bins,
)
from .data import SUPERCLASSES
from .thresholds import operating_point_table, threshold_at_sensitivity


def evaluate(run_dir: Path, sensitivity_target: float = 0.90) -> pd.DataFrame:
    run_dir = Path(run_dir)
    val_logits = np.load(run_dir / "val_logits.npy")
    val_labels = np.load(run_dir / "val_labels.npy")
    test_logits = np.load(run_dir / "test_logits.npy")
    test_labels = np.load(run_dir / "test_labels.npy")

    temps = fit_temperature_per_class(val_logits, val_labels)
    raw = 1.0 / (1.0 + np.exp(-test_logits))
    cal = np.column_stack([apply_temperature(test_logits[:, j], temps[j]) for j in range(len(temps))])

    rows = []
    tables = {}
    for j, cls in enumerate(SUPERCLASSES):
        y = test_labels[:, j]
        tab = operating_point_table(cal[:, j], y)
        tables[cls] = tab
        op = threshold_at_sensitivity(tab, sensitivity_target)
        rows.append(
            {
                "class": cls,
                "prevalence": float(y.mean()),
                "auroc": float(roc_auc_score(y, raw[:, j])),
                "auprc": float(average_precision_score(y, raw[:, j])),
                "ece_raw": expected_calibration_error(raw[:, j], y),
                "ece_temp_scaled": expected_calibration_error(cal[:, j], y),
                "temperature": float(temps[j]),
                f"threshold_at_{int(sensitivity_target * 100)}pct_sens": float(op["threshold"]),
                "alerts_per_100_at_target": float(op["alerts_per_100"]),
                "ppv_at_target": float(op["ppv"]),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir / "summary.csv", index=False)
    for cls, tab in tables.items():
        tab.to_csv(run_dir / f"operating_points_{cls}.csv", index=False)
    (run_dir / "temperatures.json").write_text(json.dumps(dict(zip(SUPERCLASSES, map(float, temps))), indent=2))
    return summary


# summary.csv column -> README heading. Keeps summary.md paste-ready for the Results table.
_SUMMARY_HEADINGS = {
    "class": "Class",
    "prevalence": "Prevalence",
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "ece_raw": "ECE (raw)",
    "ece_temp_scaled": "ECE (temp-scaled)",
    "temperature": "Temperature",
    "alerts_per_100_at_target": "Alerts / 100 ECGs @ that threshold",
    "ppv_at_target": "PPV @ that threshold",
}


def _heading(col: str) -> str:
    if col.startswith("threshold_at_") and col.endswith("pct_sens"):
        return f"Threshold @ {col[len('threshold_at_'):-len('pct_sens')]}% sensitivity"
    return _SUMMARY_HEADINGS.get(col, col)


def markdown_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Pipe-delimited markdown table (avoids the ``tabulate`` dependency of ``to_markdown``)."""

    def cell(v: object) -> str:
        return format(v, floatfmt) if isinstance(v, float) else str(v)

    lines = [
        "| " + " | ".join(_heading(str(c)) for c in df.columns) + " |",
        "|" + "|".join("---" for _ in df.columns) + "|",
    ]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def report(run_dir: Path, out_dir: Path = Path("reports")) -> None:
    """Markdown table plus reliability diagrams and alert-vs-sensitivity curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir, out_dir = Path(run_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(run_dir / "summary.csv")
    test_logits = np.load(run_dir / "test_logits.npy")
    test_labels = np.load(run_dir / "test_labels.npy")
    temps = json.loads((run_dir / "temperatures.json").read_text())

    (out_dir / "summary.md").write_text(markdown_table(summary) + "\n")

    fig, axes = plt.subplots(1, len(SUPERCLASSES), figsize=(3.2 * len(SUPERCLASSES), 3.2), sharey=True)
    for j, (cls, ax) in enumerate(zip(SUPERCLASSES, axes)):
        y = test_labels[:, j]
        raw = 1.0 / (1.0 + np.exp(-test_logits[:, j]))
        cal = apply_temperature(test_logits[:, j], temps[cls])
        for probs, label in ((raw, "raw"), (cal, "temperature scaled")):
            rb = reliability_bins(probs, y)
            ax.plot(rb.confidence, rb.accuracy, marker="o", label=label)
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.set_title(cls)
        ax.set_xlabel("predicted probability")
    axes[0].set_ylabel("observed frequency")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "reliability.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    for cls in SUPERCLASSES:
        tab = pd.read_csv(run_dir / f"operating_points_{cls}.csv")
        ax.plot(tab["alerts_per_100"], tab["sensitivity"], label=cls)
    ax.set_xlabel("alerts per 100 recordings")
    ax.set_ylabel("sensitivity")
    ax.set_title("What each operating point costs and buys")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "alerts_vs_sensitivity.png", dpi=150)
    plt.close(fig)


__all__ = ["evaluate", "markdown_table", "report"]
