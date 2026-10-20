"""PTB-XL loading, superclass labels, and the recommended fold split.

Expects the PhysioNet layout produced by ``scripts/download_ptbxl.sh``::

    data/ptbxl/
      ptbxl_database.csv
      scp_statements.csv
      records100/   (100 Hz)
      records500/   (500 Hz)
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
TRAIN_FOLDS = list(range(1, 9))
VAL_FOLD = 9
TEST_FOLD = 10


@dataclass
class Split:
    X: np.ndarray  # (n, leads, samples), float32
    y: np.ndarray  # (n, 5), float32 multi-hot
    ids: np.ndarray  # ecg_id


def load_metadata(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    db = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)
    scp = pd.read_csv(root / "scp_statements.csv", index_col=0)
    scp = scp[scp["diagnostic"] == 1]
    return db, scp


def superclass_labels(db: pd.DataFrame, scp: pd.DataFrame) -> np.ndarray:
    """Multi-hot (n, 5) array over SUPERCLASSES from the SCP codes of each record.

    A record carries a superclass if any of its diagnostic codes maps to it
    (likelihood is ignored, matching the PTB-XL benchmark convention).
    """
    code_to_super = scp["diagnostic_class"].to_dict()
    y = np.zeros((len(db), len(SUPERCLASSES)), dtype=np.float32)
    for i, codes in enumerate(db["scp_codes"]):
        for code in codes:
            sc = code_to_super.get(code)
            if sc in SUPERCLASSES:
                y[i, SUPERCLASSES.index(sc)] = 1.0
    return y


def load_signals(db: pd.DataFrame, root: Path, sampling_rate: int = 100) -> np.ndarray:
    """Read waveforms with wfdb; returns (n, leads, samples) float32."""
    import wfdb  # imported lazily so the metrics modules stay dependency-light

    col = "filename_lr" if sampling_rate == 100 else "filename_hr"
    out = []
    for f in db[col]:
        sig, _ = wfdb.rdsamp(str(root / f))
        out.append(sig.T.astype(np.float32))
    return np.stack(out)


def load_split(root: Path, sampling_rate: int = 100) -> dict[str, Split]:
    """Train/val/test splits on the recommended strat_fold assignment."""
    root = Path(root)
    db, scp = load_metadata(root)
    y_all = superclass_labels(db, scp)
    X_all = load_signals(db, root, sampling_rate)
    folds = db["strat_fold"].to_numpy()
    ids = db.index.to_numpy()

    def take(mask: np.ndarray) -> Split:
        return Split(X=X_all[mask], y=y_all[mask], ids=ids[mask])

    return {
        "train": take(np.isin(folds, TRAIN_FOLDS)),
        "val": take(folds == VAL_FOLD),
        "test": take(folds == TEST_FOLD),
    }


def standardize(train: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    """Per-lead z-scoring with statistics from the training split only."""
    mean = train.mean(axis=(0, 2), keepdims=True)
    std = train.std(axis=(0, 2), keepdims=True) + 1e-6
    return [(a - mean) / std for a in (train, *others)]


__all__ = [
    "SUPERCLASSES",
    "Split",
    "load_metadata",
    "superclass_labels",
    "load_signals",
    "load_split",
    "standardize",
]
