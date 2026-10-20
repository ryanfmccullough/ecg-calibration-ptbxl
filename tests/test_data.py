"""Metadata parsing, superclass labels, and the fold split on a tiny synthetic PTB-XL.

The fixture writes a PhysioNet-layout directory (``ptbxl_database.csv``,
``scp_statements.csv``, and 100 Hz WFDB records) so nothing here needs the
real dataset.
"""

import numpy as np
import pandas as pd
import pytest

from ecgcal.data import (
    SUPERCLASSES,
    TEST_FOLD,
    TRAIN_FOLDS,
    VAL_FOLD,
    load_metadata,
    load_signals,
    load_split,
    standardize,
    superclass_labels,
)

LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# Subset of scp_statements.csv: code, diagnostic flag, form flag, rhythm flag, diagnostic_class.
# Only diagnostic codes carry a diagnostic_class; the real file uses 1.0 / NaN for the flags.
SCP_ROWS = [
    ("NORM", 1.0, np.nan, np.nan, "NORM"),
    ("IMI", 1.0, np.nan, np.nan, "MI"),
    ("ASMI", 1.0, np.nan, np.nan, "MI"),
    ("NDT", 1.0, np.nan, np.nan, "STTC"),
    ("LAFB", 1.0, np.nan, np.nan, "CD"),
    ("LVH", 1.0, np.nan, np.nan, "HYP"),
    ("PAC", np.nan, 1.0, np.nan, np.nan),
    ("SR", np.nan, np.nan, 1.0, np.nan),
    ("AFIB", np.nan, np.nan, 1.0, np.nan),
]

# ecg_id, strat_fold, scp_codes (code -> likelihood), expected superclasses.
RECORDS = [
    (1, 1, {"NORM": 100.0, "SR": 0.0}, {"NORM"}),
    (2, 2, {"IMI": 100.0, "SR": 0.0}, {"MI"}),
    (3, 3, {"ASMI": 50.0, "LVH": 100.0, "SR": 0.0}, {"MI", "HYP"}),
    (4, 4, {"NDT": 100.0, "LAFB": 100.0, "AFIB": 0.0}, {"STTC", "CD"}),
    (5, 5, {"SR": 0.0}, set()),  # rhythm-only record: no diagnostic superclass
    (6, 6, {"PAC": 0.0, "SR": 0.0}, set()),  # form + rhythm only
    (7, 7, {"IMI": 0.0}, {"MI"}),  # likelihood 0 still counts (benchmark convention)
    (8, 8, {"NORM": 100.0, "LVH": 15.0}, {"NORM", "HYP"}),
    (9, 9, {"NDT": 100.0}, {"STTC"}),
    (10, 9, {"LAFB": 100.0, "SR": 0.0}, {"CD"}),
    (11, 10, {"IMI": 80.0, "ASMI": 20.0}, {"MI"}),  # two codes, one superclass -> single hot
    (12, 10, {"NORM": 100.0}, {"NORM"}),
]


def _expected_labels() -> np.ndarray:
    y = np.zeros((len(RECORDS), len(SUPERCLASSES)), dtype=np.float32)
    for i, (_, _, _, supers) in enumerate(RECORDS):
        for s in supers:
            y[i, SUPERCLASSES.index(s)] = 1.0
    return y


def write_metadata(root) -> None:
    """Write scp_statements.csv and ptbxl_database.csv in the PhysioNet column layout."""
    scp = pd.DataFrame(
        SCP_ROWS, columns=["code", "diagnostic", "form", "rhythm", "diagnostic_class"]
    ).set_index("code")
    scp.index.name = None  # the real file has an unnamed first column
    scp.insert(0, "description", [f"description of {c}" for c in scp.index])
    scp.to_csv(root / "scp_statements.csv")

    rows = [
        {
            "ecg_id": ecg_id,
            "patient_id": 1000 + ecg_id,
            "age": 60,
            "sex": ecg_id % 2,
            "scp_codes": repr(codes),  # stored as a Python dict literal, e.g. "{'NORM': 100.0}"
            "strat_fold": fold,
            "filename_lr": f"records100/00000/{ecg_id:05d}_lr",
            "filename_hr": f"records500/00000/{ecg_id:05d}_hr",
        }
        for ecg_id, fold, codes, _ in RECORDS
    ]
    pd.DataFrame(rows).to_csv(root / "ptbxl_database.csv", index=False)


@pytest.fixture
def ptbxl_metadata(tmp_path):
    root = tmp_path / "ptbxl"
    root.mkdir()
    write_metadata(root)
    return root


@pytest.fixture
def ptbxl_root(ptbxl_metadata):
    """Metadata plus one synthetic 12-lead, 10 s, 100 Hz WFDB record per row."""
    wfdb = pytest.importorskip("wfdb")
    rec_dir = ptbxl_metadata / "records100" / "00000"
    rec_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    signals = {}
    for ecg_id, *_ in RECORDS:
        sig = rng.normal(0.0, 0.5, size=(1000, 12))
        wfdb.wrsamp(
            f"{ecg_id:05d}_lr",
            fs=100,
            units=["mV"] * 12,
            sig_name=LEADS,
            p_signal=sig,
            fmt=["16"] * 12,
            write_dir=str(rec_dir),
        )
        signals[ecg_id] = sig.T.astype(np.float32)  # (leads, samples), as load_signals returns
    return ptbxl_metadata, signals


def test_load_metadata_parses_codes_and_keeps_diagnostic_statements(ptbxl_metadata):
    db, scp = load_metadata(ptbxl_metadata)
    assert db.index.name == "ecg_id"
    assert list(db.index) == [r[0] for r in RECORDS]
    assert db.loc[3, "scp_codes"] == {"ASMI": 50.0, "LVH": 100.0, "SR": 0.0}
    assert set(scp.index) == {"NORM", "IMI", "ASMI", "NDT", "LAFB", "LVH"}
    assert set(scp["diagnostic_class"]) <= set(SUPERCLASSES)


def test_superclass_labels(ptbxl_metadata):
    db, scp = load_metadata(ptbxl_metadata)
    y = superclass_labels(db, scp)
    assert y.shape == (len(RECORDS), len(SUPERCLASSES))
    assert y.dtype == np.float32
    np.testing.assert_array_equal(y, _expected_labels())
    # Rhythm- and form-only records carry no superclass at all.
    assert y[[4, 5]].sum() == 0
    # A record with two codes in the same superclass is still a single hot entry.
    assert y[10].sum() == 1


def test_superclass_labels_ignore_unknown_codes(ptbxl_metadata):
    db, scp = load_metadata(ptbxl_metadata)
    db["scp_codes"] = [
        {**codes, "NOT_A_CODE": 100.0} if ecg_id == 1 else codes
        for ecg_id, codes in zip(db.index, db["scp_codes"])
    ]
    y = superclass_labels(db, scp)
    assert y[0].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]


def test_load_signals_shape_and_order(ptbxl_root):
    root, signals = ptbxl_root
    db, _ = load_metadata(root)
    X = load_signals(db, root, sampling_rate=100)
    assert X.shape == (len(RECORDS), 12, 1000)
    assert X.dtype == np.float32
    for i, ecg_id in enumerate(db.index):
        np.testing.assert_allclose(X[i], signals[ecg_id], atol=1e-3)


def test_load_split_follows_strat_fold(ptbxl_root):
    root, signals = ptbxl_root
    splits = load_split(root, sampling_rate=100)
    assert set(splits) == {"train", "val", "test"}

    folds = {ecg_id: fold for ecg_id, fold, *_ in RECORDS}
    expected = {
        "train": [i for i, f in folds.items() if f in TRAIN_FOLDS],
        "val": [i for i, f in folds.items() if f == VAL_FOLD],
        "test": [i for i, f in folds.items() if f == TEST_FOLD],
    }
    assert expected["val"] == [9, 10] and expected["test"] == [11, 12]
    y_all = _expected_labels()
    for name, ids in expected.items():
        s = splits[name]
        assert s.ids.tolist() == ids
        assert s.X.shape == (len(ids), 12, 1000)
        assert s.y.shape == (len(ids), len(SUPERCLASSES))
        np.testing.assert_array_equal(s.y, y_all[[i - 1 for i in ids]])
        for k, ecg_id in enumerate(ids):
            np.testing.assert_allclose(s.X[k], signals[ecg_id], atol=1e-3)

    # Every record lands in exactly one split.
    all_ids = sorted(np.concatenate([splits[n].ids for n in splits]).tolist())
    assert all_ids == [r[0] for r in RECORDS]


def test_standardize_uses_train_statistics_only():
    rng = np.random.default_rng(0)
    train = rng.normal(2.0, 3.0, size=(50, 12, 100)).astype(np.float32)
    other = rng.normal(-1.0, 0.5, size=(10, 12, 100)).astype(np.float32)
    ztrain, zother = standardize(train, other)
    np.testing.assert_allclose(ztrain.mean(axis=(0, 2)), 0.0, atol=1e-4)
    np.testing.assert_allclose(ztrain.std(axis=(0, 2)), 1.0, atol=1e-3)
    # The other split is shifted by the training statistics, not re-centred on itself.
    assert abs(zother.mean()) > 0.5
