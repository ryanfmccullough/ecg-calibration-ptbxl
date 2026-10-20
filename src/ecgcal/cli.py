"""Command-line entry point: ``ecgcal train | evaluate | report``."""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ecgcal", description="Calibrated ECG classification on PTB-XL")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train the baseline and write logits to a run directory")
    t.add_argument("--data", type=Path, required=True, help="PTB-XL root (contains ptbxl_database.csv)")
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--sampling-rate", type=int, default=100, choices=[100, 500])
    t.add_argument("--device", default=None)

    e = sub.add_parser("evaluate", help="discrimination, calibration, operating points")
    e.add_argument("--run", type=Path, required=True)
    e.add_argument("--sensitivity", type=float, default=0.90)

    r = sub.add_parser("report", help="figures and markdown tables")
    r.add_argument("--run", type=Path, required=True)
    r.add_argument("--out", type=Path, default=Path("reports"))

    a = p.parse_args(argv)
    if a.cmd == "train":
        from .train import train

        train(a.data, a.out, a.epochs, a.batch, a.lr, a.seed, a.sampling_rate, a.device)
    elif a.cmd == "evaluate":
        from .report import evaluate

        print(evaluate(a.run, a.sensitivity).to_string(index=False))
    elif a.cmd == "report":
        from .report import report

        report(a.run, a.out)


if __name__ == "__main__":
    main()
