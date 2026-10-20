# Calibrated ECG classification on PTB-XL

A small, reproducible study of what happens to an ECG classifier *after* the accuracy number: how well its probabilities are calibrated, and what each operating threshold costs in alerts per hundred recordings against the sensitivity it buys.

Accuracy ranks models. Calibration and operating points decide whether a model is usable at the bedside. This repository keeps the two questions separate and reports both.

## What it does

1. Downloads [PTB-XL](https://physionet.org/content/ptb-xl/) (21,799 twelve-lead ECGs, 10 s, open access under CC BY 4.0) and builds the five diagnostic superclass labels (NORM, MI, STTC, CD, HYP) from the SCP statements.
2. Trains a compact 1-D residual network on the recommended folds (1 to 8 train, 9 validation, 10 test).
3. Measures discrimination (AUROC, AUPRC) and calibration (expected calibration error, reliability diagrams) on the held-out fold, before and after temperature scaling fit on the validation fold.
4. Produces an operating-point table per class: for each threshold, sensitivity, positive predictive value, and alerts per 100 recordings. The table is the artifact a reviewer would actually use to pick a threshold.

## Quickstart

```bash
pip install -e ".[dev]"
scripts/download_ptbxl.sh data/            # about 1.7 GB, one time
ecgcal train   --data data/ptbxl --out runs/baseline
ecgcal evaluate --run runs/baseline         # metrics, calibration, operating points
ecgcal report  --run runs/baseline          # figures and tables under reports/
```

Everything is CPU-runnable; a GPU shortens training from hours to minutes.

## Results

Populated by `ecgcal report`. The headline table has one row per superclass:

| Class | AUROC | AUPRC | ECE (raw) | ECE (temp-scaled) | Threshold @ 90% sensitivity | Alerts / 100 ECGs @ that threshold |
|---|---|---|---|---|---|---|

Reliability diagrams and the full operating-point curves are written to `reports/`.

## Why this shape

A model that looks strong on AUROC can still be badly calibrated, and two models with the same AUROC can impose very different alert loads at the sensitivity a clinic requires. Reporting the operating-point table alongside discrimination makes that visible before anyone argues about which model is "better".

## Layout

```
src/ecgcal/
  data.py        download, label construction, fold splits
  model.py       1-D ResNet
  train.py       training loop and CLI
  calibrate.py   ECE, reliability bins, temperature scaling
  thresholds.py  operating-point table
  report.py      figures and markdown tables
tests/           unit tests on synthetic signals (no dataset needed)
scripts/         dataset download
```

## Data and citation

PTB-XL: Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F. I., Samek, W., & Schaeffter, T. (2020). PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data*, 7, 154. https://doi.org/10.1038/s41597-020-0495-6

The dataset is not redistributed here; the download script fetches it from PhysioNet.

## Status

Scaffold and unit tests complete; baseline training and the results table are in progress. Engineering verification only; nothing here is clinically validated.

## Author

Ryan F. McCullough, [ryanfmccullough.com](https://ryanfmccullough.com). MIT License.
