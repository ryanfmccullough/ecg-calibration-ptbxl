# Status

Last updated 2026-10-20.

## Done

- Download: `scripts/download_ptbxl.sh` fetches PTB-XL 1.0.3 from PhysioNet's public S3 mirror via the `aws` CLI (wget fallback). Verified against `SHA256SUMS.txt`.
- Data: superclass labels (NORM, MI, STTC, CD, HYP) from the SCP statements, recommended fold split (1 to 8 train, 9 validation, 10 test), per-lead standardization from training statistics.
- Baseline: compact 1-D ResNet, 20 epochs, seed 0, 100 Hz; the checkpoint with the lowest validation loss (epoch 12) is kept. Device defaults to MPS, then CUDA, then CPU.
- Evaluation: AUROC, AUPRC, ECE before and after per-class temperature scaling fit on fold 9, and per-threshold operating-point tables with alerts per 100 recordings. Results table and figures are in the README and `reports/`.
- Tests: 17 tests run without the dataset (calibration, thresholds, model shape, device choice, and data loading on a synthetic PTB-XL fixture). `ruff check` and `ruff format` are clean.

## Open ideas

- Overfitting after epoch 12: try augmentation (random time shift, lead dropout, baseline wander) or early stopping and see whether the best checkpoint moves.
- 500 Hz: compare `--sampling-rate 500` with the 100 Hz baseline on the same model.
- Calibration beyond one temperature: Platt scaling or isotonic regression per class, especially for HYP, where temperature scaling barely changed ECE.
- Uncertainty: bootstrap fold 10 for AUROC, ECE, and alerts per 100 so differences between classes and methods can be read with intervals.
- Seeds: repeat the baseline over several seeds and report mean and spread.
- Alert budgets: `threshold_at_alert_budget` exists in `thresholds.py` but is not surfaced by `ecgcal evaluate`; add a `--alert-budget` option.
- Speed: cache the loaded waveforms as `.npy` so repeated runs skip the ~45 s wfdb read.
