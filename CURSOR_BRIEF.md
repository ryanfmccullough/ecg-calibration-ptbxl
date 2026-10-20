# Brief for the next working session

The scaffold is complete and the dataset-free tests pass (`pytest`). Remaining work, in order:

1. **Download PTB-XL**: `scripts/download_ptbxl.sh data/` (needs `wget`; about 1.7 GB).
2. **Smoke-train** on a subset to confirm the pipeline end to end:
   `ecgcal train --data data/ptbxl --out runs/smoke --epochs 1`.
   Fix anything that breaks in `data.py` loading (wfdb path handling is the likely spot).
3. **Baseline run**: `ecgcal train --data data/ptbxl --out runs/baseline --epochs 20` (GPU if available; CPU is fine overnight).
4. `ecgcal evaluate --run runs/baseline` then `ecgcal report --run runs/baseline`.
5. Paste `reports/summary.md` into the README Results table; commit `reports/reliability.png` and `reports/alerts_vs_sensitivity.png` (remove those two patterns from `.gitignore` first).
6. Add `tests/test_data.py` with a tiny synthetic `ptbxl_database.csv` / `scp_statements.csv` fixture so `superclass_labels` and the fold split are covered without the dataset.

Constraints:
- Keep it CPU-runnable; no new heavy dependencies.
- Reported numbers are engineering verification on a public dataset, nothing clinical. Keep that sentence in the README.
- Do not import or reference anything from the private praxis repositories.
