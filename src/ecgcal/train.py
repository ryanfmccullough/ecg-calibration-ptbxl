"""Training loop. Writes model weights, validation/test logits, and labels to a run directory."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import SUPERCLASSES, load_split, standardize
from .model import ECGResNet


def _loader(X: np.ndarray, y: np.ndarray, batch: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0)


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    for xb, _ in loader:
        out.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(out)


def train(
    data_root: Path,
    out_dir: Path,
    epochs: int = 20,
    batch: int = 128,
    lr: float = 1e-3,
    seed: int = 0,
    sampling_rate: int = 100,
    device: str | None = None,
) -> Path:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = load_split(Path(data_root), sampling_rate)
    Xtr, Xva, Xte = standardize(splits["train"].X, splits["val"].X, splits["test"].X)
    tr = _loader(Xtr, splits["train"].y, batch, True)
    va = _loader(Xva, splits["val"].y, batch, False)
    te = _loader(Xte, splits["test"].y, batch, False)

    model = ECGResNet(n_leads=Xtr.shape[1], n_classes=len(SUPERCLASSES)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(tr))
    loss_fn = nn.BCEWithLogitsLoss()

    best_val, best_state = float("inf"), None
    history = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in tr:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item() * xb.size(0)
        val_logits = predict_logits(model, va, dev)
        val_loss = float(loss_fn(torch.from_numpy(val_logits), torch.from_numpy(splits["val"].y)))
        history.append({"epoch": ep + 1, "train_loss": running / len(tr.dataset), "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {ep + 1:3d}  train {history[-1]['train_loss']:.4f}  val {val_loss:.4f}")

    model.load_state_dict(best_state)
    torch.save(best_state, out_dir / "model.pt")
    np.save(out_dir / "val_logits.npy", predict_logits(model, va, dev))
    np.save(out_dir / "val_labels.npy", splits["val"].y)
    np.save(out_dir / "test_logits.npy", predict_logits(model, te, dev))
    np.save(out_dir / "test_labels.npy", splits["test"].y)
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "epochs": epochs,
                "batch": batch,
                "lr": lr,
                "seed": seed,
                "sampling_rate": sampling_rate,
                "device": str(dev),
                "classes": SUPERCLASSES,
                "train_seconds": round(time.time() - t0, 1),
                "history": history,
            },
            indent=2,
        )
    )
    return out_dir


__all__ = ["train", "predict_logits"]
