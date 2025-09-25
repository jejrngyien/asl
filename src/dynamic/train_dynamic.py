#!/usr/bin/env python3
"""
Training script for ASL classification on dynamic clips organized as frame folders.
Note: All clips are cached into RAM up-front to avoid disk bottlenecks and duplication across workers.

Expected training layout:
  <TRAIN_ROOT> = <DATA_ROOT>/augmented
    > <class_name>/
            > <clip_id>_aug0000/ frame_0001.png, frame_0002.png, ...
            > <clip_id>_aug0001/ ...
            > ...
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from ..models import build_model
from ..metrics import topk_accuracy, ConfusionMatrix, EpochMetrics, plot_confusion_matrix, save_confusion_assets



### Constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


### Utils
def set_seed(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def pil_to_tensor_normalized(im: Image.Image) -> torch.Tensor:
    """PIL RGB -> float tensor [C,H,W] normalized by ImageNet stats."""
    x = torch.from_numpy(np.array(im, dtype=np.uint8))  # H,W,C
    x = x.permute(2, 0, 1).float() / 255.0  # C,H,W
    mean = torch.tensor(IMAGENET_MEAN).view(-1,1,1)
    std  = torch.tensor(IMAGENET_STD).view(-1,1,1)
    return (x - mean) / std


def resize_center_crop(im: Image.Image, size: int) -> Image.Image:
    """Deterministic resize then center-crop to square 'size'."""
    w, h = im.size
    if min(w, h) != size:
        scale = size / min(w, h)
        im = im.resize((int(round(w*scale)), int(round(h*scale))), Image.BILINEAR)
    w, h = im.size
    left = (w - size) // 2
    top  = (h - size) // 2
    return im.crop((left, top, left + size, top + size))


def uniform_indices(num_frames: int, T: int) -> np.ndarray:
    """
    Evenly sample T indices from [0, num_frames-1]. If num_frames < T, repeat last frame.
    Deterministic (no jitter) because offline augmentation already covered variability.
    """
    if num_frames <= 0:
        return np.zeros((T,), dtype=int)
    if T == 1:
        return np.array([num_frames // 2], dtype=int)
    base = np.linspace(0, num_frames - 1, num=T)
    idxs = np.round(base).astype(int)
    idxs = np.clip(idxs, 0, num_frames - 1)
    if num_frames < T:
        # repeat last index to fill length T (already handled by linspace, but be explicit)
        idxs[-1] = num_frames - 1
    return idxs


### Dataset
class FramesClipDataset(Dataset):
    """
    Loads *clips as frame folders* and caches all clips into RAM as [C,T,H,W] tensors.
    Structure:
      root/
        class_a/ clip_0001/ frame_*.png
        class_b/ clip_0002/ frame_*.png
    """
    def __init__(self, root: Path, img_size: int = 112, T_frames: int = 16):
        self.root = Path(root)
        self.img_size = int(img_size)
        self.T = int(T_frames)

        self.classes = [d.name for d in sorted(self.root.iterdir()) if d.is_dir()]
        if not self.classes:
            raise RuntimeError(f"No class folders found under: {self.root}")
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # discover clips
        self._frame_lists: List[List[Path]] = []
        self._labels: List[int] = []
        for cls in self.classes:
            for clip_dir in sorted((self.root / cls).iterdir()):
                if not clip_dir.is_dir():
                    continue
                frames = sorted([p for p in clip_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
                if len(frames) == 0:
                    continue
                self._frame_lists.append(frames)
                self._labels.append(self.class_to_idx[cls])

        if len(self._frame_lists) == 0:
            raise RuntimeError(f"No clips with frames found under: {self.root}")

        # Preload everything into RAM deterministically
        self._clips_ram: List[torch.Tensor] = []
        for frames in self._frame_lists:
            self._clips_ram.append(self._load_and_process_clip(frames))

    def __len__(self) -> int:
        return len(self._clips_ram)

    def _load_and_process_clip(self, frame_files: List[Path]) -> torch.Tensor:
        n = len(frame_files)
        idxs = uniform_indices(n, self.T)
        frames: List[torch.Tensor] = []
        for k in idxs:
            p = frame_files[int(k)]
            im = Image.open(p).convert("RGB")
            im = resize_center_crop(im, self.img_size)
            x = pil_to_tensor_normalized(im)  # [C,H,W]
            frames.append(x)
        clip = torch.stack(frames, dim=1)  # [C,T,H,W]
        return clip

    def __getitem__(self, index: int):
        return self._clips_ram[index], self._labels[index]

    @property
    def classes_(self) -> List[str]:
        return self.classes


### Data builders
def discover_val_root(train_root: Path, val_root_arg: Optional[Path]) -> Path:
    if val_root_arg is not None:
        return val_root_arg
    # heuristic fallbacks:
    cand = train_root.parent / "preprocessed"
    if cand.exists():
        return cand
    cand = train_root.parent / "test"
    if cand.exists():
        return cand
    raise RuntimeError(
        "Could not infer --val-root. Please pass --val-root explicitly (e.g., <DATA_ROOT>/preprocessed)."
    )


def build_loaders(train_root: Path, val_root: Path, img_size: int, T_frames: int, batch_size: int, workers: int,):
    train_ds = FramesClipDataset(train_root, img_size=img_size, T_frames=T_frames)
    val_ds   = FramesClipDataset(val_root,   img_size=img_size, T_frames=T_frames)

    # sanity: same class set
    if train_ds.classes_ != val_ds.classes_:
        missing = set(train_ds.classes_) - set(val_ds.classes_)
        extra   = set(val_ds.classes_) - set(train_ds.classes_)
        if missing or extra:
            raise RuntimeError(
                f"Class mismatch between train and val.\nMissing in val: {sorted(missing)}\nExtra in val: {sorted(extra)}"
            )
    classes = train_ds.classes_

    def collate(batch):
        clips, labels = zip(*batch)
        x = torch.stack(clips, dim=0)  # [B,C,T,H,W]
        y = torch.tensor(labels, dtype=torch.long)
        return x, y

    # NOTE: workers=0 recommended to avoid duplicating RAM cache in child processes.
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, persistent_workers=False,
        collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True, persistent_workers=False,
        collate_fn=collate
    )
    return train_loader, val_loader, classes


### Train/Eval
def train_one_epoch(model, loader, criterion, optimizer, device, scaler: GradScaler, amp: bool = True):
    model.train()
    meter = EpochMetrics(num_classes=model.fc.out_features if hasattr(model, "fc") else None, track_confusion=False)

    for clips, targets in loader:
        clips = clips.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if amp:
            with autocast():
                outputs = model(clips)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(clips)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        meter.update(outputs.detach(), targets, loss.detach())

    return meter.compute()


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes: int):
    model.eval()
    meter = EpochMetrics(num_classes=num_classes, track_confusion=True, device=device)

    for clips, targets in loader:
        clips = clips.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(clips)
        loss = criterion(outputs, targets)
        meter.update(outputs, targets, loss)

    return meter.compute()


### Main
def main():
    ap = argparse.ArgumentParser(description="ASL training on frame-folder clips (C3D / R(2+1)D) with full RAM cache.")
    ap.add_argument("--data-root", type=str, required=True, help="DATA_ROOT containing 'train' and 'test' subfolders.")
    ap.add_argument("--model", type=str, default="r2plus1d", choices=["c3d", "r2plus1d"], help="Backbone")
    ap.add_argument("--save-dir", type=str, default="./runs/exp_asl_dyn")
    args = ap.parse_args()

    frames        = 16
    img_size      = 112
    epochs        = 30
    batch_size    = 16
    learning_rate = 3e-4
    weight_decay  = 1e-4
    dropout       = 0.5
    workers       = 0  # avoid duplicating RAM cache across workers

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    set_seed(42)
    data_root  = Path(args.data_root)
    train_root = data_root / "train"
    test_root  = data_root / "test"

    # Data
    train_loader, test_loader, class_names = build_loaders(
        train_root=train_root, val_root=test_root,  # reuse builder; 'val_root' is our 'test'
        img_size=img_size, T_frames=frames,
        batch_size=batch_size, workers=workers
    )
    num_classes = len(class_names)

    # Save dir and metadata
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "classes.json").write_text(json.dumps({"idx_to_class": {i: c for i, c in enumerate(class_names)}}, indent=2))

    # Model / Opt / Sched / AMP
    model = build_model(args.model, num_classes=num_classes, in_channels=3, dropout=dropout)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # Logging
    writer = SummaryWriter(log_dir=str(save_dir / "tb"))
    history: List[Dict] = []
    csv_path = save_dir / "history.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "epoch","train_loss","train_acc1","train_acc5",
                "test_loss","test_acc1","test_acc5","test_macro_f1",
                "lr","time_sec"
            ])


    best_test_macro_f1 = -1.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, amp=scaler.is_enabled())
        test_stats  = evaluate(model, test_loader, criterion, device, num_classes=num_classes)  # test each epoch
        scheduler.step()
        dt = time.time() - t0

        log = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_acc1": train_stats["acc1"],
            "train_acc5": train_stats["acc5"],
            "test_loss":  test_stats["loss"],
            "test_acc1":  test_stats["acc1"],
            "test_acc5":  test_stats["acc5"],
            "test_macro_f1": test_stats.get("macro_f1", 0.0),
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": dt,
        }
        history.append(log)

        # Console
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss {log['train_loss']:.4f} acc1 {log['train_acc1']:.2f}% acc5 {log['train_acc5']:.2f}% | "
            f"TEST loss {log['test_loss']:.4f} acc1 {log['test_acc1']:.2f}% acc5 {log['test_acc5']:.2f}% "
            f"macroF1 {log['test_macro_f1']:.2f}% | time {dt:.1f}s"
        )

        # TensorBoard scalars
        writer.add_scalar("train/loss", log["train_loss"], epoch)
        writer.add_scalar("train/acc1", log["train_acc1"], epoch)
        writer.add_scalar("train/acc5", log["train_acc5"], epoch)
        writer.add_scalar("test/loss",  log["test_loss"],  epoch)
        writer.add_scalar("test/acc1",  log["test_acc1"],  epoch)
        writer.add_scalar("test/acc5",  log["test_acc5"],  epoch)
        writer.add_scalar("test/macro_f1", log["test_macro_f1"], epoch)
        writer.add_scalar("opt/lr", log["lr"], epoch)

        # Confusion matrix assets & TB figure (test of this epoch)
        cm_tensor = test_stats["confusion_matrix"]
        torch.save(cm_tensor, save_dir / f"confmat_epoch_{epoch:03d}_test.pt")
        asset_dir = save_dir / f"confmat_epoch_{epoch:03d}_test"
        asset_dir.mkdir(parents=True, exist_ok=True)
        _ = save_confusion_assets(cm_tensor, class_names, asset_dir)

        fig = plot_confusion_matrix(cm_tensor, class_names, normalize=True)
        writer.add_figure("test/confusion_matrix_norm", fig, global_step=epoch)
        import matplotlib.pyplot as _plt
        _plt.close(fig)

        # CSV row
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                epoch, log["train_loss"], log["train_acc1"], log["train_acc5"],
                log["test_loss"], log["test_acc1"], log["test_acc5"], log["test_macro_f1"],
                log["lr"], log["time_sec"]
            ])

        # Save checkpoint (optional: keep best by TEST Macro-F1)
        is_best = log["test_macro_f1"] > best_test_macro_f1
        best_test_macro_f1 = max(best_test_macro_f1, log["test_macro_f1"])
        state = {
            "epoch": epoch,
            "model": args.model,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "classes": class_names,
            "frames": frames,
            "img_size": img_size,
            # quick refs
            "train_loss": log["train_loss"], "train_acc1": log["train_acc1"], "train_acc5": log["train_acc5"],
            "test_loss": log["test_loss"],   "test_acc1": log["test_acc1"],   "test_acc5": log["test_acc5"],
            "test_macro_f1": log["test_macro_f1"],
        }
        torch.save(state, save_dir / f"epoch_{epoch:03d}.pt")
        if is_best:
            torch.save(state, save_dir / "best_test.pt")

        # Persist JSON history
        (save_dir / "history.json").write_text(json.dumps(history, indent=2))

    # Save final checkpoint
    torch.save(state, save_dir / "last.pt")
    writer.close()
    print(f"Done. Best TEST Macro-F1: {best_test_macro_f1:.2f}%")


if __name__ == "__main__":
    main()
