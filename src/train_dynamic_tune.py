#!/usr/bin/env python3
"""
Fine-tuning script for ASL classification on dynamic clips (frame folders).
"""

from __future__ import annotations
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None
from torch.amp import autocast, GradScaler
from models import build_model
from metrics import (
    topk_accuracy, ConfusionMatrix, EpochMetrics,
    plot_confusion_matrix, save_confusion_assets
)



### Hard-coded FT config 
MODEL_NAME = "r2plus1d_18_tv_kin400"
FRAMES: int = 16
IMG_SIZE: int = 112
EPOCHS: int = 20
BATCH_SIZE: int = 16
WORKERS_TRAIN: int = 4
WORKERS_TEST: int = 0

DROPOUT: float = 0.5
WEIGHT_DECAY: float = 3e-4
LABEL_SMOOTHING: float = 0.1

# Fine-tuning policy
FREEZE_UNTIL: str = "layer2"  # one of: "none", "stem", "layer1", "layer2", "layer3"
UNFREEZE_EPOCH: int = 5  # at this epoch, unfreeze whole backbone
BN_FREEZE: bool = True  # keep BatchNorm in eval mode (no updates)

# Opt: separate lrs for backbone vs head
BACKBONE_LR: float = 3e-5
HEAD_LR: float = 3e-4

# Which checkpoint to fine-tune from (loads matching tensors; skips mismatched head)
FINETUNE_CKPT: str = "/home/cee/Desktop/asl/runs/asl_dynamic_R21D_gpu/best_test.pt"

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
    x = x.permute(2, 0, 1).float() / 255.0             # C,H,W
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


def sample_indices_jitter(num_frames: int, T: int) -> np.ndarray:
    """
    Evenly sample T positions with random jitter around linspace anchors.
    Ensures monotonic indices within [0, num_frames-1].
    """
    if num_frames <= 0:
        return np.zeros((T,), dtype=int)
    if T == 1:
        return np.array([num_frames // 2], dtype=int)

    anchors = np.linspace(0, num_frames - 1, num=T)
    # jitter window ~ 0.5 of the segment length
    seg = max(1.0, (num_frames - 1) / (T - 1))
    jitter = (np.random.rand(T) - 0.5) * (0.5 * seg)  # +/- 0.25 segment
    idxs = np.clip(np.round(anchors + jitter), 0, num_frames - 1).astype(int)
    # enforce monotonic non-decreasing to avoid reversed selections
    idxs = np.maximum.accumulate(idxs)
    return idxs


### Dataset 
class FramesClipDataset(Dataset):
    """
    Loads clips as frame folders; stores filelists+labels; samples frames with temporal jitter on access.
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


    def __len__(self) -> int:
        return len(self._frame_lists)


    def _load_clip(self, frame_files: List[Path]) -> torch.Tensor:
        n = len(frame_files)
        idxs = sample_indices_jitter(n, self.T)
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
        clip = self._load_clip(self._frame_lists[index])
        label = self._labels[index]
        return clip, label


    @property
    def classes_(self) -> List[str]:
        return self.classes


### Fine-tuning helpers
def load_finetune_weights(model: nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt)  # accept raw state_dict or saved training state
    model_sd = model.state_dict()
    filtered = {}
    for k, v in state.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            filtered[k] = v
    print(f"[FT] Loading {len(filtered)}/{len(model_sd)} tensors from {ckpt_path} (head likely skipped).")
    missing = set(model_sd.keys()) - set(filtered.keys())
    if missing:
        print(f"[FT] Skipped {len(missing)} params due to shape/name mismatch (expected for new classifier).")
    model.load_state_dict(filtered, strict=False)


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def freeze_backbone_until(model: nn.Module, stage: str) -> None:
    if stage == "none":
        return
    to_freeze = ["stem"]
    if stage in ["layer1", "layer2", "layer3"]:
        to_freeze += ["layer1"]
    if stage in ["layer2", "layer3"]:
        to_freeze += ["layer2"]
    if stage == "layer3":
        to_freeze += ["layer3"]
    for name in to_freeze:
        if hasattr(model, name):
            set_requires_grad(getattr(model, name), False)
            print(f"[FT] Freeze: {name}")


def set_bn_eval(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False


def build_param_groups(model: nn.Module, head_lr: float, backbone_lr: float):
    # Identify classification head
    head_params = []
    head_prefixes = []
    if hasattr(model, "fc"):
        head_params = list(model.fc.parameters())
        head_prefixes = ["fc"]
    elif hasattr(model, "classifier"):
        head_params = list(model.classifier.parameters())
        head_prefixes = ["classifier"]
    else:
        raise RuntimeError("Could not find classification head (fc/classifier).")

    back_params = []
    for n, p in model.named_parameters():
        if any(n.startswith(pref) for pref in head_prefixes):
            continue
        if p.requires_grad:
            back_params.append(p)

    return [
        {"params": back_params, "lr": backbone_lr},
        {"params": head_params,  "lr": head_lr},
    ]


### Data builders
def build_loaders(train_root: Path, val_root: Path, img_size: int, T_frames: int, batch_size: int):
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

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=WORKERS_TRAIN, pin_memory=True, persistent_workers=False,
        collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=WORKERS_TEST, pin_memory=True, persistent_workers=False,
        collate_fn=collate
    )
    return train_loader, val_loader, classes


### Train
def train_one_epoch(model, loader, criterion, optimizer, device, scaler: GradScaler, amp: bool = True):
    model.train()
    if BN_FREEZE:
        set_bn_eval(model)  # BatchNorms in eval
    
    # num_classes for EpochMetrics only used to shape cm if needed; here track_confusion=False
    out_features = model.fc.out_features if hasattr(model, "fc") else (
        model.classifier.out_features if hasattr(model, "classifier") else None
    )
    meter = EpochMetrics(num_classes=out_features or 1, track_confusion=False)

    for clips, targets in loader:
        clips = clips.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if amp:
            with autocast('cuda'):
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


def safe_add_figure(writer, tag, fig, step):
    if writer is None:
        return
    try:
        writer.add_figure(tag, fig, global_step=step)
    except Exception as e:
        print(f"[TB] add_figure skipped: {e}")


### Main
def main():
    ap = argparse.ArgumentParser(description="ASL training on frame-folder clips (C3D / R(2+1)D) with full RAM cache.")
    ap.add_argument("--data-root", type=str, required=True, help="DATA_ROOT containing 'train' and 'test' subfolders.")
    ap.add_argument("--save-dir", type=str, default="./runs/exp_asl_dyn")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(42)

    data_root  = Path(args.data_root)
    train_root = data_root / "train"
    test_root  = data_root / "test"

    # Data
    train_loader, test_loader, class_names = build_loaders(
        train_root=train_root, val_root=test_root,
        img_size=IMG_SIZE, T_frames=FRAMES,
        batch_size=BATCH_SIZE
    )
    num_classes = len(class_names)

    # Save dir and metadata
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "classes.json").write_text(json.dumps({"idx_to_class": {i: c for i, c in enumerate(class_names)}}, indent=2))

    # Model
    #model = build_model(args.model, num_classes=num_classes, in_channels=3, dropout=DROPOUT)
    tv = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)  # Kinetics-400 pretrained
    # adjust head for ASL
    in_feats = tv.fc.in_features
    tv.fc = nn.Linear(in_feats, num_classes)
    model = tv
    model.to(device)

    # Freeze backbone
    freeze_backbone_until(model, FREEZE_UNTIL)
    if BN_FREEZE:
        set_bn_eval(model)

    # Criterion & Optimizer (param groups)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    param_groups = build_param_groups(model, head_lr=HEAD_LR, backbone_lr=BACKBONE_LR)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler('cuda', enabled=(device.type == "cuda"))

    # Logging
    writer = SummaryWriter(log_dir=str(save_dir / "tb")) if SummaryWriter else None
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
    state = None

    for epoch in range(1, EPOCHS + 1):
        # Gradual Unfreezing
        if epoch == UNFREEZE_EPOCH:
            print(f"[FT] Unfreezing complete backbone at epoch {epoch}.")
            set_requires_grad(model, True)
            if BN_FREEZE:
                set_bn_eval(model)  # keep BN frozen
            # re-build optimizer so newly trainable params are included
            param_groups = build_param_groups(model, head_lr=HEAD_LR, backbone_lr=BACKBONE_LR)
            optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)

        t0 = time.time()
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, amp=scaler.is_enabled())
        test_stats  = evaluate(model, test_loader,  criterion, device, num_classes=num_classes)
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
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train_loss {log['train_loss']:.4f} acc1 {log['train_acc1']:.2f}% acc5 {log['train_acc5']:.2f}% | "
            f"TEST loss {log['test_loss']:.4f} acc1 {log['test_acc1']:.2f}% acc5 {log['test_acc5']:.2f}% "
            f"macroF1 {log['test_macro_f1']:.2f}% | time {dt:.1f}s"
        )

        # TensorBoard scalars
        if writer:
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
        safe_add_figure(writer, "test/confusion_matrix_norm", fig, epoch)
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

        # Save checkpoint (keep best by TEST Macro-F1)
        is_best = log["test_macro_f1"] > best_test_macro_f1
        best_test_macro_f1 = max(best_test_macro_f1, log["test_macro_f1"])
        state = {
            "epoch": epoch,
            "model": MODEL_NAME,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "classes": class_names,
            "frames": FRAMES,
            "img_size": IMG_SIZE,
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
    if state is not None:
        torch.save(state, save_dir / "last.pt")
    if writer:
        writer.close()
    print(f"Done. Best TEST Macro-F1: {best_test_macro_f1:.2f}%")


if __name__ == "__main__":
    main()
