#!/usr/bin/env python3
"""
Training script for ASL classification using 3D CNNs (C3D or R(2+1)D).

Expected dataset layout:
    root/
      train/<class>/*  # images for static, videos for dynamic
      test/<class>/*

# Static images, T=1
python training.py \
  --data-type static \

# Dynamic videos, T=16
python training.py \
  --data /path/to/asl_dynamic_root \
  --data-type dynamic \
"""
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
#from torch.cuda.amp import autocast, GradScaler
#from torch.amp import autocast, GradScaler
try:
   from torch.amp import autocast, GradScaler   # PyTorch ≥ 2.0
   USING_TORCH2 = True
except Exception:
   from torch.cuda.amp import autocast, GradScaler  # PyTorch 1.x
   USING_TORCH2 = False


from models import build_model
from metrics import topk_accuracy, ConfusionMatrix

import cv2  # only required for dynamic (video) mode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


### Utilities
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class To3DClip:
    """Convert [C,H,W] -> [C,T,H,W] by repeating along T."""
    def __init__(self, num_frames: int = 1):
        self.num_frames = max(1, int(num_frames))
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(1).repeat(1, self.num_frames, 1, 1)


def pil_to_tensor_normalized(im: Image.Image) -> torch.Tensor:
    """PIL RGB -> float tensor [C,H,W] normalized by ImageNet stats."""
    x = torch.from_numpy(np.array(im, dtype=np.uint8))  # H,W,C uint8
    x = x.permute(2, 0, 1).float() / 255.0             # C,H,W
    mean = torch.tensor(IMAGENET_MEAN).view(-1,1,1)
    std  = torch.tensor(IMAGENET_STD).view(-1,1,1)
    return (x - mean) / std


def resize_center_crop(im: Image.Image, size: int) -> Image.Image:
    """Deterministic resize -> center crop to exact square size."""
    w, h = im.size
    if min(w, h) != size:
        scale = size / min(w, h)
        im = im.resize((int(round(w*scale)), int(round(h*scale))), Image.BILINEAR)
    w, h = im.size
    left = (w - size) // 2
    top  = (h - size) // 2
    return im.crop((left, top, left + size, top + size))


def discover_split_dirs(root: Path) -> Tuple[Path, Path]:
    """Require exactly 'train/' and 'test/' under root; abort if missing."""
    train_dir = root / "train"
    test_dir  = root / "test"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise RuntimeError(f"Expected '{root}/train' and '{root}/test' to exist.")
    return train_dir, test_dir


def list_class_files(split_dir: Path, exts: set) -> Tuple[Dict[str, List[Path]], List[str]]:
    """Return mapping class->files and the sorted class list."""
    classes = [d.name for d in sorted(split_dir.iterdir()) if d.is_dir()]
    if not classes:
        raise RuntimeError(f"No class subfolders found in {split_dir}")
    class_to_files = {}
    for c in classes:
        files = []
        for p in sorted((split_dir / c).rglob("*")):
            if p.is_file() and p.suffix.lower() in exts:
                files.append(p)
        if not files:
            raise RuntimeError(f"No files found for class '{c}' in {split_dir / c}")
        class_to_files[c] = files
    return class_to_files, classes


def sample_frame_indices(num_frames: int, T: int) -> np.ndarray:
    """Evenly sample T indices from [0, num_frames-1]."""
    if num_frames <= 0:
        return np.array([], dtype=int)
    return np.linspace(0, max(0, num_frames - 1), num=T).astype(int)



### Datasets (always-preload)
class StaticImageDataset(Dataset):
    """Static images → preload all into RAM as [C,T,H,W] + labels."""
    def __init__(self, files: List[Path], class_to_idx: Dict[str,int], img_size: int, T_frames: int):
        self.data: List[Tuple[torch.Tensor, int]] = []
        T_frames = max(1, int(T_frames))
        to3d = To3DClip(T_frames)
        for p in files:
            im = Image.open(p).convert("RGB")
            im = resize_center_crop(im, img_size)
            x = pil_to_tensor_normalized(im)  # [C,H,W]
            x = to3d(x)                       # [C,T,H,W]
            label = class_to_idx[p.parent.name]
            self.data.append((x, label))
    def __len__(self) -> int: return len(self.data)
    def __getitem__(self, i: int): return self.data[i]

class DynamicVideoDataset(Dataset):
    """Videos → preload all into RAM as [C,T,H,W] + labels (uniform sampling)."""
    def __init__(self, files: List[Path], class_to_idx: Dict[str,int], img_size: int, T_frames: int):
        if cv2 is None:
            raise ImportError("OpenCV (cv2) is required for dynamic video loading. Install opencv-python.")
        self.data: List[Tuple[torch.Tensor, int]] = []
        self.img_size = img_size
        self.T = max(1, int(T_frames))
        for p in files:
            clip = self._load_video_clip(p)         # [C,T,H,W]
            label = class_to_idx[p.parent.name]
            self.data.append((clip, label))
    def __len__(self) -> int: return len(self.data)
    def __getitem__(self, i: int): return self.data[i]
    def _load_video_clip(self, path: Path) -> torch.Tensor:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            # fall back to a black clip
            return torch.zeros(3, self.T, self.img_size, self.img_size, dtype=torch.float32)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = sample_frame_indices(length, self.T)
        frames = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(frame)
            im = resize_center_crop(im, self.img_size)
            x = pil_to_tensor_normalized(im)  # [C,H,W]
            frames.append(x)
        cap.release()
        if not frames:
            return torch.zeros(3, self.T, self.img_size, self.img_size, dtype=torch.float32)
        while len(frames) < self.T:
            frames.append(frames[-1])
        clip = torch.stack(frames[:self.T], dim=1)  # [C,T,H,W]
        return clip


### Build loaders (train/test)
def build_loaders(data_root: Path, data_type: str, img_size: int, T_frames: int, batch_size: int, workers: int):
    train_dir, test_dir = discover_split_dirs(data_root)

    # Map classes to files
    if data_type == "static":
        train_map, train_classes = list_class_files(train_dir, IMG_EXTS)
        test_map,  test_classes  = list_class_files(test_dir,  IMG_EXTS)
    elif data_type == "dynamic":
        train_map, train_classes = list_class_files(train_dir, VID_EXTS)
        test_map,  test_classes  = list_class_files(test_dir,  VID_EXTS)
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

    classes = sorted(set(train_classes) | set(test_classes))
    class_to_idx = {c: i for i, c in enumerate(classes)}

    def flatten(m: Dict[str, List[Path]]) -> List[Path]:
        out = []
        for _, files in m.items():
            out.extend(files)
        return out

    train_files = flatten(train_map)
    test_files  = flatten(test_map)

    # Always preload
    if data_type == "static":
        train_ds = StaticImageDataset(train_files, class_to_idx, img_size, T_frames)
        test_ds  = StaticImageDataset(test_files,  class_to_idx, img_size, T_frames)
    else:  # dynamic
        train_ds = DynamicVideoDataset(train_files, class_to_idx, img_size, T_frames)
        test_ds  = DynamicVideoDataset(test_files,  class_to_idx, img_size, T_frames)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True, persistent_workers=(workers>0))
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=workers, pin_memory=True, persistent_workers=(workers>0))
    return train_loader, test_loader, classes



### Checkpointing
def save_checkpoint(state: Dict, is_best: bool, save_dir: Path, epoch: int):
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"epoch_{epoch:03d}.pt"
    torch.save(state, ckpt_path)
    if is_best:
        torch.save(state, save_dir / "best.pt")



### Train / Eval
def train_one_epoch(model, loader, criterion, optimizer, device, scaler: GradScaler, amp: bool = True):
    model.train()
    epoch_loss = 0.0
    n_samples = 0
    correct_top1 = 0.0
    correct_top5 = 0.0

    for clips, targets in loader:
        clips = clips.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if amp:
            ctx = autocast(device_type=device.type, enabled=amp) if USING_TORCH2 else autocast(enabled=amp)
            with ctx:
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

        batch_size = targets.size(0)
        n_samples += batch_size
        epoch_loss += loss.item() * batch_size

        kspec = (1, 5 if outputs.size(1) >= 5 else 1)
        top1, top5 = topk_accuracy(outputs, targets, topk=kspec)
        correct_top1 += (top1 / 100.0) * batch_size
        correct_top5 += (top5 / 100.0) * batch_size

    avg_loss = epoch_loss / max(1, n_samples)
    acc1 = 100.0 * correct_top1 / max(1, n_samples)
    acc5 = 100.0 * correct_top5 / max(1, n_samples)
    return {"loss": avg_loss, "acc1": acc1, "acc5": acc5}


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes: int):
    model.eval()
    epoch_loss = 0.0
    n_samples = 0
    correct_top1 = 0.0
    correct_top5 = 0.0
    confmat = ConfusionMatrix(num_classes=num_classes)

    for clips, targets in loader:
        clips = clips.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(clips)
        loss = criterion(outputs, targets)

        batch_size = targets.size(0)
        n_samples += batch_size
        epoch_loss += loss.item() * batch_size

        kspec = (1, 5 if outputs.size(1) >= 5 else 1)
        top1, top5 = topk_accuracy(outputs, targets, topk=kspec)
        correct_top1 += (top1 / 100.0) * batch_size
        correct_top5 += (top5 / 100.0) * batch_size

        preds = outputs.argmax(dim=1)
        confmat.update(preds, targets)

    avg_loss = epoch_loss / max(1, n_samples)
    acc1 = 100.0 * correct_top1 / max(1, n_samples)
    acc5 = 100.0 * correct_top5 / max(1, n_samples)

    conf_tensor = confmat.compute()
    macro_f1 = confmat.macro_f1()                 # percent
    prec, rec, f1, support = confmat.per_class_metrics()

    return { "loss": avg_loss, "acc1": acc1, "acc5": acc5, "confusion_matrix": conf_tensor, "macro_f1": macro_f1,
            "f1_per_class": f1.tolist(), "support": support.tolist()}



### Main
def main():
    parser = argparse.ArgumentParser(description="ASL training with 3D CNNs (C3D / R(2+1)D)")
    parser.add_argument("--data", type=str, required=True, help="Root path that contains 'train/' and 'test/' subfolders.")
    parser.add_argument("--data-type", type=str, default="static", choices=["static", "dynamic"], help="Static images or dynamic videos.")
    parser.add_argument("--model", type=str, default="r2plus1d", choices=["c3d", "r2plus1d"], help="Model architecture")
    parser.add_argument("--save-dir", type=str, default="./runs/exp")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    epochs: int = 40
    batch_size: int = 32
    workers: int = 8
    img_size: int = 112
    frames: int = 1 if args.data_type == "static" else 16
    learning_rate: float = 3e-4
    dropout: float = 0.5
    weight_decay: float = 1e-4
    

    # Device selection
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu": workers = 0 

    set_seed(42)

    # Data
    data_root = Path(args.data)
    train_loader, val_loader, class_names = build_loaders(data_root, args.data_type, img_size, frames, batch_size, workers)

    print(f"Classes ({len(class_names)}): {class_names}")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "classes.json", "w") as f:
        json.dump({"idx_to_class": {i: c for i, c in enumerate(class_names)}}, f, indent=2)

    # Model
    num_classes = len(class_names)
    model = build_model(args.model, num_classes=num_classes, in_channels=3, dropout=dropout)
    model.to(device)

    # Loss / Optim / Sched / AMP
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    #scaler = GradScaler(device_type='cuda', enabled=(device.type == 'cuda'))
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # Train loop (select best by Macro-F1 for class fairness)
    best_metric = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, amp=scaler.is_enabled())
        val_stats   = evaluate(model, val_loader, criterion, device, num_classes=num_classes)
        scheduler.step()
        dt = time.time() - t0

        log = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_acc1": train_stats["acc1"],
            "train_acc5": train_stats["acc5"],
            "val_loss": val_stats["loss"],
            "val_acc1": val_stats["acc1"],
            "val_acc5": val_stats["acc5"],
            "val_macro_f1": val_stats["macro_f1"],
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": dt,
        }
        history.append(log)

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss {log['train_loss']:.4f} acc1 {log['train_acc1']:.2f}% acc5 {log['train_acc5']:.2f}% | "
            f"val_loss {log['val_loss']:.4f} acc1 {log['val_acc1']:.2f}% acc5 {log['val_acc5']:.2f}% "
            f"macroF1 {log['val_macro_f1']:.2f}% | "
            f"time {dt:.1f}s"
        )

        # Save checkpoint
        is_best = val_stats["macro_f1"] > best_metric
        best_metric = max(best_metric, val_stats["macro_f1"])
        state = {
            "epoch": epoch,
            "model": args.model,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "val_acc1": val_stats["acc1"],
            "val_acc5": val_stats["acc5"],
            "val_macro_f1": val_stats["macro_f1"],
            "val_loss": val_stats["loss"],
            "classes": class_names,
            "data_type": args.data_type,
            "frames": frames,
            "img_size": img_size,
        }
        save_checkpoint(state, is_best, save_dir, epoch)

        # Save confusion matrix for this epoch
        torch.save(val_stats["confusion_matrix"], save_dir / f"confmat_epoch_{epoch:03d}.pt")

        # Persist history
        with open(save_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)


    print(f"Training complete. Best val Macro-F1: {best_metric:.2f}%")



if __name__ == "__main__":
    main()
