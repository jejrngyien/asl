"""
Basic metrics: Top-k accuracy and a running confusion matrix.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


### core metrics
@torch.no_grad()
def topk_accuracy( logits: torch.Tensor, targets: torch.Tensor, topk: Sequence[int] = (1, 5), ) -> List[float]:
    """
    Compute top-k accuracies (in percent).
    If a requested k > num_classes, it is clipped to num_classes.
    """
    maxk = min(max(topk), logits.size(1))
    _, pred = logits.topk(k=maxk, dim=1, largest=True, sorted=True)  # [B, maxk]
    pred = pred.t()                                                  # [maxk, B]
    correct = pred.eq(targets.view(1, -1).expand_as(pred))           # [maxk, B]

    res = []
    batch_size = targets.size(0)
    for k in topk:
        k = min(k, logits.size(1))
        if k <= 0:
            res.append(0.0)
            continue
        correct_k = correct[:k].reshape(-1).float().sum(0).item()
        res.append(100.0 * correct_k / max(1, batch_size))
    return res


class ConfusionMatrix:
    def __init__(self, num_classes: int, device: torch.device | None = None):
        self.num_classes = int(num_classes)
        self.device = device if device is not None else torch.device("cpu")
        self.mat = torch.zeros((self.num_classes, self.num_classes), dtype=torch.long, device=self.device)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        preds: [B] predicted class indices
        targets: [B] true class indices
        """
        preds = preds.view(-1).to(self.device)
        targets = targets.view(-1).to(self.device)
        k = self.num_classes
        inds = targets * k + preds
        cm = torch.bincount(inds, minlength=k * k)
        cm = cm.view(k, k)
        self.mat += cm

    @torch.no_grad()
    def compute(self) -> torch.Tensor:
        return self.mat.clone()

    @torch.no_grad()
    def per_class_metrics(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (precision, recall, f1, support) as float tensors on CPU.
        Values are in percent for precision/recall/f1.
        """
        cm = self.mat.to(torch.float32)
        tp = torch.diag(cm)
        support = cm.sum(dim=1)  # true count per class

        # Precision: TP / (TP + FP)  => sum over column = predicted positives
        pred_pos = cm.sum(dim=0).clamp(min=1.0)
        precision = (tp / pred_pos) * 100.0

        # Recall: TP / (TP + FN) => sum over row = actual positives
        actual_pos = cm.sum(dim=1).clamp(min=1.0)
        recall = (tp / actual_pos) * 100.0

        # F1: harmonic mean (guard division by zero)
        denom = (precision + recall).clamp(min=1e-6)
        f1 = (2.0 * precision * recall) / denom

        return precision.cpu(), recall.cpu(), f1.cpu(), support.cpu()

    @torch.no_grad()
    def macro_f1(self) -> float:
        _, _, f1, _ = self.per_class_metrics()
        return float(f1.mean().item())


# -------------------- Epoch accumulator --------------------
class EpochMetrics:
    """
    Accumulates running metrics over an epoch for train/eval phases.
    Works with logits/targets batches and optional confusion matrix.
    """
    def __init__(self, num_classes: int, track_confusion: bool = True, device: torch.device | None = None):
        self.device = device if device is not None else torch.device("cpu")
        self.num_classes = int(num_classes)
        self.track_confusion = bool(track_confusion)

        self.loss_sum = 0.0
        self.n = 0
        self.correct_top1 = 0.0
        self.correct_top5 = 0.0
        self.cm = ConfusionMatrix(num_classes, device=self.device) if self.track_confusion else None

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss: torch.Tensor | float) -> None:
        bs = targets.size(0)
        self.n += bs
        self.loss_sum += float(loss) * bs

        kspec = (1, 5 if logits.size(1) >= 5 else 1)
        acc1, acc5 = topk_accuracy(logits, targets, topk=kspec)
        self.correct_top1 += (acc1 / 100.0) * bs
        self.correct_top5 += (acc5 / 100.0) * bs

        if self.cm is not None:
            preds = logits.argmax(dim=1)
            self.cm.update(preds, targets)

    @torch.no_grad()
    def compute(self) -> Dict:
        out = {
            "loss": self.loss_sum / max(1, self.n),
            "acc1": 100.0 * self.correct_top1 / max(1, self.n),
            "acc5": 100.0 * self.correct_top5 / max(1, self.n),
        }
        if self.cm is not None:
            cm = self.cm.compute()
            out["confusion_matrix"] = cm
            out["macro_f1"] = self.cm.macro_f1()
            # Per-class if needed:
            prec, rec, f1, support = self.cm.per_class_metrics()
            out["f1_per_class"] = f1.tolist()
            out["support"] = support.tolist()
        return out


### Plot and save confusion matrix
def plot_confusion_matrix(cm: torch.Tensor, class_names: List[str], normalize: bool = False):
    """
    Create and return a Matplotlib Figure for a confusion matrix (optionally normalized).
    No filesystem side-effects; the caller decides where/how to save/log.
    """
    cm_np = cm.detach().cpu().numpy().astype(np.float32)
    if normalize:
        row_sums = cm_np.sum(axis=1, keepdims=True).clip(min=1.0)
        cm_np = cm_np / row_sums

    fig = plt.figure(figsize=(max(6, 0.45 * len(class_names)), max(5, 0.45 * len(class_names))))
    ax = plt.gca()
    im = ax.imshow(cm_np, interpolation="nearest")
    ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks); ax.set_yticklabels(class_names)
    ax.set_ylabel("True label"); ax.set_xlabel("Predicted label")
    ax.set_ylim(len(class_names) - 0.5, -0.5)
    plt.tight_layout()
    return fig


def save_confusion_assets(cm: torch.Tensor, class_names: List[str], out_dir: Path) -> Dict[str, str]:
    """
    Save confusion matrix as PNGs (raw & normalized). Returns paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # raw
    fig_raw = plot_confusion_matrix(cm, class_names, normalize=False)
    raw_path = out_dir / "confusion_matrix.png"
    fig_raw.savefig(raw_path, dpi=200)
    plt.close(fig_raw)
    # normalized
    fig_norm = plot_confusion_matrix(cm, class_names, normalize=True)
    norm_path = out_dir / "confusion_matrix_norm.png"
    fig_norm.savefig(norm_path, dpi=200)
    plt.close(fig_norm)
    return {"png": str(raw_path), "png_norm": str(norm_path)}
