# -*- coding: utf-8 -*-
"""
make_plots.py
Erzeugt Poster-taugliche Plots:
- Training/Validation Loss & Accuracy aus history.json
- Farbige Confusion-Matrix 

Hinweis:
- Für die Confusion-Matrix kannst du Klassenlabels entweder direkt übergeben
  ODER automatisch aus einem Checkpoint (.pt) laden, der ein 'classes'-Feld enthält.
"""

import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Literal

# -----------------------------
# Kurvenplots aus history.json
# -----------------------------
def plot_training_curves(history_path: str, title_prefix: str, outdir: str = "."):
    """Zeichnet Loss- und Accuracy1-Kurven aus einem history.json."""
    history_path = Path(history_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    # Tolerant lesen, falls Keys mal fehlen
    epochs    = [h.get("epoch", i+1) for i, h in enumerate(history)]
    train_loss = [h.get("train_loss", np.nan) for h in history]
    val_loss   = [h.get("val_loss",   np.nan) for h in history]
    train_acc1 = [h.get("train_acc1", np.nan) for h in history]
    val_acc1   = [h.get("val_acc1",   np.nan) for h in history]

    # Loss
    plt.figure(figsize=(7,4.5))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss,   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} – Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"{title_prefix.lower().replace(' ','_')}_loss.png", dpi=300)
    plt.close()

    # Accuracy@1
    plt.figure(figsize=(7,4.5))
    plt.plot(epochs, train_acc1, label="Train Acc1")
    plt.plot(epochs, val_acc1,   label="Val Acc1")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy1 (%)")
    plt.title(f"{title_prefix} – Accuracy1")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"{title_prefix.lower().replace(' ','_')}_acc1.png", dpi=300)
    plt.close()


# ---------------------------------------------------
# Confusion-Matrix (ohne seaborn, rein matplotlib)
# ---------------------------------------------------
def _normalize_cm(cm: np.ndarray, how: Optional[Literal["row","all"]]) -> np.ndarray:
    if how is None:
        return cm
    cm = cm.astype(np.float64)
    if how == "row":
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return cm / row_sums
    if how == "all":
        s = cm.sum()
        return cm / (s if s != 0 else 1.0)
    return cm

def _format_annot(val: float, normalized: bool) -> str:
    return f"{val:.2f}" if normalized else f"{int(val)}"

def load_classes_from_ckpt(ckpt_path: str) -> Optional[List[str]]:
    """Versucht Klassenlabels aus einem .pt Checkpoint zu lesen."""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "classes" in ckpt:
            return list(ckpt["classes"])
    except Exception:
        pass
    return None

def plot_confusion_matrix(
    confmat_path: str,
    classes: Optional[List[str]] = None,
    *,
    title: str = "Confusion Matrix",
    out_path: Optional[str] = None,
    cmap: str = "plasma",
    normalize: Optional[Literal["row","all"]] = None,
    ckpt_for_classes: Optional[str] = None,
    annotate: bool = True,
):
    """
    confmat_path: Pfad zu gespeicherter Torch-Tensor-Matrix [num_classes, num_classes]
    classes:     Klassenlabels in passender Reihenfolge
    normalize:   None = Rohwerte, 'row' = zeilenweise (per Klasse), 'all' = global
    ckpt_for_classes: optionaler .pt Checkpoint, um classes automatisch zu laden
    """
    cm_t = torch.load(confmat_path, map_location="cpu")
    cm = cm_t.numpy() if hasattr(cm_t, "numpy") else np.array(cm_t)

    # Klassenlabels besorgen
    if classes is None and ckpt_for_classes is not None:
        classes = load_classes_from_ckpt(ckpt_for_classes)
    if classes is None:
        classes = [str(i) for i in range(cm.shape[0])]

    if len(classes) != cm.shape[0]:
        raise ValueError(f"Anzahl classes ({len(classes)}) passt nicht zur Matrix ({cm.shape[0]}).")

    normed_cm = _normalize_cm(cm, normalize)
    normalized = normalize is not None

    # Plot
    fig_w = max(6, min(12, 0.28*len(classes)+4))  # dynamische Größe
    fig_h = fig_w
    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(normed_cm, interpolation="nearest", cmap=cmap,
                    vmin=0.0, vmax=(1.0 if normalized else normed_cm.max() or 1.0))
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(range(len(classes)), classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # Beschriftungen
    if annotate:
        thresh = normed_cm.max() / 2.0 if normed_cm.size > 0 else 0.5
        for i in range(normed_cm.shape[0]):
            for j in range(normed_cm.shape[1]):
                txt = _format_annot(normed_cm[i, j], normalized)
                plt.text(
                    j, i, txt,
                    ha="center", va="center",
                    color="white" if normed_cm[i, j] > thresh else "black",
                    fontsize=8
                )

    plt.tight_layout()
    if out_path is None:
        out_path = Path(confmat_path).with_suffix("")
        tag = "_norm-row" if normalize == "row" else ("_norm-all" if normalize == "all" else "_raw")
        out_path = f"{out_path}{tag}_heatmap.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[saved] {out_path}")


# -----------------------------
# Beispiel-Nutzung
# -----------------------------
if __name__ == "__main__":

    # history.json (aus deinem training.py) – je Modell
    c3d_history      = r"D:\Masterstudium\Semester1\MS\WLASL\asl\runs\c3d\history.json"
    r2plus1d_history = r"D:\Masterstudium\Semester1\MS\WLASL\asl\runs\r2plus1d\history.json"  # Pfad prüfen!

    # Confusion-Matrix-Tensor pro Modell (z. B. von der besten Epoche)
    c3d_confmat_pt      = r"D:\Masterstudium\Semester1\MS\WLASL\asl\runs\c3d\confmat_epoch_011.pt"
    r2plus1d_confmat_pt = r"D:\Masterstudium\Semester1\MS\WLASL\asl\runs\r2plus1d\confmat_epoch_033.pt"

    #Checkpoints, um Klassenlabels automatisch zu ziehen
    c3d_ckpt_for_labels      = r"D:\Masterstudium\Semester1\MS\WLASL\asl\runs\c3d\epoch_011.pt"
    r2plus1d_ckpt_for_labels = r"D:\Masterstudium\Semester1\MS\WLASL\asl\runs\r2plus1d\epoch_033.pt"


    outdir = "metrics_plots"
    Path(outdir).mkdir(exist_ok=True)

    # --- Kurven ---
    try:
        plot_training_curves(c3d_history,      title_prefix="C3D",      outdir=outdir)
    except Exception as e:
        print(f"[warn] C3D curves: {e}")
    try:
        plot_training_curves(r2plus1d_history, title_prefix="R(2+1)D", outdir=outdir)
    except Exception as e:
        print(f"[warn] R(2+1)D curves: {e}")

    # --- Confusion-Matrices ---
    try:
        plot_confusion_matrix(
            c3d_confmat_pt,
            classes=None,                      # None → aus Checkpoint laden
            title="C3D Confusion Matrix",
            out_path=str(Path(outdir) / "c3d_confmat_heatmap.png"),
            cmap="plasma",
            normalize="row",                   # 'row' für pro-Klasse-Normalisierung
            ckpt_for_classes=c3d_ckpt_for_labels
        )
    except Exception as e:
        print(f"[warn] C3D confmat: {e}")

    try:
        plot_confusion_matrix(
            r2plus1d_confmat_pt,
            classes=None,
            title="R(2+1)D Confusion Matrix",
            out_path=str(Path(outdir) / "r2p1d_confmat_heatmap.png"),
            cmap="viridis",
            normalize="row",
            ckpt_for_classes=r2plus1d_ckpt_for_labels
        )
    except Exception as e:
        print(f"[warn] R(2+1)D confmat: {e}")
