#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive ASL Inference
- Erstes Menü: [1] statisch  [2] dynamisch  [0] exit
- Danach: .pt-Modelle aus --runs wählen
- Lädt automatisch entweder:
    A) Static ResNet18-Checkpoint: {'state_dict','classes'}
    B) 3D-CNN-Checkpoint:         {'model_state','classes','model','frames','img_size'}
- Live-Stream mit Mediapipe-Bounding-Box (ohne gelbe Punkte)
- Beenden: ESC oder Taste 'q' → beendet das GESAMTE Programm
"""

import os, sys, glob
from pathlib import Path
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import mediapipe as mp

# --- optional: dein 3D-Builder (für C3D / R(2+1)D) ---
try:
    from models import build_model as build_model_3d   # aus deinem 3D-Training
except Exception:
    build_model_3d = None

# --- optional: dein Custom-ResNet18 ---
try:
    from models.static_cnn import build_resnet18 as build_resnet18_custom
except Exception:
    build_resnet18_custom = None

# --- torchvision Fallback für ResNet18 ---
from torchvision.models import resnet18, ResNet18_Weights


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ---------------- Utils ----------------
def pick_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if arg not in {"cpu","cuda"}:
        raise ValueError("Ungültiges --device (auto|cpu|cuda).")
    return torch.device(arg)

def to_tensor_normalized(rgb_uint8: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(rgb_uint8).permute(2,0,1).float()/255.0
    mean = torch.tensor(IMAGENET_MEAN).view(-1,1,1)
    std  = torch.tensor(IMAGENET_STD).view(-1,1,1)
    return (x - mean) / std

def resize_center_crop(img_rgb: np.ndarray, size: int) -> np.ndarray:
    h,w = img_rgb.shape[:2]
    s = min(h,w)
    cy,cx = h//2, w//2
    y1 = max(0, cy - s//2); y2 = y1 + s
    x1 = max(0, cx - s//2); x2 = x1 + s
    sq = img_rgb[y1:y2, x1:x2]
    if sq.shape[0]!=size or sq.shape[1]!=size:
        sq = cv2.resize(sq, (size,size), interpolation=cv2.INTER_LINEAR)
    return sq

def bbox_from_landmarks(lms, img_w, img_h, pad=0.35):
    xs = [int(p.x * img_w) for p in lms]
    ys = [int(p.y * img_h) for p in lms]
    x1, x2 = max(min(xs),0), min(max(xs), img_w-1)
    y1, y2 = max(min(ys),0), min(max(ys), img_h-1)
    w, h = x2-x1, y2-y1
    cx, cy = x1 + w//2, y1 + h//2
    side = int(max(w, h) * (1.0 + pad))
    x1n, y1n = max(cx - side//2, 0), max(cy - side//2, 0)
    x2n, y2n = min(x1n + side, img_w-1), min(y1n + side, img_h-1)
    return x1n, y1n, x2n, y2n

def list_models(base_dir, pattern):
    files = sorted(
        glob.glob(str(Path(base_dir) / "**" / pattern), recursive=True),
        key=lambda p: -Path(p).stat().st_mtime
    )
    return files

# ---------- Modell-Builds ----------
def build_resnet18_any(num_classes: int, freeze_backbone: bool = False):
    # Nutze dein Custom-ResNet, wenn vorhanden
    if build_resnet18_custom is not None:
        return build_resnet18_custom(num_classes=num_classes, freeze_backbone=freeze_backbone)
    # Fallback: torchvision
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for p in m.parameters(): p.requires_grad = False
        for p in m.fc.parameters(): p.requires_grad = True
    m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
    return m

def load_any_pt(weights_path: str, device: torch.device):
    """
    Erkenne .pt-Format automatisch:
      - Static ResNet: {'state_dict','classes'}
      - 3D-CNN:        {'model_state','classes', 'model','frames','img_size'}
    """
    ckpt = torch.load(weights_path, map_location="cpu")

    # A) Static ResNet
    if isinstance(ckpt, dict) and "state_dict" in ckpt and "classes" in ckpt:
        classes = ckpt["classes"]
        model = build_resnet18_any(num_classes=len(classes), freeze_backbone=False)
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        if unexpected: print(f"[Warnung] Unerwartete Keys: {unexpected}")
        if missing:   print(f"[Warnung] Fehlende Keys: {missing}")
        meta = {"type": "resnet_static", "frames": 1, "img_size": 224}

    # B) 3D-CNN
    elif isinstance(ckpt, dict) and "model_state" in ckpt and "classes" in ckpt:
        if build_model_3d is None:
            raise RuntimeError("3D-Modelle gefunden, aber models.build_model ist nicht importierbar.")
        classes = ckpt["classes"]
        model_name = ckpt.get("model", "r2plus1d")
        frames = int(ckpt.get("frames", 16))
        img_size = int(ckpt.get("img_size", 112))
        model = build_model_3d(model_name, num_classes=len(classes), in_channels=3, dropout=0.5)
        # DataParallel-Präfix entfernen, falls nötig
        state = ckpt["model_state"]
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k,v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected: print(f"[Warnung] Unerwartete Keys: {unexpected}")
        if missing:   print(f"[Warnung] Fehlende Keys: {missing}")
        meta = {"type": "cnn3d", "frames": frames, "img_size": img_size}

    else:
        raise RuntimeError("Unbekanntes .pt-Format. Erwarte {state_dict,classes} oder {model_state,classes}.")

    model = model.to(device).eval()
    return model, classes, meta


import torch.nn as nn

def is_3d_model(model: nn.Module) -> bool:
    # True, wenn irgendein Conv3d im Modell steckt
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            return True
    return False

def stack_T(frames_tensors):
    """
    frames_tensors: Liste von T Tensors je [C,H,W]
    Rückgabe: [1, C, T, H, W]
    """
    x = torch.stack(frames_tensors, dim=0)  # [T, C, H, W]
    x = x.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, T, H, W]
    return x


# ---------------- Live-Loops ----------------
def webcam_loop(model, classes, mode_type, device, use_mediapipe, img_size, T, mirror, conf) -> bool:
    """
    mode_type: 'static' oder 'dynamic'
    Gibt True zurück, wenn der Nutzer 'q' gedrückt hat (Programm hart beendet werden soll).
    """
    mp_h = None
    if use_mediapipe:
        mp_h = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
                                        model_complexity=0, min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam nicht gefunden.")
        return False

    font = cv2.FONT_HERSHEY_SIMPLEX
    buffer = []  # nur für 3D/dynamic
    model_is_3d = is_3d_model(model)

    # Wenn Modus "static": T=1 verwenden, aber bei 3D-Modellen korrekt als Zeitdimension einbauen
    effective_T = 1 if mode_type == "static" else max(1, int(T))

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if mirror:
            frame_bgr = cv2.flip(frame_bgr, 1)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Optionaler Hand-Crop via Mediapipe, sonst center-crop
        crop_rgb = None
        box = None
        if mp_h is not None:
            res = mp_h.process(rgb)
            if res.multi_hand_landmarks:
                h,w = frame_bgr.shape[:2]
                lms = res.multi_hand_landmarks[0].landmark
                x1,y1,x2,y2 = bbox_from_landmarks(lms, w, h, pad=0.35)
                crop_rgb = rgb[y1:y2, x1:x2].copy()
                box = (x1,y1,x2,y2)
        if crop_rgb is None:
            crop_rgb = resize_center_crop(rgb, img_size)

        crop_rgb = resize_center_crop(crop_rgb, img_size)

        # ==== Eingabe bauen, je nach Modelltyp und Modus ====
        if not model_is_3d:
            # 2D CNN (ResNet): immer [1, C, H, W]
            x = to_tensor_normalized(crop_rgb).unsqueeze(0).to(device)  # [1,C,H,W]

        else:
            # 3D CNN: [1, C, T, H, W]
            if mode_type == "static":
                # Nur 1 Frame, aber korrekt als Zeitdimension T=1 aufbauen
                ft = to_tensor_normalized(crop_rgb)            # [C,H,W]
                x = stack_T([ft]).to(device)                   # [1,C,1,H,W]
            else:
                # dynamisch: Sliding Window sammeln
                buffer.append(crop_rgb)
                if len(buffer) > effective_T:
                    buffer = buffer[-effective_T:]
                if len(buffer) < effective_T:
                    # Noch nicht genug Frames → HUD zeigen und Frame skippen
                    label = f"(sammle {len(buffer)}/{effective_T})"
                    cv2.putText(frame_bgr, label, (20,60), font, 2, (0,255,0), 3, cv2.LINE_AA)
                    if box is not None:
                        x1,y1,x2,y2 = box
                        cv2.rectangle(frame_bgr, (x1,y1),(x2,y2), (0,255,0), 2)
                    cv2.imshow(f"ASL Inference ({mode_type})", frame_bgr)
                    k = cv2.waitKey(1) & 0xFF
                    if k == 27 or k == ord('q'):
                        quit_all = (k == ord('q'))
                        cap.release(); cv2.destroyAllWindows()
                        return quit_all
                    continue
                # genug Frames: T Stack
                frames_t = [to_tensor_normalized(f) for f in buffer[-effective_T:]]  # je [C,H,W]
                x = stack_T(frames_t).to(device)                                     # [1,C,T,H,W]

        # ==== Inferenz ====
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
        top = int(np.argmax(probs)); ptop = float(probs[top])
        label = f"{classes[top]} ({ptop:.2f})" if ptop >= conf else "(unsicher)"

        # ==== HUD ====
        cv2.putText(frame_bgr, label, (20,60), font, 2, (0,255,0), 3, cv2.LINE_AA)
        if box is not None:
            x1,y1,x2,y2 = box
            cv2.rectangle(frame_bgr, (x1,y1),(x2,y2), (0,255,0), 2)
        mode_info = f"{mode_type} | {'3D' if model_is_3d else '2D'} | T={effective_T} | img={img_size}"
        cv2.putText(frame_bgr, mode_info, (20, frame_bgr.shape[0]-20), font, 0.7, (200,200,200), 1, cv2.LINE_AA)

        cv2.imshow(f"ASL Inference ({mode_type})", frame_bgr)
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord('q'):   # ESC oder 'q'
            quit_all = (k == ord('q'))
            cap.release(); cv2.destroyAllWindows()
            return quit_all

    cap.release(); cv2.destroyAllWindows()
    return False


# ---------------- Menü & Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs",   type=str, default=str(Path(__file__).resolve().parents[1] / "runs"),
                    help="Wurzelordner zum Durchsuchen (.pt)")
    ap.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    ap.add_argument("--mirror", type=int, default=1, help="1=Selfie-Spiegelung, 0=aus")
    ap.add_argument("--conf",   type=float, default=0.35, help="Schwellwert für Anzeige")
    ap.add_argument("--no-mp",  action="store_true", help="ohne Mediapipe (nur Center-Crop)")
    args = ap.parse_args()

    device = pick_device(args.device)
    mirror = bool(args.mirror)
    use_mediapipe = not args.no_mp

    while True:
        print("\n=== Hauptmenü ===")
        print("[1] Statisch (T=1)")
        print("[2] Dynamisch (T>1, falls Modell 3D)")
        print("[0] Exit")
        choice = input("Wähle Modus: ").strip().lower()
        if choice in {"0","exit","q"}:
            break
        if choice not in {"1","2"}:
            print("Ungültige Auswahl.")
            continue
        mode_type = "static" if choice == "1" else "dynamic"

        # Modelle auflisten
        models = list_models(args.runs, "*.pt")
        if not models:
            print(f"Keine .pt-Modelle in {args.runs} gefunden.")
            continue
        print("\nVerfügbare Modelle:")
        for i,pth in enumerate(models):
            print(f"[{i}] {pth}")
        sel = input("Index wählen (oder 'back'): ").strip().lower()
        if sel in {"back","b"}:
            continue
        try:
            idx = int(sel)
            weights_path = models[idx]
        except Exception:
            print("Ungültige Auswahl.")
            continue

        # Modell laden (ResNet18 static ODER 3D-CNN)
        try:
            model, classes, meta = load_any_pt(weights_path, device)
        except Exception as e:
            print(f"Fehler beim Laden: {e}")
            continue

        # Falls dynamisch gewählt, aber Modell ist statisch: T=1 bleibt, läuft trotzdem
        T = int(meta.get("frames", 1))
        img_size = int(meta.get("img_size", 224 if meta["type"]=="resnet_static" else 112))

        # Live-Loop
        quit_all = webcam_loop(model, classes, mode_type, device, use_mediapipe, img_size, T, mirror, args.conf)
        if quit_all:
            # Taste 'q' wurde gedrückt: komplette App beenden
            break

if __name__ == "__main__":
    main()
