#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, sys, glob
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional MediaPipe (für Hand-Box)
try:
    import mediapipe as mp
except Exception:
    mp = None

# Optional: torchvision R(2+1)D
try:
    from torchvision.models.video import r2plus1d_18 as tv_r2plus1d_18
except Exception:
    tv_r2plus1d_18 = None

# Deine eigenen Modelle
BUILD_MODEL = None
try:
    from models import build_model as _bm
    BUILD_MODEL = _bm
except Exception:
    BUILD_MODEL = None

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# -------------------- Utils --------------------
def to_tensor_normalized(rgb_uint8: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(rgb_uint8).permute(2, 0, 1).float() / 255.0  # [C,H,W]
    mean = torch.tensor(IMAGENET_MEAN).view(-1,1,1)
    std  = torch.tensor(IMAGENET_STD).view(-1,1,1)
    return (x - mean) / std

def resize_center_crop(rgb: np.ndarray, size: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = min(h, w)
    y1 = max(0, (h - s) // 2); y2 = y1 + s
    x1 = max(0, (w - s) // 2); x2 = x1 + s
    sq = rgb[y1:y2, x1:x2]
    return cv2.resize(sq, (size, size), interpolation=cv2.INTER_LINEAR)

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

def any_conv3d(model: nn.Module) -> bool:
    return any(isinstance(m, nn.Conv3d) for m in model.modules())

def list_pt_files(runs_dir: Path):
    return sorted(glob.glob(str(runs_dir / "*.pt")))

def force_py_str_list(lst):
    return [str(x) for x in lst]

def stack_clip(frames):  # frames: list of [C,H,W] tensors
    x = torch.stack(frames, dim=0)       # [T,C,H,W]
    x = x.permute(1,0,2,3).unsqueeze(0)  # [1,C,T,H,W]
    return x

# -------------------- Checkpoint Loader --------------------
def _is_state_dict_like(obj) -> bool:
    return hasattr(obj, "keys") and all(isinstance(k, str) for k in obj.keys())

def _strip_module_prefix(state: dict) -> dict:
    if any(k.startswith("module.") for k in state.keys()):
        return {k.replace("module.", "", 1): v for k, v in state.items()}
    return state

def _find_classes_next_to(weights_path: Path):
    cand = [weights_path.with_name("classes.json"), weights_path.parent / "classes.json"]
    for c in cand:
        if c.exists():
            try:
                meta = json.loads(c.read_text(encoding="utf-8"))
                if isinstance(meta, dict) and "idx_to_class" in meta:
                    idx_to = meta["idx_to_class"]
                    keys = sorted(map(int, idx_to.keys()))
                    return [str(idx_to[str(i)] if str(i) in idx_to else idx_to[i]) for i in keys]
                if isinstance(meta, dict) and "classes" in meta and isinstance(meta["classes"], list):
                    return force_py_str_list(meta["classes"])
            except Exception:
                pass
    return None

def _detect_arch_from_state(state: dict, fname_hint: str):
    name = fname_hint.lower()
    kset = set(state.keys())

    # torchvision r2plus1d_18 typische Keys
    tv_hints = [
        "stem.3.weight",
        "stem.4.running_mean",
        "layer1.0.conv1.0.0.weight",
        "layer1.0.conv2.0.0.weight",
    ]
    if any(h in kset for h in tv_hints):
        return "tv_r2plus1d_18"

    # eigenes R2Plus1D (dein models.py): conv_s/conv_t vorhanden?
    if any(k.startswith("layer1.0.conv_s") for k in kset) or any(k.startswith("layer2.0.conv_s") for k in kset):
        return "r2plus1d"

    # C3D typische "features.0.weight" etc.
    if any(k.startswith("features.0") for k in kset) and any(k.startswith("classifier") for k in kset):
        return "c3d"

    # Fallback über Dateiname
    if "c3d" in name:
        return "c3d"
    if "r2" in name or "r(2+1)" in name or "r2plus1" in name:
        # könnte TV oder eigenes sein; wenn wir hier sind, nehmen eigenes
        return "r2plus1d"

    # letzer Notnagel
    return "r2plus1d"

def _infer_num_classes_from_state(state: dict):
    # gängige End-Layer
    for key in ["fc.weight", "classifier.weight"]:
        if key in state and isinstance(state[key], torch.Tensor) and state[key].ndim == 2:
            return int(state[key].shape[0])
    # sonst: nicht zuverlässig inferierbar
    return None

def load_any_checkpoint(weights_path: str, device: torch.device):
    wp = Path(weights_path)
    raw = torch.load(wp, map_location="cpu")

    # Entpacken
    state = None
    classes = None
    frames = 16
    img_size = 112
    raw_name = None
    width_mult = None

    if isinstance(raw, dict):
        raw_name = raw.get("model", None)
        if "classes" in raw and isinstance(raw["classes"], (list, tuple)):
            classes = force_py_str_list(raw["classes"])
        if "frames" in raw:
            try: frames = int(raw["frames"])
            except Exception: pass
        if "img_size" in raw:
            try: img_size = int(raw["img_size"])
            except Exception: pass
        if "width_mult" in raw:
            try: width_mult = float(raw["width_mult"])
            except Exception: pass

        # state dict suchen
        if "model_state" in raw and _is_state_dict_like(raw["model_state"]):
            state = raw["model_state"]
        elif "state_dict" in raw and _is_state_dict_like(raw["state_dict"]):
            state = raw["state_dict"]
        elif _is_state_dict_like(raw):
            state = raw
        else:
            raise RuntimeError("Unbekanntes Checkpoint-Format: kein state_dict gefunden.")
    elif _is_state_dict_like(raw):
        state = raw
    else:
        raise RuntimeError("Checkpoint ist weder Dict noch State-Dict.")

    state = _strip_module_prefix(state)
    if classes is None:
        classes = _find_classes_next_to(wp)
    if classes is None:
        # letzte Chance: Anzahl Klassen aus Gewichtsmatrix
        ncls = _infer_num_classes_from_state(state)
        if ncls is None:
            raise RuntimeError("Klassen nicht gefunden und nicht inferierbar.")
        classes = [str(i) for i in range(ncls)]

    # Architektur erkennen
    arch = _detect_arch_from_state(state, wp.name if raw_name is None else str(raw_name))
    classes = force_py_str_list(classes)

    # Modell bauen
    if arch == "tv_r2plus1d_18":
        if tv_r2plus1d_18 is None:
            raise RuntimeError("torchvision r2plus1d_18 nicht verfügbar.")
        model = tv_r2plus1d_18(weights=None)
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            model.fc = nn.Linear(model.fc.in_features, len(classes))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:   print(f"[TV] fehlende Keys: {list(missing)[:6]}{' …' if len(missing)>6 else ''}")
        if unexpected:print(f"[TV] unerwartete Keys: {list(unexpected)[:6]}{' …' if len(unexpected)>6 else ''}")
        meta = {"type": "tv_r2plus1d_18", "frames": frames, "img_size": img_size}
        return model.to(device).eval(), classes, meta

    if BUILD_MODEL is None:
        raise RuntimeError("Eigenes Modell erkannt, aber models.build_model ist nicht importierbar.")

    if arch == "r2plus1d":
        # width_mult aus stem ableiten, falls nötig
        if width_mult is None:
            w = state.get("stem.0.weight", None)
            if isinstance(w, torch.Tensor) and w.ndim == 5:
                out_ch = int(w.shape[0])
                width_mult = max(8, out_ch) / 64.0
            else:
                width_mult = 1.0
        model = BUILD_MODEL("r2plus1d", num_classes=len(classes), in_channels=3, dropout=0.5, width_mult=width_mult)
    elif arch == "c3d":
        model = BUILD_MODEL("c3d", num_classes=len(classes), in_channels=3, dropout=0.5)
        # für C3D statisch oft img_size=112
        img_size = max(1, img_size)
    else:
        # Fallback: R2+1D
        model = BUILD_MODEL("r2plus1d", num_classes=len(classes), in_channels=3, dropout=0.5)

    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        print(f"[Warnung] strict=True fehlgeschlagen ({e}). Versuche strict=False.")
        model.load_state_dict(state, strict=False)

    meta = {"type": arch, "frames": frames, "img_size": img_size}
    return model.to(device).eval(), classes, meta

# -------------------- Laufzeit: Static (bewegliche Box) --------------------
def run_static(model, classes, img_size, device, conf=0.35, mirror=True, use_mediapipe=True):
    # bewegliche, geglättete Box
    class SmoothBox:
        def __init__(self, alpha=0.25):
            self.alpha = float(alpha)
            self.prev = None
        def update(self, box, W, H):
            if self.prev is None:
                self.prev = box
            else:
                ax = self.alpha
                self.prev = tuple(int(ax*b + (1-ax)*p) for b, p in zip(box, self.prev))
            x1,y1,x2,y2 = self.prev
            x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W-1))
            y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H-1))
            if x2 <= x1: x2 = min(x1+1, W-1)
            if y2 <= y1: y2 = min(y1+1, H-1)
            self.prev = (x1,y1,x2,y2)
            return self.prev
        def reset(self): self.prev = None
        def value(self): return self.prev

    mp_hands = None
    if use_mediapipe and mp is not None:
        mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
                                            model_complexity=0, min_detection_confidence=0.5,
                                            min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam nicht gefunden."); return
    font = cv2.FONT_HERSHEY_SIMPLEX
    smoother = SmoothBox(alpha=0.25)

    while True:
        ok, frame_bgr = cap.read()
        if not ok: break
        if mirror: frame_bgr = cv2.flip(frame_bgr, 1)
        rgb_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb_full.shape[:2]

        # Hand finden
        hand_box = None
        if mp_hands is not None:
            res = mp_hands.process(rgb_full)
            if res.multi_hand_landmarks:
                lms = res.multi_hand_landmarks[0].landmark
                x1,y1,x2,y2 = bbox_from_landmarks(lms, W, H, pad=0.35)
                if (x2-x1) > 4 and (y2-y1) > 4:
                    hand_box = (x1,y1,x2,y2)

        if hand_box is not None:
            gx1,gy1,gx2,gy2 = smoother.update(hand_box, W, H)
        else:
            side = int(min(H, W) * 0.6)
            cx, cy = W//2, H//2
            gx1 = max(0, cx - side//2); gy1 = max(0, cy - side//2)
            gx2 = min(W-1, gx1 + side); gy2 = min(H-1, gy1 + side)
            smoother.reset()

        # Crop & infer
        crop_rgb = rgb_full[gy1:gy2, gx1:gx2].copy()
        if crop_rgb.size == 0: crop_rgb = rgb_full
        crop_rgb = resize_center_crop(crop_rgb, img_size)
        x = to_tensor_normalized(crop_rgb).unsqueeze(0).to(device)  # [1,C,H,W]
        if any_conv3d(model):
            x = x.unsqueeze(2)  # [1,C,1,H,W]

        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy().astype(float)

        top_idx = np.argsort(-probs)[:3]
        top = [(str(classes[i]), float(probs[i])) for i in top_idx]
        label_main = f"{top[0][0]} ({top[0][1]:.2f})" if top[0][1] >= float(conf) else "(unsicher)"
        top_lines = [f"{str(n)}: {float(p):.2f}" for n,p in top]

        # Overlay
        cv2.rectangle(frame_bgr, (gx1,gy1), (gx2,gy2), (0,255,0), 2)
        cv2.putText(frame_bgr, str(label_main), (20,60), font, 1.6, (0,255,0), 3, cv2.LINE_AA)
        for i, line in enumerate(top_lines, start=1):
            cv2.putText(frame_bgr, str(line), (20, 60 + i*28), font, 0.9, (0,255,0), 2, cv2.LINE_AA)
        if hand_box is None:
            cv2.putText(frame_bgr, "Hand ins Bild & ruhig halten", (20,120), font, 0.9, (0,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, "ESC/q: Beenden", (20, H-20), font, 0.8, (0,255,0), 2)

        cv2.imshow("ASL static (tracked box)", frame_bgr)
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord('q'): break

    cap.release(); cv2.destroyAllWindows()

# -------------------- Laufzeit: Dynamic (kein Rechteck) --------------------
def run_dynamic(model, classes, img_size, T, device, conf=0.35, mirror=True, use_mediapipe=True):
    mp_hands = None
    if use_mediapipe and mp is not None:
        mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
                                            model_complexity=0, min_detection_confidence=0.5,
                                            min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam nicht gefunden."); return
    font = cv2.FONT_HERSHEY_SIMPLEX
    buffer = []

    while True:
        ok, frame_bgr = cap.read()
        if not ok: break
        if mirror: frame_bgr = cv2.flip(frame_bgr, 1)
        rgb_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb_full.shape[:2]

        crop_rgb = None
        if mp_hands is not None:
            res = mp_hands.process(rgb_full)
            if res.multi_hand_landmarks:
                lms = res.multi_hand_landmarks[0].landmark
                x1,y1,x2,y2 = bbox_from_landmarks(lms, W, H, pad=0.35)
                tmp = rgb_full[y1:y2, x1:x2].copy()
                if tmp.size > 0:
                    crop_rgb = tmp
        if crop_rgb is None:
            crop_rgb = resize_center_crop(rgb_full, img_size)
        crop_rgb = resize_center_crop(crop_rgb, img_size)

        buffer.append(to_tensor_normalized(crop_rgb))
        if len(buffer) < int(T):
            cv2.putText(frame_bgr, f"Fülle Clip: {len(buffer)}/{int(T)}", (20,40), font, 0.9, (0,255,255), 2)
            cv2.putText(frame_bgr, "ESC/q: Beenden", (20, H-20), font, 0.8, (0,255,0), 2)
            cv2.imshow("ASL dynamic (T>1)", frame_bgr)
            if cv2.waitKey(1) & 0xFF in [27, ord('q')]: break
            continue

        clip = stack_clip(buffer[-int(T):]).to(device)  # [1,C,T,H,W]
        with torch.no_grad():
            logits = model(clip)
            probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy().astype(float)

        top_idx = np.argsort(-probs)[:3]
        top = [(str(classes[i]), float(probs[i])) for i in top_idx]
        label_main = f"{top[0][0]} ({top[0][1]:.2f})" if top[0][1] >= float(conf) else "(unsicher)"
        top_lines = [f"{str(n)}: {float(p):.2f}" for n,p in top]

        cv2.putText(frame_bgr, str(label_main), (20,60), font, 1.4, (0,255,0), 3)
        for i, line in enumerate(top_lines, start=1):
            cv2.putText(frame_bgr, str(line), (20, 60 + i*28), font, 0.9, (0,255,0), 2)
        cv2.putText(frame_bgr, "ESC/q: Beenden", (20, H-20), font, 0.8, (0,255,0), 2)

        cv2.imshow("ASL dynamic (T>1)", frame_bgr)
        if cv2.waitKey(1) & 0xFF in [27, ord('q')]: break

    cap.release(); cv2.destroyAllWindows()

# -------------------- 2D-Wrapper für dynamischen Modus --------------------
def run_dynamic_2d_wrapper(model2d, classes, img_size, T, device, conf=0.35, mirror=True, use_mediapipe=True):
    # wertet pro Frame 2D aus und mittelt die Logits über T
    mp_hands = None
    if use_mediapipe and mp is not None:
        mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
                                            model_complexity=0, min_detection_confidence=0.5,
                                            min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam nicht gefunden."); return
    font = cv2.FONT_HERSHEY_SIMPLEX
    buffer = []

    while True:
        ok, frame_bgr = cap.read()
        if not ok: break
        if mirror: frame_bgr = cv2.flip(frame_bgr, 1)
        rgb_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb_full.shape[:2]

        crop_rgb = None
        if mp_hands is not None:
            res = mp_hands.process(rgb_full)
            if res.multi_hand_landmarks:
                lms = res.multi_hand_landmarks[0].landmark
                x1,y1,x2,y2 = bbox_from_landmarks(lms, W, H, pad=0.35)
                tmp = rgb_full[y1:y2, x1:x2].copy()
                if tmp.size > 0:
                    crop_rgb = tmp
        if crop_rgb is None:
            crop_rgb = resize_center_crop(rgb_full, img_size)
        crop_rgb = resize_center_crop(crop_rgb, img_size)

        buffer.append(to_tensor_normalized(crop_rgb))
        if len(buffer) < int(T):
            cv2.putText(frame_bgr, f"Fülle Clip: {len(buffer)}/{int(T)}", (20,40), font, 0.9, (0,255,255), 2)
            cv2.putText(frame_bgr, "ESC/q: Beenden", (20, H-20), font, 0.8, (0,255,0), 2)
            cv2.imshow("ASL dynamic (2D-Wrapper)", frame_bgr)
            if cv2.waitKey(1) & 0xFF in [27, ord('q')]: break
            continue

        xs = torch.stack(buffer[-int(T):], dim=0)  # [T,C,H,W]
        with torch.no_grad():
            logits_seq = []
            for i in range(xs.size(0)):
                logits_seq.append(model2d(xs[i].unsqueeze(0).to(device)))
            logits = torch.stack(logits_seq, dim=0).mean(dim=0)  # [1,num_classes]
            probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy().astype(float)

        top_idx = np.argsort(-probs)[:3]
        top = [(str(classes[i]), float(probs[i])) for i in top_idx]
        label_main = f"{top[0][0]} ({top[0][1]:.2f})" if top[0][1] >= float(conf) else "(unsicher)"
        top_lines = [f"{str(n)}: {float(p):.2f}" for n,p in top]

        cv2.putText(frame_bgr, str(label_main), (20,60), font, 1.4, (0,255,0), 3)
        for i, line in enumerate(top_lines, start=1):
            cv2.putText(frame_bgr, str(line), (20, 60 + i*28), font, 0.9, (0,255,0), 2)
        cv2.putText(frame_bgr, "ESC/q: Beenden", (20, H-20), font, 0.8, (0,255,0), 2)

        cv2.imshow("ASL dynamic (2D-Wrapper)", frame_bgr)
        if cv2.waitKey(1) & 0xFF in [27, ord('q')]: break

    cap.release(); cv2.destroyAllWindows()

# -------------------- Menü / Main --------------------
def main():
    ap = argparse.ArgumentParser(description="ASL Demo (statisch + dynamisch) robust")
    ap.add_argument("--runs", type=str, required=True, help="Ordner mit .pt-Checkpoints")
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--mode", type=str, default=None, choices=[None, "static", "dynamic"], help="Optional Modus vorwählen")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--mirror", type=int, default=1)
    ap.add_argument("--mediapipe", type=int, default=1)
    ap.add_argument("--force-T", type=int, default=None, help="Clip-Länge überschreiben (nur 3D oder 2D-Wrapper)")
    args = ap.parse_args()

    # 🔧 FIX: device-Auswahl
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    runs_dir = Path(args.runs)
    files = list_pt_files(runs_dir)
    if not files:
        print(f"Keine .pt Dateien in {runs_dir}")
        sys.exit(1)

    print("Verfügbare Modelle:")
    for i,p in enumerate(files):
        print(f"[{i}] {p}")
    s = input("Index wählen (oder 'back'): ").strip().lower()
    if s == "back": return
    try:
        idx = int(s)
        ckpt_path = files[idx]
    except Exception:
        print("Ungültige Auswahl.")
        return

    print(f"Lade Modell {ckpt_path} ...")
    try:
        model, classes, meta = load_any_checkpoint(ckpt_path, device)
    except Exception as e:
        print(f"Fehler beim Laden: {e}")
        return

    # Normalisiere Klassen zu echten Strings
    classes = force_py_str_list(classes)

    arch = meta.get("type", "?")
    frames = int(meta.get("frames", 16))
    img_size = int(meta.get("img_size", 112))
    print(f"Geladen: {arch} | T={frames} | img_size={img_size} | Klassen={len(classes)}")

    # Modus bestimmen
    mode = args.mode
    if mode is None:
        print("\n=== Hauptmenü ===")
        print("[1] Statisch (Box folgt der Hand)")
        print("[2] Dynamisch (Clip T Frames)")
        print("[0] Exit")
        m = input("Wähle Modus: ").strip()
        mode = "static" if m=="1" else "dynamic" if m=="2" else "exit"
        if mode == "exit": return

    mirror = bool(args.mirror)
    use_mediapipe = bool(args.mediapipe)
    force_T = args.force_T

    # Lauf
    if mode == "static":
        # Für 3D-Modelle in „statisch“ empfehle ich T aus dem Checkpoint (bessere Qualität)
        if any_conv3d(model):
            T = frames if force_T is None else int(force_T)
            print(f"Hinweis: 3D-Modell im statischen Menü, wir verwenden T={T} intern per dynamischem Pfad.")
            run_dynamic(model, classes, img_size, T, device, conf=args.conf, mirror=mirror, use_mediapipe=use_mediapipe)
        else:
            run_static(model, classes, img_size if img_size>0 else 224, device, conf=args.conf, mirror=mirror, use_mediapipe=use_mediapipe)
    else:
        if any_conv3d(model):
            T = frames if force_T is None else int(force_T)
            run_dynamic(model, classes, img_size, T, device, conf=args.conf, mirror=mirror, use_mediapipe=use_mediapipe)
        else:
            T = frames if force_T is None else int(force_T)
            print(f"Info: 2D-Checkpoint im dynamischen Modus (Wrapper). T={T}")
            run_dynamic_2d_wrapper(model, classes, img_size if img_size>0 else 224, T, device, conf=args.conf, mirror=mirror, use_mediapipe=use_mediapipe)

if __name__ == "__main__":
    main()
