"""
FastAPI service for the static ASL fingerspelling model (C3D).

Exposes a JSON prediction API over the same C3D checkpoint used by the demos.
The model is reused from `src/models.py`; weights are loaded from a local file
if present, otherwise downloaded once from the project's GitHub Release.

Run locally (from the repo root):
    uvicorn api.serve:app --reload
"""
import io
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image

from src.models import build_model

MODEL_URL = os.environ.get(
    "ASL_MODEL_URL",
    "https://github.com/jejrngyien/asl-demo/releases/download/v1.0/model.pt",
)
MODEL_PATH = os.environ.get("ASL_MODEL_PATH", "model.pt")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    import mediapipe as mp
    _MP_OK = hasattr(mp, "solutions")
except Exception:
    mp = None
    _MP_OK = False


# -------------------- Preprocessing --------------------
def _resize_center_crop(rgb: np.ndarray, size: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = min(h, w)
    y1, x1 = (h - s) // 2, (w - s) // 2
    return cv2.resize(rgb[y1:y1 + s, x1:x1 + s], (size, size), interpolation=cv2.INTER_LINEAR)


def _bbox_from_landmarks(lms, W, H, pad=0.35):
    xs = [int(p.x * W) for p in lms]
    ys = [int(p.y * H) for p in lms]
    x1, x2 = max(min(xs), 0), min(max(xs), W - 1)
    y1, y2 = max(min(ys), 0), min(max(ys), H - 1)
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w // 2, y1 + h // 2
    side = int(max(w, h) * (1.0 + pad))
    x1n, y1n = max(cx - side // 2, 0), max(cy - side // 2, 0)
    return x1n, y1n, min(x1n + side, W - 1), min(y1n + side, H - 1)


def _to_tensor(rgb: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(-1, 1, 1)
    return (x - mean) / std


# -------------------- Model --------------------
def _load_model():
    if not os.path.exists(MODEL_PATH):
        torch.hub.download_url_to_file(MODEL_URL, MODEL_PATH, progress=False)
    try:
        ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError):
        ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    classes = [str(c) for c in ckpt["classes"]]
    state = ckpt.get("model_state", ckpt)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model = build_model(ckpt.get("model", "c3d"), num_classes=len(classes))
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    return model, classes, int(ckpt.get("img_size", 112))


app = FastAPI(
    title="ASL Fingerspelling API",
    description="Classify an American Sign Language fingerspelling handshape from an image.",
    version="1.0.0",
)
MODEL, CLASSES, IMG_SIZE = _load_model()
_HANDS = (mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1,
                                   model_complexity=1, min_detection_confidence=0.5)
          if _MP_OK else None)


def _crop_hand(rgb: np.ndarray):
    H, W = rgb.shape[:2]
    if _HANDS is not None:
        res = _HANDS.process(rgb)
        if res.multi_hand_landmarks:
            x1, y1, x2, y2 = _bbox_from_landmarks(res.multi_hand_landmarks[0].landmark, W, H)
            if (x2 - x1) > 4 and (y2 - y1) > 4:
                return rgb[y1:y2, x1:x2].copy(), True
    side = int(min(H, W) * 0.7)
    cx, cy = W // 2, H // 2
    return rgb[max(0, cy - side // 2):cy + side // 2, max(0, cx - side // 2):cx + side // 2].copy(), False


@app.get("/health")
def health():
    return {"status": "ok", "classes": len(CLASSES), "device": str(DEVICE)}


@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 3):
    try:
        rgb = np.array(Image.open(io.BytesIO(await file.read())).convert("RGB"))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded image.")

    crop, hand_detected = _crop_hand(rgb)
    if crop.size == 0:
        crop = rgb
    crop = _resize_center_crop(crop, IMG_SIZE)
    x = _to_tensor(crop).unsqueeze(0).unsqueeze(2).to(DEVICE)  # [1, C, 1, H, W]

    with torch.no_grad():
        probs = F.softmax(MODEL(x), dim=1)[0].cpu().numpy()

    k = max(1, min(int(top_k), len(CLASSES)))
    order = probs.argsort()[::-1][:k]
    return {
        "hand_detected": bool(hand_detected),
        "predictions": [
            {"label": CLASSES[i], "probability": round(float(probs[i]), 4)} for i in order
        ],
    }
