#!/usr/bin/env python3
"""
dataset_root/
    class_a/*.jpg|.png|.jpeg
    class_b/*.jpg|.png|.jpeg
      ...

1) Preprocessing with MediaPipe Hands: robust crop around one/both hands,
   margin, resize to target size. Safe fallback if no hands are detected.
2) Augmentation: color jitter, (optional) horizontal flip,
   random resized crop, small rotations in {0°, ±5°, ±10°}, and gaussian blur.
3) On‑disk outputs (next to the input root by default):
    dataset_processed/
        preprocessed/<class>/*.png     # 1:1 processed originals
        augmented/<class>/*_aug####.png# generated augmentations
        metadata.json                  # per-sample metadata
        splits.json                    # train/test split; each original's
                                    # augmentations stay with the same split

4) get_data(...): loads the processed data back into memory as **tensors**
   suitable for 3D CNNs: returns two dicts `(train, test)` where each contains:
       {
         'clips':  FloatTensor [N, 3, T, H, W]
         'labels': LongTensor  [N]
         'class_names': List[str]
       }
   - Basic support for videos is included (expects already‑cropped clips in the
     processed folder). Images remain the expected primary use case here.
    - as_tensors=True will load all images into RAM
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps
import mediapipe as mp
from torchvision import transforms as T


### Utility
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUPPORTED_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUPPORTED_VID_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
PIPELINE_VERSION = "pre_v1.0_aug_v1.0"



def _ensure_rgb(im: Image.Image) -> Image.Image:
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im


def _safe_float(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


@dataclass
class PreprocessConfig:
    target_size: int = 112
    margin: float = 0.25  # 25% expansion around the union hand bbox
    include_face: bool = False
    min_det_conf: float = 0.5


@dataclass
class AugmentConfig:
    augs_per_image: int = 4
    allow_horizontal_flip: bool = False  # better off?
    flip_p: float = 0.5
    jitter_p: float = 1.0
    jitter_bcs: Tuple[float, float, float] = (0.2, 0.2, 0.2)  # brightness, contrast, saturation
    rrc_p: float = 0.5
    rrc_scale: Tuple[float, float] = (0.8, 1.0)
    rrc_ratio: Tuple[float, float] = (0.9, 1.1)
    rotate_p: float = 0.3
    rotations: Sequence[int] = (0, 5, -5, 10, -10)
    blur_p: float = 0.15
    blur_radius: float = 1.0


@dataclass
class SplitConfig:
    train_ratio: float = 0.9
    seed: int = 42


### MediaPipe Hand Detector
class HandDetector:
    """
    Thin wrapper around MediaPipe Hands for static images.
    """

    def __init__(self, min_det_conf: float = 0.5):
        if mp is None:
            raise ImportError(
                "mediapipe is not installed. Please `pip install mediapipe` first."
            )
        self.min_det_conf = min_det_conf
        self._hands = None

    def __enter__(self):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=self.min_det_conf,
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._hands is not None:
            self._hands.close()
        self._hands = None

    def detect_bbox(self, image_rgb_np: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Return union hand bounding box as (x1, y1, x2, y2) in *pixel coords*.
        None if nothing detected.
        """
        h, w = image_rgb_np.shape[:2]
        results = self._hands.process(image_rgb_np)
        if not results.multi_hand_landmarks:
            return None

        xs, ys = [], []
        for hand_lms in results.multi_hand_landmarks:
            for lm in hand_lms.landmark:
                xs.append(_safe_float(lm.x * w, 0, w - 1))
                ys.append(_safe_float(lm.y * h, 0, h - 1))
        if not xs:
            return None
        x1, x2 = int(min(xs)), int(math.ceil(max(xs)))
        y1, y2 = int(min(ys)), int(math.ceil(max(ys)))
        return x1, y1, x2, y2


#### Core Preprocessing and Augmentation
def preprocess_image(im: Image.Image, det: HandDetector, cfg: PreprocessConfig) -> Tuple[Image.Image, Dict]:
    """
    Crop around detected hand(s) with margin, resize to cfg.target_size.
    """
    im = _ensure_rgb(im)
    W, H = im.size
    np_rgb = np.array(im)

    bbox = det.detect_bbox(np_rgb)
    used_fallback = False

    if bbox is None:
        # Fallback: center square crop of the shortest side
        used_fallback = True
        side = min(W, H)
        cx, cy = W // 2, H // 2
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(W, x1 + side)
        y2 = min(H, y1 + side)
    else:
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1
        # Expand with margin
        mx = int(bw * cfg.margin)
        my = int(bh * cfg.margin)
        x1 = max(0, x1 - mx)
        y1 = max(0, y1 - my)
        x2 = min(W, x2 + mx)
        y2 = min(H, y2 + my)

        # Ensure square-ish by padding the shorter side
        bw, bh = (x2 - x1), (y2 - y1)
        if bw > bh:
            diff = bw - bh
            y1 = max(0, y1 - diff // 2)
            y2 = min(H, y2 + math.ceil(diff / 2))
        elif bh > bw:
            diff = bh - bw
            x1 = max(0, x1 - diff // 2)
            x2 = min(W, x2 + math.ceil(diff / 2))


    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        # Extreme edge case fallback
        used_fallback = True
        side = min(W, H)
        im_crop = im.crop(((W - side) // 2, (H - side) // 2, (W + side) // 2, (H + side) // 2))
    else:
        im_crop = im.crop((x1, y1, x2, y2))

    # Resize to target
    resize = T.Compose([
        T.Resize(cfg.target_size),
        T.CenterCrop(cfg.target_size),
    ])
    out = resize(im_crop)

    meta = {
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "fallback": bool(used_fallback),
    }
    return out, meta


def augment_once(im: Image.Image, cfg: AugmentConfig, target_size: int) -> Image.Image:
    """
    Apply one stochastic augmentation sample and return an image of target_size.
    """
    im = _ensure_rgb(im.copy())

    # 1) Optional horizontal flip
    if cfg.allow_horizontal_flip and random.random() < cfg.flip_p:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)

    # 2) Color jitter
    if random.random() < cfg.jitter_p:
        b, c, s = cfg.jitter_bcs
        jitter = T.ColorJitter(brightness=b, contrast=c, saturation=s)
        im = jitter(im)

    # 3) Random resized crop
    if random.random() < cfg.rrc_p:
        rrc = T.RandomResizedCrop(size=target_size, scale=cfg.rrc_scale, ratio=cfg.rrc_ratio, antialias=True)
        im = rrc(im)
    else:
        im = im.resize((target_size, target_size), Image.BILINEAR)

    # 4) Small rotation from discrete set
    if random.random() < cfg.rotate_p:
        angle = random.choice(cfg.rotations)
        if angle != 0:
            im = im.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))

    # 5) Blur
    if random.random() < cfg.blur_p:
        im = im.filter(ImageFilter.GaussianBlur(radius=cfg.blur_radius))

    # Ensure final size exactly target_size
    if im.size != (target_size, target_size):
        im = im.resize((target_size, target_size), Image.BILINEAR)
    return im


#### Disk Pipeline
def run_offline_pipeline(src_root: str | Path, dst_root: Optional[str | Path] = None, preprocess_cfg: Optional[PreprocessConfig] = None,
    augment_cfg: Optional[AugmentConfig] = None, split_cfg: Optional[SplitConfig] = None, img_exts: Optional[Sequence[str]] = None,) -> Dict:
    """
    Process every image in `src_root` and write outputs under `dst_root`.
    """
    src_root = Path(src_root)
    if dst_root is None:
        dst_root = src_root.with_name(src_root.name + "_processed")
    dst_root = Path(dst_root)

    preprocess_cfg = preprocess_cfg or PreprocessConfig()
    augment_cfg = augment_cfg or AugmentConfig()
    split_cfg = split_cfg or SplitConfig()
    img_exts = set(e.lower() for e in (img_exts or SUPPORTED_IMG_EXTS))

    # Prepare output dirs
    pre_dir = dst_root / "preprocessed"
    aug_dir = dst_root / "augmented"
    pre_dir.mkdir(parents=True, exist_ok=True)
    aug_dir.mkdir(parents=True, exist_ok=True)

    # Discover classes
    classes = [d.name for d in sorted(src_root.iterdir()) if d.is_dir()]
    if not classes:
        raise RuntimeError(f"No class subfolders found under: {src_root}")
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Collect samples (class and path)
    originals: List[Tuple[Path, str]] = []
    for c in classes:
        for p in sorted((src_root / c).rglob("*")):
            if p.is_file() and p.suffix.lower() in img_exts:
                originals.append((p, c))
    if not originals:
        raise RuntimeError("No input images found.")

    # split at original level
    rng = random.Random(split_cfg.seed)
    per_class: Dict[str, List[int]] = {c: [] for c in classes}
    for i, (_, c) in enumerate(originals):
        per_class[c].append(i)
    train_idx_set = set()
    test_idx_set = set()
    for c, idxs in per_class.items():
        rng.shuffle(idxs)
        n_train = int(len(idxs) * split_cfg.train_ratio)
        train_idx_set.update(idxs[:n_train])
        test_idx_set.update(idxs[n_train:])

    # MediaPipe detector
    with HandDetector(min_det_conf=preprocess_cfg.min_det_conf) as det:
        metadata = []
        n_processed = 0
        n_augmented = 0

        for k, (in_path, cls) in enumerate(originals):
            # Load
            try:
                im = Image.open(in_path)
            except Exception:
                # Skip unreadable files
                print(f"[INFO] can't read \"{in_path}\"")
                continue

            # Preprocess
            try:
                im_proc, meta = preprocess_image(im, det, preprocess_cfg)
            except Exception:
                # Skip if anything goes wrong irrecoverably
                continue

            # Save preprocessed
            rel_name = in_path.stem + ".png"
            out_pre_dir = pre_dir / cls
            out_pre_dir.mkdir(parents=True, exist_ok=True)
            out_pre_path = out_pre_dir / rel_name
            im_proc.save(out_pre_path, format="PNG", optimize=True)

            # Determine split for this original
            split = "train" if k in train_idx_set else "test"

            # Augmentations (train only)
            aug_paths: List[str] = []
            if split == "train" and augment_cfg.augs_per_image > 0:
                for j in range(augment_cfg.augs_per_image):
                    try:
                        im_aug = augment_once(im_proc, augment_cfg, preprocess_cfg.target_size)
                        out_aug_dir = aug_dir / cls
                        out_aug_dir.mkdir(parents=True, exist_ok=True)
                        out_aug_path = out_aug_dir / f"{in_path.stem}_aug{j:04d}.png"
                        im_aug.save(out_aug_path, format="PNG", optimize=True)
                        aug_paths.append(str(out_aug_path.relative_to(dst_root)))
                        n_augmented += 1
                    except Exception:
                        continue

            n_processed += 1

            # Bookkeeping
            rec = {
                "source": str(in_path.relative_to(src_root)),
                "class": cls,
                "class_idx": class_to_idx[cls],
                "preprocessed": str(out_pre_path.relative_to(dst_root)),
                "augmented": aug_paths,
                "split": split,
                "meta": meta,
                "version": PIPELINE_VERSION,
            }
            metadata.append(rec)

    # Write metadata & splits
    dst_root.mkdir(parents=True, exist_ok=True)
    (dst_root / "classes.json").write_text(json.dumps({"classes": classes, "class_to_idx": class_to_idx}, indent=2))

    (dst_root / "metadata.json").write_text(json.dumps({
        "root": str(src_root),
        "processed_root": str(dst_root),
        "version": PIPELINE_VERSION,
        "preprocess_cfg": vars(preprocess_cfg),
        "augment_cfg": vars(augment_cfg),
        "split_cfg": vars(split_cfg),
        "records": metadata,
    }, indent=2))

    # Also emit a flattened split list for quick loading
    splits = {"train": [], "test": []}
    for rec in metadata:
        # originals always included
        splits[rec["split"]].append(rec["preprocessed"])  # preprocessed path (relative)
        if rec["split"] == "train":
            splits["train"].extend(rec["augmented"])  # add augmentations to train only

    (dst_root / "splits.json").write_text(json.dumps(splits, indent=2))

    summary = {
        "num_classes": len(classes),
        "num_originals": len(originals),
        "num_processed": n_processed,
        "num_augmented": n_augmented,
        "dst_root": str(dst_root),
    }
    return summary



### CLI (optional): preprocess & summary
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="ASL offline preprocessing &ánd augmentation -> on-disk outputs")
    p.add_argument("src_root", type=str, help="Path to dataset root (ImageFolder style)")
    p.add_argument("--dst_root", type=str, default=None, help="Where to write processed outputs (default: <src>_processed)")
    p.add_argument("--train_ratio", type=float, default=0.9, help="Train proportion for the split")
    p.add_argument("--seed", type=int, default=42, help="Split RNG seed")
    p.add_argument("--augs_per_image", type=int, default=4, help="How many augmented images per training original")
    args = p.parse_args()

    preprocess_cfg = PreprocessConfig()
    augment_cfg = AugmentConfig(augs_per_image=args.augs_per_image)
    split_cfg = SplitConfig(train_ratio=args.train_ratio, seed=args.seed)

    summary = run_offline_pipeline(src_root=args.src_root, dst_root=args.dst_root, preprocess_cfg=preprocess_cfg, augment_cfg=augment_cfg, split_cfg=split_cfg)
    print(json.dumps(summary, indent=2))
