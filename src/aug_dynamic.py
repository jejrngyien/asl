#!/usr/bin/env python3
"""
Dynamic preprocessing & augmentation for ASL clips.

Pipeline (per clip):
  1) load video
  2) Detect hand bbox on a stride of frames, compute union bbox, expand by margin, make square, clamp → apply SAME crop to all frames.
  3) Resize to target size
  4) Save preprocessed frames to "processed/preprocessed/<class>/<clip_id>/".
  5) For each augmentation sample, apply temporally-consistent transforms (same params for every frame!), and save to
    "processed/augmented/<class>/<clip_id>_aug####/".
  6) Write meta infos

Note: All augmentations are mild and temporally consistent. square crop determined from the first frame.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter
import cv2
import mediapipe as mp
import torchvision.transforms.functional as F



### Configuration
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUPPORTED_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUPPORTED_VID_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
PIPELINE_VERSION = "dyn_pre_v1.0_aug_v1.0"


def _ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


@dataclass
class PreprocessConfig:
    target_size: int = 112
    margin: float = 0.25
    min_det_conf: float = 0.5
    det_stride: int = 3


@dataclass
class AugmentConfig:
    augs_per_clip: int = 2
    allow_horizontal_flip: bool = False
    flip_p: float = 0.5
    jitter_p: float = 1.0
    jitter_bcs: Tuple[float, float, float] = (0.2, 0.2, 0.2)
    rrc_p: float = 0.5
    rrc_scale: Tuple[float, float] = (0.8, 1.0)
    rrc_ratio: Tuple[float, float] = (0.9, 1.1)
    rotate_p: float = 0.3
    rotations: Sequence[int] = (0, 5, -5, 10, -10)
    blur_p: float = 0.15
    blur_radius: float = 1.0


@dataclass
class SplitConfig:
    train_ratio: float = 1.0
    seed: int = 42


### MediaPipe Hand Detector for ctopping
class HandDetector:
    def __init__(self, min_det_conf: float = 0.5):
        if mp is None:
            raise ImportError("mediapipe is not installed. pip install mediapipe")
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
        h, w = image_rgb_np.shape[:2]
        results = self._hands.process(image_rgb_np)
        if not results.multi_hand_landmarks:
            return None
        xs, ys = [], []
        for hand_lms in results.multi_hand_landmarks:
            for lm in hand_lms.landmark:
                xs.append(_clamp(lm.x * w, 0, w - 1))
                ys.append(_clamp(lm.y * h, 0, h - 1))
        if not xs:
            return None
        x1, x2 = int(min(xs)), int(math.ceil(max(xs)))
        y1, y2 = int(min(ys)), int(math.ceil(max(ys)))
        return x1, y1, x2, y2


### IO helpers
def _list_images(p: Path) -> List[Path]:
    return [q for q in sorted(p.iterdir()) if q.suffix.lower() in SUPPORTED_IMG_EXTS]


def _decode_video(vid_path: Path) -> List[Image.Image]:
    cap = cv2.VideoCapture(str(vid_path))
    frames: List[Image.Image] = []
    ok, frame = cap.read()
    while ok:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
        ok, frame = cap.read()
    cap.release()
    return frames


def _load_clip(clip_path: Path) -> List[Image.Image]:
    if clip_path.is_dir():
        imgs = _list_images(clip_path)
        return [_ensure_rgb(Image.open(p)) for p in imgs]
    if clip_path.is_file() and clip_path.suffix.lower() in SUPPORTED_VID_EXTS:
        return _decode_video(clip_path)
    raise RuntimeError(f"Unsupported clip path: {clip_path}")


def _save_frames(frames: List[Image.Image], out_dir: Path, prefix: str = "frame_") -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rels = []
    for i, img in enumerate(frames):
        out_path = out_dir / f"{prefix}{i:05d}.png"
        img.save(out_path, format="PNG", optimize=True)
        rels.append(str(out_path))
    return rels


### Cropping across frames
def _square_with_margin(x1, y1, x2, y2, W, H, margin: float):
    bw, bh = (x2 - x1), (y2 - y1)
    # expand
    mx, my = int(bw * margin), int(bh * margin)
    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(W, x2 + mx), min(H, y2 + my)
    # make square
    bw, bh = (x2 - x1), (y2 - y1)
    if bw > bh:
        d = bw - bh
        y1, y2 = max(0, y1 - d // 2), min(H, y2 + math.ceil(d / 2))
    elif bh > bw:
        d = bh - bw
        x1, x2 = max(0, x1 - d // 2), min(W, x2 + math.ceil(d / 2))
    # clamp
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    return x1, y1, x2, y2


def compute_union_bbox(frames: List[Image.Image], det: HandDetector, stride: int) -> Optional[Tuple[int, int, int, int]]:
    x1u = y1u = 1_000_000
    x2u = y2u = -1_000_000
    found = False
    for idx in range(0, len(frames), max(1, stride)):
        np_rgb = np.array(_ensure_rgb(frames[idx]))
        bbox = det.detect_bbox(np_rgb)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        x1u, y1u = min(x1u, x1), min(y1u, y1)
        x2u, y2u = max(x2u, x2), max(y2u, y2)
        found = True
    if not found:
        return None
    return int(x1u), int(y1u), int(x2u), int(y2u)


def crop_clip(frames: List[Image.Image], cfg: PreprocessConfig, det: HandDetector) -> Tuple[List[Image.Image], Dict]:
    if not frames:
        return [], {"fallback": True, "bbox": None}
    W, H = frames[0].size
    bbox = compute_union_bbox(frames, det, cfg.det_stride)
    fallback = False
    if bbox is None:
        # center square on first frame
        s = min(W, H)
        cx, cy = W // 2, H // 2
        x1 = max(0, cx - s // 2)
        y1 = max(0, cy - s // 2)
        x2 = min(W, x1 + s)
        y2 = min(H, y1 + s)
        fallback = True
    else:
        x1, y1, x2, y2 = _square_with_margin(*bbox, W, H, cfg.margin)
    # crop all frames with SAME box
    out = [img.crop((x1, y1, x2, y2)).resize((cfg.target_size, cfg.target_size), Image.BILINEAR)
           for img in frames]
    meta = {"bbox": [int(x1), int(y1), int(x2), int(y2)], "fallback": bool(fallback)}
    return out, meta


### Clip-wide consistent augmentations
class VideoAugmentor:
    def __init__(self, cfg: AugmentConfig, target_size: int):
        self.cfg = cfg
        self.target_size = target_size

    def _sample_params(self):
        rng = random.random
        params = {}
        # flip
        params["do_flip"] = self.cfg.allow_horizontal_flip and rng() < self.cfg.flip_p
        # jitter factors
        if rng() < self.cfg.jitter_p:
            b, c, s = self.cfg.jitter_bcs
            params["bright"] = random.uniform(max(0.0, 1 - b), 1 + b)
            params["contrast"] = random.uniform(max(0.0, 1 - c), 1 + c)
            params["saturation"] = random.uniform(max(0.0, 1 - s), 1 + s)
        else:
            params["bright"] = params["contrast"] = params["saturation"] = None
        # random resized crop (on first frame)
        params["use_rrc"] = random.random() < self.cfg.rrc_p
        # rotation
        params["angle"] = random.choice(self.cfg.rotations) if (random.random() < self.cfg.rotate_p) else 0
        # blur
        params["blur"] = (random.random() < self.cfg.blur_p)
        params["blur_radius"] = self.cfg.blur_radius
        return params

    def _apply_to_frame(self, img: Image.Image, params, rrc_rect=None) -> Image.Image:
        img = _ensure_rgb(img)
        if params["do_flip"]:
            img = F.hflip(img)
        if params["bright"] is not None:
            img = F.adjust_brightness(img, params["bright"])
            img = F.adjust_contrast(img, params["contrast"])
            img = F.adjust_saturation(img, params["saturation"])
        if params["use_rrc"] and rrc_rect is not None:
            i, j, h, w = rrc_rect
            img = F.resized_crop(img, i, j, h, w, (self.target_size, self.target_size))
        else:
            if img.size != (self.target_size, self.target_size):
                img = img.resize((self.target_size, self.target_size), Image.BILINEAR)
        angle = params["angle"]
        if angle:
            img = F.rotate(img, angle, interpolation=Image.BILINEAR, fill=0)
        if params["blur"]:
            img = img.filter(ImageFilter.GaussianBlur(radius=params["blur_radius"]))
        return img

    def __call__(self, frames: List[Image.Image]) -> List[Image.Image]:
        if not frames:
            return frames
        params = self._sample_params()
        # determine RRC window once using first frame
        rrc_rect = None
        if params["use_rrc"]:
            # emulate RandomResizedCrop.get_params
            scale, ratio = self.cfg.rrc_scale, self.cfg.rrc_ratio
            width, height = frames[0].size
            area = height * width
            for _ in range(10):
                target_area = random.uniform(*scale) * area
                log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
                aspect_ratio = math.exp(random.uniform(*log_ratio))
                w = int(round(math.sqrt(target_area * aspect_ratio)))
                h = int(round(math.sqrt(target_area / aspect_ratio)))
                if 0 < w <= width and 0 < h <= height:
                    i = random.randint(0, height - h)
                    j = random.randint(0, width - w)
                    rrc_rect = (i, j, h, w)
                    break
            if rrc_rect is None:
                rrc_rect = (0, 0, height, width)
        return [self._apply_to_frame(f, params, rrc_rect) for f in frames]


### Dataset discovery and metadata
def discover_clips(src_root: Path) -> List[Tuple[Path, str, str]]:
    """
    Return list of (clip_path, class_name, clip_id).
    clip_id is derived from the filename/folder name without extension.
    """
    items: List[Tuple[Path, str, str]] = []
    for cls_dir in sorted([d for d in src_root.iterdir() if d.is_dir()]):
        cls = cls_dir.name
        for p in sorted(cls_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED_VID_EXTS:
                items.append((p, cls, p.stem))
            elif p.is_dir():
                imgs = _list_images(p)
                if imgs:
                    items.append((p, cls, p.name))
    return items


### Pipeline
def run_offline_pipeline(src_root: str | Path, dst_root: Optional[str | Path] = None, preprocess_cfg: Optional[PreprocessConfig] = None,
        augment_cfg: Optional[AugmentConfig] = None, split_cfg: Optional[SplitConfig] = None):
    src_root = Path(src_root)
    if dst_root is None:
        dst_root = src_root.with_name(src_root.name + "_processed_dyn")
    dst_root = Path(dst_root)

    preprocess_cfg = preprocess_cfg or PreprocessConfig()
    augment_cfg = augment_cfg or AugmentConfig()
    split_cfg = split_cfg or SplitConfig()

    pre_dir = dst_root / "preprocessed"
    aug_dir = dst_root / "augmented"
    pre_dir.mkdir(parents=True, exist_ok=True)
    aug_dir.mkdir(parents=True, exist_ok=True)

    classes = [d.name for d in sorted(src_root.iterdir()) if d.is_dir()]
    if not classes:
        raise RuntimeError(f"No class subfolders under {src_root}")
    class_to_idx = {c: i for i, c in enumerate(classes)}

    clips = discover_clips(src_root)
    if not clips:
        raise RuntimeError("No input clips found (videos or frame-folders).")

    # split per class
    rng = random.Random(split_cfg.seed)
    per_class: Dict[str, List[int]] = {c: [] for c in classes}
    for i, (_, cls, _) in enumerate(clips):
        per_class[cls].append(i)
    train_set, test_set = set(), set()
    for c, idxs in per_class.items():
        rng.shuffle(idxs)
        n_train = int(len(idxs) * split_cfg.train_ratio)
        train_set.update(idxs[:n_train])
        test_set.update(idxs[n_train:])

    augmenter = VideoAugmentor(augment_cfg, preprocess_cfg.target_size)
    metadata = []
    n_processed = 0
    n_augmented = 0

    with HandDetector(min_det_conf=preprocess_cfg.min_det_conf) as det:
        for k, (clip_path, cls, clip_id) in enumerate(clips):
            split = "train" if k in train_set else "test"

            # load clip
            frames = _load_clip(clip_path)
            if not frames: continue

            # crop+resize consistently
            frames_proc, meta = crop_clip(frames, preprocess_cfg, det)

            # save preprocessed frames
            out_pre = pre_dir / cls / clip_id
            saved_pre_abs = _save_frames(frames_proc, out_pre)
            saved_pre_rel = [str(Path(p).relative_to(dst_root)) for p in saved_pre_abs]

            # augment (train only)
            aug_dirs_rel: List[str] = []
            if split == "train" and augment_cfg.augs_per_clip > 0:
                for j in range(augment_cfg.augs_per_clip):
                    frames_aug = augmenter(frames_proc)
                    out_aug = aug_dir / cls / f"{clip_id}_aug{j:04d}"
                    saved_aug_abs = _save_frames(frames_aug, out_aug)
                    aug_dirs_rel.append(str(out_aug.relative_to(dst_root)))
                    n_augmented += 1

            n_processed += 1

            rec = {
                "source": str(clip_path.relative_to(src_root)),
                "class": cls,
                "class_idx": class_to_idx[cls],
                "preprocessed_dir": str((pre_dir / cls / clip_id).relative_to(dst_root)),
                "frames": saved_pre_rel,
                "augmented_dirs": aug_dirs_rel,  # directories (each contains frames)
                "num_frames": len(frames_proc),
                "split": split,
                "meta": meta,
                "version": PIPELINE_VERSION,
            }
            metadata.append(rec)

    # write metadata
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

    # flattened split list for quick loading
    splits = {"train": [], "test": []}
    for rec in metadata:
        splits[rec["split"]].append(rec["preprocessed_dir"])  # push directory path per clip
        if rec["split"] == "train":
            splits["train"].extend(rec["augmented_dirs"])
    (dst_root / "splits.json").write_text(json.dumps(splits, indent=2))

    return {"num_classes": len(classes), "num_clips": len(clips), "num_processed": n_processed, "num_augmented": n_augmented, "dst_root": str(dst_root)}



if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ASL dynamic preprocessing & temporally-consistent augmentation")
    ap.add_argument("src_root", type=str, help="Dataset root (class subfolders with videos or frame-folders)")
    ap.add_argument("--dst_root", type=str, default=None, help="Output root (default: <src>_processed_dyn)")
    args = ap.parse_args()

    augs_per_clip = 6
    target_img_size = 112
    stride = 2  # frames between two pictures used toi compute bb for the entire clip
    pre_cfg = PreprocessConfig(target_size=target_img_size, det_stride=stride)
    aug_cfg = AugmentConfig(augs_per_clip=augs_per_clip)
    split_cfg = SplitConfig(train_ratio=1.0, seed=42)

    summary = run_offline_pipeline(
        src_root=args.src_root,
        dst_root=args.dst_root,
        preprocess_cfg=pre_cfg,
        augment_cfg=aug_cfg,
        split_cfg=split_cfg,
    )
    print(json.dumps(summary, indent=2))
