#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import cv2
from PIL import Image



VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
RESIZE_SHORTER = 128  
CROP_SIZE = 112      

def list_videos(root: Path):
    return [p for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS]

def resize_keep_aspect_to_short(img_rgb, target_short=RESIZE_SHORTER):
    h, w = img_rgb.shape[:2]
    if h == 0 or w == 0:
        return None
    short = min(h, w)
    if short == target_short:
        return img_rgb
    scale = target_short / short
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)

def center_crop(img_rgb, size=CROP_SIZE):
    h, w = img_rgb.shape[:2]
    y0 = max(0, (h - size) // 2)
    x0 = max(0, (w - size) // 2)
    crop = img_rgb[y0:y0 + size, x0:x0 + size]
    if crop.shape[0] != size or crop.shape[1] != size:
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    return crop

def preprocess_frame(bgr):
    # BGR -> RGB
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # wie preprocessing.py: Resize(shorter=128) -> CenterCrop(112)
    rgb = resize_keep_aspect_to_short(rgb, RESIZE_SHORTER)
    if rgb is None:
        return None
    rgb = center_crop(rgb, CROP_SIZE)
    return rgb

def save_png(path: Path, rgb_u8):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb_u8).save(path, format="PNG", optimize=True)

def relative_class(src_root: Path, video_path: Path):
    try:
        rel = video_path.relative_to(src_root)
        return rel.parts[0] if len(rel.parts) > 1 else video_path.parent.name
    except Exception:
        return video_path.parent.name

def process_video(src_root: Path, dst_root: Path, vpath: Path, overwrite: bool, stride: int):
    cls = relative_class(src_root, vpath)
    out_dir = dst_root / cls / vpath.stem

    if out_dir.exists() and not overwrite:
        existing = list(out_dir.glob("frame_*.png"))
        if existing:
            print(f"[skip] {vpath} (found {len(existing)} frames; use --overwrite to force)")
            return

    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        print(f"[warn] Cannot open video: {vpath}", file=sys.stderr)
        return

    pad = 5
    saved = 0
    idx = 0

    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        idx += 1
        if (idx - 1) % stride != 0:
            continue

        rgb = preprocess_frame(bgr)
        if rgb is None:
            print(f"[warn] Invalid frame in {vpath} at idx {idx}", file=sys.stderr)
            continue

        out_path = out_dir / f"frame_{saved+1:0{pad}d}.png"
        save_png(out_path, rgb)
        saved += 1

    cap.release()
    print(f"[ok] {vpath} -> {out_dir} ({saved} frames saved, stride={stride})")

def main():
    ap = argparse.ArgumentParser(description="Extract frames with Resize(128 short) + CenterCrop(112).")
    ap.add_argument("--src", type=str, required=True, help="Source root with class subfolders and videos.")
    ap.add_argument("--dst", type=str, required=True, help="Destination root for extracted frames.")
    ap.add_argument("--stride", type=int, default=1, help="Keep every N-th frame (2 = jeder zweite).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing extracted frames.")
    args = ap.parse_args()

    if args.stride < 1:
        print("[error] --stride must be >= 1", file=sys.stderr)
        sys.exit(2)

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    if not src_root.is_dir():
        print(f"[error] Not a directory: {src_root}", file=sys.stderr)
        sys.exit(1)

    videos = list_videos(src_root)
    if not videos:
        print(f"[warn] No videos found under: {src_root}", file=sys.stderr)
        sys.exit(0)

    for vp in videos:
        try:
            process_video(src_root, dst_root, vp, args.overwrite, args.stride)
        except KeyboardInterrupt:
            print("\n[info] Interrupted by user.")
            break
        except Exception as e:
            print(f"[warn] Failed {vp}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
