#!/usr/bin/env python3
"""
Organizes videos from a *flat* folder into per-class subfolders using
WLASL / WLASL-processed annotations.

Supported annotations:
- JSON in (near) official WLASL format:
  [
    {
      "gloss": "book",
      "instances": [{"video_id": "book_01234", ...}, ...]
    },
    ...
  ]

Default behavior: **copy** files. Alternative move files.
"""
from pathlib import Path
import textwrap
import argparse
import csv
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, List



VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def norm_id(s: str) -> str:
    """Normalize a video id / file stem for robust matching."""
    s = s.lower()
    s = os.path.splitext(s)[0]
    return re.sub(r"[^a-z0-9]", "", s)


def safe_label(label: str) -> str:
    """Sanitize label to a filesystem-safe folder name."""
    label = label.strip().lower()
    label = re.sub(r"\s+", "_", label)
    label = re.sub(r"[^a-z0-9_\-]", "", label)
    label = re.sub(r"_+", "_", label).strip("_")
    return label or "unknown"


def parse_json_annotations(path: Path) -> Dict[str, str]:
    """
    Parse JSON annotations and return mapping: normalized_video_id -> label (gloss).
    Accepts official WLASL format or a flat list/dict with keys
    ('video_id'/'video'/'id' and 'gloss'/'label'/'class').
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    id2label: Dict[str, str] = {}

    def add_pair(vid: str, lab: str):
        if not vid or not lab:
            return
        id2label[norm_id(vid)] = lab

    if isinstance(data, list):
        # Either WLASL top-level list with "instances" per gloss or a flat list of dicts.
        if data and isinstance(data[0], dict) and "instances" in data[0] and "gloss" in data[0]:
            # WLASL format
            for entry in data:
                gloss = entry.get("gloss") or entry.get("label")
                for inst in entry.get("instances", []):
                    vid = inst.get("video_id") or inst.get("video") or inst.get("id")
                    add_pair(vid, gloss)
        else:
            # Flat list of dicts
            for row in data:
                vid = row.get("video_id") or row.get("video") or row.get("id")
                gloss = row.get("gloss") or row.get("label") or row.get("class")
                add_pair(vid, gloss)
    elif isinstance(data, dict):
        # Possibly a dict with a list under one of these keys.
        items = None
        for key in ["annotations", "instances", "data", "items"]:
            if key in data and isinstance(data[key], list):
                items = data[key]
                break
        if items is None:
            # Maybe direct mapping: {video_id: label, ...}
            for k, v in data.items():
                if isinstance(v, (str, int)):
                    add_pair(k, str(v))
            return id2label
        # Process list of dicts
        for row in items:
            vid = row.get("video_id") or row.get("video") or row.get("id")
            gloss = row.get("gloss") or row.get("label") or row.get("class")
            add_pair(vid, gloss)
    else:
        raise ValueError("Unknown JSON structure in annotations.")

    return id2label


def build_id2label(ann_path: Path) -> Dict[str, str]:
    if ann_path.suffix.lower() == ".json":
        return parse_json_annotations(ann_path)
    else: raise ValueError("Only .json or .csv annotations are supported.")


def index_source_videos(src_dir: Path) -> Dict[str, Path]:
    """Index source files: normalized_basename -> absolute Path."""
    index: Dict[str, Path] = {}
    for p in src_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in VIDEO_EXTS:
            continue
        nid = norm_id(p.stem)
        index[nid] = p
    return index


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def write_csv(rows: List[List[str]], header: List[str], out_path: Path):
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def transfer(src: Path, dst: Path, mode: str):
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(src, dst)
    else:
        raise ValueError(f"Unknown transfer mode: {mode}")



def main():
    ap = argparse.ArgumentParser(description="Organize WLASL/WLASL-processed videos into per-class folders.")
    ap.add_argument("--src", required=True, type=Path, help="Source folder with videos in a flat structure")
    ap.add_argument("--ann", required=True, type=Path, help="Path to annotations (.json or .csv)")
    ap.add_argument("--out", required=True, type=Path, help="Destination root folder where class subfolders will be created")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--copy", action="store_true", help="Copy files (default)")
    group.add_argument("--move", action="store_true", help="Move files")
    ap.add_argument("--preview_only", action="store_true", help="Only produce mapping_preview.csv (no file transfer)")
    args = ap.parse_args()

    mode = "copy"
    if args.move: mode = "move"

    # Load annotations and source index
    id2label = build_id2label(args.ann)
    src_index = index_source_videos(args.src)

    # Stats and preview
    class_counts = defaultdict(int)
    matched_rows: List[List[str]] = []
    unmatched_rows: List[List[str]] = []
    missing_rows: List[List[str]] = []

    # Match files -> labels
    for nid, path in src_index.items():
        label = id2label.get(nid)
        if label is None:
            # Heuristic: try trailing digits as id (e.g., "..._12345")
            m = re.search(r"(\d+)$", path.stem)
            alt = norm_id(m.group(1)) if m else None
            if alt and alt in id2label:
                label = id2label[alt]
        if label is None:
            unmatched_rows.append([str(path)])
        else:
            class_counts[label] += 1
            matched_rows.append([str(path), label])

    # Missing files for annotations
    for nid, label in id2label.items():
        if nid not in src_index:
                missing_rows.append([nid, label])

    # Write logs
    ensure_dir(args.out)
    write_csv(matched_rows, ["video_path", "label"], args.out / "mapping_preview.csv")
    if unmatched_rows:
        write_csv(unmatched_rows, ["video_without_annotation"], args.out / "unmatched_videos.csv")
    if missing_rows:
        write_csv(missing_rows, ["annotation_video_id_missing", "label"], args.out / "missing_files.csv")

    if args.preview_only:
        print(f"[Preview] {len(matched_rows)} videos would be organized. Classes: {len(set(l for _, l in matched_rows))}")
        return

    # Transfer files
    created = 0
    for src_path_str, label in matched_rows:
        src_path = Path(src_path_str)
        dst_dir = args.out / safe_label(label)
        ensure_dir(dst_dir)
        dst_path = dst_dir / src_path.name
        if dst_path.exists():
            base, ext = os.path.splitext(src_path.name)
            k = 1
            while True:
                candidate = dst_dir / f"{base}_{k}{ext}"
                if not candidate.exists():
                    dst_path = candidate
                    break
                k += 1
        transfer(src_path, dst_path, mode)
        created += 1

    print(f"Done: {created} files into {len(set(safe_label(l) for _, l in matched_rows))} classes at '{args.out}' ({mode}).")
    if unmatched_rows:
        print(f"Note: {len(unmatched_rows)} source videos without annotations (see unmatched_videos.csv).")
    if missing_rows:
        print(f"Note: {len(missing_rows)} annotations without files (see missing_files.csv).")


if __name__ == "__main__":
    main()
