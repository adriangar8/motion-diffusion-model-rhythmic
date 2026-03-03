#!/usr/bin/env python3
"""
Build style-subset CSVs from the Retargeted 100STYLE dataset.

Maps 100STYLE style labels to our 4 categories using the primary/secondary
mapping, then writes one CSV per category with columns:
    motion_id, style, caption, source_file

Usage:
    python scripts/style_subsets/build_100style_subsets.py \
        --dataset_root /path/to/RETARGETED_100STYLE \
        --out_dir outputs/style_subsets
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

STYLE_MAP: Dict[str, Dict[str, List[str]]] = {
    "old_elderly": {
        "primary": ["Old"],
        "secondary": ["SlideFeet"],
    },
    "angry_aggressive": {
        "primary": ["Angry"],
        "secondary": [],
    },
    "proud_confident": {
        "primary": ["Proud"],
        "secondary": ["Strutting", "Elated"],
    },
    "robot_mechanical": {
        "primary": ["Robot"],
        "secondary": ["Stiff"],
    },
}


def parse_name_dict(path: str) -> List[Tuple[str, str, str]]:
    """Parse 100STYLE_name_dict.txt → list of (motion_id, bvh_name, style_name)."""
    entries = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            motion_id = parts[0]
            bvh_name = parts[1]
            style_name = bvh_name.split("_")[0]
            entries.append((motion_id, bvh_name, style_name))
    return entries


def load_captions(texts_dir: str, motion_id: str) -> List[str]:
    """Load all captions for a motion_id from texts/{motion_id}.txt."""
    txt_path = os.path.join(texts_dir, motion_id + ".txt")
    if not os.path.isfile(txt_path):
        return []
    captions = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if line:
                captions.append(line)
    return captions


def build_reverse_map(style_map: Dict) -> Dict[str, str]:
    """100STYLE name → our category name."""
    rev = {}
    for category, mapping in style_map.items():
        for name in mapping["primary"] + mapping["secondary"]:
            rev[name] = category
    return rev


def main():
    parser = argparse.ArgumentParser(
        description="Build style-subset CSVs from Retargeted 100STYLE"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/Data/yash.bhardwaj/datasets/Retargeted100Style/RETARGETED_100STYLE",
        help="Path to RETARGETED_100STYLE root",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/style_subsets",
        help="Output directory (CSVs go into out_dir/<style>/)",
    )
    args = parser.parse_args()

    name_dict_path = os.path.join(args.dataset_root, "100STYLE_name_dict.txt")
    texts_dir = os.path.join(args.dataset_root, "texts")

    if not os.path.isfile(name_dict_path):
        print(f"ERROR: {name_dict_path} not found", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(texts_dir):
        print(f"ERROR: {texts_dir} not found", file=sys.stderr)
        sys.exit(1)

    entries = parse_name_dict(name_dict_path)
    rev_map = build_reverse_map(STYLE_MAP)

    # Group by our category
    grouped: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    for motion_id, bvh_name, style_name in entries:
        if style_name in rev_map:
            grouped[rev_map[style_name]].append((motion_id, bvh_name, style_name))

    for category in STYLE_MAP:
        cat_dir = os.path.join(args.out_dir, category)
        os.makedirs(cat_dir, exist_ok=True)
        out_csv = os.path.join(cat_dir, "100style_retrieved.csv")
        out_ids = os.path.join(cat_dir, "100style_motion_ids.txt")

        rows = []
        motion_ids = []
        for motion_id, bvh_name, style_name in grouped.get(category, []):
            source_file = os.path.join(texts_dir, motion_id + ".txt")
            captions = load_captions(texts_dir, motion_id)
            caption = captions[0] if captions else ""
            rows.append({
                "motion_id": motion_id,
                "style": category,
                "caption": caption,
                "source_file": source_file,
            })
            motion_ids.append(motion_id)

        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["motion_id", "style", "caption", "source_file"]
            )
            writer.writeheader()
            writer.writerows(rows)

        with open(out_ids, "w") as f:
            f.write("\n".join(motion_ids) + "\n")

        primary = STYLE_MAP[category]["primary"]
        secondary = STYLE_MAP[category]["secondary"]
        print(
            f"{category}: {len(rows)} motions "
            f"(primary={primary}, secondary={secondary}) → {out_csv}"
        )


if __name__ == "__main__":
    main()
