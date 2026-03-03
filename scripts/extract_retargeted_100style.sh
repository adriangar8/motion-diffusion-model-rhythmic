#!/usr/bin/env bash
# Extract each RETARGETED_100STYLE zip into its own folder (no merging).
# Run from repo root after download from:
#   https://drive.google.com/drive/folders/1P_aQdSuiht3gh1kjGkK4KBt_9i9ARawy
#
# Usage:
#   bash scripts/extract_retargeted_100style.sh /Data/yash.bhardwaj/datasets/Retargeted100Style/RETARGETED_100STYLE

set -e
DIR="${1:-/Data/yash.bhardwaj/datasets/Retargeted100Style/RETARGETED_100STYLE}"
cd "$DIR"

for zip in 100STYLE_SMPL_BVH.zip new_joint_vecs.zip new_joints.zip texts.zip; do
  if [[ -f "$zip" ]]; then
    name="${zip%.zip}"
    echo "Extracting $zip -> $name/"
    mkdir -p "$name"
    unzip -o -q "$zip" -d "$name"
    echo "  done."
  else
    echo "Skip (not found): $zip"
  fi
done

echo "100STYLE_name_dict.txt stays in $DIR (no zip)."
echo "Done."
