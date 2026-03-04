"""
Extract 519-d (Wav2CLIP + 7-d librosa) audio features for existing AIST processed data.
Uses motion lengths from processed/motions_263/ and WAVs from audio/.
No conversion repo or SMPL needed.

Usage:
    python scripts/extract_aist_audio_519.py --aist_dir /Data/yash.bhardwaj/datasets/aist [--device cuda]
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

def get_music_name(sequence_name):
    for part in sequence_name.split('_'):
        if part.startswith('m') and len(part) >= 3 and part[1:3].isalpha():
            return part
    return None

def main():
    parser = argparse.ArgumentParser(description='Extract 519-d audio for AIST (Wav2CLIP + 7-d librosa)')
    parser.add_argument('--aist_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--target_fps', type=int, default=20)
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model.audio_features_wav2clip import extract_wav2clip_plus_librosa

    motion_dir = os.path.join(args.aist_dir, 'processed', 'motions_263')
    audio_wav_dir = os.path.join(args.aist_dir, 'audio')
    out_dir = os.path.join(args.aist_dir, 'processed', 'audio_feats_519')
    os.makedirs(out_dir, exist_ok=True)

    split_files = [
        os.path.join(args.aist_dir, 'splits', 'crossmodal_train.txt'),
        os.path.join(args.aist_dir, 'splits', 'crossmodal_val.txt'),
        os.path.join(args.aist_dir, 'splits', 'crossmodal_test.txt'),
    ]
    names = set()
    for p in split_files:
        if os.path.exists(p):
            with open(p) as f:
                for line in f:
                    n = line.strip()
                    if n:
                        names.add(n)

    names = sorted(names)
    print(f"Extracting 519-d audio for {len(names)} sequences (device={args.device})")

    ok, skip, fail = 0, 0, 0
    for name in tqdm(names, desc='519-d audio'):
        motion_path = os.path.join(motion_dir, f'{name}.npy')
        out_path = os.path.join(out_dir, f'{name}.npy')
        if not os.path.exists(motion_path):
            skip += 1
            continue
        if os.path.exists(out_path):
            ok += 1
            continue
        motion = np.load(motion_path)
        T = motion.shape[0]
        duration = T / args.target_fps + 0.5
        music_id = get_music_name(name)
        if not music_id:
            fail += 1
            continue
        wav_path = os.path.join(audio_wav_dir, f'{music_id}.wav')
        if not os.path.exists(wav_path):
            fail += 1
            continue
        try:
            feats = extract_wav2clip_plus_librosa(
                wav_path, target_fps=args.target_fps, duration=duration, device=args.device
            )
            feats = feats[:T].astype(np.float32)
            if feats.shape[0] < T:
                feats = np.pad(feats, ((0, T - feats.shape[0]), (0, 0)), mode='edge')
            np.save(out_path, feats)
            ok += 1
        except Exception as e:
            tqdm.write(f"FAIL {name}: {e}")
            fail += 1

    print(f"Done: {ok} ok, {skip} skipped (no motion), {fail} failed")
    print(f"Output: {out_dir}")

if __name__ == '__main__':
    main()
