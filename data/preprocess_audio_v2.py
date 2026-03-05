####################################################################################[start]
####################################################################################[start]

"""

Re-extract audio features using v2 pipeline for AIST++ dataset.

Reads existing .wav files and saves 52-dim features to audio_feats_v2/ directory.

Usage:
    python -m data.preprocess_audio_v2 \
        --aist_dir ./dataset/aist \
        --output_subdir audio_feats_v2

"""

import os
import sys
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm

sys.path.insert(0, '.')

from model.audio_features_v2 import extract_audio_features_v2

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--aist_dir', type=str, required=True)
    parser.add_argument('--output_subdir', type=str, default='audio_feats_v2')
    parser.add_argument('--target_fps', type=int, default=20)
    args = parser.parse_args()

    audio_dir = os.path.join(args.aist_dir, 'audio')
    output_dir = os.path.join(args.aist_dir, 'processed', args.output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    # -- find all wav files --
    
    wav_files = sorted(glob(os.path.join(audio_dir, '*.wav')))
    print(f"Found {len(wav_files)} audio files")

    # -- also get list of motion files to know which sequences we have --
    
    motion_dir = os.path.join(args.aist_dir, 'processed', 'motions_263')
    motion_names = set()
    
    if os.path.exists(motion_dir):
        for f in os.listdir(motion_dir):
            if f.endswith('.npy'):
                motion_names.add(f.replace('.npy', ''))

    processed = 0
    skipped = 0

    music_map = {}

    for wav_path in wav_files:
        music_id = os.path.splitext(os.path.basename(wav_path))[0]
        music_map[music_id] = wav_path

    # -- extract features for each sequence that has a motion file --
    
    for seq_name in tqdm(sorted(motion_names), desc="Extracting v2 features"):
    
        output_path = os.path.join(output_dir, f'{seq_name}.npy')

        if os.path.exists(output_path):
            processed += 1
            continue

        # -- extract music ID from sequence name (5th component) --
        
        parts = seq_name.split('_')
        
        if len(parts) >= 5:
            music_id = parts[4]
        
        else:
            skipped += 1
            continue

        if music_id not in music_map:
            skipped += 1
            continue

        try:
            features = extract_audio_features_v2(
                music_map[music_id],
                target_fps=args.target_fps
            )
            np.save(output_path, features)
            processed += 1
        except Exception as e:
            print(f"  Error processing {seq_name}: {e}")
            skipped += 1

    print(f"\nDone: {processed} processed, {skipped} skipped")
    print(f"Output: {output_dir}")

    # -- verify one sample --
    
    sample_files = [f for f in os.listdir(output_dir) if f.endswith('.npy')]
    
    if sample_files:
        sample = np.load(os.path.join(output_dir, sample_files[0]))
        print(f"Sample shape: {sample.shape} (expected (T, 52))")
        print(f"Sample range: [{sample.min():.3f}, {sample.max():.3f}]")

if __name__ == '__main__':
    main()
    
####################################################################################[end]
####################################################################################[end]

