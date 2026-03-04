####################################################################################[start]
####################################################################################[start]

"""

AIST++ Dataset for audio-conditioned motion generation training.

Loads AIST++ dance motions (SMPL format) paired with music,
converts motions to HumanML3D 263-dim representation,
and extracts librosa audio features at 20fps.

Expected directory structure:
    <aist_dir>/
    ├── motions/          ← AIST++ SMPL motion .pkl files
    ├── wav/              ← Music .wav files (from AIST Dance DB)
    └── splits/
        ├── crossmodal_train.txt
        ├── crossmodal_val.txt
        └── crossmodal_test.txt

Preprocessing:
    Run preprocess_aist.py first to convert SMPL → HumanML3D features
    and extract audio features. This creates:
    <aist_dir>/
    ├── processed/
    │   ├── motions_263/   ← .npy files, each (T, 263)
    │   └── audio_feats/   ← .npy files, each (T, 145)
    └── processed_stats.npz  ← mean/std for normalization
    
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class AISTDataset(Dataset):
    
    """
    
    PyTorch Dataset for preprocessed AIST++ data.
    Returns (motion, audio_features, text, length) tuples.
    
    """

    def __init__(self, aist_dir, split='train', max_motion_length=196,
                 min_motion_length=40, fps=20,
                 humanml_mean=None, humanml_std=None, use_wav2clip=False):
    
        """
    
        Args:
            aist_dir: Path to AIST++ root directory
            split: 'train', 'val', or 'test'
            max_motion_length: Maximum frames to use (196 = 9.8s at 20fps)
            min_motion_length: Minimum frames required (40 = 2.0s)
            fps: Frame rate (should be 20 to match HumanML3D)
            humanml_mean: Mean for normalization (from HumanML3D), shape (263,)
            humanml_std: Std for normalization (from HumanML3D), shape (263,)
            use_wav2clip: If True, load 519-d features from processed/audio_feats_519/
                         (Wav2CLIP 512 + librosa 7). Else load 145-d from audio_feats/.
    
        """
    
        self.aist_dir = aist_dir
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.fps = fps
        self.use_wav2clip = use_wav2clip
        self.audio_feat_dim = 519 if use_wav2clip else 145

        # -- load normalization stats from HumanML3D --
        # -- must match what MDM was trained with --
        
        if humanml_mean is not None and humanml_std is not None:
            
            self.mean = humanml_mean
            self.std = humanml_std
            
        else:
            
            # -- try to load from HumanML3D dataset directory --
            
            stats_path = os.path.join(aist_dir, 'processed_stats.npz')
            
            if os.path.exists(stats_path):
            
                stats = np.load(stats_path)
            
                self.mean = stats['mean']
                self.std = stats['std']
            
            else:
            
                raise ValueError(
                    "Must provide humanml_mean/std or have processed_stats.npz. "
                    "Copy Mean.npy and Std.npy from your HumanML3D dataset directory."
                )

        # -- paths to preprocessed data --
        
        self.motion_dir = os.path.join(aist_dir, 'processed', 'motions_263')
        self.audio_dir = os.path.join(aist_dir, 'processed', 'audio_feats_519' if use_wav2clip else 'audio_feats')

        # -- load split file --
        
        split_map = {
            'train': 'crossmodal_train.txt',
            'val': 'crossmodal_val.txt',
            'test': 'crossmodal_test.txt',
        }
        
        split_file = os.path.join(aist_dir, 'splits', split_map[split])
        
        with open(split_file, 'r') as f:
            sequence_names = [line.strip() for line in f if line.strip()]

        # -- filter to sequences that have both motion and audio preprocessed --
        
        self.data = []
        
        skipped = 0
        
        for name in sequence_names:
            
            motion_path = os.path.join(self.motion_dir, f'{name}.npy')
            audio_path = os.path.join(self.audio_dir, f'{name}.npy')

            if not os.path.exists(motion_path) or not os.path.exists(audio_path):
                
                skipped += 1
                continue

            # -- check length --
            
            motion = np.load(motion_path) # (T, 263)
            
            if motion.shape[0] < min_motion_length:
            
                skipped += 1
                continue

            self.data.append({
                'name': name,
                'motion_path': motion_path,
                'audio_path': audio_path,
                'length': motion.shape[0],
            })

        print(f"AIST++ {split}: {len(self.data)} sequences loaded, {skipped} skipped")

        # -- generic text descriptions per dance genre --
        # -- must match AIST++ naming: g{GENRE}_{...} --
        
        self.genre_texts = {
            'gBR': 'a person performs breakdancing moves to music',
            'gPO': 'a person performs popping dance moves to music',
            'gLO': 'a person performs locking dance moves to music',
            'gMH': 'a person performs hip hop dance moves to music',
            'gLH': 'a person performs hip hop dance moves to music',
            'gHO': 'a person performs house dance moves to music',
            'gWA': 'a person performs waacking dance moves to music',
            'gKR': 'a person performs krumping dance moves to music',
            'gJS': 'a person performs jazz dance moves to music',
            'gJB': 'a person performs ballet jazz dance moves to music',
        }
        
        self.default_text = 'a person dances to music'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        entry = self.data[idx]

        # -- load motion and audio --
        
        motion = np.load(entry['motion_path']).astype(np.float32) # (T_full, 263)
        audio = np.load(entry['audio_path']).astype(np.float32)    # (T_full, 145)

        # -- determine usable length (min of motion and audio) --
        
        usable_len = min(motion.shape[0], audio.shape[0], self.max_motion_length)

        # -- random crop if longer than max --
        
        if motion.shape[0] > self.max_motion_length:
        
            start = np.random.randint(0, motion.shape[0] - self.max_motion_length)
            motion = motion[start:start + self.max_motion_length]
            audio = audio[start:start + self.max_motion_length]
            usable_len = self.max_motion_length
        
        else:
        
            motion = motion[:usable_len]
            audio = audio[:usable_len]

        # -- normalize motion using HumanML3D stats --
        
        motion = (motion - self.mean) / self.std

        # -- get text description based on genre code --
        
        genre_code = entry['name'][:3]
        text = self.genre_texts.get(genre_code, self.default_text)

        return {
            'motion': motion,          # (T, 263) normalized
            'audio': audio,            # (T, 145) raw librosa features
            'text': text,              # str
            'length': usable_len,      # int
            'name': entry['name'],     # str
        }

def aist_collate_fn(batch):
    
    """
    
    Collate function that pads motion and audio to the max length in the batch.

    Returns a dict compatible with MDM's training loop:
        - motion: (B, 263, 1, T_max) — MDM's expected format
        - y: dict with 'text', 'audio_features', 'mask', 'lengths'
    Audio feature dim is 145 (librosa) or 519 (Wav2CLIP+librosa), inferred from batch.
    
    """
    
    batch_size = len(batch)
    max_len = max(item['length'] for item in batch)
    audio_dim = batch[0]['audio'].shape[1]  # 145 or 519

    # -- pad motion: (B, T_max, 263) → then reshape to (B, 263, 1, T_max) for MDM --
    
    motion_padded = np.zeros((batch_size, max_len, 263), dtype=np.float32)
    audio_padded = np.zeros((batch_size, max_len, audio_dim), dtype=np.float32)
    
    mask = np.zeros((batch_size, max_len), dtype=bool)
    
    lengths = []
    texts = []
    names = []

    for i, item in enumerate(batch):
        
        L = item['length']
        motion_padded[i, :L] = item['motion']
        audio_padded[i, :L] = item['audio']
        mask[i, :L] = True
        lengths.append(L)
        texts.append(item['text'])
        names.append(item['name'])

    # -- convert to tensors --
    # -- MDM expects motion as (B, njoints, nfeats, T) where njoints*nfeats = 263 --
    # -- for HumanML3D: njoints=263, nfeats=1 --
    
    motion_tensor = torch.from_numpy(motion_padded).permute(0, 2, 1).unsqueeze(2)
    
    # → (B, 263, 1, T_max)

    audio_tensor = torch.from_numpy(audio_padded) # (B, T_max, 145)
    mask_tensor = torch.from_numpy(mask).unsqueeze(1).unsqueeze(1) # (B, 1, 1, T_max)

    cond = {
        'text': texts,
        'audio_features': audio_tensor,
        'mask': mask_tensor,
        'lengths': torch.tensor(lengths),
        'names': names,
    }

    return motion_tensor, cond

def get_aist_dataloader(aist_dir, split='train', batch_size=32,
                         max_motion_length=196, num_workers=4,
                         humanml_mean=None, humanml_std=None, use_wav2clip=False):
    
    """
    
    Convenience function to create AIST++ dataloader.
    use_wav2clip: load 519-d audio from processed/audio_feats_519/ when True.
    
    """
    
    dataset = AISTDataset(
        aist_dir=aist_dir,
        split=split,
        max_motion_length=max_motion_length,
        humanml_mean=humanml_mean,
        humanml_std=humanml_std,
        use_wav2clip=use_wav2clip,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=aist_collate_fn,
        drop_last=(split == 'train'),
        pin_memory=True,
    )
    
    return loader

####################################################################################[end]
####################################################################################[end]