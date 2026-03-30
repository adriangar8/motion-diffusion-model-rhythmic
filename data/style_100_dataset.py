"""
Datasets for Stage 3 DoRA style training.

- Style100Dataset: 100STYLE retargeted motion clips for one style.
- HumanML3DRetrievedDataset: CLIP-retrieved HumanML3D motions for one style.
- CombinedStyleDataset: Concatenates both for training.

Both use the same 263-d HumanML3D representation and Mean/Std normalization.
"""

import os
import numpy as np
import torch
from torch.utils import data
from os.path import join as pjoin
import codecs as cs


class Style100Dataset(data.Dataset):
    """100STYLE clips for one style: motion (T, 263) + text."""

    def __init__(self, motion_ids_file, style_root, mean, std, max_length=196):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.max_length = max_length
        self.motion_dir = pjoin(style_root, 'new_joint_vecs')
        self.text_dir = pjoin(style_root, 'texts')

        with open(motion_ids_file, 'r') as f:
            ids = [line.strip() for line in f if line.strip()]
        self.ids = [mid for mid in ids
                    if os.path.exists(pjoin(self.motion_dir, f'{mid}.npy'))
                    and os.path.exists(pjoin(self.text_dir, f'{mid}.txt'))]
        if len(self.ids) < len(ids):
            print(f"Style100: using {len(self.ids)} of {len(ids)} IDs (missing motion or text)")
        assert len(self.ids) > 0, f"No valid samples in {motion_ids_file}"

    def __len__(self):
        return len(self.ids)

    def _load_motion(self, mid):
        return np.load(pjoin(self.motion_dir, f'{mid}.npy')).astype(np.float32)

    def _load_caption(self, mid):
        with cs.open(pjoin(self.text_dir, f'{mid}.txt'), 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        return np.random.choice(lines) if lines else "a person moves."

    def _process(self, motion, caption, mid):
        T = motion.shape[0]
        if T > self.max_length:
            start = np.random.randint(0, T - self.max_length + 1)
            motion = motion[start:start + self.max_length]
            T = self.max_length
        motion = (motion - self.mean.numpy()) / (self.std.numpy() + 1e-8)
        if T < self.max_length:
            pad = np.zeros((self.max_length - T, 263), dtype=np.float32)
            motion = np.concatenate([motion, pad], axis=0)
        motion = torch.from_numpy(motion).float()
        return {
            'inp': motion.T.unsqueeze(1),
            'text': caption,
            'lengths': T,
            'key': mid,
        }

    def __getitem__(self, idx):
        mid = self.ids[idx]
        motion = self._load_motion(mid)
        caption = self._load_caption(mid)
        return self._process(motion, caption, mid)

    def inv_transform(self, data):
        return data * self.std + self.mean


class HumanML3DRetrievedDataset(data.Dataset):
    """
    CLIP-retrieved HumanML3D motions for one style.
    IDs from retrieved_motion_ids.txt, data from HumanML3D/new_joint_vecs/ and texts/.
    HumanML3D text format: caption#pos_tags#start#end — we extract caption only.
    """

    def __init__(self, motion_ids_file, humanml_dir, mean, std, max_length=196):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.max_length = max_length
        self.motion_dir = pjoin(humanml_dir, 'new_joint_vecs')
        self.text_dir = pjoin(humanml_dir, 'texts')

        with open(motion_ids_file, 'r') as f:
            ids = [line.strip() for line in f if line.strip()]
        self.ids = [mid for mid in ids
                    if os.path.exists(pjoin(self.motion_dir, f'{mid}.npy'))
                    and os.path.exists(pjoin(self.text_dir, f'{mid}.txt'))]
        if len(self.ids) < len(ids):
            print(f"HumanML3D retrieved: using {len(self.ids)} of {len(ids)} IDs")
        assert len(self.ids) > 0, f"No valid samples in {motion_ids_file}"

    def __len__(self):
        return len(self.ids)

    def _load_motion(self, mid):
        return np.load(pjoin(self.motion_dir, f'{mid}.npy')).astype(np.float32)

    def _load_caption(self, mid):
        with cs.open(pjoin(self.text_dir, f'{mid}.txt'), 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        captions = []
        for line in lines:
            cap = line.split('#')[0].strip()
            if cap:
                captions.append(cap)
        return np.random.choice(captions) if captions else "a person moves."

    def _process(self, motion, caption, mid):
        T = motion.shape[0]
        if T > self.max_length:
            start = np.random.randint(0, T - self.max_length + 1)
            motion = motion[start:start + self.max_length]
            T = self.max_length
        motion = (motion - self.mean.numpy()) / (self.std.numpy() + 1e-8)
        if T < self.max_length:
            pad = np.zeros((self.max_length - T, 263), dtype=np.float32)
            motion = np.concatenate([motion, pad], axis=0)
        motion = torch.from_numpy(motion).float()
        return {
            'inp': motion.T.unsqueeze(1),
            'text': caption,
            'lengths': T,
            'key': mid,
        }

    def __getitem__(self, idx):
        mid = self.ids[idx]
        motion = self._load_motion(mid)
        caption = self._load_caption(mid)
        return self._process(motion, caption, mid)


class CombinedStyleDataset(data.Dataset):
    """
    Concatenates 100STYLE + CLIP-retrieved HumanML3D for a single style.
    Optional style_upsample: repeat 100STYLE indices N times so style is a larger
    fraction of the data (stronger style gradient).
    """

    def __init__(self, style_ds, humanml_ds, style_upsample=1):
        self.style_ds = style_ds
        self.humanml_ds = humanml_ds
        self.style_upsample = max(1, int(style_upsample))
        self._style_len = len(style_ds) * self.style_upsample
        self._hml_len = len(humanml_ds)
        total = self._style_len + self._hml_len
        print(f"CombinedStyleDataset: {len(style_ds)} 100STYLE x{self.style_upsample} + {len(humanml_ds)} HumanML3D = {total} total")

    def __len__(self):
        return self._style_len + self._hml_len

    def __getitem__(self, idx):
        if idx < self._style_len:
            return self.style_ds[idx % len(self.style_ds)]
        return self.humanml_ds[idx - self._style_len]


def style_100_collate(batch):
    """Collate to match t2m format: motion (B, 263, 1, T), mask, lengths, text. No audio."""
    from data_loaders.tensors import lengths_to_mask
    inp = torch.stack([b['inp'] for b in batch], dim=0)
    lengths = torch.tensor([b['lengths'] for b in batch], dtype=torch.long)
    B, J, _, T = inp.shape
    mask = lengths_to_mask(lengths, T).unsqueeze(1).unsqueeze(1)
    cond = {
        'y': {
            'mask': mask,
            'lengths': lengths,
            'text': [b['text'] for b in batch],
        }
    }
    return inp, cond
