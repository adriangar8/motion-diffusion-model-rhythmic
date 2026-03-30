"""
Evaluate audio-conditioned MDM on HumanML3D standard benchmarks.

Reproduces the metrics from the original MDM paper (Table 1):
  FID, R-Precision (top 1/2/3), Matching Score, Diversity, MultiModality

Supports three evaluation conditions:
  1. text_only:    audio_guidance=0, null audio features
                   → proves we didn't degrade text-to-motion quality
  2. text+audio:   audio_guidance=2.5, real AIST++ tracks cycling per batch
                   → shows impact of audio conditioning on motion quality
  3. fixed_audio:  audio_guidance=2.5, single AIST++ track for all batches
                   → controlled single-track evaluation

This script is SELF-CONTAINED: it includes a modified CompMDMGeneratedDataset
that injects audio features into model_kwargs['y'] before generation, and an
AudioProvider that handles feature extraction from AIST++ tracks.

No existing files are modified.

Usage:
    # 1. Text-only baseline (recommended first run)
    python -m eval.eval_audio_humanml_v2 \
        --model_path ./save/audio_stage2_v2/model_final.pt \
        --audio_mode none \
        --eval_mode wo_mm \
        --guidance_param 2.5

    # 2. Text + cycling AIST++ audio
    python -m eval.eval_audio_humanml_v2 \
        --model_path ./save/audio_stage2_v2/model_final.pt \
        --audio_mode random_aist \
        --audio_guidance_param 2.5 \
        --eval_mode wo_mm \
        --guidance_param 2.5

    # 3. Text + fixed audio track
    python -m eval.eval_audio_humanml_v2 \
        --model_path ./save/audio_stage2_v2/model_final.pt \
        --audio_mode fixed \
        --audio_path ./dataset/aist/audio/mBR0.wav \
        --audio_guidance_param 2.5 \
        --eval_mode wo_mm

    # Debug mode (fast, 5 reps, 1000 samples)
    python -m eval.eval_audio_humanml_v2 \
        --model_path ./save/audio_stage2_v2/model_final.pt \
        --audio_mode none \
        --eval_mode debug
"""

import os
import sys
import json
import torch
import numpy as np
from argparse import ArgumentParser
from datetime import datetime
from collections import OrderedDict
from glob import glob
from copy import deepcopy
from types import SimpleNamespace

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# MDM imports
from utils.fixseed import fixseed
from data_loaders.humanml.utils.metrics import *
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from data_loaders.humanml.scripts.motion_process import *
from data_loaders.humanml.utils.utils import *
from diffusion import logger
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_gaussian_diffusion
from train.train_platforms import (ClearmlPlatform, TensorboardPlatform,
                                   NoPlatform, WandBPlatform)

torch.multiprocessing.set_sharing_strategy('file_system')


# ============================================================================
# AUDIO PROVIDER — loads AIST++ audio, extracts features, serves per-batch
# ============================================================================

class AudioProvider:
    """Pre-extracts audio features from AIST++ tracks and serves them per batch.

    For 'random_aist' mode: cycles through all available tracks.
    For 'fixed' mode: uses a single track for every batch.
    For 'none' mode: returns None (no audio injection).
    """

    def __init__(self, mode, audio_feat_dim, device,
                 audio_path=None, aist_audio_dir=None, fps=20):
        """
        Args:
            mode: 'none', 'fixed', or 'random_aist'
            audio_feat_dim: 52 (v2), 145 (v1), or 519 (wav2clip)
            device: torch device
            audio_path: path to specific .wav (for 'fixed' mode)
            aist_audio_dir: directory with AIST++ .wav files (for 'random_aist')
            fps: target frame rate (default 20)
        """
        self.mode = mode
        self.audio_feat_dim = audio_feat_dim
        self.device = device
        self.fps = fps
        self.features_cache = {}  # path -> (T, D) numpy array
        self._cycle_idx = 0
        self._track_paths = []

        if mode == 'none':
            print("[AudioProvider] Mode: none (text-only evaluation)")
            return

        if mode == 'fixed':
            assert audio_path and os.path.exists(audio_path), \
                f"Fixed mode requires valid --audio_path, got: {audio_path}"
            self._track_paths = [audio_path]
            self._extract(audio_path)
            print(f"[AudioProvider] Mode: fixed, track={os.path.basename(audio_path)}")

        elif mode == 'random_aist':
            if aist_audio_dir is None:
                aist_audio_dir = './dataset/aist/audio'
            wav_files = sorted(glob(os.path.join(aist_audio_dir, '*.wav')))
            assert len(wav_files) > 0, \
                f"No .wav files found in {aist_audio_dir}"
            self._track_paths = wav_files
            # Pre-extract all tracks (takes a few seconds)
            for wp in wav_files:
                self._extract(wp)
            print(f"[AudioProvider] Mode: random_aist, "
                  f"{len(wav_files)} tracks from {aist_audio_dir}")
            for wp in wav_files:
                name = os.path.basename(wp)
                T = self.features_cache[wp].shape[0]
                print(f"  - {name}: {T} frames ({T/fps:.1f}s)")

        else:
            raise ValueError(f"Unknown audio mode: {mode}")

    def _extract(self, wav_path):
        """Extract audio features from a .wav file and cache them."""
        if wav_path in self.features_cache:
            return

        dim = self.audio_feat_dim

        if dim == 519:
            from model.audio_features_wav2clip import extract_wav2clip_plus_librosa
            feat = extract_wav2clip_plus_librosa(
                wav_path, target_fps=self.fps,
                duration=None, device=self.device,
            )
        elif dim == 52:
            from model.audio_features_v2 import extract_audio_features_v2
            feat = extract_audio_features_v2(
                wav_path, target_fps=self.fps, duration=None,
            )
        else:
            from model.audio_features import extract_audio_features
            feat = extract_audio_features(
                wav_path, target_fps=self.fps, duration=None,
            )

        self.features_cache[wav_path] = feat  # (T, D) numpy
        return feat

    def _get_beat_frames(self, feat):
        """Extract beat frame indices from audio feature array."""
        dim = self.audio_feat_dim
        if dim == 519:
            beat_signal = feat[:, 512 + 1]
            return list(np.where(beat_signal > 0.5)[0])
        elif dim == 52:
            beat_signal = feat[:, 34]
            return list(np.where(beat_signal > 0.3)[0])
        elif dim == 145:
            beat_signal = feat[:, 129]
            return list(np.where(beat_signal > 0.5)[0])
        return []

    def get_batch_audio(self, batch_size, n_frames):
        """Get audio features for a generation batch.

        Args:
            batch_size: number of samples in batch
            n_frames: temporal length of the motion (max_motion_length)

        Returns:
            dict with 'audio_features' tensor and 'beat_frames' list,
            or None if mode is 'none'.
        """
        if self.mode == 'none':
            return None

        # Select track (cycle for random_aist, always [0] for fixed)
        track_path = self._track_paths[self._cycle_idx % len(self._track_paths)]
        self._cycle_idx += 1

        feat_full = self.features_cache[track_path]  # (T_audio, D)

        # Trim or pad to match motion length
        if feat_full.shape[0] >= n_frames:
            feat = feat_full[:n_frames]
        else:
            # Pad with zeros (silence) if audio is shorter than motion
            pad = np.zeros((n_frames - feat_full.shape[0], feat_full.shape[1]))
            feat = np.concatenate([feat_full, pad], axis=0)

        beat_frames = self._get_beat_frames(feat)

        # (1, T, D) -> (B, T, D)
        audio_tensor = torch.from_numpy(feat).float().unsqueeze(0)
        audio_tensor = audio_tensor.repeat(batch_size, 1, 1).to(self.device)

        return {
            'audio_features': audio_tensor,
            'beat_frames': beat_frames,
        }


# ============================================================================
# AUDIO CFG WRAPPER — decomposed text + audio guidance
# ============================================================================

class AudioCFGSampleModel(torch.nn.Module):
    """Classifier-free guidance wrapper for audio-conditioned MDM.

    output = uncond + s_text*(text_only - uncond) + s_audio*(full - text_only)

    When audio_scale=0, this reduces to standard text-only CFG.
    """

    def __init__(self, model, text_scale=2.5, audio_scale=2.5):
        super().__init__()
        self.model = model
        self.text_scale = text_scale
        self.audio_scale = audio_scale

        # Diffusion compatibility pointers
        self.rot2xyz = model.rot2xyz
        self.translation = model.translation
        self.njoints = model.njoints
        self.nfeats = model.nfeats
        self.data_rep = model.data_rep
        self.cond_mode = model.cond_mode
        self.encode_text = model.encode_text

    def forward(self, x, timesteps, y=None):
        has_audio = (
            getattr(self.model, 'audio_conditioning', False)
            and y is not None
            and 'audio_features' in y
            and y['audio_features'] is not None
            and self.audio_scale > 0
        )

        if has_audio:
            # 1. Full conditioning (text + audio)
            out_full = self.model(x, timesteps, y)

            # 2. Text only (audio masked)
            y_no_audio = deepcopy(y)
            y_no_audio['uncond_audio'] = True
            out_text = self.model(x, timesteps, y_no_audio)

            # 3. Fully unconditional
            y_uncond = deepcopy(y)
            y_uncond['uncond'] = True
            y_uncond['uncond_audio'] = True
            out_uncond = self.model(x, timesteps, y_uncond)

            return (out_uncond
                    + self.text_scale * (out_text - out_uncond)
                    + self.audio_scale * (out_full - out_text))
        else:
            # Text-only CFG (2 forward passes)
            out = self.model(x, timesteps, y)

            y_uncond = deepcopy(y)
            y_uncond['uncond'] = True
            if getattr(self.model, 'audio_conditioning', False):
                y_uncond['uncond_audio'] = True
            out_uncond = self.model(x, timesteps, y_uncond)

            return out_uncond + self.text_scale * (out - out_uncond)


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_audio_mdm(model_path, device):
    """Load an audio-conditioned MDM checkpoint."""
    from model.mdm import MDM

    ckpt = torch.load(model_path, map_location='cpu')

    if 'args' in ckpt:
        ckpt_args = ckpt['args']
    else:
        args_path = os.path.join(os.path.dirname(model_path), 'args.json')
        with open(args_path, 'r') as f:
            ckpt_args = json.load(f)

    model = MDM(
        modeltype='',
        njoints=263,
        nfeats=1,
        num_actions=1,
        translation=True,
        pose_rep='rot6d',
        glob=True,
        glob_rot=[3.141592653589793, 0, 0],
        latent_dim=ckpt_args.get('latent_dim', 512),
        ff_size=ckpt_args.get('ff_size', 1024),
        num_layers=ckpt_args.get('num_layers', 8),
        num_heads=ckpt_args.get('num_heads', 4),
        dropout=ckpt_args.get('dropout', 0.1),
        activation=ckpt_args.get('activation', 'gelu'),
        data_rep='hml_vec',
        dataset='humanml',
        clip_dim=512,
        arch='trans_enc',
        clip_version=ckpt_args.get('clip_version', 'ViT-B/32'),
        cond_mode='text',
        cond_mask_prob=ckpt_args.get('text_cond_mask_prob', 0.1),
        audio_conditioning=True,
        audio_feat_dim=ckpt_args.get('audio_feat_dim', 519),
        audio_cond_mask_prob=ckpt_args.get('audio_cond_mask_prob', 0.15),
        use_audio_token_concat=ckpt_args.get('use_audio_token_concat', False),
        temporal_sigma=ckpt_args.get('temporal_sigma', 4.0),
        beat_weight=ckpt_args.get('beat_weight', 2.0),
    )

    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    audio_feat_dim = ckpt_args.get('audio_feat_dim', 519)
    print(f"Audio MDM loaded from {model_path}")
    print(f"  audio_feat_dim={audio_feat_dim}, "
          f"use_wav2clip={ckpt_args.get('use_wav2clip', False)}, "
          f"use_audio_token_concat={ckpt_args.get('use_audio_token_concat', False)}")
    return model, ckpt_args


# ============================================================================
# MODIFIED GENERATION DATASET — injects audio features before sampling
# ============================================================================

class CompAudioMDMGeneratedDataset(Dataset):
    """CompMDMGeneratedDataset with audio feature injection.

    This is a modified copy of CompMDMGeneratedDataset from
    comp_v6_model_dataset.py. The only change is at the audio injection
    point (marked with >>> AUDIO INJECTION <<<) where we optionally add
    audio_features and beat_frames to model_kwargs['y'] before calling
    sample_fn.
    """

    def __init__(self, args, model, diffusion, dataloader,
                 mm_num_samples, mm_num_repeats, max_motion_length,
                 num_samples_limit, scale=1., audio_provider=None):
        self.args = args
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.model = model
        self.audio_provider = audio_provider
        assert mm_num_samples < len(dataloader.dataset)

        use_ddim = False
        clip_denoised = False
        self.max_motion_length = max_motion_length
        sample_fn = (
            diffusion.p_sample_loop if not use_ddim
            else diffusion.ddim_sample_loop
        )

        real_num_batches = len(dataloader)
        if num_samples_limit is not None:
            real_num_batches = min(
                num_samples_limit // dataloader.batch_size + 1,
                real_num_batches
            )
        print('real_num_batches', real_num_batches)

        generated_motion = []
        mm_generated_motions = []
        if mm_num_samples > 0:
            mm_idxs = np.random.choice(
                real_num_batches,
                mm_num_samples // dataloader.batch_size + 1,
                replace=False
            )
            mm_idxs = np.sort(mm_idxs)
        else:
            mm_idxs = []
        print('mm_idxs', mm_idxs)

        model.eval()

        with torch.no_grad():
            for i, (motion, model_kwargs) in tqdm(enumerate(dataloader)):

                if (num_samples_limit is not None
                        and len(generated_motion) >= num_samples_limit):
                    break

                model_kwargs['y'] = {
                    key: val.to(dist_util.dev()) if torch.is_tensor(val) else val
                    for key, val in model_kwargs['y'].items()
                }
                motion = motion.to(dist_util.dev())

                tokens = [t.split('_') for t in model_kwargs['y']['tokens']]

                # Add CFG scale to batch
                if scale != 1.:
                    model_kwargs['y']['scale'] = torch.ones(
                        motion.shape[0], device=dist_util.dev()
                    ) * scale

                # >>>>>>>>>>>>>>>>>> AUDIO INJECTION <<<<<<<<<<<<<<<<<<
                # This is the ONLY modification vs the original
                # CompMDMGeneratedDataset. We inject audio features
                # into model_kwargs so the AudioCFGSampleModel can use
                # them during generation.
                if self.audio_provider is not None:
                    n_frames = motion.shape[-1]  # temporal dimension
                    batch_size = motion.shape[0]
                    audio_data = self.audio_provider.get_batch_audio(
                        batch_size, n_frames
                    )
                    if audio_data is not None:
                        model_kwargs['y']['audio_features'] = \
                            audio_data['audio_features']
                        model_kwargs['y']['beat_frames'] = \
                            audio_data['beat_frames']
                # <<<<<<<<<<<<<<<<<<< END INJECTION >>>>>>>>>>>>>>>>>>>

                mm_num_now = len(mm_generated_motions) // dataloader.batch_size
                is_mm = i in mm_idxs
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []

                for t in range(repeat_times):
                    sample = sample_fn(
                        model,
                        motion.shape,
                        clip_denoised=clip_denoised,
                        model_kwargs=model_kwargs,
                        skip_timesteps=0,
                        init_image=None,
                        progress=False,
                        dump_steps=None,
                        noise=None,
                        const_noise=False,
                    )

                    if 'prefix' in model_kwargs['y'].keys():
                        model_kwargs['y']['lengths'] = \
                            model_kwargs['y']['orig_lengths']

                    if t == 0:
                        sub_dicts = [{
                            'motion': sample[bs_i].squeeze().permute(1, 0).cpu().numpy(),
                            'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                            'caption': model_kwargs['y']['text'][bs_i],
                            'tokens': tokens[bs_i],
                            'cap_len': tokens[bs_i].index('eos/OTHER') + 1,
                        } for bs_i in range(dataloader.batch_size)]
                        generated_motion += sub_dicts

                    if is_mm:
                        for bs_i in range(dataloader.batch_size):
                            mm_motion = sample[bs_i].squeeze().permute(1, 0).cpu().numpy()
                            if self.dataset.mode == 'eval':
                                mm_motion = self.dataset.t2m_dataset.inv_transform(mm_motion)
                                mm_motion = ((mm_motion - self.dataset.mean_for_eval)
                                             / self.dataset.std_for_eval)
                            mm_motions.append({
                                'motion': mm_motion,
                                'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                            })

                if is_mm:
                    mm_generated_motions += [{
                        'caption': model_kwargs['y']['text'][bs_i],
                        'tokens': tokens[bs_i],
                        'cap_len': len(tokens[bs_i]),
                        'mm_motions': mm_motions[bs_i::dataloader.batch_size],
                    } for bs_i in range(dataloader.batch_size)]

        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.w_vectorizer = dataloader.dataset.w_vectorizer

    def __len__(self):
        return len(self.generated_motion)

    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion = data['motion']
        m_length = data['length']
        caption = data['caption']
        tokens = data['tokens']
        sent_len = data['cap_len']

        if self.dataset.mode == 'eval':
            normed_motion = motion
            denormed_motion = self.dataset.t2m_dataset.inv_transform(normed_motion)
            renormed_motion = ((denormed_motion - self.dataset.mean_for_eval)
                               / self.dataset.std_for_eval)
            motion = renormed_motion

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return (word_embeddings, pos_one_hots, caption,
                sent_len, motion, m_length, '_'.join(tokens))


# ============================================================================
# MODIFIED LOADER — passes audio_provider to dataset
# ============================================================================

class MMGeneratedDataset(Dataset):
    """Unchanged from model_motion_loaders.py."""
    def __init__(self, opt, motion_dataset, w_vectorizer):
        self.opt = opt
        self.dataset = motion_dataset.mm_generated_motion
        self.w_vectorizer = w_vectorizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        mm_motions = data['mm_motions']
        m_lens = []
        motions = []
        for mm_motion in mm_motions:
            m_lens.append(mm_motion['length'])
            motion = mm_motion['motion']
            motion = motion[None, :]
            motions.append(motion)
        m_lens = np.array(m_lens, dtype=np.int64)
        motions = np.concatenate(motions, axis=0)
        sort_indx = np.argsort(m_lens)[::-1].copy()
        m_lens = m_lens[sort_indx]
        motions = motions[sort_indx]
        return motions, m_lens


def collate_fn(batch):
    from torch.utils.data._utils.collate import default_collate
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


def get_audio_mdm_loader(args, model, diffusion, batch_size,
                          ground_truth_loader, mm_num_samples, mm_num_repeats,
                          max_motion_length, num_samples_limit, scale,
                          audio_provider=None):
    """Modified get_mdm_loader that passes audio_provider to dataset."""
    opt = {'name': 'test'}
    print('Generating %s ...' % opt['name'])

    dataset = CompAudioMDMGeneratedDataset(
        args, model, diffusion, ground_truth_loader,
        mm_num_samples, mm_num_repeats,
        max_motion_length, num_samples_limit, scale,
        audio_provider=audio_provider,
    )

    mm_dataset = MMGeneratedDataset(
        opt, dataset, ground_truth_loader.dataset.w_vectorizer
    )

    motion_loader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn,
        drop_last=True, num_workers=4
    )
    mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=1)

    print('Generated Dataset Loading Completed!!!')
    return motion_loader, mm_motion_loader


# ============================================================================
# METRIC FUNCTIONS — identical to eval_humanml.py (do NOT modify)
# ============================================================================

def evaluate_matching_score(eval_wrapper, motion_loaders, file):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    print('========== Evaluating Matching Score ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        all_size = 0
        matching_score_sum = 0
        top_k_count = 0
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _ = batch
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                    word_embs=word_embeddings,
                    pos_ohot=pos_one_hots,
                    cap_lens=sent_lens,
                    motions=motions,
                    m_lens=m_lens
                )
                dist_mat = euclidean_distance_matrix(
                    text_embeddings.cpu().numpy(),
                    motion_embeddings.cpu().numpy()
                )
                matching_score_sum += dist_mat.trace()
                argsmax = np.argsort(dist_mat, axis=1)
                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0)
                all_size += text_embeddings.shape[0]
                all_motion_embeddings.append(motion_embeddings.cpu().numpy())

            all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
            matching_score = matching_score_sum / all_size
            R_precision = top_k_count / all_size
            match_score_dict[motion_loader_name] = matching_score
            R_precision_dict[motion_loader_name] = R_precision
            activation_dict[motion_loader_name] = all_motion_embeddings

        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}')
        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}',
              file=file, flush=True)
        line = f'---> [{motion_loader_name}] R_precision: '
        for j in range(len(R_precision)):
            line += '(top %d): %.4f ' % (j+1, R_precision[j])
        print(line)
        print(line, file=file, flush=True)

    return match_score_dict, R_precision_dict, activation_dict


def evaluate_fid(eval_wrapper, groundtruth_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print('========== Evaluating FID ==========')
    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            _, _, _, sent_lens, motions, m_lens, _ = batch
            motion_embeddings = eval_wrapper.get_motion_embeddings(
                motions=motions, m_lens=m_lens
            )
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        print(f'---> [{model_name}] FID: {fid:.4f}')
        print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
        eval_dict[model_name] = fid
    return eval_dict


def evaluate_diversity(activation_dict, file, diversity_times):
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
        print(f'---> [{model_name}] Diversity: {diversity:.4f}',
              file=file, flush=True)
    return eval_dict


def evaluate_multimodality(eval_wrapper, mm_motion_loaders, file, mm_num_times):
    eval_dict = OrderedDict({})
    print('========== Evaluating MultiModality ==========')
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        with torch.no_grad():
            for idx, batch in enumerate(mm_motion_loader):
                motions, m_lens = batch
                motion_embedings = eval_wrapper.get_motion_embeddings(
                    motions[0], m_lens[0]
                )
                mm_motion_embeddings.append(motion_embedings.unsqueeze(0))
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(
                mm_motion_embeddings, dim=0
            ).cpu().numpy()
            multimodality = calculate_multimodality(
                mm_motion_embeddings, mm_num_times
            )
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}',
              file=file, flush=True)
        eval_dict[model_name] = multimodality
    return eval_dict


def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file,
               replication_times, diversity_times, mm_num_times,
               run_mm=False, eval_platform=None):
    with open(log_file, 'w') as f:
        all_metrics = OrderedDict({
            'Matching Score': OrderedDict({}),
            'R_precision': OrderedDict({}),
            'FID': OrderedDict({}),
            'Diversity': OrderedDict({}),
            'MultiModality': OrderedDict({}),
        })

        for replication in range(replication_times):
            motion_loaders = {}
            mm_motion_loaders = {}
            motion_loaders['ground truth'] = gt_loader
            for name, getter in eval_motion_loaders.items():
                motion_loader, mm_motion_loader = getter()
                motion_loaders[name] = motion_loader
                mm_motion_loaders[name] = mm_motion_loader

            print(f'==================== Replication {replication} '
                  f'====================')
            print(f'==================== Replication {replication} '
                  f'====================', file=f, flush=True)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)

            mat_score_dict, R_precision_dict, acti_dict = \
                evaluate_matching_score(eval_wrapper, motion_loaders, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            fid_score_dict = evaluate_fid(
                eval_wrapper, gt_loader, acti_dict, f
            )

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            div_score_dict = evaluate_diversity(acti_dict, f, diversity_times)

            if run_mm:
                print(f'Time: {datetime.now()}')
                print(f'Time: {datetime.now()}', file=f, flush=True)
                mm_score_dict = evaluate_multimodality(
                    eval_wrapper, mm_motion_loaders, f, mm_num_times
                )

            print(f'!!! DONE !!!')
            print(f'!!! DONE !!!', file=f, flush=True)

            for key, item in mat_score_dict.items():
                if key not in all_metrics['Matching Score']:
                    all_metrics['Matching Score'][key] = [item]
                else:
                    all_metrics['Matching Score'][key] += [item]
            for key, item in R_precision_dict.items():
                if key not in all_metrics['R_precision']:
                    all_metrics['R_precision'][key] = [item]
                else:
                    all_metrics['R_precision'][key] += [item]
            for key, item in fid_score_dict.items():
                if key not in all_metrics['FID']:
                    all_metrics['FID'][key] = [item]
                else:
                    all_metrics['FID'][key] += [item]
            for key, item in div_score_dict.items():
                if key not in all_metrics['Diversity']:
                    all_metrics['Diversity'][key] = [item]
                else:
                    all_metrics['Diversity'][key] += [item]
            if run_mm:
                for key, item in mm_score_dict.items():
                    if key not in all_metrics['MultiModality']:
                        all_metrics['MultiModality'][key] = [item]
                    else:
                        all_metrics['MultiModality'][key] += [item]

        mean_dict = {}
        for metric_name, metric_dict in all_metrics.items():
            print('========== %s Summary ==========' % metric_name)
            print('========== %s Summary ==========' % metric_name,
                  file=f, flush=True)
            for model_name, values in metric_dict.items():
                mean, conf_interval = get_metric_statistics(
                    np.array(values), replication_times
                )
                mean_dict[metric_name + '_' + model_name] = mean
                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    print(f'---> [{model_name}] Mean: {mean:.4f} '
                          f'CInterval: {conf_interval:.4f}')
                    print(f'---> [{model_name}] Mean: {mean:.4f} '
                          f'CInterval: {conf_interval:.4f}',
                          file=f, flush=True)
                elif isinstance(mean, np.ndarray):
                    line = f'---> [{model_name}]'
                    for j in range(len(mean)):
                        line += '(top %d) Mean: %.4f CInt: %.4f;' % (
                            j+1, mean[j], conf_interval[j]
                        )
                    print(line)
                    print(line, file=f, flush=True)

        if eval_platform is not None:
            for k, v in mean_dict.items():
                if k.startswith('R_precision'):
                    for j in range(len(v)):
                        eval_platform.report_scalar(
                            name=f'top{j+1}_' + k, value=v[j],
                            iteration=1, group_name='Eval'
                        )
                else:
                    eval_platform.report_scalar(
                        name=k, value=v,
                        iteration=1, group_name='Eval'
                    )

        return mean_dict


# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def parse_args():
    parser = ArgumentParser(
        description='Evaluate audio-conditioned MDM on HumanML3D benchmarks'
    )

    # Model
    parser.add_argument('--model_path', required=True, type=str,
                        help='Path to Stage 2 audio-conditioned checkpoint')

    # Audio mode
    parser.add_argument('--audio_mode', default='none',
                        choices=['none', 'random_aist', 'fixed'],
                        help='none=text-only, random_aist=cycle AIST++ tracks, '
                             'fixed=single track')
    parser.add_argument('--audio_path', default='', type=str,
                        help='Path to .wav file (for --audio_mode fixed)')
    parser.add_argument('--aist_audio_dir', default='', type=str,
                        help='Directory with AIST++ .wav files '
                             '(for --audio_mode random_aist). '
                             'Default: ./dataset/aist/audio')

    # Guidance
    parser.add_argument('--guidance_param', default=2.5, type=float,
                        help='Text CFG scale (s_text)')
    parser.add_argument('--audio_guidance_param', default=2.5, type=float,
                        help='Audio CFG scale (s_audio). '
                             'Ignored when --audio_mode none.')

    # Eval settings
    parser.add_argument('--eval_mode', default='debug',
                        choices=['debug', 'wo_mm', 'mm_short'],
                        help='debug=5 reps, wo_mm=20 reps, mm_short=5 reps+MM')
    parser.add_argument('--dataset', default='humanml', type=str)
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Must be 32 for correct R-precision')
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--train_platform_type', default='NoPlatform',
                        choices=['NoPlatform', 'ClearmlPlatform',
                                 'TensorboardPlatform', 'WandBPlatform'])

    # Data paths (override if your datasets are not at the default locations)
    parser.add_argument('--humanml_dir', default='', type=str,
                        help='Path to HumanML3D dataset directory. '
                             'If empty, uses the repo default.')

    # Prefix completion (usually unused for this eval)
    parser.add_argument('--context_len', default=0, type=int)
    parser.add_argument('--pred_len', default=0, type=int)
    parser.add_argument('--autoregressive', action='store_true')

    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    args = parse_args()
    fixseed(args.seed)
    args.batch_size = 32  # Must be 32 for R-precision

    # If audio_mode is none, force audio guidance to 0
    if args.audio_mode == 'none':
        args.audio_guidance_param = 0.0

    # Build log file name
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    log_name = f'eval_audio_humanml_{name}_{niter}'
    log_name += f'_tscale{args.guidance_param}'
    log_name += f'_ascale{args.audio_guidance_param}'
    log_name += f'_amode_{args.audio_mode}'
    log_name += f'_{args.eval_mode}'
    log_file = os.path.join(os.path.dirname(args.model_path), log_name + '.log')
    save_dir = os.path.dirname(log_file)

    print(f'\n{"="*70}')
    print(f'  Audio-Conditioned MDM — HumanML3D Evaluation')
    print(f'{"="*70}')
    print(f'  Model:          {args.model_path}')
    print(f'  Audio mode:     {args.audio_mode}')
    print(f'  Text guidance:  {args.guidance_param}')
    print(f'  Audio guidance: {args.audio_guidance_param}')
    print(f'  Eval mode:      {args.eval_mode}')
    if args.humanml_dir:
        print(f'  HumanML3D dir:  {args.humanml_dir}')
    print(f'  Log file:       {log_file}')
    print(f'{"="*70}\n')

    eval_platform_type = eval(args.train_platform_type)
    eval_platform = eval_platform_type(save_dir, name=log_name)

    # Eval mode settings
    if args.eval_mode == 'debug':
        num_samples_limit = 1000
        run_mm = False
        mm_num_samples = 0
        mm_num_repeats = 0
        mm_num_times = 0
        diversity_times = 300
        replication_times = 5
    elif args.eval_mode == 'wo_mm':
        num_samples_limit = 1000
        run_mm = False
        mm_num_samples = 0
        mm_num_repeats = 0
        mm_num_times = 0
        diversity_times = 300
        replication_times = 20
    elif args.eval_mode == 'mm_short':
        num_samples_limit = 1000
        run_mm = True
        mm_num_samples = 100
        mm_num_repeats = 30
        mm_num_times = 10
        diversity_times = 300
        replication_times = 5
    else:
        raise ValueError(f'Unknown eval_mode: {args.eval_mode}')

    dist_util.setup_dist(args.device)
    logger.configure()

    # --- Data loaders (standard HumanML3D, unchanged) ---
    logger.log("Creating data loaders...")
    split = 'test'

    # Build extra kwargs for branches that support humanml_dir
    _extra_loader_kwargs = {}
    if args.humanml_dir:
        _extra_loader_kwargs['humanml_dir'] = args.humanml_dir

    gt_loader = get_dataset_loader(
        name=args.dataset, batch_size=args.batch_size,
        num_frames=None, split=split, hml_mode='gt',
        **_extra_loader_kwargs
    )
    gen_loader = get_dataset_loader(
        name=args.dataset, batch_size=args.batch_size,
        num_frames=None, split=split, hml_mode='eval',
        fixed_len=args.context_len + args.pred_len,
        pred_len=args.pred_len, device=dist_util.dev(),
        autoregressive=args.autoregressive,
        **_extra_loader_kwargs
    )

    # --- Load audio-conditioned model ---
    logger.log("Loading audio-conditioned model...")
    model, ckpt_args = load_audio_mdm(args.model_path, dist_util.dev())
    audio_feat_dim = ckpt_args.get('audio_feat_dim', 519)

    # --- Create diffusion ---
    logger.log("Creating diffusion...")

    class _DiffusionArgs:
        diffusion_steps = 1000
        noise_schedule = 'cosine'
        sigma_small = True
        lambda_vel = 0.0
        lambda_rcxyz = 0.0
        lambda_fc = 0.0
        lambda_target_loc = 0.0

    diffusion = create_gaussian_diffusion(_DiffusionArgs())

    # --- Wrap model with audio CFG ---
    model = AudioCFGSampleModel(
        model,
        text_scale=args.guidance_param,
        audio_scale=args.audio_guidance_param,
    )
    model.to(dist_util.dev())
    model.eval()

    # --- Create audio provider ---
    aist_dir = args.aist_audio_dir or './dataset/aist/audio'
    audio_provider = AudioProvider(
        mode=args.audio_mode,
        audio_feat_dim=audio_feat_dim,
        device=dist_util.dev(),
        audio_path=args.audio_path if args.audio_path else None,
        aist_audio_dir=aist_dir,
        fps=20,
    ) if args.audio_mode != 'none' else None

    # --- Build eval motion loader with audio injection ---
    eval_motion_loaders = {
        'vald': lambda: get_audio_mdm_loader(
            args,
            model=model,
            diffusion=diffusion,
            batch_size=args.batch_size,
            ground_truth_loader=gen_loader,
            mm_num_samples=mm_num_samples,
            mm_num_repeats=mm_num_repeats,
            max_motion_length=gt_loader.dataset.opt.max_motion_length,
            num_samples_limit=num_samples_limit,
            scale=args.guidance_param,
            audio_provider=audio_provider,
        )
    }

    # --- Run evaluation ---
    eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())
    mean_dict = evaluation(
        eval_wrapper, gt_loader, eval_motion_loaders, log_file,
        replication_times, diversity_times, mm_num_times,
        run_mm=run_mm, eval_platform=eval_platform,
    )
    eval_platform.close()

    print(f'\n{"="*70}')
    print(f'  EVALUATION COMPLETE')
    print(f'{"="*70}')
    print(f'  Results saved to: {log_file}')
    print(f'  Audio mode: {args.audio_mode}')
    print(f'  Text guidance: {args.guidance_param}')
    print(f'  Audio guidance: {args.audio_guidance_param}')
    print(f'{"="*70}\n')