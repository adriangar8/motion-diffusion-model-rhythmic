"""
Evaluate audio-conditioned MDM variants on HumanML3D using standard metrics:
FID, R-Precision, Matching Score, Diversity, MultiModality.

This is a modified version of eval/eval_humanml.py that loads Stage 2 audio
models (wav2clip, beataware, mospa, etc.) and wraps them with an audio-aware
CFG sampler, while sending null (silent) audio features so that the model
runs in text-only mode on the HumanML3D benchmark.

Usage:
    python -m eval.eval_audio_humanml \
        --model_path ./save/audio_stage2_wav2clip/model_final.pt \
        --eval_mode debug \
        --guidance_param 2.5 \
        --audio_guidance_param 0.0
"""

import os
import sys
import json
import torch
import numpy as np
from argparse import ArgumentParser
from datetime import datetime
from collections import OrderedDict

from utils.fixseed import fixseed
from data_loaders.humanml.motion_loaders.model_motion_loaders import get_mdm_loader
from data_loaders.humanml.utils.metrics import *
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from data_loaders.humanml.scripts.motion_process import *
from data_loaders.humanml.utils.utils import *
from diffusion import logger
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_gaussian_diffusion
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform, WandBPlatform

torch.multiprocessing.set_sharing_strategy('file_system')

# ---------------------------------------------------------------------------
# Audio model loading (mirrors sample/refine_with_audio.py::load_model)
# ---------------------------------------------------------------------------

def load_audio_mdm(model_path, device):
    """Load an audio-conditioned MDM checkpoint and its args dict."""
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

    print(f"Audio MDM loaded from {model_path}")
    print(f"  audio_feat_dim={ckpt_args.get('audio_feat_dim', 519)}, "
          f"use_wav2clip={ckpt_args.get('use_wav2clip', False)}, "
          f"use_audio_token_concat={ckpt_args.get('use_audio_token_concat', False)}")
    return model, ckpt_args

# ---------------------------------------------------------------------------
# Null audio feature provider (text-only evaluation with audio architecture)
# ---------------------------------------------------------------------------

AUDIO_FEAT_DIM = 519  # 512 Wav2CLIP + 7 librosa

def get_null_audio_features(batch_size, max_frames, device):
    """Return silent/null audio tensors so the model sees no audio signal.

    Returns:
        audio_features: (batch_size, max_frames, 519) all zeros
        audio_mask:     (batch_size,) all False -- tells the model every
                        sample in the batch has no valid audio
    """
    audio_features = torch.zeros(batch_size, max_frames, AUDIO_FEAT_DIM, device=device)
    audio_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    return audio_features, audio_mask

# ---------------------------------------------------------------------------
# Audio classifier-free guidance wrapper
# ---------------------------------------------------------------------------

# TODO: AudioClassifierFreeSampleModel should live in utils/sampler_util.py
#       alongside ClassifierFreeSampleModel.  For now we import the version
#       that already exists in sample/generate_audio.py.
from sample.generate_audio import AudioCFGSampleModel as AudioClassifierFreeSampleModel

# ---------------------------------------------------------------------------
# Metric functions -- identical to eval/eval_humanml.py, do NOT modify
# ---------------------------------------------------------------------------

def evaluate_matching_score(eval_wrapper, motion_loaders, file):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    print('========== Evaluating Matching Score ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        score_list = []
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
                dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(),
                                                     motion_embeddings.cpu().numpy())
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
        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}', file=file, flush=True)

        line = f'---> [{motion_loader_name}] R_precision: '
        for i in range(len(R_precision)):
            line += '(top %d): %.4f ' % (i+1, R_precision[i])
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
                motions=motions,
                m_lens=m_lens
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
        print(f'---> [{model_name}] Diversity: {diversity:.4f}', file=file, flush=True)
    return eval_dict


def evaluate_multimodality(eval_wrapper, mm_motion_loaders, file, mm_num_times):
    eval_dict = OrderedDict({})
    print('========== Evaluating MultiModality ==========')
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        with torch.no_grad():
            for idx, batch in enumerate(mm_motion_loader):
                motions, m_lens = batch
                motion_embedings = eval_wrapper.get_motion_embeddings(motions[0], m_lens[0])
                mm_motion_embeddings.append(motion_embedings.unsqueeze(0))
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}', file=file, flush=True)
        eval_dict[model_name] = multimodality
    return eval_dict


def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file, replication_times,
               diversity_times, mm_num_times, run_mm=False, eval_platform=None):
    with open(log_file, 'w') as f:
        all_metrics = OrderedDict({'Matching Score': OrderedDict({}),
                                   'R_precision': OrderedDict({}),
                                   'FID': OrderedDict({}),
                                   'Diversity': OrderedDict({}),
                                   'MultiModality': OrderedDict({})})
        for replication in range(replication_times):
            motion_loaders = {}
            mm_motion_loaders = {}
            motion_loaders['ground truth'] = gt_loader
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                motion_loader, mm_motion_loader = motion_loader_getter()
                motion_loaders[motion_loader_name] = motion_loader
                mm_motion_loaders[motion_loader_name] = mm_motion_loader

            print(f'==================== Replication {replication} ====================')
            print(f'==================== Replication {replication} ====================', file=f, flush=True)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(eval_wrapper, motion_loaders, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            fid_score_dict = evaluate_fid(eval_wrapper, gt_loader, acti_dict, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            div_score_dict = evaluate_diversity(acti_dict, f, diversity_times)

            if run_mm:
                print(f'Time: {datetime.now()}')
                print(f'Time: {datetime.now()}', file=f, flush=True)
                mm_score_dict = evaluate_multimodality(eval_wrapper, mm_motion_loaders, f, mm_num_times)

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
            print('========== %s Summary ==========' % metric_name, file=f, flush=True)
            for model_name, values in metric_dict.items():
                mean, conf_interval = get_metric_statistics(np.array(values), replication_times)
                mean_dict[metric_name + '_' + model_name] = mean
                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}')
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}', file=f, flush=True)
                elif isinstance(mean, np.ndarray):
                    line = f'---> [{model_name}]'
                    for i in range(len(mean)):
                        line += '(top %d) Mean: %.4f CInt: %.4f;' % (i+1, mean[i], conf_interval[i])
                    print(line)
                    print(line, file=f, flush=True)

        if eval_platform is not None:
            for k, v in mean_dict.items():
                if k.startswith('R_precision'):
                    for i in range(len(v)):
                        eval_platform.report_scalar(name=f'top{i + 1}_' + k, value=v[i],
                                                            iteration=1, group_name='Eval')
                else:
                    eval_platform.report_scalar(name=k, value=v, iteration=1, group_name='Eval')

        return mean_dict


# ---------------------------------------------------------------------------
# Argument parser (audio-specific, replaces evaluation_parser)
# ---------------------------------------------------------------------------

def audio_evaluation_parser():
    parser = ArgumentParser(description='Evaluate audio-conditioned MDM on HumanML3D metrics')

    parser.add_argument('--model_path', required=True, type=str,
                        help='Path to Stage 2 model checkpoint (e.g. model_final.pt)')
    parser.add_argument('--eval_mode', default='debug', choices=['wo_mm', 'mm_short', 'debug'],
                        type=str, help='debug (5 reps), wo_mm (20 reps), mm_short (5 reps + MM)')
    parser.add_argument('--dataset', default='humanml', choices=['humanml', 'kit'], type=str)
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Must be 32 for correct R-precision calculation')
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--train_platform_type', default='NoPlatform',
                        choices=['NoPlatform', 'ClearmlPlatform', 'TensorboardPlatform', 'WandBPlatform'])

    parser.add_argument('--guidance_param', default=2.5, type=float,
                        help='Text classifier-free guidance scale (s_text)')
    parser.add_argument('--audio_guidance_param', default=0.0, type=float,
                        help='Audio classifier-free guidance scale (s_audio). '
                             '0.0 = text-only evaluation (recommended for HumanML3D benchmark)')

    parser.add_argument('--context_len', default=0, type=int)
    parser.add_argument('--pred_len', default=0, type=int)
    parser.add_argument('--autoregressive', action='store_true')

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args = audio_evaluation_parser()
    fixseed(args.seed)
    args.batch_size = 32  # Must be 32 for R-precision

    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    log_name = f'eval_audio_humanml_{name}_{niter}'
    if args.guidance_param != 1.:
        log_name += f'_tscale{args.guidance_param}'
    if args.audio_guidance_param != 0.:
        log_name += f'_ascale{args.audio_guidance_param}'
    log_name += f'_{args.eval_mode}'
    log_file = os.path.join(os.path.dirname(args.model_path), log_name + '.log')
    save_dir = os.path.dirname(log_file)

    print(f'Will save to log file [{log_file}]')

    eval_platform_type = eval(args.train_platform_type)
    eval_platform = eval_platform_type(save_dir, name=log_name)

    print(f'Eval mode [{args.eval_mode}]')
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

    # -- Data loaders (standard HumanML3D, same as eval_humanml.py) --
    logger.log("creating data loader...")
    split = 'test'
    gt_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size,
                                   num_frames=None, split=split, hml_mode='gt')
    gen_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size,
                                   num_frames=None, split=split, hml_mode='eval',
                                   fixed_len=args.context_len + args.pred_len,
                                   pred_len=args.pred_len, device=dist_util.dev(),
                                   autoregressive=args.autoregressive)

    # -- Load audio-conditioned model --
    logger.log("Loading audio-conditioned model...")
    model, ckpt_args = load_audio_mdm(args.model_path, dist_util.dev())

    # -- Create diffusion --
    logger.log("Creating diffusion...")

    class _DiffusionArgs:
        """Minimal namespace for create_gaussian_diffusion."""
        diffusion_steps = 1000
        noise_schedule = 'cosine'
        sigma_small = True
        lambda_vel = 0.0
        lambda_rcxyz = 0.0
        lambda_fc = 0.0
        lambda_target_loc = 0.0

    diffusion = create_gaussian_diffusion(_DiffusionArgs())

    # -- Wrap with audio CFG sampler --
    if args.guidance_param != 1 or args.audio_guidance_param != 0:
        model = AudioClassifierFreeSampleModel(
            model,
            text_scale=args.guidance_param,
            audio_scale=args.audio_guidance_param,
        )
    model.to(dist_util.dev())
    model.eval()

    # -- Build eval motion loader --
    # TODO: get_mdm_loader (in model_motion_loaders.py) needs to be modified
    #       separately to accept audio_features_fn and audio_scale kwargs, and
    #       inject y['audio_features'] and y['audio_mask'] into each batch's
    #       conditioning dict inside CompMDMGeneratedDataset.__init__.
    #       Until then, the model will not receive audio features and the audio
    #       branch will see whatever default the model's forward() uses for
    #       missing audio (typically zeros / unconditional).
    eval_motion_loaders = {
        'vald': lambda: get_mdm_loader(
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
            # TODO: pass these once get_mdm_loader supports them:
            # audio_features_fn=get_null_audio_features,
            # audio_scale=args.audio_guidance_param,
        )
    }

    eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())
    evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file, replication_times,
               diversity_times, mm_num_times, run_mm=run_mm, eval_platform=eval_platform)
    eval_platform.close()

    print(f'\nResults saved to [{log_file}]')
