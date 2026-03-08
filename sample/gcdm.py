"""
GCDM: Composite Classifier-Free Guidance for Multi-Modal Diffusion.

Implements the GCDM formulation (ECCV 2024) for principled multi-conditional
generation with text + audio (+ implicit style via DoRA):

    eps_tilde = eps(empty) + alpha * [
        lambda(t) * (eps(text,audio) - eps(empty))
      + (1 - lambda(t)) * (beta_text * (eps(text) - eps(empty))
                         + beta_audio * (eps(audio) - eps(empty)))
    ]

lambda(t) is timestep-dependent:
  - Early denoising (high t): lambda high (joint term dominates,
    global structure + style character established)
  - Late denoising (low t): lambda low (independent terms dominate,
    audio refines beat alignment)

Requires model trained with independent condition dropout
(text ~15%, audio ~15%) AND joint dropout (~5%).
"""

import torch
import torch.nn as nn
from copy import deepcopy


class GCDMSampleModel(nn.Module):
    """
    Drop-in replacement for AudioCFGSampleModel that uses GCDM composite
    guidance with timestep-dependent condition weighting.

    Four forward passes per denoising step:
        1. eps(text, audio)   fully conditioned
        2. eps(text)          audio masked
        3. eps(audio)         text masked
        4. eps(empty)         both masked

    Style is implicit (DoRA weights baked in) and always active.
    """

    def __init__(self, model, alpha=3.0, beta_text=1.0, beta_audio=1.5,
                 lambda_start=0.8, lambda_end=0.2, diffusion_steps=1000):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.beta_text = beta_text
        self.beta_audio = beta_audio
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
        self.diffusion_steps = diffusion_steps

        self.rot2xyz = model.rot2xyz
        self.translation = model.translation
        self.njoints = model.njoints
        self.nfeats = model.nfeats
        self.data_rep = model.data_rep
        self.cond_mode = model.cond_mode
        self.encode_text = model.encode_text

    def _lambda_t(self, timesteps):
        """Linear schedule: lambda_start at t=T, lambda_end at t=0."""
        t_frac = timesteps.float() / self.diffusion_steps
        return self.lambda_start * t_frac + self.lambda_end * (1.0 - t_frac)

    def forward(self, x, timesteps, y=None):
        has_audio = (
            getattr(self.model, 'audio_conditioning', False)
            and y is not None
            and 'audio_features' in y
            and y['audio_features'] is not None
        )

        if has_audio:
            out_full = self.model(x, timesteps, y)

            y_text = deepcopy(y)
            y_text['uncond_audio'] = True
            out_text = self.model(x, timesteps, y_text)

            y_audio = deepcopy(y)
            y_audio['uncond'] = True
            out_audio = self.model(x, timesteps, y_audio)

            y_uncond = deepcopy(y)
            y_uncond['uncond'] = True
            y_uncond['uncond_audio'] = True
            out_uncond = self.model(x, timesteps, y_uncond)

            lam = self._lambda_t(timesteps).view(-1, 1, 1, 1)

            joint_term = out_full - out_uncond
            indep_term = (self.beta_text * (out_text - out_uncond)
                          + self.beta_audio * (out_audio - out_uncond))

            return out_uncond + self.alpha * (
                lam * joint_term + (1.0 - lam) * indep_term
            )

        else:
            out = self.model(x, timesteps, y)
            y_uncond = deepcopy(y)
            y_uncond['uncond'] = True
            out_uncond = self.model(x, timesteps, y_uncond)
            return out_uncond + self.alpha * (out - out_uncond)
