####################################################################################[start]
####################################################################################[start]

"""

Custom Transformer Encoder with Audio Cross-Attention v2.

Changes from v1:
  - Beat-aware attention bias: temporal locality Gaussian centered on each
    motion frame's corresponding audio frame, with extra weight at beat positions.
  - Configurable temporal_sigma for the locality window.

The bias is added to cross-attention logits before softmax, so:
  - Each motion frame attends most strongly to nearby audio frames
  - Beat frames get extra attention weight
  - Distant audio frames are suppressed but not hard-masked

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

def build_temporal_bias(T_motion, T_audio, sigma=4.0, device='cpu'):

    """

    Build a Gaussian temporal locality bias for cross-attention.

    Each motion frame i attends to audio frames with a Gaussian centered at
    the temporally corresponding audio frame.

    Args:
        T_motion: number of motion frames (seq_len including condition token)
        T_audio: number of audio frames
        sigma: standard deviation in frames (~200ms at sigma=4, 20fps)

    Returns:
        bias: (T_motion, T_audio) — add to attention logits

    """

    # -- motion frame indices (skip first token = condition token) --
    
    motion_idx = torch.arange(T_motion, device=device, dtype=torch.float32)
    audio_idx = torch.arange(T_audio, device=device, dtype=torch.float32)

    if T_motion > 1 and T_audio > 1:
        scale = (T_audio - 1) / (T_motion - 1)
    else:
        scale = 1.0

    # -- corresponding audio position for each motion position --
    
    audio_pos = motion_idx * scale  # (T_motion,)

    # -- gaussian: -(motion_pos - audio_idx)^2 / (2*sigma^2) --
    
    diff = audio_pos.unsqueeze(1) - audio_idx.unsqueeze(0) # (T_motion, T_audio)
    bias = -diff.pow(2) / (2 * sigma ** 2)

    return bias

def build_beat_bias(T_audio, beat_frames, beat_weight=2.0, device='cpu'):

    """

    Build a beat emphasis bias for cross-attention.

    Adds extra weight to audio frames that are beat positions.

    Args:
        T_audio: number of audio frames
        beat_frames: list/array of beat frame indices
        beat_weight: extra logit added at beat positions

    Returns:
        bias: (T_audio,) — broadcast over T_motion dimension

    """

    bias = torch.zeros(T_audio, device=device)

    for b in beat_frames:

        if 0 <= b < T_audio:

            bias[b] = beat_weight

            # -- also slight boost to neighbors --

            if b > 0:
                bias[b - 1] = max(bias[b - 1], beat_weight * 0.5)
            if b < T_audio - 1:
                bias[b + 1] = max(bias[b + 1], beat_weight * 0.5)
                
    return bias

class AudioCondTransformerEncoderLayer(nn.Module):

    """

    Transformer block with self-attention, beat-aware audio cross-attention, and FFN.

    The cross-attention uses an attention bias that combines:
      1. Temporal locality (Gaussian centered on corresponding time)
      2. Beat emphasis (extra weight at detected beat positions)

    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "gelu",
                 temporal_sigma: float = 4.0, beat_weight: float = 2.0):

        super().__init__()

        self.nhead = nhead
        self.temporal_sigma = temporal_sigma
        self.beat_weight = beat_weight

        # -- self-attention (same as nn.TransformerEncoderLayer) --
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # -- audio cross-attention --
        
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm_cross = nn.LayerNorm(d_model)
        self.dropout_cross = nn.Dropout(dropout)

        # -- gating: starts at 0 for identity at init --
        
        self.cross_attn_gate = nn.Parameter(torch.zeros(1))

        # -- feed-forward network --
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, src: torch.Tensor,
                audio_memory: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                audio_key_padding_mask: Optional[torch.Tensor] = None,
                beat_frames: Optional[list] = None) -> torch.Tensor:
        
        """
        
        Args:
            src: (seq_len, batch_size, d_model)
            audio_memory: (T_audio, batch_size, d_model) or None
            src_key_padding_mask: (batch_size, seq_len) or None
            audio_key_padding_mask: (batch_size, T_audio) or None
            beat_frames: list of beat frame indices for beat-aware masking

        Returns:
            output: (seq_len, batch_size, d_model)
        
        """
        
        # -- 1. self-attention --
        
        src2 = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # -- 2. audio cross-attention with beat-aware bias --
        
        if audio_memory is not None:

            T_motion = src.shape[0]
            T_audio = audio_memory.shape[0]
            B = src.shape[1]

            # -- build attention bias --
            
            attn_bias = build_temporal_bias(
                T_motion, T_audio,
                sigma=self.temporal_sigma,
                device=src.device
            ) # (T_motion, T_audio)

            # -- add beat emphasis --
            
            if beat_frames is not None:
                beat_bias = build_beat_bias(
                    T_audio, beat_frames,
                    beat_weight=self.beat_weight,
                    device=src.device
                ) # (T_audio,)
                attn_bias = attn_bias + beat_bias.unsqueeze(0) # broadcast

            # -- expand for multi-head: (B*nhead, T_motion, T_audio) --
            
            attn_mask = attn_bias.unsqueeze(0).expand(B * self.nhead, -1, -1)

            cross_out = self.cross_attn(
                query=src,
                key=audio_memory,
                value=audio_memory,
                attn_mask=attn_mask,
                key_padding_mask=audio_key_padding_mask
            )[0]

            src = src + self.dropout_cross(torch.tanh(self.cross_attn_gate) * cross_out)
            src = self.norm_cross(src)

        # -- 3. FFN --
        
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)

        return src

class AudioCondTransformerEncoder(nn.Module):

    """

    Stack of AudioCondTransformerEncoderLayers.
    Drop-in replacement for nn.TransformerEncoder.

    """

    def __init__(self, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "gelu",
                 temporal_sigma: float = 4.0, beat_weight: float = 2.0):

        super().__init__()

        self.layers = nn.ModuleList([
            AudioCondTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                temporal_sigma=temporal_sigma,
                beat_weight=beat_weight,
            )
            for _ in range(num_layers)
        ])

        self.num_layers = num_layers

    def forward(self, src: torch.Tensor,
                audio_memory: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                audio_key_padding_mask: Optional[torch.Tensor] = None,
                beat_frames: Optional[list] = None) -> torch.Tensor:

        output = src

        for layer in self.layers:

            output = layer(output,
                           audio_memory=audio_memory,
                           src_key_padding_mask=src_key_padding_mask,
                           audio_key_padding_mask=audio_key_padding_mask,
                           beat_frames=beat_frames)

        return output

    def load_pretrained_weights(self, pretrained_encoder: nn.TransformerEncoder):

        """Load weights from pretrained nn.TransformerEncoder."""

        for our_layer, pretrained_layer in zip(self.layers, pretrained_encoder.layers):
            our_layer.self_attn.load_state_dict(pretrained_layer.self_attn.state_dict())
            our_layer.norm1.load_state_dict(pretrained_layer.norm1.state_dict())
            our_layer.linear1.load_state_dict(pretrained_layer.linear1.state_dict())
            our_layer.linear2.load_state_dict(pretrained_layer.linear2.state_dict())
            our_layer.norm2.load_state_dict(pretrained_layer.norm2.state_dict())

        print(f"Loaded pretrained weights into {self.num_layers} layers "
              f"(self-attn + FFN). Cross-attention layers are freshly initialized.")
        
####################################################################################[end]
####################################################################################[end]
