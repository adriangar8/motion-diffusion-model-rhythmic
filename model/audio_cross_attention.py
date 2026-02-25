####################################################################################[start]
####################################################################################[start]

"""

Custom Transformer Encoder with Audio Cross-Attention.

Replaces nn.TransformerEncoder in MDM when audio conditioning is enabled.
Each block: Self-Attention → LN → Audio Cross-Attention → LN → FFN → LN

The self-attention and FFN match the original TransformerEncoderLayer behavior,
so pretrained weights load directly into the corresponding submodules.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class AudioCondTransformerEncoderLayer(nn.Module):

    """

    A single transformer block with:

      1. Self-attention (weights loadable from pretrained MDM)
      2. Audio cross-attention (new, randomly initialized)
      3. Feed-forward network (weights loadable from pretrained MDM)

    Matches the PyTorch TransformerEncoderLayer interface for self-attention + FFN,
    with the cross-attention inserted in between.
    
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "gelu"):
        
        super().__init__()

        # -- self-attention (same as nn.TransformerEncoderLayer) --
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # -- audio cross-attention (NEW) --
        
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.norm_cross = nn.LayerNorm(d_model)
        self.dropout_cross = nn.Dropout(dropout)

        # -- gating parameter: starts at 0 so cross-attention has no effect initially --
        # -- this ensures the model behaves identically to pretrained MDM at initialization --
        
        self.cross_attn_gate = nn.Parameter(torch.zeros(1))

        # -- feed-forward network (same as nn.TransformerEncoderLayer) --
        
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
                audio_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        """
        
        Args:
            src: (seq_len, batch_size, d_model) — motion tokens + condition token
            audio_memory: (T_audio, batch_size, d_model) — projected audio features
                          If None, cross-attention is skipped (text-only mode)
            src_key_padding_mask: (batch_size, seq_len) — padding mask for self-attention
            audio_key_padding_mask: (batch_size, T_audio) — padding mask for audio

        Returns:
            output: (seq_len, batch_size, d_model)
        
        """
        
        # -- 1. self-attention (pre-norm style matching PyTorch's TransformerEncoderLayer) --
        
        src2 = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # -- 2. Audio cross-attention (only if audio is provided) --
        
        if audio_memory is not None:
        
            # -- queries: motion tokens, Keys/Values: audio features --
            
            cross_out = self.cross_attn(
                query=src,
                key=audio_memory,
                value=audio_memory,
                key_padding_mask=audio_key_padding_mask
            )[0]
            
            # -- gated residual: gate starts at 0, so initially this has no effect --
            
            src = src + self.dropout_cross(torch.tanh(self.cross_attn_gate) * cross_out)
            src = self.norm_cross(src)

        # -- 3. feed-forward network --
        
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)

        return src

class AudioCondTransformerEncoder(nn.Module):

    """

    Stack of AudioCondTransformerEncoderLayers.
    Drop-in replacement for nn.TransformerEncoder when audio conditioning is used.

    """

    def __init__(self, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "gelu"):
        
        super().__init__()
        
        self.layers = nn.ModuleList([
            AudioCondTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation
            )
            for _ in range(num_layers)
        ])
        
        self.num_layers = num_layers

    def forward(self, src: torch.Tensor,
                audio_memory: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                audio_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        """
        
        Args:
        
            src: (seq_len, batch_size, d_model)
            audio_memory: (T_audio, batch_size, d_model) or None
            src_key_padding_mask: (batch_size, seq_len) or None
            audio_key_padding_mask: (batch_size, T_audio) or None

        Returns:
            output: (seq_len, batch_size, d_model)
        
        """
        
        output = src
        
        for layer in self.layers:
        
            output = layer(output,
                           audio_memory=audio_memory,
                           src_key_padding_mask=src_key_padding_mask,
                           audio_key_padding_mask=audio_key_padding_mask)
        
        return output

    def load_pretrained_weights(self, pretrained_encoder: nn.TransformerEncoder):
        
        """
        
        Load weights from a pretrained nn.TransformerEncoder into the self-attention
        and FFN submodules. Cross-attention weights remain randomly initialized.

        This is the key function that lets us start from pretrained MDM and only
        train the new cross-attention layers.
        
        """
        
        for our_layer, pretrained_layer in zip(self.layers, pretrained_encoder.layers):
        
            # -- self-attention weights --
            
            our_layer.self_attn.load_state_dict(pretrained_layer.self_attn.state_dict())
            our_layer.norm1.load_state_dict(pretrained_layer.norm1.state_dict())

            # -- FFN weights --
            
            our_layer.linear1.load_state_dict(pretrained_layer.linear1.state_dict())
            our_layer.linear2.load_state_dict(pretrained_layer.linear2.state_dict())
            our_layer.norm2.load_state_dict(pretrained_layer.norm2.state_dict())

        print(f"Loaded pretrained weights into {self.num_layers} layers "
              f"(self-attn + FFN). Cross-attention layers are freshly initialized.")
        
####################################################################################[end]
####################################################################################[end]
