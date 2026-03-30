"""
DoRA (Weight-Decomposed Low-Rank Adaptation) for self-attention Q and V.

DoRA decomposes weight W into magnitude m and direction V, then applies
low-rank update to direction only: W' = m * (V + B*A) / ||V + B*A||_F.
Ref: Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation," ICML 2024.

Used in Stage 3 style adapters on Q and V projections of transformer self-attention only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


def decompose_weight(W: torch.Tensor):
    """Decompose W into magnitude (scalar) and direction (normalized matrix)."""
    m = W.norm(p='fro')
    if m < 1e-8:
        m = torch.ones_like(m, device=W.device, dtype=W.dtype)
        V = W
    else:
        V = W / m
    return m, V


class DoRALinear(nn.Module):
    """
    DoRA wrapper for a linear projection: W_eff = m * (V + B@A) / ||V + B@A||_F.
    Original W is decomposed into m (magnitude) and V (direction).
    m, B, A are trainable; V (direction) stays frozen.
    """

    def __init__(self, in_features: int, out_features: int, rank: int, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, device: Optional[torch.device] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        dev = device or weight.device
        assert weight.shape == (out_features, in_features)
        w = weight.detach().float().to(dev)
        m, V = decompose_weight(w)
        # Trainable magnitude (scaling factor m in DoRA)
        self.magnitude = nn.Parameter(m.detach().clone().to(dev))
        self.register_buffer('direction', V)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        if bias is not None:
            self.register_buffer('bias', bias.detach().to(dev))
        else:
            self.register_buffer('bias', torch.zeros(out_features, device=dev))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features). Ensure all operands on x.device for correct backward.
        dev = x.device
        magnitude = self.magnitude.to(dev)  # Parameter; .to(dev) for device safety
        direction = self.direction.to(dev)
        bias = self.bias.to(dev)
        delta = self.lora_B @ self.lora_A  # (out_features, in_features)
        V_updated = direction + delta
        V_updated_norm = V_updated / (V_updated.norm(p='fro', keepdim=True) + 1e-8)
        W_eff = magnitude * V_updated_norm
        out = F.linear(x, W_eff, bias)
        return out


class DoRAMultiheadAttention(nn.Module):
    """
    MultiheadAttention with DoRA applied to Q and V projections only (K stays frozen).
    Wraps nn.MultiheadAttention and replaces in_proj for Q and V with DoRA versions.
    """

    @classmethod
    def from_mha(cls, mha: nn.MultiheadAttention, rank: int = 8, alpha: float = 16.0, device: Optional[torch.device] = None) -> 'DoRAMultiheadAttention':
        """Build DoRA MHA from an existing nn.MultiheadAttention (copies weights into DoRA)."""
        drop = getattr(mha.dropout, 'p', mha.dropout) if hasattr(mha.dropout, 'p') else 0.0
        out = cls(embed_dim=mha.embed_dim, num_heads=mha.num_heads, dropout=drop, rank=rank, alpha=alpha)
        out._mha.load_state_dict(mha.state_dict())
        dev = device if device is not None else next(mha.parameters()).device
        out._init_dora(rank, alpha, device=dev)
        return out

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Placeholder MHA; real weights set in from_mha or _init_dora
        self._mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self._init_dora(rank, alpha)

    def _init_dora(self, rank: int, alpha: float, device: Optional[torch.device] = None):
        W = self._mha.in_proj_weight  # (3*embed_dim, embed_dim)
        d = self.embed_dim
        dev = device or W.device
        W = W.to(dev)
        q_bias = self._mha.in_proj_bias[:d].clone().to(dev) if self._mha.in_proj_bias is not None else None
        v_bias = self._mha.in_proj_bias[2*d:].clone().to(dev) if self._mha.in_proj_bias is not None else None
        self.dora_q = DoRALinear(self.embed_dim, self.embed_dim, rank, W[:d].clone(), q_bias, device=dev)
        self.dora_v = DoRALinear(self.embed_dim, self.embed_dim, rank, W[2*d:].clone(), v_bias, device=dev)
        self.register_buffer('_k_weight', W[d:2*d].clone())
        if self._mha.in_proj_bias is not None:
            self.register_buffer('_k_bias', self._mha.in_proj_bias[d:2*d].clone().to(dev))
        else:
            self.register_buffer('_k_bias', torch.zeros(d, device=dev))
        with torch.no_grad():
            self.out_proj.weight.copy_(self._mha.out_proj.weight.to(dev))
            if self._mha.out_proj.bias is not None:
                self.out_proj.bias.copy_(self._mha.out_proj.bias.to(dev))
        self.out_proj.weight.requires_grad = False
        if self.out_proj.bias is not None:
            self.out_proj.bias.requires_grad = False

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None):
        # query, key, value: (T, B, embed_dim)
        T, B, _ = query.shape
        dev = query.device
        q = self.dora_q(query)   # (T, B, embed_dim)
        k = F.linear(query, self._k_weight.to(dev), self._k_bias.to(dev))  # K from query for self-attn
        v = self.dora_v(query)
        q = q.reshape(T, B * self.num_heads, self.head_dim).transpose(0, 1)  # (B*nhead, T, head_dim)
        k = k.reshape(T, B * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.reshape(T, B * self.num_heads, self.head_dim).transpose(0, 1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn = attn + attn_mask
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(0, 1).reshape(T, B, self.embed_dim)
        out = self.out_proj(out)
        return out, None
