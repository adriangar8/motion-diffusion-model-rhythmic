"""
Precise parameter counting for all model variants.
Counts parameters per component (audio_encoder, cross_attn, self_attn, FFN, etc.)
for each audio_feat_dim variant.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from model.audio_encoder import AudioEncoder
from model.audio_cross_attention_v2 import AudioCondTransformerEncoderLayer

LATENT_DIM = 512
NUM_HEADS = 4
FF_SIZE = 1024
NUM_LAYERS = 8
DROPOUT = 0.1

def count(module):
    return sum(p.numel() for p in module.parameters())

def count_trainable(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

print("=" * 80)
print("EXACT PARAMETER COUNTS")
print("=" * 80)

# --- 1. Audio Encoder variants ---
for name, feat_dim in [("Librosa-only (52-dim)", 52),
                        ("Wav2CLIP + 7d Librosa (519-dim)", 519)]:
    enc = AudioEncoder(audio_feat_dim=feat_dim, latent_dim=LATENT_DIM, dropout=DROPOUT)
    
    # break down per conv layer
    conv_layers = list(enc.conv_layers.children())
    total = 0
    print(f"\n--- Audio Encoder: {name} ---")
    for i, layer in enumerate(conv_layers):
        n = count(layer)
        if n > 0:
            print(f"  Layer {i} ({layer.__class__.__name__}): {n:>10,}")
            total += n
    print(f"  TOTAL audio_encoder: {count(enc):>10,}")

# --- 2. Cross-attention per layer ---
print(f"\n--- Cross-Attention per Transformer Layer ---")
layer = AudioCondTransformerEncoderLayer(
    d_model=LATENT_DIM, nhead=NUM_HEADS,
    dim_feedforward=FF_SIZE, dropout=DROPOUT, activation="gelu"
)

cross_attn_params = count(layer.cross_attn)
norm_cross_params = count(layer.norm_cross)
gate_params = layer.cross_attn_gate.numel()
cross_total_per_layer = cross_attn_params + norm_cross_params + gate_params

print(f"  cross_attn (MHA):   {cross_attn_params:>10,}")
print(f"  norm_cross (LN):    {norm_cross_params:>10,}")
print(f"  cross_attn_gate:    {gate_params:>10,}")
print(f"  Cross-attn total/layer: {cross_total_per_layer:>10,}")
print(f"  Cross-attn total × {NUM_LAYERS} layers: {cross_total_per_layer * NUM_LAYERS:>10,}")

# --- 3. Self-attention + FFN per layer (backbone) ---
self_attn_params = count(layer.self_attn)
norm1_params = count(layer.norm1)
ffn_params = count(layer.linear1) + count(layer.linear2)
norm2_params = count(layer.norm2)
backbone_per_layer = self_attn_params + norm1_params + ffn_params + norm2_params

print(f"\n--- Backbone (Self-Attn + FFN) per Transformer Layer ---")
print(f"  self_attn (MHA):    {self_attn_params:>10,}")
print(f"  norm1 (LN):         {norm1_params:>10,}")
print(f"  linear1 + linear2 (FFN): {ffn_params:>10,}")
print(f"  norm2 (LN):         {norm2_params:>10,}")
print(f"  Backbone total/layer:    {backbone_per_layer:>10,}")
print(f"  Backbone total × {NUM_LAYERS} layers: {backbone_per_layer * NUM_LAYERS:>10,}")

# --- 4. Full layer ---
full_per_layer = count(layer)
print(f"\n  Full layer params:  {full_per_layer:>10,}")
print(f"  Full × {NUM_LAYERS} layers:    {full_per_layer * NUM_LAYERS:>10,}")

# --- 5. Other MDM components (input_process, output_process, embed_timestep, embed_text, pos_enc) ---
print(f"\n--- Other MDM components (approximate) ---")

njoints, nfeats = 263, 1
input_feats = njoints * nfeats

input_process = nn.Linear(input_feats, LATENT_DIM)
output_process = nn.Linear(LATENT_DIM, input_feats)
embed_text = nn.Linear(512, LATENT_DIM)

# TimestepEmbedder: 2 Linear layers (latent_dim -> latent_dim)
ts_fc1 = nn.Linear(LATENT_DIM, LATENT_DIM)
ts_fc2 = nn.Linear(LATENT_DIM, LATENT_DIM)

print(f"  input_process (Linear {input_feats}->{LATENT_DIM}):  {count(input_process):>10,}")
print(f"  output_process (Linear {LATENT_DIM}->{input_feats}): {count(output_process):>10,}")
print(f"  embed_text (Linear 512->{LATENT_DIM}):    {count(embed_text):>10,}")
print(f"  timestep_embed (2×Linear {LATENT_DIM}->{LATENT_DIM}): {count(ts_fc1)+count(ts_fc2):>10,}")

other_total = count(input_process) + count(output_process) + count(embed_text) + count(ts_fc1) + count(ts_fc2)
print(f"  Other total (excl CLIP): {other_total:>10,}")

# --- 6. CLIP ViT-B/32 ---
import clip
clip_model, _ = clip.load("ViT-B/32", device="cpu", jit=False)
clip_params = sum(p.numel() for p in clip_model.parameters())
print(f"\n--- CLIP ViT-B/32 ---")
print(f"  CLIP total:         {clip_params:>10,}")

# --- 7. Grand total for each variant ---
print(f"\n{'='*80}")
print("GRAND TOTAL SUMMARIES")
print(f"{'='*80}")

# Base MDM (no audio)
base_backbone = backbone_per_layer * NUM_LAYERS  # using standard TransformerEncoderLayer
base_mdm_no_clip = other_total + base_backbone
base_mdm_total = base_mdm_no_clip + clip_params
print(f"\nBase MDM (no audio conditioning):")
print(f"  Backbone (8 layers self-attn+FFN):  {base_backbone:>12,}")
print(f"  Other (input/output/text/timestep):  {other_total:>12,}")
print(f"  MDM excl. CLIP:                      {base_mdm_no_clip:>12,}")
print(f"  CLIP ViT-B/32 (frozen):              {clip_params:>12,}")
print(f"  TOTAL:                               {base_mdm_total:>12,}")

for name, feat_dim in [("Librosa-only", 52), ("Wav2CLIP variants", 519)]:
    enc = AudioEncoder(audio_feat_dim=feat_dim, latent_dim=LATENT_DIM, dropout=DROPOUT)
    enc_params = count(enc)
    cross_total = cross_total_per_layer * NUM_LAYERS
    new_trainable = enc_params + cross_total
    
    # The AudioCondTransformerEncoderLayer replaces TransformerEncoderLayer,
    # so backbone self-attn+FFN params are same; cross_attn is added ON TOP
    full_backbone = (backbone_per_layer + cross_total_per_layer) * NUM_LAYERS
    total_no_clip = other_total + full_backbone + enc_params
    total = total_no_clip + clip_params
    
    print(f"\n{name} (audio_feat_dim={feat_dim}):")
    print(f"  Audio encoder:                       {enc_params:>12,}")
    print(f"  Cross-attn (8 layers):               {cross_total:>12,}")
    print(f"  NEW trainable total:                 {new_trainable:>12,}")
    print(f"  Backbone self-attn+FFN (frozen):     {base_backbone:>12,}")
    print(f"  Other (frozen):                      {other_total:>12,}")
    print(f"  CLIP ViT-B/32 (frozen):              {clip_params:>12,}")
    print(f"  TOTAL in checkpoint:                 {total:>12,}")
    pct = 100.0 * new_trainable / total
    print(f"  Trainable %:                         {pct:>11.1f}%")
