# Stage 2: Audio-Conditioned Motion Diffusion — Architecture Deep Dive

## Table of Contents

1. [Overview](#1-overview)
2. [Stage 1 Recap (Pretrained MDM)](#2-stage-1-recap-pretrained-mdm)
3. [Audio Feature Extractors](#3-audio-feature-extractors)
4. [Audio Encoder (1D CNN)](#4-audio-encoder-1d-cnn)
5. [Audio Cross-Attention Transformer](#5-audio-cross-attention-transformer)
6. [Beat-Aware Attention Bias](#6-beat-aware-attention-bias)
7. [Full Forward Pass (MDM with Audio)](#7-full-forward-pass-mdm-with-audio)
8. [MOSPA Variant (Token Concatenation)](#8-mospa-variant-token-concatenation)
9. [Loss Functions](#9-loss-functions)
10. [Classifier-Free Guidance (CFG)](#10-classifier-free-guidance-cfg)
11. [GCDM: Composite Guidance](#11-gcdm-composite-guidance)
12. [SDEdit Refinement Pipeline](#12-sdedit-refinement-pipeline)
13. [Training Procedure](#13-training-procedure)
14. [Stage 2 Variant Summary](#14-stage-2-variant-summary)
15. [Model Diagrams](#15-model-diagrams)

---

## 1. Overview

Stage 2 extends the pretrained **MDM (Motion Diffusion Model)** to support **audio conditioning** for music-driven dance generation. The core idea:

- **Stage 1**: A text-conditioned diffusion model (MDM) trained on HumanML3D — generates motion from text prompts.
- **Stage 2**: Injects audio understanding by adding cross-attention layers inside the transformer. Only the new audio modules are trained; the pretrained text/motion backbone is frozen.

The result is a model that can generate dance motion conditioned on **both** a text description (e.g., "a person performs breakdancing moves") **and** music audio.

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Audio integration | Cross-attention (not concat to embedding) | Preserves per-frame temporal alignment between motion and audio |
| Initialization | Zero-gated residual (`tanh(0) = 0`) | Model starts identical to pretrained MDM, avoids catastrophic forgetting |
| Frozen backbone | Self-attention + FFN frozen | Only ~15% of params trained — efficient, preserves motion quality |
| Beat awareness | Gaussian attention bias + beat emphasis | Encourages motion-audio synchronization at beat positions |
| Diffusion | Cosine schedule, predict x₀ | Inherited from MDM Stage 1 |

---

## 2. Stage 1 Recap (Pretrained MDM)

Before diving into Stage 2, here's the Stage 1 model that we build upon:

```
Input:  x_t ∈ ℝ^(B × 263 × 1 × T)     (noisy HumanML3D motion)
        t ∈ ℤ^B                          (diffusion timestep)
        text ∈ List[str]                 (text prompt)

Output: x̂₀ ∈ ℝ^(B × 263 × 1 × T)      (predicted clean motion)
```

**Stage 1 Architecture:**
- `InputProcess`: Linear(263 → 512) per frame
- `TimestepEmbedder`: sinusoidal position → MLP(512 → 512 → 512)
- `CLIP ViT-B/32`: text → 512-d embedding → Linear(512 → 512)
- Embedding policy: `emb = text_emb + time_emb` (additive)
- 8 × `TransformerEncoderLayer(d=512, heads=4, ff=1024, dropout=0.1, GELU)`
- `OutputProcess`: Linear(512 → 263) per frame

**Sequence layout (Stage 1):**
```
Position:  [ 0     | 1    | 2    | ... | T    ]
Token:     [ emb   | m_1  | m_2  | ... | m_T  ]
           ↑ condition token (time+text embedding)
```

---

## 3. Audio Feature Extractors

We explored three audio feature representations at 20 fps (matching motion frame rate):

### 3.1. Librosa 145-d (`model/audio_features.py`)

| Channel | Dims | Description |
|---------|------|-------------|
| Mel spectrogram | 128 | 128 mel bands, power-to-dB, range ~[-80, 0] |
| Onset strength | 1 | Librosa onset detection |
| Beat indicator | 1 | Binary: 1.0 at detected beat frames |
| RMS energy | 1 | Root mean square energy |
| Chroma | 12 | 12-bin chromagram |
| Spectral centroid | 1 | Normalized by Nyquist |
| Tempo | 1 | Global BPM / 200 (repeated per frame) |
| **Total** | **145** | |

**Beat indicator index**: 129 (0-indexed)

### 3.2. Librosa v2 52-d (`model/audio_features_v2.py`)

Improved version addressing mel spectrogram dominance (was 88% of features → now 62%):

| Channel | Dims | Description |
|---------|------|-------------|
| Mel spectrogram (reduced) | 32 | 32 mel bands, normalized to [0,1] |
| Onset strength | 1 | Normalized by max |
| Onset envelope | 1 | Gaussian-smoothed onset (σ=3 frames, ~150ms) |
| Beat indicator (soft) | 1 | Gaussian around beats (σ=1 frame) instead of binary |
| Beat distance past | 1 | Frames since last beat / avg_beat_period |
| Beat distance future | 1 | Frames until next beat / avg_beat_period |
| RMS energy | 1 | Normalized by max |
| Chroma | 12 | Chromagram (already [0,1]) |
| Spectral centroid | 1 | Normalized by Nyquist |
| Tempo | 1 | BPM / 200 |
| **Total** | **52** | |

Key improvements: beat distance features (inspired by Beat-It, ECCV 2024), all features normalized to [0,1], soft beat indicator.

### 3.3. Wav2CLIP + Librosa 519-d (`model/audio_features_wav2clip.py`)

Combines semantic audio understanding with explicit rhythmic features:

| Channel | Dims | Description |
|---------|------|-------------|
| Wav2CLIP embedding | 512 | Frame-level CLIP-aligned audio embeddings (0.5s window, 20fps hop) |
| Onset strength | 1 | Librosa onset |
| Beat indicator | 1 | Binary beat |
| Beat distance past | 1 | Seconds to last beat |
| Beat distance future | 1 | Seconds to next beat |
| RMS energy | 1 | Energy envelope |
| Spectral centroid | 1 | Normalized |
| Tempo | 1 | BPM / 200 |
| **Total** | **519** | |

**Beat indicator index**: 513 (512 + 1, 0-indexed)

**Wav2CLIP**: A pretrained model (from Lyrebird) that embeds audio into the same space as CLIP visual embeddings. Provides semantic understanding of audio content (genre, instrumentation, mood) beyond low-level spectral features.

---

## 4. Audio Encoder (1D CNN)

**File**: `model/audio_encoder.py`

The audio encoder projects frame-level audio features into the transformer's latent space using a 3-layer 1D CNN.

```
Input:  (B, T_audio, D_audio)     e.g., (32, 196, 519)
Output: (B, T_audio, 512)         projected to transformer dimension
```

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   AudioEncoder (1D CNN)                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Input: (B, T, D_audio) ──transpose──▸ (B, D_audio, T)  │
│                                                          │
│  ┌─────────────────────────────────────────────────┐     │
│  │ Conv1d(D_audio → 256, k=5, pad=2)               │     │
│  │ BatchNorm1d(256)                                 │     │
│  │ GELU                                             │     │
│  │ Dropout(0.1)                                     │     │
│  └─────────────────────────────────────────────────┘     │
│                         ↓                                │
│  ┌─────────────────────────────────────────────────┐     │
│  │ Conv1d(256 → 512, k=5, pad=2)                    │     │
│  │ BatchNorm1d(512)                                  │     │
│  │ GELU                                              │     │
│  │ Dropout(0.1)                                      │     │
│  └─────────────────────────────────────────────────┘     │
│                         ↓                                │
│  ┌─────────────────────────────────────────────────┐     │
│  │ Conv1d(512 → 512, k=5, pad=2)                    │     │
│  │ BatchNorm1d(512)                                  │     │
│  │ GELU                                              │     │
│  └─────────────────────────────────────────────────┘     │
│                                                          │
│  Output: (B, 512, T) ──transpose──▸ (B, T, 512)         │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Receptive field**: kernel=5, 3 layers → 13 frames = 0.65s at 20fps. Each output frame has context from roughly ±0.3s of surrounding audio, capturing local beat/onset context.

**Note**: `padding=2` with `kernel_size=5` preserves temporal dimension (T_in = T_out). The CNN does not downsample — there is a 1:1 correspondence between input and output frames.

---

## 5. Audio Cross-Attention Transformer

**File**: `model/audio_cross_attention_v2.py`

The standard `nn.TransformerEncoder` is replaced with `AudioCondTransformerEncoder` — a custom encoder where each layer has an additional cross-attention block for audio.

### Single Layer: `AudioCondTransformerEncoderLayer`

Each of the 8 layers follows this pattern:

```
┌──────────────────────────────────────────────────────────────┐
│           AudioCondTransformerEncoderLayer                     │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Input: src (1+T, B, 512)     audio_memory (T_audio, B, 512) │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  1. SELF-ATTENTION (frozen from Stage 1)              │     │
│  │     Q, K, V = src                                     │     │
│  │     src = src + Dropout(SelfAttn(Q, K, V))            │     │
│  │     src = LayerNorm(src)                              │     │
│  └──────────────────────────────────────────────────────┘     │
│                          ↓                                    │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  2. AUDIO CROSS-ATTENTION (new, trainable)            │     │
│  │     Q = src (motion tokens)                           │     │
│  │     K, V = audio_memory (projected audio features)    │     │
│  │                                                       │     │
│  │     attn_bias = temporal_gaussian + beat_emphasis      │     │
│  │     cross_out = CrossAttn(Q, K, V, attn_mask=bias)    │     │
│  │                                                       │     │
│  │     src = src + Dropout(tanh(gate) × cross_out)       │     │
│  │     src = LayerNorm(src)          ↑                   │     │
│  │                            starts at 0                │     │
│  └──────────────────────────────────────────────────────┘     │
│                          ↓                                    │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  3. FEED-FORWARD NETWORK (frozen from Stage 1)        │     │
│  │     src = src + Dropout(Linear₂(Dropout(GELU(         │     │
│  │                   Linear₁(src)))))                    │     │
│  │     src = LayerNorm(src)                              │     │
│  │     Linear₁: 512 → 1024, Linear₂: 1024 → 512        │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                               │
│  Output: src (1+T, B, 512)                                    │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Gated Residual Connection

The cross-attention output is gated by a learnable scalar parameter:

```python
self.cross_attn_gate = nn.Parameter(torch.zeros(1))  # initialized to 0
src = src + dropout(tanh(gate) * cross_out)
```

**At initialization**: `tanh(0) = 0`, so cross-attention has zero effect. The model starts functionally identical to the pretrained Stage 1 MDM. During training, the gate gradually opens as the cross-attention layers learn useful audio-motion correspondences.

### Weight Loading

The `load_pretrained_weights()` method copies from a standard `nn.TransformerEncoder`:

| Submodule | Source | Trainable in Stage 2? |
|-----------|--------|-----------------------|
| `self_attn` (Q,K,V,out projections) | Pretrained MDM | **No** (frozen) |
| `norm1` (post-self-attn LayerNorm) | Pretrained MDM | **No** (frozen) |
| `cross_attn` (Q,K,V,out projections) | Random init | **Yes** |
| `norm_cross` (post-cross-attn LayerNorm) | Random init | **Yes** |
| `cross_attn_gate` | Zeros | **Yes** |
| `linear1` (FFN up-projection) | Pretrained MDM | **No** (frozen) |
| `linear2` (FFN down-projection) | Pretrained MDM | **No** (frozen) |
| `norm2` (post-FFN LayerNorm) | Pretrained MDM | **No** (frozen) |

---

## 6. Beat-Aware Attention Bias

**File**: `model/audio_cross_attention_v2.py`

The cross-attention uses an additive bias on the attention logits (before softmax) to encourage temporal locality and beat synchronization.

### 6.1. Temporal Gaussian Bias

Each motion frame `i` should attend most strongly to its temporally corresponding audio frame, with attention falling off for distant frames:

```
bias(i, j) = -(audio_pos(i) - j)² / (2σ²)
```

where:
- `audio_pos(i) = i × (T_audio - 1) / (T_motion - 1)` — linear time mapping
- `σ = 4.0` frames (default) ≈ 200ms at 20fps
- Shape: `(T_motion, T_audio)`

This creates a diagonal band of high attention along the temporal correspondence line. Frames far from the diagonal get strongly negative bias → near-zero attention weight.

```
             Audio frames →
           0    5   10   15   20
Motion  0 [████░░░░░░░░░░░░░░░░]
frames  5 [░░░████░░░░░░░░░░░░░]
↓      10 [░░░░░░░████░░░░░░░░░]
       15 [░░░░░░░░░░░░████░░░░]
       20 [░░░░░░░░░░░░░░░░████]

       ████ = high attention (Gaussian peak)
       ░░░░ = suppressed attention
```

### 6.2. Beat Emphasis Bias

An additional 1D bias that boosts attention to audio frames at detected beat positions:

```python
for each beat frame b:
    bias[b]     += beat_weight      # default: 2.0
    bias[b-1]   += beat_weight × 0.5  # neighbor boost
    bias[b+1]   += beat_weight × 0.5  # neighbor boost
```

This is broadcast across all motion frames → every motion token pays extra attention to beat-aligned audio frames.

### 6.3. Combined Bias

```python
attn_mask = temporal_bias + beat_bias.unsqueeze(0)  # (T_motion, T_audio)
attn_mask = attn_mask.unsqueeze(0).expand(B * nhead, -1, -1)  # multi-head
```

The combined bias is passed as `attn_mask` to `nn.MultiheadAttention`, which adds it to the raw attention logits before softmax.

---

## 7. Full Forward Pass (MDM with Audio)

**File**: `model/mdm.py`

Here is the complete data flow through the Stage 2 model:

```
INPUTS
──────
x_t:              (B, 263, 1, T)       noisy motion at diffusion step t
timesteps:        (B,)                 integer diffusion timestep
y['text']:        List[str], len=B     text prompts
y['audio_features']: (B, T, D_audio)  frame-level audio features
y['beat_frames']: List[int]            detected beat frame indices
y['mask']:        (B, 1, 1, T)        padding mask


FORWARD PASS
────────────

1. TIME EMBEDDING
   timesteps → sinusoidal_pos_enc → MLP(512→512→512)
   → time_emb: (1, B, 512)

2. TEXT EMBEDDING
   texts → CLIP ViT-B/32 (frozen) → (1, B, 512)
   → text_cond_dropout (p=0.10–0.15 during training)
   → Linear(512→512)
   → text_emb: (1, B, 512)

3. CONDITION TOKEN
   emb = text_emb + time_emb    # (1, B, 512)

4. AUDIO ENCODING
   y['audio_features']           # (B, T, D_audio)
   → AudioEncoder (1D CNN)       # (B, T, 512)
   → permute to seq-first        # (T, B, 512)
   → audio_cond_dropout (p=0.15 during training)
   → audio_memory: (T, B, 512)

5. MOTION EMBEDDING
   x_t: (B, 263, 1, T)
   → InputProcess: permute + Linear(263→512)
   → x: (T, B, 512)

6. SEQUENCE CONSTRUCTION
   xseq = concat([emb, x], dim=0)  # (1+T, B, 512)
   xseq = PositionalEncoding(xseq) # add sinusoidal pos

7. TRANSFORMER (8 layers)
   For each layer:
     a. Self-Attention(xseq, xseq, xseq)        [frozen]
     b. Cross-Attention(xseq, audio_memory,      [trainable]
                        attn_mask=temporal+beat)
     c. FFN(xseq)                                [frozen]

8. OUTPUT
   output = xseq[1:]              # drop condition token → (T, B, 512)
   → OutputProcess: Linear(512→263) + reshape
   → x̂₀: (B, 263, 1, T)         predicted clean motion
```

---

## 8. MOSPA Variant (Token Concatenation)

When `use_audio_token_concat=True`, instead of (or in addition to) cross-attention, audio tokens are **concatenated** with motion tokens in the sequence dimension:

```
Standard:
  xseq = [emb | m_1 | m_2 | ... | m_T]                    (1+T tokens)
  audio_memory is only accessed via cross-attention

MOSPA:
  xseq = [emb | m_1 | m_2 | ... | m_T | a_1 | a_2 | ... | a_T']  (1+T+T' tokens)
  audio tokens participate in self-attention with motion tokens
  cross-attention also still runs on audio_memory
  output = xseq[1 : 1+T]   (slice out only motion tokens)
```

This allows audio to influence motion through both self-attention (global mixing) and cross-attention (local temporal alignment). The padding mask is extended to mark audio tokens as valid.

---

## 9. Loss Functions

### 9.1. Primary: Masked L2 on x₀ Prediction

The model is trained with the standard DDPM loss using x₀-prediction (not ε-prediction):

```
L_main = masked_l2(x̂₀, x₀, mask)
```

where:
- `x₀` is the ground-truth clean motion
- `x̂₀` is the model's prediction of x₀ from noised input x_t
- `mask` is the padding mask (handles variable-length sequences)
- `masked_l2` computes mean squared error only over valid (non-padded) frames

### 9.2. Diffusion Process

| Parameter | Value |
|-----------|-------|
| Diffusion steps | 1000 |
| Noise schedule | Cosine |
| Model mean type | `START_X` (predict x₀) |
| Model var type | Fixed small |
| Timestep sampling | Uniform |

### 9.3. Auxiliary Losses (Stage 2)

All auxiliary loss lambdas are set to **0.0** in Stage 2 training by default:

| Loss | Lambda | Description |
|------|--------|-------------|
| `rot_mse` | 1.0 | Primary: masked L2 on 263-d HumanML3D repr |
| `vel_mse` | 0.0 | Velocity consistency (disabled) |
| `rcxyz_mse` | 0.0 | Joint position in XYZ (disabled) |
| `fc` | 0.0 | Foot contact (disabled) |

### 9.4. Optional Physics Losses

When `--use_physics_losses` is enabled:

```
L_total = L_main + λ_contact × L_contact + λ_penetrate × L_penetrate + λ_skating × L_skating
```

| Loss | Default λ | Description |
|------|-----------|-------------|
| Contact | 0.5 | Ground contact consistency |
| Penetration | 1.0 | Ground penetration penalty |
| Skating | 0.5 | Foot sliding penalty |

### 9.5. Condition Dropout (for CFG training)

During training, conditions are randomly dropped to enable classifier-free guidance at inference:

| Dropout | Probability | Effect |
|---------|-------------|--------|
| Text dropout | 10–15% | Text embedding zeroed → model learns unconditional motion |
| Audio dropout | 15% | Audio memory zeroed → model learns text-only motion |
| Joint dropout (GCDM) | 5% | Both text AND audio zeroed simultaneously |

The joint dropout is essential for GCDM (see Section 11) — it teaches the model a fully unconditional mode, enabling the four-way decomposition at inference.

---

## 10. Classifier-Free Guidance (CFG)

### 10.1. Legacy Audio CFG (Decomposed)

**File**: `sample/refine_with_audio.py` → `AudioCFGModel`

Three forward passes:
```
ε̂ = ε(∅, ∅)                                              # unconditional
    + s_text  × (ε(text, ∅)    - ε(∅, ∅))                 # text direction
    + s_audio × (ε(text, audio) - ε(text, ∅))             # audio direction
```

This decomposes guidance into independent text and audio axes, allowing separate control of text fidelity and audio responsiveness.

### 10.2. Text-Only CFG

**File**: `sample/refine_with_audio.py` → `TextOnlyCFGModel`

For the initial text-only generation in the SDEdit pipeline:
```
ε̂ = ε(∅) + s_text × (ε(text) - ε(∅))
```

Both text and audio are masked for the unconditional pass.

---

## 11. GCDM: Composite Guidance

**File**: `sample/gcdm.py`

GCDM (Generalized Composite Diffusion Models, ECCV 2024) provides a principled framework for multi-conditional guidance with **timestep-dependent** mixing:

### Formula

```
ε̂ = ε(∅) + α × [
    λ(t) × (ε(text, audio) - ε(∅))                        # joint term
  + (1 - λ(t)) × (
        β_text  × (ε(text)  - ε(∅))                       # text-only direction
      + β_audio × (ε(audio) - ε(∅))                       # audio-only direction
    )
]
```

### Parameters

| Parameter | Default | Role |
|-----------|---------|------|
| α | 3.0 | Overall guidance strength |
| β_text | 1.0 | Text guidance weight (in independent term) |
| β_audio | 1.5 | Audio guidance weight (in independent term) |
| λ_start | 0.8 | Lambda at t=T (early denoising = high noise) |
| λ_end | 0.2 | Lambda at t=0 (late denoising = low noise) |

### Timestep-Dependent Schedule

```
λ(t) = λ_start × (t/T) + λ_end × (1 - t/T)
```

- **Early denoising** (high t, high noise): λ ≈ 0.8 → joint term dominates. The model establishes global structure (pose, trajectory) using the full text+audio signal.
- **Late denoising** (low t, low noise): λ ≈ 0.2 → independent terms dominate. Audio separately refines beat alignment and rhythmic details.

### Four Forward Passes

| Pass | Text | Audio | Name |
|------|------|-------|------|
| 1 | ✓ | ✓ | Fully conditioned: `ε(text, audio)` |
| 2 | ✓ | ✗ | Text only: `ε(text)` |
| 3 | ✗ | ✓ | Audio only: `ε(audio)` |
| 4 | ✗ | ✗ | Unconditional: `ε(∅)` |

This requires 4× the compute of a single forward pass but provides superior quality and control compared to the 3-pass decomposed CFG.

---

## 12. SDEdit Refinement Pipeline

**File**: `sample/refine_with_audio.py`

An alternative to direct audio-conditioned generation. Instead of generating dance from scratch (which requires the model to compose text+audio distributions), SDEdit uses a two-phase approach:

### Phase 1: Text-Only Generation
```
z ~ N(0, I)
  → DDPM sampling with text-only CFG (1000 steps)
  → x₀^text    (clean text-only motion)
```

### Phase 2: Audio Refinement
```
x₀^text
  → q(x_t | x₀^text) at t = T - skip_timesteps    (add partial noise)
  → DDPM sampling with text+audio CFG (T-skip steps)
  → x₀^refined  (audio-aligned motion)
```

### Skip Timesteps Trade-off

| skip_timesteps | Noise Level | Effect |
|----------------|-------------|--------|
| 800 | 20% noise | Subtle rhythmic nudges, text fully preserved |
| 500 | 50% noise | Clear rhythmic modulation, text mostly preserved |
| 200 | 80% noise | Strong audio influence, text structure may degrade |

```
Text-only motion:   ──────────────────────────▸  smooth walking

                         ┌─── add noise ───┐
                         │   (skip=500)     │
                         ▼                  │
Noised motion:      ~~~~~~~~~~~~~~~~~~~~~~~~   50% corrupted

                         │
                         ▼ denoise with audio
                         │
Refined motion:     ─∿─∿─∿─∿─∿─∿─∿─∿─∿──▸  rhythmic walking
                     ↑   ↑   ↑   ↑   ↑
                     beat beat beat beat beat
```

---

## 13. Training Procedure

**File**: `train/train_audio.py`

### Step-by-Step

1. **Load pretrained MDM** — `model000200000.pt` (Stage 1, trained on HumanML3D)
2. **Build MDM with `audio_conditioning=True`** — creates `AudioCondTransformerEncoder` with cross-attention layers
3. **Load weights** — copy self-attention + FFN weights from pretrained checkpoint; cross-attention layers start randomly initialized (with zero gate)
4. **Freeze backbone** — `model.freeze_non_audio()` sets `requires_grad=False` on all params except audio encoder, cross-attention, and gate
5. **Train on AIST++** — paired (motion, audio) data with genre-based text labels

### Trainable Parameters

| Module | Parameters | % of Total |
|--------|------------|------------|
| AudioEncoder (3-layer CNN) | ~1.3M | ~2.5% |
| Cross-attention Q,K,V,out × 8 layers | ~8.4M | ~16% |
| Cross-attention LayerNorm × 8 layers | ~8K | <0.1% |
| Cross-attention gates × 8 layers | 8 | <0.01% |
| **Total trainable** | **~9.7M** | **~18%** |
| Total model parameters | ~54M | 100% |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| LR schedule | Cosine annealing → 1e-6 |
| Batch size | 32 |
| Training steps | 100,000 |
| Gradient clipping | max_norm=1.0 |
| Max motion length | 196 frames (9.8s) |
| Save interval | 10,000 steps |

### Dataset: AIST++

| Property | Value |
|----------|-------|
| Format | SMPL → HumanML3D 263-d |
| Audio | 10 genres of dance music |
| Normalization | HumanML3D Mean/Std |
| Text labels | Genre-based: "a person performs {genre} dance moves to music" |
| Motion FPS | 20 |
| Split | crossmodal_train / val / test |

### Training Loop Pseudocode

```python
for (motion, cond) in aist_loader:
    # motion: (B, 263, 1, T) normalized
    # cond: {text, audio_features, mask, lengths}

    # Extract beat frames from audio features for beat-aware bias
    beat_idx = 513 if wav2clip else 129
    cond['beat_frames'] = where(audio[:, beat_idx] > 0.5)

    # GCDM joint dropout (5% chance)
    if random() < 0.05:
        cond['uncond'] = True
        cond['uncond_audio'] = True

    # Sample random timestep
    t ~ Uniform(0, 999)

    # Forward diffusion: add noise
    x_t = sqrt(α̅_t) × motion + sqrt(1 - α̅_t) × ε

    # Model predicts x₀
    x̂₀ = model(x_t, t, y=cond)

    # Loss
    loss = masked_l2(x̂₀, motion, mask)
    loss.backward()
    optimizer.step()
```

---

## 14. Stage 2 Variant Summary

| Variant | Audio Features | Beat-Aware | MOSPA | GCDM | Script |
|---------|----------------|------------|-------|------|--------|
| **Librosa** | 145-d | v1 (no bias) | No | No | — |
| **Librosa v2** | 52-d | Yes | No | No | `train_audio_v2.py` |
| **Wav2CLIP** | 519-d | No | No | No | `resume_stage2.sh` |
| **Wav2CLIP + Beat-aware** | 519-d | Yes | No | Yes | `train_stage2_wav2clip_beataware.sh` |
| **Wav2CLIP + MOSPA** | 519-d | No | Yes | No | `train_stage2_wav2clip_mospa.sh` |

### Variant: `train_audio_v2.py`

Additional changes:
- `--unfreeze_top_n`: Optionally unfreezes the top N backbone layers' self-attention + FFN (default: 2)
- Separate `audio_lr` and `backbone_lr` param groups
- Logs cross-attention gate values during training

---

## 15. Model Diagrams

### 15.1. Full Stage 2 Pipeline

```
                          ┌─────────────────────────────────┐
                          │        AIST++ Dataset            │
                          │   (motion + audio + genre text)  │
                          └──────┬──────────────┬───────────┘
                                 │              │
                          motion │              │ audio
                          (T,263)│              │ (T, D_audio)
                                 │              │
                          ┌──────▼──────┐  ┌────▼─────────────┐
                          │  Normalize   │  │  AudioEncoder     │
                          │  (HumanML3D  │  │  (1D CNN)         │
                          │   Mean/Std)  │  │  D_audio → 512    │
                          └──────┬──────┘  └────┬─────────────┘
                                 │              │
                          (B,263,1,T)    (T, B, 512)
                                 │              │
                     ┌───────────▼──────┐       │ audio_memory
                     │   Diffusion      │       │
                     │   q(x_t|x_0)     │       │
                     │   add noise at t │       │
                     └───────────┬──────┘       │
                                 │              │
                          x_t    │              │
                    (B,263,1,T)  │              │
                                 │              │
        ┌────────────────────────▼──────────────▼─────────────────┐
        │                                                          │
        │                  MDM (Stage 2)                           │
        │                                                          │
        │   ┌─────────┐  ┌──────────┐  ┌──────────────┐           │
        │   │ Timestep │  │   CLIP   │  │ InputProcess │           │
        │   │ Embedder │  │ Text Enc │  │ Linear       │           │
        │   └────┬────┘  └────┬─────┘  │ (263→512)    │           │
        │        │            │         └──────┬───────┘           │
        │   (1,B,512)    (1,B,512)        (T,B,512)               │
        │        │            │               │                    │
        │        └─────┬──────┘               │                    │
        │              │ add                  │                    │
        │              ▼                      │                    │
        │         emb (1,B,512)               │                    │
        │              │                      │                    │
        │              └────────┬─────────────┘                    │
        │                       │ concat                           │
        │                       ▼                                  │
        │              xseq (1+T, B, 512)                          │
        │                       │                                  │
        │                       ▼                                  │
        │              PositionalEncoding                           │
        │                       │                                  │
        │         ┌─────────────▼──────────────────┐               │
        │         │                                │               │
        │         │   ×8 Transformer Layers        │               │
        │         │                                │               │
        │         │   ┌──────────────────┐         │               │
        │         │   │  Self-Attention   │ FROZEN  │               │
        │         │   │  + LayerNorm      │         │               │
        │         │   └────────┬─────────┘         │               │
        │         │            │                   │               │
        │         │   ┌────────▼─────────┐         │               │
        │         │   │ Audio Cross-Attn  │◄── audio_memory        │
        │         │   │ + Beat Bias       │ TRAINABLE              │
        │         │   │ + Gated Residual  │         │               │
        │         │   │ + LayerNorm       │         │               │
        │         │   └────────┬─────────┘         │               │
        │         │            │                   │               │
        │         │   ┌────────▼─────────┐         │               │
        │         │   │  Feed-Forward     │ FROZEN  │               │
        │         │   │  512→1024→512     │         │               │
        │         │   │  + LayerNorm      │         │               │
        │         │   └────────┬─────────┘         │               │
        │         │            │                   │               │
        │         └────────────┼───────────────────┘               │
        │                      │                                   │
        │                      ▼                                   │
        │             output[1:] (T, B, 512)                       │
        │                      │                                   │
        │              ┌───────▼────────┐                          │
        │              │ OutputProcess   │                          │
        │              │ Linear(512→263) │                          │
        │              └───────┬────────┘                          │
        │                      │                                   │
        └──────────────────────┼───────────────────────────────────┘
                               │
                               ▼
                        x̂₀ (B, 263, 1, T)
                        predicted clean motion
```

### 15.2. Beat-Aware Cross-Attention Detail

```
  Motion tokens                          Audio memory
  (1+T, B, 512)                         (T_audio, B, 512)
       │                                      │
       ├──── Q = W_q × src ──────┐            │
       │                          │            ├──── K = W_k × audio
       │                          │            │
       │                          │            ├──── V = W_v × audio
       │                          │            │
       │                          ▼            ▼
       │                    ┌─────────────────────┐
       │                    │   Attention Logits   │
       │                    │   A = Q × K^T / √d   │
       │                    └──────────┬──────────┘
       │                               │
       │                    ┌──────────▼──────────┐
       │                    │   + Temporal Bias    │
       │                    │   (Gaussian, σ=4)    │
       │                    │                      │
       │                    │   + Beat Bias        │
       │                    │   (weight=2.0 at     │
       │                    │    beat frames)      │
       │                    └──────────┬──────────┘
       │                               │
       │                    ┌──────────▼──────────┐
       │                    │   Softmax            │
       │                    └──────────┬──────────┘
       │                               │
       │                    ┌──────────▼──────────┐
       │                    │   × V                │
       │                    │   cross_out           │
       │                    └──────────┬──────────┘
       │                               │
       │                    ┌──────────▼──────────┐
       │                    │ × tanh(gate)         │
       │                    │ gate initialized = 0 │
       │                    └──────────┬──────────┘
       │                               │
       └───────── + (residual) ◄───────┘
                       │
                       ▼
                  LayerNorm
                       │
                       ▼
                  output (1+T, B, 512)
```

### 15.3. GCDM Inference (4-Pass Guidance)

```
                    x_t, t
                      │
          ┌───────────┼───────────┬───────────┐
          │           │           │           │
          ▼           ▼           ▼           ▼
     ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
     │ text ✓  │ │ text ✓  │ │ text ✗  │ │ text ✗  │
     │ audio ✓ │ │ audio ✗ │ │ audio ✓ │ │ audio ✗ │
     └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
          │           │           │           │
          ▼           ▼           ▼           ▼
       ε(t,a)      ε(t)       ε(a)        ε(∅)
          │           │           │           │
          └─────┬─────┴─────┬─────┘           │
                │           │                 │
                ▼           ▼                 │
          joint_term   indep_term             │
         ε(t,a)-ε(∅)  β_t(ε(t)-ε(∅))        │
                       +β_a(ε(a)-ε(∅))       │
                │           │                 │
                ▼           ▼                 │
           × λ(t)    × (1-λ(t))              │
                │           │                 │
                └─────┬─────┘                 │
                      │                       │
                   × α                        │
                      │                       │
                      └───────── + ◄──────────┘
                                │
                                ▼
                         ε̂ (guided output)
```

### 15.4. SDEdit Refinement Flow

```
              Text prompt                    Audio file
            "walks forward                   music.wav
             and waves"                         │
                │                               │
                ▼                               ▼
     ┌─────────────────────┐          ┌──────────────────┐
     │  Phase 1: Text-Only │          │ Audio Features    │
     │  DDPM (1000 steps)  │          │ (Wav2CLIP+librosa)│
     │  with text CFG      │          └────────┬─────────┘
     └──────────┬──────────┘                   │
                │                               │
                ▼                               │
         x₀^text (clean                        │
         text-only motion)                      │
                │                               │
                ▼                               │
     ┌──────────────────────┐                   │
     │  Add noise to level  │                   │
     │  t = T - skip        │                   │
     │  (e.g., skip=500     │                   │
     │   → 50% noise)       │                   │
     └──────────┬───────────┘                   │
                │                               │
                ▼                               │
         x_t (partially                         │
         noised motion)                         │
                │                               │
                ▼                               ▼
     ┌──────────────────────────────────────────────────┐
     │  Phase 2: Audio-Conditioned DDPM                  │
     │  Denoise from t = T-skip → 0                     │
     │  with text+audio CFG                              │
     │  (only T-skip steps, not full 1000)               │
     └──────────────────────┬───────────────────────────┘
                            │
                            ▼
                    x₀^refined
               (rhythmic, text-faithful
                motion output)
```

---

## Appendix: File Reference

| File | Purpose |
|------|---------|
| `model/mdm.py` | Main MDM model with audio conditioning support |
| `model/audio_encoder.py` | 1D CNN for audio feature projection |
| `model/audio_cross_attention.py` | Cross-attention v1 (no beat bias) |
| `model/audio_cross_attention_v2.py` | Cross-attention v2 (beat-aware bias) |
| `model/audio_features.py` | Librosa 145-d feature extraction |
| `model/audio_features_v2.py` | Librosa v2 52-d feature extraction |
| `model/audio_features_wav2clip.py` | Wav2CLIP 512-d + librosa 7-d = 519-d |
| `train/train_audio.py` | Stage 2 training script |
| `train/train_audio_v2.py` | Stage 2 v2 training (unfreeze top layers) |
| `sample/refine_with_audio.py` | SDEdit refinement pipeline |
| `sample/gcdm.py` | GCDM composite guidance |
| `sample/generate_audio.py` | Direct audio-conditioned generation |
| `data/aist_dataset.py` | AIST++ dataset and dataloader |
| `diffusion/gaussian_diffusion.py` | Diffusion training losses and sampling |
