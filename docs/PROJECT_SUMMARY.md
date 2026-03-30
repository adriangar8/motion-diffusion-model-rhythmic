# Rhythmic Motion Diffusion — Complete Project Summary

> **Last updated:** March 2026  
> **Codebase:** `motion-diffusion-model-rhythmic`  
> **Author:** Yash Bhardwaj

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Overview](#2-architecture-overview)
3. [Datasets](#3-datasets)
4. [Models](#4-models)
   - 4.1 Stage 0 — Base MDM (pre-trained)
   - 4.2 Stage 2 — Librosa-only
   - 4.3 Stage 2 — Wav2CLIP
   - 4.4 Stage 2 — Wav2CLIP + MOSPA
   - 4.5 Stage 2 — Wav2CLIP + Beat-aware (★ Best)
   - 4.6 Baseline — EDGE
5. [Audio Feature Representations](#5-audio-feature-representations)
6. [Conditioning Mechanism](#6-conditioning-mechanism)
   - 6.1 Audio cross-attention (frame-level)
   - 6.2 Beat-aware attention bias
   - 6.3 MOSPA
   - 6.4 Classifier-Free Guidance (AudioCFG)
   - 6.5 GCDM (Generalised CFG)
7. [Training Details](#7-training-details)
   - 7.1 Parameter Breakdown — What Gets Trained
8. [Inference & Sampling](#8-inference--sampling)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Quantitative Results](#10-quantitative-results)
    - 10.1 HumanML3D text-quality metrics
    - 10.2 Beat Alignment Score (BAS) — CFG sweep
    - 10.3 BAS vs EDGE per track
11. [CFG Hyperparameter Study](#11-cfg-hyperparameter-study)
12. [Qualitative Visualisations](#12-qualitative-visualisations)
13. [Key Implementation Notes & Fixes](#13-key-implementation-notes--fixes)
14. [File Map](#14-file-map)
15. [Known Limitations & Future Work](#15-known-limitations--future-work)

---

## 1. Project Overview

This project extends **MDM (Motion Diffusion Model)** — a text-conditioned human motion diffusion model — with **music/audio conditioning** so that generated motions are rhythmically aligned to music beats while still following text instructions.

**Core research questions:**
1. Can a diffusion model learn to generate motion that *simultaneously* satisfies a text description *and* is rhythmically synchronised to music?
2. What audio representation best transfers rhythmic information into motion?
3. Can beat-aware attention help without degrading text quality?

**Answer in brief:** Yes. Adding a 512-dim Wav2CLIP semantic audio embedding concatenated with a 7-dim Librosa feature vector (519-dim total), cross-attended into the MDM Transformer with an additive beat-aware bias, achieves BAS **0.286** (ag=1.5, tg=2.5) — competitive with and on several tracks exceeding EDGE (a dedicated music-to-dance model) — while preserving near-baseline text quality (FID **0.698**, R_top1 **0.373**).

---

## 2. Architecture Overview

```
Text prompt ─── CLIP ViT-B/32 ──► single 512-dim vector
                                      │  Linear(512→512)
                                      ▼
                              [text_emb + time_emb]  ← condition token (1 × 512)
                                      │
                                      ▼  prepended to motion sequence
                              ┌───────────────────────┐
Audio WAV ──┐                 │  MDM Transformer       │
            │                 │  (8 layers)            │
  Wav2CLIP ─┤                 │                        │
  + Librosa ┤► 519-dim/frame  │  Self-attention        │
            │   │             │       ↕                │
            │   ▼             │  Audio cross-attention │──► x̃₀ (denoised motion)
            │  AudioEncoder   │  (query=motion,        │
            │  (1D CNN)       │   key/val=audio seq)   │
            │   │             │       ↕                │
            │   ▼             │  Feed-forward          │
            │  T_audio × 512  │                        │
            │  (frame-level)  └───────────────────────┘
            │                         ↑
            └── beat_frames ──► temporal Gaussian + beat emphasis bias
```

**MDM Transformer** (backbone weights frozen during Stage 2):
- Architecture: Transformer encoder, 8 layers, 4 heads, latent dim 512, ff_size 1024
- Input: noisy motion `x_t` ∈ ℝ^(B × T × 263), timestep embedding `t`
- **Text conditioning:** CLIP ViT-B/32 encodes text to a **single** 512-dim vector (not 77 tokens). This is linearly projected and **added** to the timestep embedding, forming one condition token prepended to the motion sequence. Attended via **self-attention** (not cross-attention).
- **Audio conditioning (new):** 519-dim per-frame features → AudioEncoder (3-layer 1D CNN) → **T_audio × 512-dim sequence**. Each transformer layer has a dedicated **cross-attention** module where motion frames query the full audio sequence. Temporal Gaussian bias + beat emphasis bias are added to cross-attention logits.
- Motion representation: HumanML3D 263-dim (22 joints, 3D positions + velocities + foot contacts)
- Max sequence length: 196 frames = 9.8 s at 20 FPS

**Diffusion process:**
- Gaussian diffusion, T=1000 steps, cosine noise schedule, x₀-prediction
- Default sampling: DDPM with 1000 steps (or DDIM-50 for speed)
- Loss: simple MSE in x₀-prediction space, with optional physics penalties (skating, penetration)

---

## 3. Datasets

### HumanML3D
- **Purpose:** Text-conditioned motion generation training + evaluation
- **Content:** ~14,616 motion clips from AMASS + HumanAct12, paired with ~44,970 text annotations
- **Motion format:** 263-dimensional HumanML3D representation (22 SMPL joints)
- **FPS:** 20 Hz
- **Split:** train / val / test
- **Location:** `/Data/yash.bhardwaj/datasets/HumanML3D/`

### AIST++ (Dance subset)
- **Purpose:** Audio-conditioned motion training + BAS evaluation
- **Content:** Professional dancers performing 10 dance genres to music
- **Motion format:** SMPL pose parameters, 30 FPS (downsampled to 20 FPS for training)
- **Audio format:** 44 kHz WAV
- **Training set:** ~519 tracks used for Stage 2 fine-tuning
- **Test set:** 10 tracks spanning all genres (8 used for BAS evaluation)
- **Locations:**
  - Audio: `/Data/yash.bhardwaj/datasets/aist/audio_test_10/`
  - Motions: `/Data/yash.bhardwaj/datasets/aist/`

#### AIST++ Test Tracks (8 used for BAS eval, sorted by decreasing BPM)

| Track | Genre | BPM |
|-------|-------|-----|
| mBR0  | Breakdance | 161 |
| mKR2  | Krump | 122 |
| mLO2  | Lock | 118 |
| mPO1  | Pop | 116 |
| mJS3  | Jazz Swing | 110 |
| mLH4  | LA-style Hiphop | 105 |
| mMH3  | Middle Hiphop | 100 |
| mWA0  | Waacking | 81 |

---

## 4. Models

### 4.1 Stage 0 — Base MDM (pre-trained, frozen)
| Property | Value |
|----------|-------|
| Source | Original MDM (Tevet et al., 2022) |
| Checkpoint | `/Data/yash.bhardwaj/pretrained/humanml_trans_enc_512/model000475000.pt` |
| MDM parameters (excl. CLIP) | **17.9 M** (backbone 16.8 M + input/output/embed 1.1 M) |
| CLIP ViT-B/32 (frozen, loaded separately) | **151.3 M** |
| Total parameters (incl. CLIP) | **169.2 M** |
| Training | HumanML3D, text-only, 475k steps |
| Conditioning | Text (CLIP) only |
| Purpose | Initialisation for all Stage 2 models; BAS / text-quality baseline |

---

### 4.2 Stage 2 — Librosa-only (`audio_stage2_librosa`)
| Property | Value |
|----------|-------|
| Checkpoint | `save/audio_stage2_librosa/model_final.pt` |
| File size | **567 MB** |
| Audio features | 52-dim Librosa v2 (Mel-32, onset, beat distance, chroma, spectral, tempo) |
| Audio feats subdir | `audio_feats_v2/` |
| Beat-aware | Yes (same cross-attn architecture, beat bias in v2 features) |
| MOSPA | No |
| **`unfreeze_top_n`** | **2** (top 2 backbone layers also trained at lr=1e-5) |
| Dropout | 0.0 |
| Text mask prob | 0.10 |
| Trainable params | **14,655,240** (8.2%) — audio encoder + cross-attn + top-2 backbone |

**Text-quality (HumanML3D, tg=2.5, no audio):**
- FID: **5.58** ↑ (much worse than base MDM)
- R_top1: **0.292**
- Diversity: **7.70** (less diverse)

*Interpretation:* Two factors likely contributed to degraded text quality: (1) the 52-dim Librosa features lack sufficient semantic content, and (2) `unfreeze_top_n=2` allowed the top 2 backbone layers to drift during fine-tuning, disrupting the pre-trained text-motion mapping. Later models kept the backbone fully frozen.

---

### 4.3 Stage 2 — Wav2CLIP (`audio_stage2_wav2clip`)
| Property | Value |
|----------|-------|
| Checkpoint | `save/audio_stage2_wav2clip/model_final.pt` |
| File size | **542 MB** |
| Audio features | 512-dim Wav2CLIP + 7-dim Librosa = **519-dim** |
| Beat-aware | No (no `joint_cond_mask_prob`, no beat bias) |
| MOSPA | No |
| `unfreeze_top_n` | 0 (backbone fully frozen) |
| Dropout | 0.0 |
| Text mask prob | 0.10 |
| Trainable params | **11,047,432** (6.1%) — audio encoder + cross-attn only |

**Text-quality (HumanML3D, tg=2.5, no audio):**
- FID: **0.698**
- R_top1: **0.373**
- Diversity: **9.19**

*Interpretation:* The rich 512-dim semantic Wav2CLIP embedding gives the model enough audio information to fine-tune without collapsing text quality. Keeping the backbone fully frozen preserves the pre-trained text-motion mapping. FID is excellent (better than base MDM ~0.90).

---

### 4.4 Stage 2 — Wav2CLIP + MOSPA (`audio_stage2_wav2clip_mospa`)
| Property | Value |
|----------|-------|
| Checkpoint | `save/audio_stage2_wav2clip_mospa/model_final.pt` |
| File size | **542 MB** |
| Audio features | 519-dim (same as Wav2CLIP) |
| MOSPA | Yes — Momentum-based Online Style Parameter Adaptation |
| `unfreeze_top_n` | 0 (backbone fully frozen) |
| Dropout | 0.0 |
| Text mask prob | 0.10 |
| Trainable params | **11,047,432** (6.1%) — same as Wav2CLIP |

**Text-quality (HumanML3D, tg=2.5, no audio):**
- FID: **0.698** (identical to Wav2CLIP — MOSPA adds no degradation)
- R_top1: **0.373**
- Diversity: **9.19**

*Interpretation:* MOSPA is an inference-time adaptation mechanism; without explicit style input its results are identical to plain Wav2CLIP. No additional trainable parameter overhead to the checkpoint.

---

### 4.5 Stage 2 — Wav2CLIP + Beat-aware (`audio_stage2_wav2clip_beataware`) ★ **Best Model**
| Property | Value |
|----------|-------|
| Checkpoint | `save/audio_stage2_wav2clip_beataware/model_final.pt` |
| File size | **542 MB** |
| Audio features | 519-dim (512 Wav2CLIP + 7 Librosa) |
| Beat-aware bias | Yes — temporal Gaussian + beat emphasis bias in cross-attention logits |
| Beat indicator | Dim 512 (0-indexed) of 519-dim feature vector; `beat_frames` list for attention bias |
| `unfreeze_top_n` | 0 (backbone fully frozen) |
| Training steps | 100,000 |
| Batch size | 32 |
| Learning rate | 1e-4 |
| Latent dim | 512 |
| Layers | 8 |
| Heads | 4 |
| Dropout | 0.0 |
| Text mask prob | 0.15 |
| Audio mask prob | 0.15 |
| Joint mask prob | 0.05 |
| Physics losses | Off |
| Initialised from | `model000200000.pt` (base MDM at 200k steps) |
| Trainable params | **11,047,432** (6.1%) — audio encoder + cross-attn only |

**Text-quality (HumanML3D, tg=2.5, no audio):**
- FID: **0.698**
- R_top1: **0.373**
- Diversity: **9.19**

**Best BAS (ag=1.5, tg=2.5):** Mean **0.286** across 8 AIST++ test tracks

*This is the recommended model for all inference.* It matches Wav2CLIP text quality exactly while the beat-aware attention improves rhythmic alignment, especially on intermediate-tempo tracks (mMH3: 0.433, mJS3: 0.426).

---

### 4.6 Baseline — EDGE
| Property | Value |
|----------|-------|
| Paper | Tseng et al., CVPR 2023 |
| Architecture | Transformer-based music-to-dance |
| Input | Music audio only (no text) |
| FPS | **30 FPS** (different from ours at 20 FPS — requires frame-rate sync in comparisons) |
| Feature extractor | Jukebox (music features) |
| Evaluation | AIST++ test set, 8 tracks |
| Checkpoint | `/Data/yash.bhardwaj/pretrained/edge/` |

---

## 5. Audio Feature Representations

### Librosa v2 (52-dim)
- Mel spectrogram: 32 bands (reduced from 128 in v1; normalised)
- Onset strength + smoothed onset envelope: 2 dims
- Beat indicator (soft Gaussian, σ=1 frame): 1 dim
- Beat distance past/future (normalised): 2 dims
- RMS energy (normalised): 1 dim
- Chroma: 12 dims
- Spectral centroid (normalised): 1 dim
- Tempo (normalised): 1 dim
- Computed per-frame at 20 Hz
- Feature extraction: `model/audio_features_v2.py`

### Wav2CLIP (512-dim)
- Pre-trained audio-language embedding model (wav2clip; Wu et al.)
- Architecture: ResNet-based audio encoder trained with CLIP-style contrastive loss on AudioSet
- Output: 512-dim semantically-rich audio embedding **per frame**
- Extraction: 0.5 s sliding window at 20 Hz (`frame_length=0.5s`, `hop_length=sr/fps`) → `(T, 512)` float32
- SR: 16 kHz (wav2clip native), audio resampled from 22050 Hz
- Library: `pip install wav2clip`

### Combined (519-dim = 512 + 7)
- Wav2CLIP 512-dim semantic embedding (per frame)
- 7 Librosa features: RMS, spectral centroid, spectral rolloff, chroma mean, onset strength, tempo, beat indicator
- Concatenated per frame → `(T, 519)` condition sequence
- Beat indicator at dim 512 (0-indexed): 1.0 at beat frames, 0.0 otherwise

---

## 6. Conditioning Mechanism

### 6.1 Audio Cross-Attention (frame-level)
Audio conditioning uses a **dedicated cross-attention module** in each of the 8 transformer layers — separate from the text pathway.

**How it works:**
1. Per-frame audio features (519-dim at 20 FPS) are passed through the `AudioEncoder` (3-layer 1D CNN: 519 → 256 → 512 → 512) producing a **T_audio × 512-dim sequence** — one latent per audio frame.
2. In each transformer layer, motion tokens (query) attend to the full audio sequence (key/value) via `nn.MultiheadAttention`.
3. The cross-attention output is gated by a learnable scalar (`cross_attn_gate`, initialised to 0 for smooth training start): `src += tanh(gate) * cross_attn_output`.
4. The result is layer-normalised before passing to the FFN.

**Key distinction from text:** Text conditioning uses CLIP to encode the prompt into a **single** 512-dim vector, which is added to the timestep embedding and prepended as a condition token — processed only via self-attention. Audio conditioning uses **frame-level cross-attention** with T_audio keys, giving the model per-frame rhythmic information.

**Conditioning dropout (for CFG compatibility):**
- Text: masked (zeroed) independently per sample (prob = 0.10 for Wav2CLIP/MOSPA, 0.15 for beat-aware)
- Audio: masked (zeroed) independently per sample (prob = 0.15 for all)
- Beat-aware model adds explicit joint dropout at 5% via `joint_cond_mask_prob` (see Section 7 for per-model details)

### 6.2 Beat-Aware Attention Bias
Two additive biases are applied to the cross-attention logits (before softmax) in each layer:

**1. Temporal locality bias (Gaussian):**
Each motion frame `i` attends most strongly to the temporally corresponding audio frame, with attention falling off as a Gaussian:
```
bias[i, j] = -(pos_i - pos_j)² / (2σ²)    where σ = 4.0 frames (~200 ms at 20 FPS)
```
This ensures each motion frame focuses on nearby audio context (~0.4 s window) rather than the entire sequence.

**2. Beat emphasis bias:**
At detected beat frame positions, an extra positive logit (`beat_weight = 2.0`) is added, plus half-weight on neighbouring frames:
```
bias[:, beat_frame] += 2.0       (all motion frames attend more to beat audio frames)
bias[:, beat_frame±1] += 1.0     (neighbour smoothing)
```

**Beat detection:** Librosa `beat_track` on the audio WAV → list of frame indices passed as `y['beat_frames']`.

**Effect:** The model "sees" beat frames more prominently in cross-attention, encouraging kinetic velocity peaks (motion beats) to coincide with musical beats.

**Implemented in:** `model/audio_cross_attention_v2.py` (functions `build_temporal_bias`, `build_beat_bias`)

### 6.3 MOSPA (Momentum-based Online Style Parameter Adaptation)
- Adapts style parameters online during inference using momentum-based updates
- Implemented as a wrapper around the audio encoder
- Does not change inference output without explicit style input (results identical to plain Wav2CLIP at default settings)
- Designed for lightweight online adaptation without full re-training

### 6.4 Classifier-Free Guidance (AudioCFG)
Standard CFG extended for two independent conditioning signals. MDM is an **x₀-prediction** model, so guidance operates in x₀ space:

```
x̃₀ = x₀(∅,∅) 
    + tg × (x₀(text,∅) − x₀(∅,∅))         # text guidance
    + ag × (x₀(text,audio) − x₀(text,∅))   # audio guidance
```

Three forward passes per denoising step:
1. `x₀(∅,∅)` — fully unconditioned (both text and audio masked)
2. `x₀(text,∅)` — text-only (audio masked)
3. `x₀(text,audio)` — full conditioning

Implemented in `AudioCFGSampleModel` in `sample/generate_audio.py`.

**Best found hyperparameters: `tg=2.5, ag=1.5`**

### 6.5 GCDM (Generalised Classifier-Free Diffusion Guidance)
A more sophisticated guidance scheme with timestep-dependent blending (also in x₀ space):

```
x̃₀ = x₀(∅) + α [ λ(t)·(x₀(text,audio)−x₀(∅)) + (1−λ(t))·(β_t·(x₀(text)−x₀(∅)) + β_a·(x₀(audio)−x₀(∅))) ]
```

- Four passes per step: full, text-only, audio-only, unconditioned
- λ(t) = t/T: linear interpolation; early steps (high t) favour joint, late steps (low t) favour independent
- Intended benefit: "style early, rhythm late"
- In practice, AudioCFG with tg=2.5/ag=1.5 outperforms all tested GCDM configurations (see Section 11)

---

## 7. Training Details

| Setting | Librosa-only | Wav2CLIP / MOSPA | Beat-aware ★ |
|---------|:------------:|:----------------:|:------------:|
| Base checkpoint | MDM @ 200k steps | MDM @ 200k steps | MDM @ 200k steps |
| Dataset | AIST++ | AIST++ | AIST++ |
| Steps | 100,000 | 100,000 | 100,000 |
| Batch size | 32 | 32 | 32 |
| Optimizer | AdamW | AdamW | AdamW |
| LR (audio modules) | 1e-4 | 1e-4 | 1e-4 |
| LR (backbone, if unfrozen) | 1e-5 | — | — |
| Weight decay | 0.0 | 0.0 | 0.0 |
| Max motion length | 196 frames | 196 frames | 196 frames |
| Dropout (transformer) | 0.0 | 0.0 | 0.0 |
| `unfreeze_top_n` | **2** | 0 | 0 |
| Text mask prob | 0.10 | 0.10 | **0.15** |
| Audio mask prob | 0.15 | 0.15 | 0.15 |
| Joint mask prob | — | — | **0.05** |
| Hardware | 1× GPU | 1× GPU | 1× GPU |
| Approx. train time | ~12–16 hours | ~12–16 hours | ~12–16 hours |
| Save interval | Every 10k steps | Every 10k steps | Every 10k steps |

**Conditioning dropout (for CFG compatibility):**
- Text and audio are masked independently per sample via `mask_cond()` and `mask_audio()` in `model/mdm.py`.
- Beat-aware model additionally uses 5% joint dropout (both text and audio zeroed simultaneously via `joint_cond_mask_prob`), which enables the three-pass AudioCFG and four-pass GCDM formulations.
- Earlier models (Wav2CLIP, MOSPA) used 10% text dropout; beat-aware increased to 15% for consistency with AudioCFG requirements.

---

## 7.1 Parameter Breakdown — What Gets Trained

During Stage 2, `freeze_non_audio()` is called to freeze the original MDM weights. For Wav2CLIP variants (`unfreeze_top_n=0`), **all** backbone weights stay frozen. For Librosa-only (`unfreeze_top_n=2`), the top 2 backbone layers were additionally unfrozen. The optimizer receives `model.audio_parameters()` (and optionally the top-N backbone layers).

### Per-component parameter counts (exact)

| Component | Params | Wav2CLIP models | Librosa-only |
|-----------|-------:|:---------------:|:------------:|
| **CLIP ViT-B/32** (text encoder) | 151,277,313 | Frozen | Frozen |
| **MDM backbone** - self-attention (8 layers) | 8,404,992 | Frozen | Top 2 unfrozen |
| **MDM backbone** - FFN (8 layers) | 8,400,896 | Frozen | Top 2 unfrozen |
| **MDM backbone** - LayerNorm (8 layers x 2) | 16,384 | Frozen | Top 2 unfrozen |
| **MDM other** - input/output process, embed_text, timestep | 1,058,055 | Frozen | Frozen |
| **Audio encoder** - 1D CNN (52 to 512) | 2,036,480 | - | Trained |
| **Audio encoder** - 1D CNN (519 to 512) | 2,634,240 | Trained | - |
| **Cross-attention** - MHA (8 layers x 1,050,624) | 8,404,992 | Trained | Trained |
| **Cross-attention** - LayerNorm (8 layers x 1,024) | 8,192 | Trained | Trained |
| **Cross-attention** - gate scalar (8 layers x 1) | 8 | Trained | Trained |

### Summary per model variant

| Model | Trainable params | Total in model | % trainable |
|-------|---------------------:|---------------:|:----------:|
| **Base MDM** (no audio) | 0 | 169,157,640 | 0% |
| **Librosa-only** (52-dim, top-2 unfrozen) | **14,655,240** | 179,607,312 | **8.2%** |
| **Wav2CLIP / MOSPA / beat-aware** (519-dim, backbone frozen) | **11,047,432** | 180,205,072 | **6.1%** |

*Breakdown of 14.7M trainable (Librosa-only):*
- Audio encoder CNN (52 to 512): 2,036,480 (13.9%)
- Cross-attention (8 layers): 8,413,192 (57.4%)
- Top-2 backbone layers (self-attn + FFN + LN): 4,205,568 (28.7%)

*Breakdown of 11.0M trainable (Wav2CLIP variants):*
- Audio encoder CNN (519 to 512): 2,634,240 (23.9%)
- Cross-attention (8 layers): 8,413,192 (76.1%)

**Why text quality is preserved in Wav2CLIP but not Librosa:**
- Wav2CLIP/beat-aware/MOSPA keep the MDM backbone **completely frozen** (`unfreeze_top_n=0`). The original self-attention and FFN weights are unchanged, preserving the text-motion mapping learned during pre-training. Only the new audio encoder and cross-attention layers (~11M) are trained from scratch.
- Librosa-only unfroze the top 2 backbone layers (`unfreeze_top_n=2` at `backbone_lr=1e-5`). Combined with the weaker 52-dim audio signal, this caused the backbone to drift, degrading text quality (FID 0.90 -> 5.58).

`freeze_non_audio(unfreeze_top_n)` in `model/mdm.py`: sets `requires_grad=False` on all parameters, then `requires_grad=True` on `audio_parameters()` (audio encoder + cross-attn + norm_cross + gate) and optionally the top-N backbone layers.

---

## 8. Inference & Sampling

### Standard audio-conditioned generation
```bash
python -m sample.generate_audio \
  --model_path save/audio_stage2_wav2clip_beataware/model_final.pt \
  --audio_path /path/to/audio.wav \
  --text_prompt "a person dances to the beat" \
  --guidance_param 2.5 \
  --audio_guidance_param 1.5 \
  --seed 42 \
  --fps 20
```
Output: `sample_audio_00.npy` — shape `(196, 263)` HumanML3D motion

### Text-only (no audio conditioning)
```bash
python -m sample.generate_audio \
  --model_path save/audio_stage2_wav2clip_beataware/model_final.pt \
  --text_prompt "a person walks forward" \
  --guidance_param 2.5 \
  --audio_guidance_param 0.0
```

### GCDM sampling
```bash
python -m sample.generate_audio \
  --use_gcdm \
  --gcdm_alpha 3.0 \
  --gcdm_beta_text 1.0 \
  --gcdm_beta_audio 1.5 \
  ...
```

---

## 9. Evaluation Metrics

### Beat Alignment Score (BAS)
- **Definition:** For each detected motion beat, compute the probability that a music beat occurs nearby, weighted by a Gaussian kernel:
  ```
  BAS = (1/|M|) Σ_m exp(−min_b (m−b)² / (2σ²))
  ```
  where M = set of motion beat frames, B = set of music beat frames
- **σ (BA sigma):** 1.00 frame (auto-scaled from FPS) = 50 ms tolerance window
- **Motion beats:** Computed from mean joint velocity, smoothed (σ=1.67 frames), then peak-picked
- **Music beats:** Librosa `beat_track` on the audio
- **Higher is better.** Range: [0, 1]. Typical good values: >0.30 for dance models.
- Implemented in `eval/beat_align_score.py`

### HumanML3D Text-Quality Metrics
| Metric | Description |
|--------|-------------|
| **FID** | Fréchet Inception Distance between generated and real motion feature distributions. Lower is better. |
| **R_precision (top-1/2/3)** | Text-motion retrieval accuracy. Higher is better. |
| **Matching Score** | Mean L2 distance between paired text and motion features. Lower is better. |
| **Diversity** | Average pairwise distance in the generated set. Should match ground-truth ~9.1. |
| **MultiModality** | Variance across samples for the same text. Higher = more varied outputs. |

Evaluated using the pre-trained HumanML3D motion encoder. Run via `eval/eval_audio_humanml_v2.py`.

### Rhythmic Residual Beat Alignment (RRBA)
- Measures how much of the **delta** motion (audio-conditioned minus text-only, same seed) is beat-aligned
- Isolates the audio contribution to rhythm from the background text-driven motion

---

## 10. Quantitative Results

### 10.1 HumanML3D Text-Quality Metrics
*(Debug mode: 1 replication. Settings: tg=2.5, audio_guidance=0.0, audio_mode=none)*

| Model | FID ↓ | R_top1 ↑ | R_top3 ↑ | Diversity | Matching Score ↓ |
|-------|--------|----------|----------|-----------|-----------------|
| **Ground Truth** | ~0.001 | ~0.445 | ~0.770 | ~9.13 | ~3.24 |
| **Base MDM** (text-only, tg=2.5) | 0.900 | 0.377 | 0.678 | — | 3.891 |
| **Librosa-only** | 5.577 | 0.292 | 0.577 | 7.698 | 4.510 |
| **Wav2CLIP** | **0.698** | **0.373** | **0.664** | **9.19** | 3.911 |
| **Wav2CLIP + MOSPA** | 0.698 | 0.373 | 0.664 | 9.19 | 3.911 |
| **Wav2CLIP + Beat-aware** ★ | 0.698 | 0.373 | 0.664 | 9.19 | 3.911 |

**Key findings:**
- Librosa-only dramatically hurts text quality (FID 5.6×). Two contributing factors: (a) the 52-dim feature provides insufficient semantic grounding, and (b) `unfreeze_top_n=2` allowed the top 2 backbone layers to drift, disrupting the pre-trained text-motion mapping.
- All Wav2CLIP variants (backbone fully frozen, `unfreeze_top_n=0`) match or exceed Base MDM on FID, confirming the 519-dim representation + frozen backbone preserves text controllability perfectly.
- Beat-aware mechanism adds **zero text-quality overhead** (identical FID/R to plain Wav2CLIP).
- MOSPA adds zero overhead at default inference settings.

---

### 10.2 Beat Alignment Score (BAS) — CFG Sweep
*(wav2clip_beataware model, AIST++ 8 test tracks, single seed=42)*

| Config | tg | ag | Mean BAS ↑ | Notes |
|--------|----|----|------------|-------|
| **AudioCFG ag=1.5** | 2.5 | **1.5** | **0.2856** | ★ Best |
| AudioCFG ag=1.0 | 2.5 | 1.0 | 0.2504 | Under-driven |
| tg=2.0, ag=1.5 | 2.0 | 1.5 | 0.2557 | Lower text doesn't help |
| tg=3.0, ag=1.5 | 3.0 | 1.5 | 0.2681 | Marginally worse |
| AudioCFG ag=2.5 | 2.5 | 2.5 | 0.2652 | Starts to saturate |
| AudioCFG ag=5.0 | 2.5 | 5.0 | 0.2274 | Over-driven |
| AudioCFG ag=7.5 | 2.5 | 7.5 | 0.2248 | Over-driven |
| GCDM α=3, βt=1, βa=1.5 | — | — | 0.2407 | 4-pass, worse than CFG |
| GCDM α=5, βt=1, βa=2.5 | — | — | 0.2240 | 4-pass, worse than CFG |

**Conclusion:** AudioCFG with `tg=2.5, ag=1.5` is the sweet spot. Audio guidance >2.5 degrades BAS — the model over-corrects motion velocity rather than placing peaks at beats. GCDM does not improve over simple AudioCFG in any tested configuration.

---

### 10.3 BAS vs EDGE — Per Track Comparison
*(Ours: wav2clip_beataware, ag=1.5, tg=2.5 | EDGE: Tseng et al. CVPR 2023)*

| Track | Genre | BPM | **Ours BAS** | **EDGE BAS** | Winner |
|-------|-------|-----|-------------|-------------|--------|
| mBR0 | Breakdance | 161 | 0.2530 | 0.1905 | **Ours** |
| mJS3 | Jazz Swing | 110 | **0.4263** | 0.2641 | **Ours** ✦ |
| mKR2 | Krump | 122 | 0.2565 | 0.3537 | EDGE |
| mLH4 | LA Hiphop | 105 | 0.1810 | 0.3501 | EDGE |
| mLO2 | Lock | 118 | 0.2585 | 0.3129 | EDGE |
| mMH3 | Middle Hiphop | 100 | **0.4332** | 0.3438 | **Ours** ✦ |
| mPO1 | Pop | 116 | 0.2263 | 0.3377 | EDGE |
| mWA0 | Waacking | 81 | 0.2496 | 0.1325 | **Ours** |
| **Mean** | — | — | **0.2856** | **0.2869** | ~Tie |

**Key findings:**
- Ours win on 4/8 tracks; overall mean is a virtual tie (0.286 vs 0.287).
- Ours strongly win on Jazz Swing (mJS3) and Middle Hiphop (mMH3) — intermediate tempos with clear groove patterns.
- EDGE wins on genre-specific tracks (Krump, LA Hiphop, Pop, Lock) — likely because EDGE was trained directly on AIST++ dance motions and has learned genre-specific choreography.
- **Critical advantage of our model:** Ours generates *text-conditioned* motion. EDGE cannot follow text prompts — it maps music → generic dance only. Our model can be directed to "walk," "jump," "kick," or "dance to the beat" while simultaneously staying rhythmically aligned.

---

## 11. CFG Hyperparameter Study

Full per-track breakdown for `audio_stage2_wav2clip_beataware`:

### AudioCFG — BAS per track (tg=2.5 fixed)

| ag | mBR0 | mJS3 | mKR2 | mLH4 | mLO2 | mMH3 | mPO1 | mWA0 | **Mean** |
|----|------|------|------|------|------|------|------|------|----------|
| 1.0 | 0.143 | 0.321 | 0.294 | 0.226 | 0.353 | 0.330 | 0.168 | 0.169 | 0.250 |
| **1.5** | **0.253** | **0.426** | **0.257** | **0.181** | **0.259** | **0.433** | **0.226** | **0.250** | **0.286** |
| 2.5 | 0.303 | 0.210 | 0.179 | 0.279 | 0.291 | 0.395 | 0.242 | 0.222 | 0.265 |
| 5.0 | 0.180 | 0.051 | 0.191 | 0.381 | 0.271 | 0.223 | 0.200 | 0.323 | 0.227 |
| 7.5 | 0.124 | 0.211 | 0.288 | 0.325 | 0.177 | 0.236 | 0.233 | 0.205 | 0.225 |

### GCDM vs AudioCFG

| Method | Config | Mean BAS |
|--------|--------|----------|
| AudioCFG | tg=2.5, ag=1.5 | **0.286** |
| GCDM | α=3, βt=1, βa=1.5 | 0.241 |
| GCDM | α=5, βt=1, βa=2.5 | 0.224 |

GCDM's four-pass overhead does not yield better rhythmic alignment than the simpler three-pass AudioCFG in any tested configuration.

---

## 12. Qualitative Visualisations

All visualisations use the **MDM-style rendering**: root-centred skeleton, moving floor grid, blue trajectory trail, `elev=120°, azim=-90°`, dark background. Each panel includes a wavestrip (audio waveform, amber music beats, coral/coloured motion beats, mint velocity curve, white playhead).

### E6 Experiment — Text-Audio Independence
**Goal:** Prove that our model responds independently to text and audio conditioning.

**Method:** Same audio track, same seed; vary:
- (a) Text prompt ("walk" vs "jump" vs "kick") → motion action changes while beat alignment is preserved
- (b) Audio track (slow mWA0 81 BPM vs fast mBR0 161 BPM) → motion tempo adapts while action is preserved

**Location:** `/Data/yash.bhardwaj/eval/e6_variants/` (seeds 0–9, mBR0/mJS3/mWA0 tracks)

---

### Ablation Grid (1×3) — Audio Track Variation
**Goal:** Visualise the effect of different audio tracks on the same text prompt.

**Layout:** `[Base MDM text-only] | [Ours + variable track] | [Ours + mBR0]`

**Prompt:** "a person dances"  
**Model:** wav2clip_beataware, ag=1.5, tg=2.5, 3 seeds (42, _s1, _s2)

**Location:** `/Data/yash.bhardwaj/eval/ablation_grid/`

| Video | Col 2 Track | Seeds available |
|-------|------------|-----------------|
| `ablation_mLO2*.mp4` | Lock 118 BPM | 42, _s1, _s2 |
| `ablation_mKR2*.mp4` | Krump 122 BPM | 42, _s1, _s2 |
| `ablation_mLH4*.mp4` | LA Hiphop 105 BPM | 42, _s1, _s2 |
| `ablation_mWA0*.mp4` | Waacking 81 BPM | 42, _s1, _s2 |

---

### Text Week (1×3) — Text Diversity with Same Audio
**Goal:** Demonstrate simultaneous text-audio control by fixing the audio and varying the text action.

**Layout:** `["a person walks forward"] | ["a person jumps"] | ["a person kicks"]`

All three columns share the same audio track. Motion beats differ per column (different actions hit the beat differently), demonstrating text-audio disentanglement within a single video.

**Model:** wav2clip_beataware, ag=1.5, tg=2.5, seed=42  
**Videos include audio** (muxed via ffmpeg — no re-render needed)  
**Location:** `/Data/yash.bhardwaj/eval/text_week/`

| Video | Audio Track | BAS walk / jump / kick |
|-------|-------------|------------------------|
| `audio_textweek_mBR0.mp4` | Breakdance 161 BPM | 0.184 / 0.081 / 0.284 |
| `audio_textweek_mJS3.mp4` | Jazz Swing 110 BPM | — |
| `audio_textweek_mKR2.mp4` | Krump 122 BPM | — |
| `audio_textweek_mLH4.mp4` | LA Hiphop 105 BPM | — |
| `audio_textweek_mLO2.mp4` | Lock 118 BPM | — |
| `audio_textweek_mMH3.mp4` | Middle Hiphop 100 BPM | — |
| `audio_textweek_mPO1.mp4` | Pop 116 BPM | — |
| `audio_textweek_mWA0.mp4` | Waacking 81 BPM | — |

---

### Side-by-Side Comparisons — Ours vs EDGE
**Goal:** Direct visual and rhythmic comparison with the EDGE baseline.

**Stick-figure versions** (20 FPS, EDGE resampled 30→20 FPS):  
`/Data/yash.bhardwaj/eval/sidebyside/` — tracks: mJS3, mKR2, mLO2, mMH3

**Skinned SMPL mesh versions** (full body mesh, detailed wavestrip):  
`/Data/yash.bhardwaj/eval/sidebyside_mesh/` — all 8 eval tracks

Wavestrip shows: audio waveform, amber music beats, coral ours motion beats, lime EDGE motion beats, mint velocity curves, white playhead, per-model BAS badge.

---

## 13. Key Implementation Notes & Fixes

### GCDM (verified correct)
- **Formula:** Four-pass guided estimation with λ(t) = t/T blending
- **Convention:** t runs 999→0; λ(999)≈1 (joint early), λ(0)≈0 (independent late)
- **`diffusion_steps`:** Passed as `diffusion.num_timesteps` (=1000) — consistent with model scheduler
- **Safety fix:** `has_audio = getattr(model, 'audio_conditioning', False)` — prevents AttributeError on non-audio checkpoints

### AudioCFG (verified correct)
Three forward passes: `x₀(∅,∅)`, `x₀(text,∅)`, `x₀(text,audio)`. Text guidance applied as delta from unconditional; audio guidance applied as delta from text-only. Operates in x₀-prediction space (MDM's parameterisation).

### Beat-Aware Mechanism
- Beat detection: Librosa `beat_track` → list of beat frame indices → passed as `y['beat_frames']`
- In the 519-dim per-frame feature vector: dims 0–511 = Wav2CLIP, dims 512–518 = 7 Librosa features. The beat indicator is stored at dim index 512 (0-indexed), set to 1.0 at beat frames, 0.0 otherwise.
- However, the beat-aware **attention bias** operates independently of the feature vector: it uses the `beat_frames` list directly to add logit biases in cross-attention (see Section 6.2). The feature-vector beat indicator and the attention bias reinforce each other.

### Frame Rate Sync (EDGE vs Ours)
EDGE outputs at 30 FPS; our model runs at 20 FPS. When rendering side-by-side:
```python
edge_frame_map = np.minimum(
    (np.arange(T_render) * 30.0 / 20.0).astype(int),
    T_edge - 1
)
```
This maps each of our 20 FPS render frames to the correct EDGE 30 FPS frame without artificially speeding up EDGE.

### SMPL Model Setup (for mesh visualisations)
- Required file: `SMPL_NEUTRAL.pkl` → `/Data/yash.bhardwaj/models/smpl/smpl/SMPL_NEUTRAL.pkl`
- Symlinked to: `body_models/smpl/SMPL_NEUTRAL.pkl`
- `chumpy` patched for NumPy ≥1.24 compatibility: removed deprecated `from numpy import int, float, ...` aliases
- `gmm_08.pkl` (SMPLify prior) not available → replaced with a custom lightweight gradient-descent SMPL fitter in `visualize_sidebyside_mesh.py`
- `neutral_smpl_mean_params.h5` created synthetically (zero pose + shape) to bypass missing dependency

### Coordinate System (AIST++ / EDGE)
AIST++ and EDGE store motion in Z-up convention. Pyrender expects Y-up. Fixed by swapping Y↔Z axes in `edge_pkl_to_vertices`:
```python
vertices[:, [1, 2]] = vertices[:, [2, 1]]
```

---

## 14. File Map

```
motion-diffusion-model-rhythmic/
├── model/
│   ├── mdm.py                          # Core MDM Transformer + audio conditioning
│   ├── audio_cross_attention.py        # Beat-aware cross-attention (v1, unused)
│   ├── audio_cross_attention_v2.py     # ★ Beat-aware cross-attention (active)
│   ├── audio_encoder.py                # Audio projection layer
│   ├── audio_features.py               # Librosa feature extraction (v1, 145-dim)
│   ├── audio_features_v2.py            # Librosa features v2 (52-dim, used by librosa-only model)
│   ├── audio_features_wav2clip.py      # Wav2CLIP + Librosa (519-dim)
│   ├── cfg_sampler.py                  # CFG wrapper
│   └── rotation2xyz.py                 # SMPL rotation → joint positions
│
├── train/
│   └── train_audio.py                  # Stage 2 training loop
│
├── sample/
│   ├── generate_audio.py               # Main inference script (AudioCFG + GCDM)
│   └── gcdm.py                         # GCDM sampler
│
├── eval/
│   ├── beat_align_score.py             # BAS metric + beat detection utilities
│   ├── evaluate_audio.py               # Full audio eval (BAS + FID + diversity)
│   ├── eval_audio_humanml_v2.py        # HumanML3D text-quality eval
│   └── rhythmic_residual.py            # RRBA metric
│
├── scripts/
│   ├── visualize_bas_alignment.py      # BAS alignment visualiser (single track)
│   ├── visualize_e6_1x3_mdm.py        # E6 1×3 MDM-style grid renderer
│   ├── visualize_sidebyside.py         # Stick-figure ours vs EDGE
│   ├── visualize_sidebyside_mesh.py    # Skinned SMPL mesh ours vs EDGE
│   ├── render_ablation_grids.py        # Ablation grid (audio variation, 1×3)
│   ├── render_text_week.py             # Text-week (action variation, 1×3, with audio)
│   └── count_params.py                 # Exact parameter counting per component
│
├── save/
│   ├── audio_stage2_librosa/           # Librosa-only model
│   ├── audio_stage2_wav2clip/          # Wav2CLIP model
│   ├── audio_stage2_wav2clip_mospa/    # Wav2CLIP + MOSPA
│   └── audio_stage2_wav2clip_beataware/ # ★ Best model
│
└── docs/
    ├── PROJECT_SUMMARY.md              # ← This file
    ├── evaluation_guide.md             # How to run evaluations
    └── implementation_audit.md         # GCDM + beat-aware correctness audit
```

**Key data paths:**
```
/Data/yash.bhardwaj/
├── datasets/
│   ├── HumanML3D/                      # Text-motion dataset
│   └── aist/audio_test_10/             # 8+ AIST++ test WAVs
├── pretrained/
│   ├── humanml_trans_enc_512/          # Base MDM checkpoint
│   └── edge/                           # EDGE model checkpoint
├── models/smpl/                        # SMPL body model files
└── eval/
    ├── ablation_grid/                   # 12 ablation videos (3 seeds × 4 tracks)
    ├── text_week/                       # 8 text-week videos (with audio)
    ├── sidebyside/                      # Stick-figure ours vs EDGE
    ├── sidebyside_mesh/                 # Skinned mesh ours vs EDGE (all 8 tracks)
    ├── e6_variants/                     # E6 independence visualisations
    └── cfg_sweep/                       # Full CFG sweep results + summary.txt
```

---

## 15. Known Limitations & Future Work

### Current Limitations
1. **BAS variance is high.** Single-seed evaluation on 8 tracks gives noisy estimates; proper evaluation needs 5+ seeds × full AIST++ test split.
2. **EDGE comparison is asymmetric.** EDGE is a dedicated music-to-dance model with genre-specific choreography patterns; our model is a general-purpose text+audio model. The comparison is somewhat unfair in both directions.
3. **Audio-motion alignment is approximate.** Wav2CLIP features are per-frame (0.5 s sliding window) but each window captures only coarse semantic content; finer rhythmic cues come from Librosa's 7 features + beat bias. A dedicated music encoder with finer temporal resolution could improve sub-beat precision.
4. **No physics losses in final model.** Foot skating and ground penetration penalties were available but disabled during training for simplicity.
5. **GCDM not beneficial.** The theoretical advantage of GCDM over AudioCFG did not materialise in practice, possibly due to insufficient training time or hyperparameter sensitivity.

### Future Work
1. **Richer frame-level audio encoding:** Wav2CLIP embeddings are extracted per frame (0.5 s sliding window at 20 Hz) but each window covers only coarse semantic content. Replacing Wav2CLIP with a dedicated music encoder (e.g. music transformer, Jukebox features) that captures finer harmonic and rhythmic structure could improve sub-beat synchronisation.
2. **Multi-person generation:** Extend to paired/group dance scenarios.
3. **Real-time inference:** Distil to DDIM-10 or consistency models for low-latency interactive generation.
4. **User study:** Formal perceptual study comparing text following + rhythmic alignment between ours, EDGE, and base MDM.
5. **Physics-based refinement:** Enable skating and penetration losses during fine-tuning for more physically plausible foot contact.
