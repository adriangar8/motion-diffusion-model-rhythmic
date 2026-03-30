# How to Evaluate If We've Done Well

This guide covers **what to measure** and **how to run it** for the motion diffusion pipeline (Stage 2 audio conditioning + Stage 3 DoRA style adapters).

---

## 1. What “done well” means

| Stage | Goal | Good means |
|-------|-----|------------|
| **Stage 2** (audio) | Motion follows text and is rhythmically aligned to music | High Beat Alignment Score (BAS), low FID vs real dance, plausible physics |
| **Stage 3** (DoRA) | Motion has the target style (e.g. old_elderly) while staying on-prompt and on-beat | Subjective style match in grids; FID/diversity not worse than Stage 2 |

There is **no automatic “style strength” metric** in the repo. Style quality is judged by **visual comparison** (e.g. 2×2 grids) and, if you add it, by **user studies** or **retrieval** (e.g. “does this motion retrieve the right style label in CLIP?”).

---

## 2. Stage 2 (audio-conditioned) — quantitative

Use **`eval/evaluate_audio.py`**. It reports:

- **Beat Alignment Score (BAS)** — fraction of music beats with a motion kinetic peak nearby. **Higher is better** (e.g. > 0.4 is decent).
- **FID** — distribution distance (generated vs real AIST++). **Lower is better** (e.g. &lt; 10 is good).
- **Diversity** — variance across generated samples. Should be in the ballpark of real diversity (not collapse, not explosion).
- **Physical plausibility** — foot sliding (lower is better), smoothness (similar to real).

### Option A: Full pipeline (generate + evaluate)

Generates from the Stage 2 model, then computes all metrics including BAS:

```bash
python -m eval.evaluate_audio \
  --model_path ./save/audio_stage2_wav2clip/model_final.pt \
  --aist_dir /path/to/aist \
  --humanml_dir /Data/yash.bhardwaj/datasets/HumanML3D \
  --num_audio_tracks 10 \
  --samples_per_track 5 \
  --output_path ./save/audio_stage2_wav2clip/eval_results.json
```

Note: `evaluate_audio` uses the old `generate_audio` loader (no Wav2CLIP). If your Stage 2 is Wav2CLIP-based, you may need to generate samples with your usual script and then run evaluation with **Option B**.

### Option B: Pre-generated samples only (FID, diversity, physics; no BAS)

If you already have `.npy` motion files (e.g. from `generate_stage3_style` or your own script):

```bash
python -m eval.evaluate_audio \
  --humanml_dir /Data/yash.bhardwaj/datasets/HumanML3D \
  --aist_dir /path/to/aist \
  --skip_generation \
  --sample_dir ./save/audio_stage3_dora/old_elderly/samples_breakdance \
  --output_path ./save/audio_stage3_dora/old_elderly/eval_results.json
```

This computes **FID**, **diversity**, **foot sliding**, and **smoothness** (no BAS, since no per-sample audio is stored). Compare FID/diversity across styles or vs Stage-2-only samples to check that DoRA did not degrade motion quality.

---

## 3. Rhythmic residual (RRBA) — audio contribution

**`eval/rhythmic_residual.py`** measures how much of the **added** motion (audio − text-only) is beat-aligned. Useful for tuning audio guidance and checking that rhythm comes from the audio, not just the text.

- **Paired mode** (audio-conditioned vs text-only samples, same prompt/seed):

```bash
python -m eval.rhythmic_residual \
  --audio_dir ./save/audio_stage2/samples_fair_audio \
  --text_dir ./save/audio_stage2/samples_fair_noaudio \
  --audio_path /Data/yash.bhardwaj/datasets/aist/audio/mBR0.wav \
  --humanml_dir /Data/yash.bhardwaj/datasets/HumanML3D
```

- Good: RRBA clearly above random; not so high that motion looks over-driven.

---

## 4. Stage 3 (DoRA style) — qualitative + optional metrics

### Visual comparison (main criterion)

- Use **2×2 grids** from `sample/compare_4_doras.py` (same prompt + same audio, four styles).
- Check:
  - **Style**: Does old_elderly look older/calmer, angry_aggressive more aggressive, etc.?
  - **Content**: Is the motion still on-prompt (e.g. “walks forward”, “breakdancing”)?
  - **Rhythm**: Does it still align with the music (no need to be perfect)?

### Training sanity

- **Stage 3 loss**: Should decrease from ~0.06 to ~0.05 and then plateau (see `docs/stage3_training_eval.md`). No decrease at all suggests a bug or bad hyperparameters.
- **Checkpoints**: `adapter_final.pt` per style; training log in `save/audio_stage3_dora/{style}/train.log`.

### Optional: FID/diversity per style

Generate a batch of samples per DoRA (e.g. 50 samples with the same prompt/audio or multiple prompts), then run:

```bash
python -m eval.evaluate_audio \
  --humanml_dir /Data/yash.bhardwaj/datasets/HumanML3D \
  --aist_dir /path/to/aist \
  --skip_generation \
  --sample_dir ./save/audio_stage3_dora/old_elderly/eval_samples \
  --output_path ./save/audio_stage3_dora/old_elderly/eval_results.json
```

Compare `eval_results.json` across styles. You want FID/diversity in a similar range to Stage-2-only; large degradation may indicate overfitting or collapsed style.

---

## 5. HumanML3D text-to-motion metrics (optional)

If you ever train or evaluate **text-only** motion on HumanML3D (no audio), the repo supports:

- **R_precision** — text–motion retrieval (higher is better).
- **FID** — generated vs real motion distribution.
- **Diversity** — variance of generated set.
- **MultiModality** — variance across samples for the same text.

These are used by `eval/eval_humanml.py` during training with `--eval_during_training`. They are **not** required for judging Stage 2 / Stage 3 audio + style; use them only if you work on text-only HumanML3D evaluation.

---

## 6. Practical checklist

- [ ] **Stage 2**: Run `evaluate_audio` (or at least FID + BAS if you have the right generator). BAS &gt; 0.3–0.4, FID in a reasonable range (e.g. &lt; 15).
- [ ] **Stage 3 training**: Loss goes down then plateaus; no NaNs; `adapter_final.pt` saved.
- [ ] **Stage 3 style**: 2×2 grids show clear style differences; motion stays on-prompt and roughly on-beat.
- [ ] **Optional**: FID/diversity on Stage-3 samples (e.g. per style) not much worse than Stage 2.
- [ ] **Optional**: RRBA on Stage-2 (or Stage-3) paired samples to confirm rhythm comes from audio.

If all of the above hold, you can reasonably conclude that training and style adaptation have “done well” given the current metrics and visual inspection.
