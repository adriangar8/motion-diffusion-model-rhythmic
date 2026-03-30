# Implementation Audit (GCDM + Beataware)

Quick audit of the GCDM, Stage 2/3, and beataware integration. **Conclusion: implementation is correct;** one minor robustness fix was applied.

---

## 1. GCDM (`sample/gcdm.py`)

- **Formula**: Matches blueprint:  
  `ε̃ = ε(∅) + α[ λ(t)(ε(t,a)−ε(∅)) + (1−λ(t))(β_text(ε(text)−ε(∅)) + β_audio(ε(audio)−ε(∅)) ) ]`
- **Four passes**: full, text-only (`uncond_audio=True`), audio-only (`uncond=True`), both dropped. All use `deepcopy(y)` so `y` is not mutated.
- **λ(t)**: Linear in t; `t_frac = timesteps / diffusion_steps`. Sampling runs t from 999 down to 0, so early steps (high t) get high λ (joint term), late steps (low t) get low λ (independent terms). Correct for “style early, audio late”.
- **diffusion_steps**: Passed from `diffusion.num_timesteps` in `generate_audio` and `generate_stage3_style` so it matches the scheduler (1000).
- **Fix applied**: `has_audio` now uses `getattr(self.model, 'audio_conditioning', False)` so older or non-audio checkpoints do not assume the attribute exists.

---

## 2. MDM conditioning (`model/mdm.py`)

- **Text**: `force_mask = y.get('uncond', False)` → `mask_cond(enc_text, force_mask=force_mask)`. Uncond pass correctly zeros text.
- **Audio**: `mask_audio(..., force_mask=y.get('uncond_audio', False))`. Uncond pass correctly zeros audio.
- **Beat bias**: `beat_frames=y.get('beat_frames', None)` passed into cross-attention; beat index 513 (519-d) / 129 (145-d) is consistent with `train_audio.py` and generation scripts.

---

## 3. Training (`train/train_audio.py`)

- **Joint dropout (5%)**: `batch_cond['uncond'] = True` and `batch_cond['uncond_audio'] = True` for GCDM-style full dropout. Only that batch is modified; next batch is a new dict from the loader.
- **Independent dropout**: Handled inside MDM: `cond_mask_prob` (text) and `audio_cond_mask_prob` (audio). No explicit per-condition flags needed for the 95% case; `y.get('uncond', False)` is False when keys are absent.

---

## 4. AudioCFG vs GCDM

- **AudioCFG** (`generate_audio.py`): `uncond + text_scale*(text−uncond) + audio_scale*(full−text)`. Three passes; correct.
- **GCDM**: Four passes and timestep-dependent λ; used when `--use_gcdm` is set. When GCDM is used, `y['scale']` is still in `model_kwargs` but ignored by GCDM; no bug.

---

## 5. Generation scripts

- **generate_audio**: Optional `--use_gcdm`; builds `GCDMSampleModel` with `diffusion.num_timesteps`; default model path is beataware.
- **generate_stage3_style**: Passes `diffusion.num_timesteps` into GCDM; default `stage2_dir` is beataware.
- **compare_4_doras**: Forwards GCDM flags to `generate_stage3_style`; default `stage2_dir` and `adapter_root` are beataware; only builds grid when all four style videos exist.

---

## 6. Visualization

- **visualize_with_audio**: Uses `--samples_denormalized` for outputs from `generate_audio` / `generate_stage3_style` (samples are saved after `* std + mean`). Test script and `compare_4_doras` pass this flag.

---

## 7. Diffusion timestep convention

- `p_sample_loop` iterates `indices = range(num_timesteps - skip_timesteps)[::-1]`, so t = 999, 998, …, 0.
- Model is called with `model(x, self._scale_timesteps(t), **model_kwargs)`. With `rescale_timesteps=False` (default in `create_gaussian_diffusion`), the model receives integer t. GCDM’s `_lambda_t(timesteps)` with `diffusion_steps=1000` is therefore consistent.

---

## Summary

- **GCDM**: Correct formula, four passes, λ(t) and diffusion_steps usage are correct; `audio_conditioning` access made safe with `getattr`.
- **MDM**: Uncond/uncond_audio and beat_frames are wired correctly.
- **Training**: Joint and independent dropout are correct for GCDM.
- **Scripts**: Beataware defaults and GCDM wiring are consistent; no implementation mistakes found.
