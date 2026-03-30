# Stage 3 DoRA Training Evaluation

## Is training happening correctly?

**Yes.** The loss **does** go down and DoRA weights change:

- **Epoch 1 avg loss:** 0.0637  
- **Best epochs (25–26, 60, 67):** ~0.051–0.053  
- **Epoch 80:** 0.0553  

So we get roughly **~15–20% relative reduction** (0.064 → 0.051). The curve is **noisy** and the **final gain looks small** for two main reasons:

---

## 1. LR schedule is too aggressive

We use `CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)` with `total_steps = 70 * 80 = 5600`.

- By **epoch 40** LR is already ~5e-5 (half of 1e-4).
- By **epoch 60** LR ≈ 1.55e-5.
- By **epoch 70–80** LR is **1e-6** — updates are tiny.

So most of the **effective learning** happens in the first ~30–40 epochs. After that, the model barely moves and the loss just oscillates. That’s why it looks like “loss isn’t going down”: the second half of training is effectively at very low LR.

**Fix:** Use a slower decay so LR stays useful for all 80 epochs (e.g. `T_max = 2 * total_steps` or constant LR for the first half, then cosine).

---

## 2. Mixed data dilutes the style signal

- **100STYLE:** 320 samples (old_elderly style).
- **HumanML3D retrieved:** 800 samples (generic, style‑relevant but not “old_elderly” only).

So only **~28%** of batches are pure style; the rest are generic. The adapter is optimizing **average reconstruction** over both, not only “old_elderly”. So:

- Loss can decrease (better average fit).
- But the adapter is not pushed as hard toward the target style, and the loss will be noisier (style vs generic pull in different directions).

**Fix:** Upsample 100STYLE in the combined dataset (e.g. 3× so ~55% of samples are style) so the gradient is more often style‑specific.

---

## Summary

| Aspect | Status |
|--------|--------|
| Loss definition | Correct (diffusion MSE, masked). |
| DoRA params trained | Yes (magnitude + lora_A, lora_B). |
| Gradients / updates | Yes (loss and DoRA weights change). |
| Loss trend | Goes down from ~0.064 to ~0.051–0.055, then plateaus. |
| Noisy curve | Expected: mixed data + per‑batch timestep sampling. |
| Why it “stops” improving | LR decays to 1e-6 by end; second half barely updates. |
| Style strength | Limited by 28% style data; rest is generic. |

**Conclusion:** Training is **correct**. The modest and noisy improvement comes from an **overly aggressive cosine schedule** and **mixed data** with a minority of style samples. Use a **slower LR decay** and **style upsampling** (or more style data) for a clearer, stronger drop in loss and more style‑specific adapters.
