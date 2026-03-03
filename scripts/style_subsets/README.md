# Style Subset Construction for HumanML3D

Implements the **"Style Subset Construction (Improved)"** pipeline from the
Multimodal Motion Generation Blueprint:

1. **CLIP embedding retrieval** — encode all HumanML3D captions with CLIP,
   then retrieve the top-k motions most similar to style prototype sentences.
2. **Manual verification** — sample a subset for human quality-checking, then
   export the cleaned style subset.

## Prerequisites

```bash
# Core (likely already installed)
pip install torch numpy pandas tqdm pyyaml pyarrow

# CLIP backend — either open_clip (preferred) or HF transformers
pip install open_clip_torch        # preferred: more model variants
# pip install transformers         # fallback (already in this repo)
```

## Quick Start

All commands are run from the **repository root**.

### Step 1 — Build the CLIP Index

Scan every `<motion_id>.txt` caption file in HumanML3D and compute CLIP
embeddings. Results are cached so subsequent runs are instant.

```bash
python scripts/style_subsets/style_subset_construction.py build-index \
    --text_root dataset/HumanML3D/texts \
    --cache_dir cache/clip_index \
    --clip_model ViT-L-14 \
    --device auto \
    --batch_size 256
```

**Output** (in `cache/clip_index/`):

| File | Contents |
|------|----------|
| `index_meta.parquet` | Caption metadata (motion_id, caption, caption_idx, source_file) |
| `index_embeds.npy` | Float32 CLIP embeddings, shape `(N_captions, 768)` |
| `index_config.json` | Model name, backend, timestamp, counts |

### Step 2 — Retrieve Style Subsets

For each style defined in `styles.yaml`, embed the prototype sentences,
average them into a query vector, and retrieve the top-k matching motions.

```bash
python scripts/style_subsets/style_subset_construction.py retrieve \
    --cache_dir cache/clip_index \
    --styles_yaml scripts/style_subsets/styles.yaml \
    --out_dir outputs/style_subsets \
    --k 1200
```

**Output** (per style, e.g. `outputs/style_subsets/elderly_slow/`):

| File | Contents |
|------|----------|
| `retrieved.csv` | Ranked list: rank, style, motion_id, best_caption, similarity, ... |
| `retrieved_motion_ids.txt` | One motion_id per line |

### Step 3 — Manual Review

#### Option A: Edit CSV in a spreadsheet

```bash
python scripts/style_subsets/style_subset_construction.py review \
    --retrieved_csv outputs/style_subsets/elderly_slow/retrieved.csv \
    --out_dir outputs/style_subsets/elderly_slow \
    --n 50 --seed 0
```

This creates `review_sample.csv`. Open it, fill the `decision` column with
`keep` or `drop`, then finalize:

```bash
python scripts/style_subsets/style_subset_construction.py review \
    --retrieved_csv outputs/style_subsets/elderly_slow/retrieved.csv \
    --out_dir outputs/style_subsets/elderly_slow \
    --finalize
```

#### Option B: Interactive CLI

```bash
python scripts/style_subsets/style_subset_construction.py review \
    --retrieved_csv outputs/style_subsets/elderly_slow/retrieved.csv \
    --out_dir outputs/style_subsets/elderly_slow \
    --n 50 --seed 0 --interactive
```

For each sample you will see the caption and similarity, then type
`y` (keep), `n` (drop), `s` (skip), or `q` (quit and save progress).
Progress is saved after every decision; you can resume any time.

**Output** (after finalize):

| File | Contents |
|------|----------|
| `review_sample.csv` | The reviewed sample with decisions |
| `cleaned.csv` | Full retrieved set minus dropped items |
| `cleaned_motion_ids.txt` | Kept motion IDs, one per line |

## Retargeted 100STYLE and (T, 263) format

The blueprint expects **Retargeted 100STYLE** in **HumanML3D 263-dim format** (same as
`new_joint_vecs`: shape `(T, 263)` per motion). If your 100STYLE download only
has joint positions `(T, 22, 3)`, convert them with:

```bash
python scripts/convert_100style_to_joint_vecs.py \
  --humanml3d_root /path/to/HumanML3D \
  --style_root /path/to/RETARGETED_100STYLE
```

This writes `(T, 263)` `.npy` files into `style_root/new_joint_vecs/`, using the
same motion pipeline as HumanML3D so the data loader can use 100STYLE like
HumanML3D.

## Customising Styles

Edit `styles.yaml` to add or modify styles. Each style needs a
`prototypes` list of 3–5 sentences describing the target movement quality.

```yaml
my_new_style:
  description: "Short human-readable description"
  prototypes:
    - "a person moves in the desired way"
    - "another example sentence"
    - "a third prototype"
```

## Arguments Reference

| Subcommand | Key flags | Default |
|------------|-----------|---------|
| `build-index` | `--text_root` (required), `--cache_dir`, `--clip_model`, `--device`, `--batch_size` | `cache/clip_index`, `ViT-L-14`, `auto`, `256` |
| `retrieve` | `--cache_dir`, `--styles_yaml`, `--out_dir`, `--k`, `--device` | `cache/clip_index`, `styles.yaml`, `outputs/style_subsets`, `1200`, `auto` |
| `review` | `--retrieved_csv` (required), `--out_dir` (required), `--n`, `--seed`, `--interactive`, `--finalize`, `--restart` | `50`, `0` |

Add `-v` / `--verbose` to any command for debug-level logging.

## Expected Scale

With HumanML3D (~14,616 motions, ~44k captions) and `k=1200`:

- **build-index**: ~2–5 min on GPU (one-time, cached).
- **retrieve**: ~10 sec per style (just dot products).
- **review**: ~5–10 min of human time per style (50 samples).
- **Final subsets**: ~1,100–1,200 motions per style after cleaning.
