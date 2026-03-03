#!/usr/bin/env python3
"""
Style Subset Construction for HumanML3D
========================================
Implements the "Style Subset Construction (Improved)" pipeline:

  Step 1 — CLIP embedding retrieval using prototype sentences per style.
  Step 2 — Manual verification helper (CSV review + interactive CLI).

Subcommands:
  build-index   Scan HumanML3D captions, compute CLIP embeddings, cache to disk.
  retrieve      Load prototypes from styles.yaml, retrieve top-k motions per style.
  review        Sample items for manual verification; optionally run interactive CLI.

Examples:
  python style_subset_construction.py build-index \\
      --text_root dataset/HumanML3D/texts --cache_dir cache/clip_index

  python style_subset_construction.py retrieve \\
      --cache_dir cache/clip_index --styles_yaml styles.yaml --out_dir outputs --k 1200

  python style_subset_construction.py review \\
      --retrieved_csv outputs/elderly_slow/retrieved.csv \\
      --out_dir outputs/elderly_slow --n 50 --seed 0 --interactive
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

logger = logging.getLogger("style_subset")


# ---------------------------------------------------------------------------
# CLIP backend abstraction
# ---------------------------------------------------------------------------

class CLIPBackend:
    """Unified interface over open_clip or HuggingFace transformers CLIP."""

    def __init__(self, model_name: str, device: str) -> None:
        self.device = device
        self.model_name = model_name
        self._backend: str = ""
        self._model: Any = None
        self._tokenizer: Any = None
        self._load(model_name)

    # -- loading --------------------------------------------------------

    def _load(self, model_name: str) -> None:
        """Try open_clip first; fall back to HuggingFace transformers."""
        if self._try_open_clip(model_name):
            return
        if self._try_transformers(model_name):
            return
        raise RuntimeError(
            "No CLIP backend available. "
            "Install open_clip_torch  (pip install open_clip_torch) "
            "or huggingface transformers  (pip install transformers)."
        )

    def _try_open_clip(self, model_name: str) -> bool:
        try:
            import open_clip  # type: ignore[import-untyped]
        except ImportError:
            logger.info("open_clip not installed, skipping.")
            return False

        oc_model = model_name.replace("/", "-")
        try:
            pretrained_opts = open_clip.list_pretrained_tags_by_model(oc_model)
        except Exception:
            pretrained_opts = []
        if not pretrained_opts:
            logger.info(f"open_clip: no pretrained weights for {oc_model}, skipping.")
            return False

        pretrained = pretrained_opts[0]
        for tag in pretrained_opts:
            if "laion2b" in str(tag).lower():
                pretrained = tag
                break

        try:
            model, _, _ = open_clip.create_model_and_transforms(
                oc_model, pretrained=pretrained, device=self.device
            )
            tokenizer = open_clip.get_tokenizer(oc_model)
        except Exception as exc:
            logger.info(f"open_clip model creation failed ({exc}), skipping.")
            return False

        self._model = model
        self._tokenizer = tokenizer
        self._backend = "open_clip"
        logger.info(f"Loaded open_clip: {oc_model} / {pretrained} on {self.device}")
        return True

    def _try_transformers(self, model_name: str) -> bool:
        try:
            from transformers import CLIPModel, CLIPTokenizerFast  # type: ignore[import-untyped]
        except ImportError:
            logger.info("transformers not installed, skipping.")
            return False

        hf_name = self._hf_model_id(model_name)
        try:
            model = CLIPModel.from_pretrained(hf_name).to(self.device)
            tokenizer = CLIPTokenizerFast.from_pretrained(hf_name)
        except Exception as exc:
            logger.info(f"transformers CLIP load failed ({exc}), skipping.")
            return False

        self._model = model
        self._tokenizer = tokenizer
        self._backend = "transformers"
        logger.info(f"Loaded transformers CLIP: {hf_name} on {self.device}")
        return True

    @staticmethod
    def _hf_model_id(model_name: str) -> str:
        """Map short names to HuggingFace model identifiers."""
        table = {
            "ViT-L-14": "openai/clip-vit-large-patch14",
            "ViT-L/14": "openai/clip-vit-large-patch14",
            "ViT-B-32": "openai/clip-vit-base-patch32",
            "ViT-B/32": "openai/clip-vit-base-patch32",
            "ViT-B-16": "openai/clip-vit-base-patch16",
            "ViT-B/16": "openai/clip-vit-base-patch16",
        }
        return table.get(model_name, model_name)

    # -- encoding -------------------------------------------------------

    @torch.no_grad()
    def encode_texts(self, texts: List[str], batch_size: int = 256) -> np.ndarray:
        """Return L2-normalised CLIP text embeddings as float32 (N, D)."""
        self._model.eval()
        parts: List[np.ndarray] = []

        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
            batch = texts[start : start + batch_size]

            if self._backend == "open_clip":
                tokens = self._tokenizer(batch).to(self.device)
                emb = self._model.encode_text(tokens)
            else:
                tokens = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                ).to(self.device)
                emb = self._model.get_text_features(**tokens)

            emb = emb / emb.norm(dim=-1, keepdim=True)
            parts.append(emb.cpu().float().numpy())

        return np.concatenate(parts, axis=0)

    @property
    def backend_name(self) -> str:
        return self._backend


# ---------------------------------------------------------------------------
# Caption parsing
# ---------------------------------------------------------------------------

def parse_caption_file(filepath: Path) -> List[str]:
    """Parse a single HumanML3D caption file.

    Each non-empty line is ``caption#tokens#start#end``.
    We take everything before the first ``#`` as the caption.
    If there is no ``#``, the whole line is the caption.
    """
    captions: List[str] = []
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            caption = line.split("#")[0].strip()
            if caption:
                captions.append(caption)
    return captions


def build_caption_dataframe(text_root: Path) -> pd.DataFrame:
    """Scan ``text_root/*.txt`` and return a DataFrame of all captions.

    Columns: ``motion_id  caption  caption_idx  source_file``
    """
    txt_files = sorted(text_root.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(
            f"No .txt caption files found in {text_root}. "
            "Check --text_root path."
        )

    logger.info(f"Scanning {len(txt_files)} caption files in {text_root}")
    records: List[Dict[str, Any]] = []
    for fpath in tqdm(txt_files, desc="Parsing captions"):
        motion_id = fpath.stem
        for idx, cap in enumerate(parse_caption_file(fpath)):
            records.append(
                {
                    "motion_id": motion_id,
                    "caption": cap,
                    "caption_idx": idx,
                    "source_file": str(fpath),
                }
            )

    df = pd.DataFrame(records)
    logger.info(
        f"Built caption index: {len(df)} captions from "
        f"{df['motion_id'].nunique()} unique motions"
    )
    return df


# ---------------------------------------------------------------------------
# Index caching
# ---------------------------------------------------------------------------

def save_index(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    cache_dir: Path,
    model_name: str,
    backend: str,
) -> None:
    """Write caption metadata + embeddings to *cache_dir*."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    meta_path = cache_dir / "index_meta.parquet"
    embed_path = cache_dir / "index_embeds.npy"
    config_path = cache_dir / "index_config.json"

    df.to_parquet(meta_path, index=False)
    np.save(embed_path, embeddings)

    config = {
        "model_name": model_name,
        "backend": backend,
        "num_captions": len(df),
        "num_motions": int(df["motion_id"].nunique()),
        "embed_dim": int(embeddings.shape[1]),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(config_path, "w") as fh:
        json.dump(config, fh, indent=2)

    logger.info(f"Saved index ({len(df)} captions, dim={embeddings.shape[1]}) → {cache_dir}")


def load_index(cache_dir: Path) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    """Load a previously saved caption index from *cache_dir*."""
    meta_path = cache_dir / "index_meta.parquet"
    embed_path = cache_dir / "index_embeds.npy"
    config_path = cache_dir / "index_config.json"

    for p in (meta_path, embed_path, config_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Cache file missing: {p}. Run `build-index` first."
            )

    df = pd.read_parquet(meta_path)
    embeddings = np.load(embed_path)
    with open(config_path) as fh:
        config = json.load(fh)

    logger.info(
        f"Loaded cached index: {config['num_captions']} captions, "
        f"{config['num_motions']} motions, dim={config['embed_dim']} "
        f"(model={config['model_name']}, backend={config['backend']})"
    )
    return df, embeddings, config


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_style(
    style_name: str,
    prototypes: List[str],
    clip: CLIPBackend,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    k: int,
) -> pd.DataFrame:
    """Retrieve top-*k* unique motions whose captions best match *prototypes*.

    For each motion_id the single caption with the highest cosine similarity
    to the style query (mean of prototype embeddings) is kept.  The resulting
    motion-level ranking is truncated at *k*.
    """
    proto_embeds = clip.encode_texts(prototypes, batch_size=len(prototypes))
    query = proto_embeds.mean(axis=0)
    query /= np.linalg.norm(query)

    similarities = embeddings @ query  # (N_captions,)

    scored = df.copy()
    scored["similarity"] = similarities

    best_per_motion = scored.loc[scored.groupby("motion_id")["similarity"].idxmax()]
    best_per_motion = (
        best_per_motion
        .sort_values("similarity", ascending=False)
        .head(k)
        .reset_index(drop=True)
    )
    best_per_motion.insert(0, "rank", range(1, len(best_per_motion) + 1))

    result = best_per_motion[
        ["rank", "motion_id", "caption", "caption_idx", "similarity", "source_file"]
    ].rename(columns={"caption": "best_caption", "caption_idx": "best_caption_idx"})

    sim_lo, sim_hi = result["similarity"].min(), result["similarity"].max()
    logger.info(
        f"[{style_name}] Retrieved {len(result)} motions "
        f"(similarity {sim_lo:.4f} – {sim_hi:.4f})"
    )
    return result


# ---------------------------------------------------------------------------
# CLI: build-index
# ---------------------------------------------------------------------------

def cmd_build_index(args: argparse.Namespace) -> None:
    """Scan all HumanML3D captions, compute CLIP embeddings, save to cache."""
    text_root = Path(args.text_root).resolve()
    cache_dir = Path(args.cache_dir).resolve()

    if not text_root.is_dir():
        logger.error(f"text_root does not exist or is not a directory: {text_root}")
        sys.exit(1)

    df = build_caption_dataframe(text_root)

    device = _resolve_device(args.device)
    logger.info(f"Using device: {device}")

    clip = CLIPBackend(args.clip_model, device)
    embeddings = clip.encode_texts(df["caption"].tolist(), batch_size=args.batch_size)

    save_index(df, embeddings, cache_dir, args.clip_model, clip.backend_name)
    logger.info("build-index complete ✓")


# ---------------------------------------------------------------------------
# CLI: retrieve
# ---------------------------------------------------------------------------

def cmd_retrieve(args: argparse.Namespace) -> None:
    """Load cached index + styles YAML, retrieve top-k motions per style."""
    cache_dir = Path(args.cache_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    styles_path = Path(args.styles_yaml)

    if not styles_path.exists():
        logger.error(f"Styles YAML not found: {styles_path}")
        sys.exit(1)

    df, embeddings, config = load_index(cache_dir)

    with open(styles_path, "r") as fh:
        styles_cfg: Dict[str, Any] = yaml.safe_load(fh)

    device = _resolve_device(args.device)
    clip = CLIPBackend(config["model_name"], device)

    for style_name, style_def in styles_cfg.items():
        prototypes: List[str] = style_def["prototypes"]
        logger.info(
            f"Style '{style_name}': {len(prototypes)} prototypes, retrieving top-{args.k}"
        )

        result = retrieve_style(style_name, prototypes, clip, df, embeddings, k=args.k)

        style_dir = out_dir / style_name
        style_dir.mkdir(parents=True, exist_ok=True)

        result.insert(1, "style", style_name)
        result.to_csv(style_dir / "retrieved.csv", index=False)
        (style_dir / "retrieved_motion_ids.txt").write_text(
            "\n".join(result["motion_id"].tolist()) + "\n"
        )

        logger.info(f"  → {style_dir}/retrieved.csv  ({len(result)} motions)")

    logger.info("retrieve complete ✓")


# ---------------------------------------------------------------------------
# CLI: review
# ---------------------------------------------------------------------------

def cmd_review(args: argparse.Namespace) -> None:
    """Manual verification workflow: sample, review, produce cleaned output."""
    retrieved_csv = Path(args.retrieved_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not retrieved_csv.exists():
        logger.error(f"Retrieved CSV not found: {retrieved_csv}")
        sys.exit(1)

    full_df = pd.read_csv(retrieved_csv)
    review_path = out_dir / "review_sample.csv"

    # ---- create or resume review file ----
    if review_path.exists() and not args.restart:
        logger.info(f"Resuming from existing review: {review_path}")
        sample = pd.read_csv(review_path, dtype={"decision": str, "notes": str})
        sample["decision"] = sample["decision"].fillna("")
        sample["notes"] = sample["notes"].fillna("")
    else:
        n = min(args.n, len(full_df))
        rng = random.Random(args.seed)
        indices = sorted(rng.sample(range(len(full_df)), n))
        sample = full_df.iloc[indices].copy().reset_index(drop=True)
        sample = sample.rename(columns={"best_caption": "caption"})
        sample["decision"] = ""
        sample["notes"] = ""
        cols = ["motion_id", "style", "caption", "similarity", "decision", "notes"]
        available = [c for c in cols if c in sample.columns]
        sample = sample[available]
        sample.to_csv(review_path, index=False)
        logger.info(f"Created review file ({n} samples): {review_path}")

    # ---- interactive mode ----
    if args.interactive:
        _interactive_review(sample, review_path)

    # ---- finalize ----
    if args.finalize or args.interactive:
        _finalize_review(sample, review_path, full_df, out_dir)
    elif not args.interactive:
        logger.info(
            f"\nReview file ready at:\n  {review_path}\n\n"
            f"Edit the 'decision' column  (keep / drop)  in your editor,\n"
            f"then re-run with --finalize to produce cleaned outputs.\n"
            f"Or re-run with --interactive for CLI-guided review."
        )


def _interactive_review(sample: pd.DataFrame, review_path: Path) -> None:
    """Walk through each unreviewed item and prompt keep / drop / skip."""
    total = len(sample)
    done = (sample["decision"].isin(["keep", "drop"])).sum()
    logger.info(f"Interactive review: {done}/{total} already decided")

    for i in range(total):
        row = sample.iloc[i]
        if row["decision"] in ("keep", "drop"):
            continue

        print(f"\n{'=' * 64}")
        print(f"  [{i + 1}/{total}]  motion_id : {row['motion_id']}")
        print(f"  caption   : {row['caption']}")
        print(f"  similarity: {row['similarity']:.4f}")
        print(f"{'=' * 64}")

        while True:
            ans = input("  keep? [y / n / s(skip) / q(quit)]: ").strip().lower()
            if ans in ("y", "n", "s", "q"):
                break
            print("  → enter y, n, s, or q")

        if ans == "q":
            sample.to_csv(review_path, index=False)
            logger.info("Quit. Progress saved.")
            return
        elif ans == "y":
            sample.at[i, "decision"] = "keep"
        elif ans == "n":
            sample.at[i, "decision"] = "drop"

        sample.to_csv(review_path, index=False)

    logger.info("All items reviewed. Progress saved.")


def _finalize_review(
    sample: pd.DataFrame,
    review_path: Path,
    full_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Produce cleaned outputs.

    Any motion_id explicitly marked **drop** in the review sample is removed
    from the full retrieved set.  Everything else is kept — the review is a
    quality-assurance spot-check, not a whitelist.
    """
    sample = pd.read_csv(review_path, dtype={"decision": str, "notes": str})
    sample["decision"] = sample["decision"].fillna("")

    drop_ids = set(sample.loc[sample["decision"] == "drop", "motion_id"])
    kept = full_df[~full_df["motion_id"].isin(drop_ids)].copy()

    kept.to_csv(out_dir / "cleaned.csv", index=False)
    (out_dir / "cleaned_motion_ids.txt").write_text(
        "\n".join(kept["motion_id"].tolist()) + "\n"
    )

    n_reviewed = (sample["decision"] != "").sum()
    n_drop = len(drop_ids)
    logger.info(
        f"Finalized: reviewed {n_reviewed}/{len(sample)}, "
        f"dropped {n_drop} motion(s), "
        f"kept {len(kept)}/{len(full_df)} motions "
        f"→ {out_dir / 'cleaned.csv'}"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_device(device_str: str) -> str:
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Style Subset Construction for HumanML3D",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")

    sub = parser.add_subparsers(dest="command")
    sub.required = True

    # ---- build-index ----
    p_idx = sub.add_parser(
        "build-index",
        help="Build CLIP text-embedding index from HumanML3D caption files",
    )
    p_idx.add_argument(
        "--text_root", required=True,
        help="Path to HumanML3D texts/ directory containing <motion_id>.txt files",
    )
    p_idx.add_argument(
        "--cache_dir", default="cache/clip_index",
        help="Directory to store cached index (default: cache/clip_index)",
    )
    p_idx.add_argument(
        "--clip_model", default="ViT-L-14",
        help="CLIP model name (default: ViT-L-14)",
    )
    p_idx.add_argument(
        "--device", default="auto",
        help="Compute device: auto | cpu | cuda | cuda:0 (default: auto)",
    )
    p_idx.add_argument(
        "--batch_size", type=int, default=256,
        help="Batch size for CLIP text encoding (default: 256)",
    )

    # ---- retrieve ----
    p_ret = sub.add_parser(
        "retrieve",
        help="Retrieve top-k motions per style from cached CLIP index",
    )
    p_ret.add_argument(
        "--cache_dir", default="cache/clip_index",
        help="Path to cached index directory (default: cache/clip_index)",
    )
    p_ret.add_argument(
        "--styles_yaml",
        default=str(Path(__file__).resolve().parent / "styles.yaml"),
        help="Path to styles YAML file (default: styles.yaml beside this script)",
    )
    p_ret.add_argument(
        "--out_dir", default="outputs/style_subsets",
        help="Output directory for per-style results (default: outputs/style_subsets)",
    )
    p_ret.add_argument(
        "--k", type=int, default=1200,
        help="Number of unique motions to retrieve per style (default: 1200)",
    )
    p_ret.add_argument(
        "--device", default="auto",
        help="Compute device for prototype encoding (default: auto)",
    )

    # ---- review ----
    p_rev = sub.add_parser(
        "review",
        help="Manual verification workflow for a retrieved style subset",
    )
    p_rev.add_argument(
        "--retrieved_csv", required=True,
        help="Path to a style's retrieved.csv",
    )
    p_rev.add_argument(
        "--out_dir", required=True,
        help="Output directory for review artefacts",
    )
    p_rev.add_argument(
        "--n", type=int, default=50,
        help="Number of samples to review (default: 50)",
    )
    p_rev.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for reproducible sampling (default: 0)",
    )
    p_rev.add_argument(
        "--interactive", action="store_true",
        help="Run interactive CLI review (print each item, prompt keep/drop)",
    )
    p_rev.add_argument(
        "--finalize", action="store_true",
        help="Produce cleaned.csv and cleaned_motion_ids.txt from review decisions",
    )
    p_rev.add_argument(
        "--restart", action="store_true",
        help="Discard any existing review progress and re-sample",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(args.verbose)

    dispatch = {
        "build-index": cmd_build_index,
        "retrieve": cmd_retrieve,
        "review": cmd_review,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
