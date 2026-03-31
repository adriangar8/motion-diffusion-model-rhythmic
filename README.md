# Adding Musical Rhythm to Text-to-Motion Diffusion Models

Karl Zeeny, Adrián García, Yash Bhardwaj — École Polytechnique, Institut Polytechnique de Paris (2026)

We extend the [Human Motion Diffusion Model (MDM)](https://arxiv.org/abs/2209.14916) with audio conditioning via cross-attention, enabling rhythmic synchronization with music while preserving text-to-motion quality. Text controls *what* the person does; audio controls *when* emphasis occurs.

**Key results:** BAS = 0.286 (within 0.001 of EDGE), R-Prec@3 = 0.695, using only frozen MDM backbone + learned audio modules trained on AIST++.

## Setup

### Environment

```bash
conda env create -f environment.yml
conda activate mdm
python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git
```

Requires: Python 3.9+, CUDA GPU, ffmpeg.

### Weights

Download `final_weights/` from **[Google Drive](https://drive.google.com/drive/folders/1CgdWvdJcV2QTfV_lmHDF2QXhWI4dtEK-?usp=sharing)** and place at the repo root:

```
final_weights/
├── pretrained/
│   ├── humanml_trans_enc_512/
│   │   └── model000200000.pt      # base MDM (Stage 1 init + text-only baseline)
│   └── edge/
│       └── checkpoint.pt           # EDGE baseline for BAS comparison
├── stage2/
│   ├── audio_stage2_wav2clip_beataware/   # best model (BAS=0.286)
│   ├── audio_stage2_wav2clip/
│   ├── audio_stage2_wav2clip_mospa/
│   └── audio_stage2_librosa/
└── evaluators/
    ├── t2m/        # HumanML3D text-motion evaluator networks
    └── glove/      # GloVe word embeddings for evaluator
```

The demo notebook auto-detects `final_weights/` and falls back to `./save/` if absent. For CLI commands below, replace `./final_weights/stage2/` with `./save/` if you trained locally.

### Data

1. **HumanML3D** — follow [original instructions](https://github.com/EricGuo5513/HumanML3D) and place in `./dataset/HumanML3D/`
2. **AIST++** — download from [AIST++](https://google.github.io/aistplusplus_dataset/), preprocess with `python data/preprocess_aist.py`, place in `./dataset/aist/`

The evaluation code loads evaluator weights from `./t2m/` and `./glove/` at the repo root. After downloading `final_weights/`, create symlinks:
```bash
ln -s ./final_weights/evaluators/t2m ./t2m
ln -s ./final_weights/evaluators/glove ./glove
```

## Training

All scripts use environment variable overrides. Set `PRETRAINED`, `AIST_DIR`, `HUMANML_DIR` if your paths differ from the defaults.

### Wav2CLIP + Beat-aware (best model)
```bash
bash scripts/train_stage2_wav2clip_beataware.sh
```

### Wav2CLIP (vanilla cross-attention)
```bash
# Same as above but without beat-aware bias or joint dropout:
python train/train_audio.py \
  --pretrained_path ./final_weights/pretrained/humanml_trans_enc_512/model000200000.pt \
  --aist_dir ./dataset/aist \
  --humanml_dir ./dataset/HumanML3D \
  --save_dir ./save/audio_stage2_wav2clip \
  --use_wav2clip \
  --batch_size 32 --lr 1e-4 --num_steps 100000
```

### Wav2CLIP + MOSPA (input token concatenation)
```bash
bash scripts/train_stage2_wav2clip_mospa.sh
```

### Librosa-only (52-dim, no Wav2CLIP)
```bash
python train/train_audio.py \
  --pretrained_path ./final_weights/pretrained/humanml_trans_enc_512/model000200000.pt \
  --aist_dir ./dataset/aist \
  --humanml_dir ./dataset/HumanML3D \
  --save_dir ./save/audio_stage2_librosa \
  --batch_size 32 --lr 1e-4 --num_steps 100000
```

Training takes ~3 hours on a single RTX 4000 Ada for 100k steps.

## Evaluation

### Text quality (HumanML3D benchmark)

Evaluates FID, R-Precision, Matching Score on HumanML3D test set (text-only mode, no audio):

```bash
python -m eval.eval_audio_humanml_v2 \
  --model_path ./final_weights/stage2/audio_stage2_wav2clip_beataware/model_final.pt \
  --audio_mode none \
  --audio_guidance_param 0.0 \
  --guidance_param 2.5 \
  --eval_mode debug
```

Or run all model variants:
```bash
bash scripts/eval_text_quality_all.sh
```

### Beat Alignment Score (BAS)

Generate motions on AIST++ test songs and compute BAS:

```bash
bash scripts/gen_stage2_aist_bas.sh
```

### CFG sweep

Sweep audio/text guidance scales and report BAS for each configuration:

```bash
bash scripts/cfg_sweep_bas.sh
```

### RRBA (Rhythmic Residual Beat Alignment)

```bash
python -m eval.rhythmic_residual \
  --model_path ./final_weights/stage2/audio_stage2_wav2clip_beataware/model_final.pt \
  --audio_path ./dataset/aist/audio_test_10/mBR0.wav \
  --text_prompt "a person walks forward"
```

## Demo

Interactive Jupyter notebook with text sweep, audio sweep, and free-form generation (with model selector):

```bash
jupyter notebook demo.ipynb
```

## Code Attribution

### Our code (new)

| File | Description |
|---|---|
| `model/audio_encoder.py` | 1D CNN audio encoder (519→512 dim) |
| `model/audio_cross_attention.py`, `audio_cross_attention_v2.py` | Beat-aware cross-attention with temporal locality and beat emphasis biases |
| `model/audio_features_v2.py` | Librosa v2 feature extraction (52-dim) |
| `model/audio_features_wav2clip.py` | Wav2CLIP + librosa feature extraction (519-dim) |
| `train/train_audio.py` | Stage 2 training loop (frozen backbone, audio modules only) |
| `sample/generate_audio.py` | Audio-conditioned generation with decomposed CFG |
| `eval/evaluate_audio.py` | Audio-conditioned evaluation pipeline |
| `eval/evaluate_all.py` | Combined text + audio evaluation |
| `eval/beat_align_score.py` | Beat Alignment Score computation |
| `eval/rhythmic_residual.py` | RRBA metric |
| `data/aist_dataset.py` | AIST++ data loader |
| `data/preprocess_aist.py`, `data/preprocess_audio_v2.py` | AIST++ preprocessing |
| `demo.ipynb` | Interactive demo notebook |
| `scripts/train_stage2_*.sh` | Training scripts for all model variants |
| `scripts/cfg_sweep_bas.sh`, `scripts/gen_stage2_aist_bas.sh` | Evaluation scripts |
| Audio conditioning integration in `model/mdm.py` | Cross-attention insertion, zero-init gating, audio encoder integration into existing MDM transformer blocks |

### Reused code (with modifications)

| Source | Files | What we used |
|---|---|---|
| [MDM (Tevet et al., ICLR 2023)](https://github.com/GuyTevet/motion-diffusion-model) | `model/mdm.py` (backbone), `diffusion/`, `data_loaders/`, `utils/`, `visualize/`, `eval/eval_humanml.py`, `train/train_mdm.py` | Transformer backbone, diffusion process, HumanML3D data loading, evaluation framework, SMPL visualization. We froze the backbone and inserted our audio modules. |
| [HumanML3D (Guo et al., CVPR 2022)](https://github.com/EricGuo5513/HumanML3D) | `data_loaders/humanml/` | Motion representation (263-dim), text-motion evaluator networks (t2m), GloVe embeddings |
| [CLIP (Radford et al., 2021)](https://github.com/openai/CLIP) | Used via MDM | Text encoding (ViT-B/32, 512-dim) |
| [Wav2CLIP (Wu et al., ICASSP 2022)](https://github.com/descriptinc/lyrebird-wav2clip) | Called in `model/audio_features_wav2clip.py` | Pretrained audio encoder for 512-dim semantic embeddings |
| [librosa (McFee et al., 2015)](https://librosa.org/) | Called in `model/audio_features*.py` | Audio feature extraction (onset, beats, RMS, chroma, spectral) |
| [EDGE (Tseng et al., CVPR 2023)](https://github.com/Stanford-TML/EDGE) | Used for baseline comparison only | BAS baseline (0.287); checkpoint in `final_weights/pretrained/edge/` |

## License

This code is distributed under an [MIT LICENSE](LICENSE). Depends on CLIP, SMPL, librosa, and Wav2CLIP — each with their own licenses.

## Citation

```bibtex
@techreport{zeeny2026rhythmic,
  title={Adding Musical Rhythm to Text-to-Motion Diffusion Models},
  author={Zeeny, Karl and Garc{\'i}a, Adri{\'a}n and Bhardwaj, Yash},
  institution={{\'E}cole Polytechnique, Institut Polytechnique de Paris},
  year={2026}
}
```
