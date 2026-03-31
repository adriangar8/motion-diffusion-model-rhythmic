#!/usr/bin/env bash
# Text-only HumanML3D evaluation (debug mode) for all Stage-2 audio models.
# audio_mode=none, audio_guidance=0.0, text_guidance=2.5 — matches beataware eval.
# Run from project root: bash scripts/eval_text_quality_all.sh

set -euo pipefail
cd "$(dirname "$0")/.."

GUIDANCE=2.5

# Auto-detect weights location (Google Drive or local)
if [ -d "final_weights/stage2" ]; then
    W="final_weights/stage2"
else
    W="save"
fi

declare -A MODELS
MODELS["audio_stage2_librosa"]="$W/audio_stage2_librosa/model_final.pt"
MODELS["audio_stage2_wav2clip"]="$W/audio_stage2_wav2clip/model_final.pt"
MODELS["audio_stage2_wav2clip_mospa"]="$W/audio_stage2_wav2clip_mospa/model_final.pt"

for NAME in audio_stage2_librosa audio_stage2_wav2clip audio_stage2_wav2clip_mospa; do
    MODEL_PATH="${MODELS[$NAME]}"
    LOG="save/${NAME}/eval_text_quality_debug.log"

    echo ""
    echo "=========================================="
    echo "Evaluating: $NAME"
    echo "  model:    $MODEL_PATH"
    echo "  log:      $LOG"
    echo "=========================================="

    python -u -m eval.eval_audio_humanml_v2 \
        --model_path          "$MODEL_PATH" \
        --audio_mode          none \
        --audio_guidance_param 0.0 \
        --guidance_param      "$GUIDANCE" \
        --eval_mode           debug \
        2>&1 | tee "$LOG"

    echo ""
    echo "=== $NAME DONE ==="
done

echo ""
echo "ALL EVALUATIONS COMPLETE"
