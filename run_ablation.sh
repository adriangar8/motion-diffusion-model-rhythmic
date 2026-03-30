####################################################################################[start]
####################################################################################[start]

#!/bin/bash
###############################################################################
# ABLATION STUDY: Adrian v2 vs Yash wav2clip
#
# Sweeps three hyperparameters:
#   1. Skip timesteps (SDEdit noise level): 200, 300, 400, 500, 600, 700, 800
#   2. Audio guidance scale: 0.5, 1.0, 1.5, 2.5, 3.5, 5.0
#   3. Text guidance scale: 1.0, 1.5, 2.5, 3.5, 5.0
#
# For each sweep, the other hyperparameters are held at defaults:
#   - Default skip: 500
#   - Default audio guidance: 2.5
#   - Default text guidance: 2.5
#
# Estimated runtime: ~16-20 hours on a single GPU
#
# Output structure:
#   save/ablation/
#   ├── adrian_v2/
#   │   ├── skip_sweep/
#   │   │   └── {music}__{prompt}/          # sweep_skip with default guidance
#   │   ├── audio_guidance_sweep/
#   │   │   └── {music}__{prompt}_ag{scale}/ # fixed skip=500, vary audio guidance
#   │   └── text_guidance_sweep/
#   │       └── {music}__{prompt}_tg{scale}/ # fixed skip=500, vary text guidance
#   ├── yash_wav2clip/
#   │   └── (same structure)
#   └── results/
#       └── *.json
###############################################################################

set -e

# -- configuration --

ADRIAN_REPO="/Data/adrian.garcia/motion-diffusion-model-rhythmic"
ADRIAN_MODEL="${ADRIAN_REPO}/save/audio_stage2_v2/model_final.pt"

YASH_REPO="/Data/adrian.garcia/branch/motion-diffusion-model-rhythmic"
YASH_MODEL="${YASH_REPO}/save/audio_stage2_wav2clip_beataware/model_final.pt"

HUMANML_DIR="${ADRIAN_REPO}/dataset/HumanML3D"
AIST_AUDIO_DIR="${ADRIAN_REPO}/dataset/aist/audio"
GT_DANCE_DIR="${ADRIAN_REPO}/dataset/aist/processed/motions_263"

ABLATION_DIR="${ADRIAN_REPO}/save/ablation"
RESULTS_DIR="${ABLATION_DIR}/results"

NUM_SAMPLES=5
SEED=42
SMOOTH_SIGMA=1.0

# -- sweep configuration values --

# -- refinement skip sweep (default guidance) --

SKIP_VALUES="200 300 400 500 600 700 800"
DEFAULT_SKIP=500

# -- audio guidance sweep (fixed skip=500, default text guidance) --

AUDIO_GUIDANCE_VALUES="0.5 1.0 1.5 2.5 3.5 5.0"
DEFAULT_AUDIO_GUIDANCE=2.5

# -- text guidance sweep (fixed skip=500, default audio guidance) --

TEXT_GUIDANCE_VALUES="1.0 1.5 2.5 3.5 5.0"
DEFAULT_TEXT_GUIDANCE=2.5

# -- prompts and music tracks --

declare -a PROMPTS=(
    "a person walks forward and waves"
    "a person jogs in place"
    "a person stands and looks around"
)

declare -a PROMPT_SHORTS=(
    "walk_forward"
    "jog_in_place"
    "stand_look"
)

declare -a MUSIC_IDS=(
    "mBR0"
    "mPO0"
    "mHO0"
)

# -- helper: check audio exists --

check_audio() {
    [ -f "${AIST_AUDIO_DIR}/$1.wav" ]
}

# -- generation functions --

run_skip_sweep() {
    local repo=$1 model=$2 model_name=$3 music=$4 prompt="$5" pshort=$6
    local audio="${AIST_AUDIO_DIR}/${music}.wav"
    local outdir="${ABLATION_DIR}/${model_name}/skip_sweep/${music}__${pshort}"

    if [ -f "${outdir}/meta_refinement.json" ]; then
        echo "  [DONE] skip_sweep: ${model_name}/${music}__${pshort}"
        return
    fi

    echo ""
    echo "=== SKIP SWEEP: ${model_name} | ${music} | ${pshort} ==="
    cd "${repo}"

    python -m sample.refine_with_audio \
        --model_path "${model}" \
        --audio_path "${audio}" \
        --humanml_dir "${HUMANML_DIR}" \
        --text_prompt "${prompt}" \
        --output_dir "${outdir}" \
        --sweep_skip ${SKIP_VALUES} \
        --save_text_only \
        --save_full_audio \
        --num_samples ${NUM_SAMPLES} \
        --seed ${SEED} \
        --smooth_sigma ${SMOOTH_SIGMA} \
        --text_guidance ${DEFAULT_TEXT_GUIDANCE} \
        --audio_guidance ${DEFAULT_AUDIO_GUIDANCE}
}

run_audio_guidance_sweep() {
    local repo=$1 model=$2 model_name=$3 music=$4 prompt="$5" pshort=$6
    local audio="${AIST_AUDIO_DIR}/${music}.wav"

    for ag in ${AUDIO_GUIDANCE_VALUES}; do
        local outdir="${ABLATION_DIR}/${model_name}/audio_guidance_sweep/${music}__${pshort}_ag${ag}"

        if [ -f "${outdir}/meta_refinement.json" ]; then
            echo "  [DONE] ag=${ag}: ${model_name}/${music}__${pshort}"
            continue
        fi

        echo ""
        echo "=== AUDIO GUIDANCE ${ag}: ${model_name} | ${music} | ${pshort} ==="
        cd "${repo}"

        python -m sample.refine_with_audio \
            --model_path "${model}" \
            --audio_path "${audio}" \
            --humanml_dir "${HUMANML_DIR}" \
            --text_prompt "${prompt}" \
            --output_dir "${outdir}" \
            --skip_timesteps ${DEFAULT_SKIP} \
            --save_text_only \
            --num_samples ${NUM_SAMPLES} \
            --seed ${SEED} \
            --smooth_sigma ${SMOOTH_SIGMA} \
            --text_guidance ${DEFAULT_TEXT_GUIDANCE} \
            --audio_guidance ${ag}
    done
}

run_text_guidance_sweep() {
    local repo=$1 model=$2 model_name=$3 music=$4 prompt="$5" pshort=$6
    local audio="${AIST_AUDIO_DIR}/${music}.wav"

    for tg in ${TEXT_GUIDANCE_VALUES}; do
        local outdir="${ABLATION_DIR}/${model_name}/text_guidance_sweep/${music}__${pshort}_tg${tg}"

        if [ -f "${outdir}/meta_refinement.json" ]; then
            echo "  [DONE] tg=${tg}: ${model_name}/${music}__${pshort}"
            continue
        fi

        echo ""
        echo "=== TEXT GUIDANCE ${tg}: ${model_name} | ${music} | ${pshort} ==="
        cd "${repo}"

        python -m sample.refine_with_audio \
            --model_path "${model}" \
            --audio_path "${audio}" \
            --humanml_dir "${HUMANML_DIR}" \
            --text_prompt "${prompt}" \
            --output_dir "${outdir}" \
            --skip_timesteps ${DEFAULT_SKIP} \
            --save_text_only \
            --num_samples ${NUM_SAMPLES} \
            --seed ${SEED} \
            --smooth_sigma ${SMOOTH_SIGMA} \
            --text_guidance ${tg} \
            --audio_guidance ${DEFAULT_AUDIO_GUIDANCE}
    done
}

# -- evaluation functions --

eval_skip_sweep() {
    local repo=$1 model_name=$2 music=$3 pshort=$4
    local sample_dir="${ABLATION_DIR}/${model_name}/skip_sweep/${music}__${pshort}"
    local audio="${AIST_AUDIO_DIR}/${music}.wav"
    local outfile="${RESULTS_DIR}/${model_name}__${music}__${pshort}__skip_sweep.json"

    if [ -f "${outfile}" ]; then echo "  [DONE] ${outfile}"; return; fi
    if [ ! -d "${sample_dir}" ]; then return; fi

    echo "  Eval skip sweep: ${model_name} | ${music} | ${pshort}"
    cd "${repo}"

    python -m eval.evaluate_all \
        --sweep_mode skip \
        --sweep_dir "${sample_dir}" \
        --sweep_values ${SKIP_VALUES} \
        --text_dir "${sample_dir}" \
        --text_pattern "sample_text_only_*.npy" \
        --audio_pattern "sample_refined_skip{}_*.npy" \
        --gt_dance_dir "${GT_DANCE_DIR}" \
        --music_id "${music}" \
        --audio_path "${audio}" \
        --humanml_dir "${HUMANML_DIR}" \
        --output_path "${outfile}"
}

eval_guidance_sweep() {

    # -- evaluates audio_guidance or text_guidance sweep --

    local repo=$1 model_name=$2 music=$3 pshort=$4 sweep_type=$5
    
    # -- sweep_type: "audio_guidance" or "text_guidance" --

    local prefix="ag"
    local values="${AUDIO_GUIDANCE_VALUES}"
    local sweep_mode="audio_guidance"
    if [ "${sweep_type}" = "text_guidance" ]; then
        prefix="tg"
        values="${TEXT_GUIDANCE_VALUES}"
        sweep_mode="text_guidance"
    fi

    local outfile="${RESULTS_DIR}/${model_name}__${music}__${pshort}__${sweep_type}_sweep.json"
    if [ -f "${outfile}" ]; then echo "  [DONE] ${outfile}"; return; fi

    echo "  Eval ${sweep_type} sweep: ${model_name} | ${music} | ${pshort}"

    local audio="${AIST_AUDIO_DIR}/${music}.wav"
    local tmp_results=()

    cd "${repo}"

    python -c "
import json, os, sys, numpy as np
sys.path.insert(0, '.')

from eval.evaluate_all import (
    recover_joints, analyze_audio, load_ground_truth_dances,
    compute_average_dance_ke, evaluate_sample_set, metric_tf,
    print_sweep_table
)
from glob import glob

mean = np.load('${HUMANML_DIR}/Mean.npy')
std = np.load('${HUMANML_DIR}/Std.npy')
audio_info = analyze_audio('${audio}', 20)

gt_dir = '${GT_DANCE_DIR}'
profiles = load_ground_truth_dances(gt_dir, '${music}', mean, std)
if profiles:
    avg_ke = compute_average_dance_ke(profiles, max(len(k) for k in profiles))
else:
    avg_ke = np.array([])

# First pass: compute TF_max across all guidance values
all_tfs = []
values_str = '${values}'.split()
for val in values_str:
    d = '${ABLATION_DIR}/${model_name}/${sweep_type}_sweep/${music}__${pshort}_${prefix}' + val
    af = sorted(glob(os.path.join(d, 'sample_refined_skip${DEFAULT_SKIP}_*.npy')))
    tf = sorted(glob(os.path.join(d, 'sample_text_only_*.npy')))
    for a, t in zip(af, tf):
        ja = recover_joints(np.load(a), mean, std)
        jt = recover_joints(np.load(t), mean, std)
        all_tfs.append(metric_tf(ja, jt))
tf_max = max(all_tfs) if all_tfs else 1.0

# Second pass: evaluate each guidance value
sweep_results = {}
for val in values_str:
    d = '${ABLATION_DIR}/${model_name}/${sweep_type}_sweep/${music}__${pshort}_${prefix}' + val
    af = sorted(glob(os.path.join(d, 'sample_refined_skip${DEFAULT_SKIP}_*.npy')))
    tf = sorted(glob(os.path.join(d, 'sample_text_only_*.npy')))
    if not af or not tf:
        print(f'  No samples for {val}')
        continue
    print(f'\n--- ${sweep_type}={val} ---')
    agg = evaluate_sample_set(af, tf, audio_info, avg_ke, mean, std, 20, tf_max)
    sweep_results[val] = agg

results = {'sweep': sweep_results, 'sweep_mode': '${sweep_mode}', 'tf_max': tf_max}
print_sweep_table(sweep_results, '${sweep_type}')

with open('${outfile}', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f'Saved: ${outfile}')
"
}

eval_full_audio() {
    local repo=$1 model_name=$2 music=$3 pshort=$4
    local sample_dir="${ABLATION_DIR}/${model_name}/skip_sweep/${music}__${pshort}"
    local audio="${AIST_AUDIO_DIR}/${music}.wav"
    local outfile="${RESULTS_DIR}/${model_name}__${music}__${pshort}__full_audio.json"

    if [ -f "${outfile}" ]; then echo "  [DONE] ${outfile}"; return; fi

    local tmp_a=$(mktemp -d)
    local tmp_t=$(mktemp -d)
    ln -sf "${sample_dir}"/sample_full_audio_*.npy "${tmp_a}/" 2>/dev/null || true
    ln -sf "${sample_dir}"/sample_text_only_*.npy "${tmp_t}/" 2>/dev/null || true

    if [ -z "$(ls ${tmp_a}/*.npy 2>/dev/null)" ]; then
        rm -rf "${tmp_a}" "${tmp_t}"
        return
    fi

    echo "  Eval full audio: ${model_name} | ${music} | ${pshort}"
    cd "${repo}"

    python -m eval.evaluate_all \
        --audio_dir "${tmp_a}" \
        --text_dir "${tmp_t}" \
        --gt_dance_dir "${GT_DANCE_DIR}" \
        --music_id "${music}" \
        --audio_path "${audio}" \
        --humanml_dir "${HUMANML_DIR}" \
        --output_path "${outfile}"

    rm -rf "${tmp_a}" "${tmp_t}"
}

eval_comparison() {
    local skip=$1 music=$2 pshort=$3
    local outfile="${RESULTS_DIR}/comparison__${music}__${pshort}__skip${skip}.json"
    local audio="${AIST_AUDIO_DIR}/${music}.wav"

    if [ -f "${outfile}" ]; then echo "  [DONE] ${outfile}"; return; fi

    local a_dir="${ABLATION_DIR}/adrian_v2/skip_sweep/${music}__${pshort}"
    local y_dir="${ABLATION_DIR}/yash_wav2clip/skip_sweep/${music}__${pshort}"
    [ ! -d "${a_dir}" ] || [ ! -d "${y_dir}" ] && return

    local tmp_a=$(mktemp -d) tmp_y=$(mktemp -d) tmp_ta=$(mktemp -d) tmp_ty=$(mktemp -d)
    ln -sf "${a_dir}"/sample_refined_skip${skip}_*.npy "${tmp_a}/" 2>/dev/null || true
    ln -sf "${y_dir}"/sample_refined_skip${skip}_*.npy "${tmp_y}/" 2>/dev/null || true
    ln -sf "${a_dir}"/sample_text_only_*.npy "${tmp_ta}/" 2>/dev/null || true
    ln -sf "${y_dir}"/sample_text_only_*.npy "${tmp_ty}/" 2>/dev/null || true

    echo "  Compare at skip=${skip}: ${music} | ${pshort}"
    cd "${ADRIAN_REPO}"

    python -m eval.evaluate_all \
        --compare_mode \
        --model_dirs "${tmp_a}" "${tmp_y}" \
        --model_names "Adrian_v2" "Yash_wav2clip" \
        --text_dirs "${tmp_ta}" "${tmp_ty}" \
        --gt_dance_dir "${GT_DANCE_DIR}" \
        --music_id "${music}" \
        --audio_path "${audio}" \
        --humanml_dir "${HUMANML_DIR}" \
        --output_path "${outfile}"

    rm -rf "${tmp_a}" "${tmp_y}" "${tmp_ta}" "${tmp_ty}"
}

# -- main execution --

mkdir -p "${RESULTS_DIR}"

# -- filter available audio tracks --

VALID_MUSIC=()

for m in "${MUSIC_IDS[@]}"; do
    if check_audio "$m"; then VALID_MUSIC+=("$m"); fi
done

N_CONFIGS=$((2 * ${#VALID_MUSIC[@]} * ${#PROMPTS[@]}))
N_SKIP_RUNS=$((N_CONFIGS * 7))
N_AG_RUNS=$((N_CONFIGS * 6))
N_TG_RUNS=$((N_CONFIGS * 5))
N_TOTAL=$((N_SKIP_RUNS + N_AG_RUNS + N_TG_RUNS))

echo ""
echo "###############################################################################"
echo "# ABLATION STUDY CONFIGURATION"
echo "###############################################################################"
echo ""
echo "Models:        Adrian v2, Yash wav2clip"
echo "Audio tracks:  ${VALID_MUSIC[*]} (${#VALID_MUSIC[@]})"
echo "Text prompts:  ${#PROMPTS[@]}"
echo "Samples/config: ${NUM_SAMPLES}"
echo ""
echo "Sweeps:"
echo "  Skip:           ${SKIP_VALUES}  (${N_SKIP_RUNS} runs)"
echo "  Audio guidance:  ${AUDIO_GUIDANCE_VALUES}  (${N_AG_RUNS} runs)"
echo "  Text guidance:   ${TEXT_GUIDANCE_VALUES}  (${N_TG_RUNS} runs)"
echo ""
echo "Total generation runs: ${N_TOTAL}"
echo ""


# -- phase 1: generation --

echo "###############################################################################"
echo "# PHASE 1A: SKIP SWEEP GENERATION"
echo "###############################################################################"

for model_info in "adrian_v2:${ADRIAN_REPO}:${ADRIAN_MODEL}" "yash_wav2clip:${YASH_REPO}:${YASH_MODEL}"; do
    IFS=':' read -r mname mrepo mmodel <<< "${model_info}"
    for mid in "${VALID_MUSIC[@]}"; do
        for i in "${!PROMPTS[@]}"; do
            run_skip_sweep "${mrepo}" "${mmodel}" "${mname}" "${mid}" "${PROMPTS[$i]}" "${PROMPT_SHORTS[$i]}"
        done
    done
done

echo ""
echo "###############################################################################"
echo "# PHASE 1B: AUDIO GUIDANCE SWEEP GENERATION"
echo "###############################################################################"

for model_info in "adrian_v2:${ADRIAN_REPO}:${ADRIAN_MODEL}" "yash_wav2clip:${YASH_REPO}:${YASH_MODEL}"; do
    IFS=':' read -r mname mrepo mmodel <<< "${model_info}"
    for mid in "${VALID_MUSIC[@]}"; do
        for i in "${!PROMPTS[@]}"; do
            run_audio_guidance_sweep "${mrepo}" "${mmodel}" "${mname}" "${mid}" "${PROMPTS[$i]}" "${PROMPT_SHORTS[$i]}"
        done
    done
done

echo ""
echo "###############################################################################"
echo "# PHASE 1C: TEXT GUIDANCE SWEEP GENERATION"
echo "###############################################################################"

for model_info in "adrian_v2:${ADRIAN_REPO}:${ADRIAN_MODEL}" "yash_wav2clip:${YASH_REPO}:${YASH_MODEL}"; do
    IFS=':' read -r mname mrepo mmodel <<< "${model_info}"
    for mid in "${VALID_MUSIC[@]}"; do
        for i in "${!PROMPTS[@]}"; do
            run_text_guidance_sweep "${mrepo}" "${mmodel}" "${mname}" "${mid}" "${PROMPTS[$i]}" "${PROMPT_SHORTS[$i]}"
        done
    done
done

echo ""
echo "Phase 1 complete."

# -- phase 2: evaluation --

echo ""
echo "###############################################################################"
echo "# PHASE 2: EVALUATION"
echo "###############################################################################"

for model_info in "adrian_v2:${ADRIAN_REPO}" "yash_wav2clip:${YASH_REPO}"; do
    IFS=':' read -r mname mrepo <<< "${model_info}"
    for mid in "${VALID_MUSIC[@]}"; do
        for i in "${!PROMPTS[@]}"; do

            # -- skip sweep eval --
        
            eval_skip_sweep "${mrepo}" "${mname}" "${mid}" "${PROMPT_SHORTS[$i]}"
            
            # -- full audio eval --
        
            eval_full_audio "${mrepo}" "${mname}" "${mid}" "${PROMPT_SHORTS[$i]}"
            
            # -- audio guidance sweep eval --
        
            eval_guidance_sweep "${mrepo}" "${mname}" "${mid}" "${PROMPT_SHORTS[$i]}" "audio_guidance"
        
            # -- text guidance sweep eval --
        
            eval_guidance_sweep "${mrepo}" "${mname}" "${mid}" "${PROMPT_SHORTS[$i]}" "text_guidance"
        
        done
    done
done

echo ""
echo "Phase 2 complete."

# -- phase 3: model comparison --

echo ""
echo "###############################################################################"
echo "# PHASE 3: MODEL COMPARISON"
echo "###############################################################################"

for skip in 300 500 700; do
    for mid in "${VALID_MUSIC[@]}"; do
        for i in "${!PROMPTS[@]}"; do
            eval_comparison "${skip}" "${mid}" "${PROMPT_SHORTS[$i]}"
        done
    done
done

echo ""
echo "Phase 3 complete."

# -- phase 4: aggregate results --

echo ""
echo "###############################################################################"
echo "# PHASE 4: AGGREGATE RESULTS"
echo "###############################################################################"

cd "${ADRIAN_REPO}"

python -c "
import json, os, numpy as np
from glob import glob

results_dir = '${RESULTS_DIR}'
files = sorted(glob(os.path.join(results_dir, '*.json')))
print(f'Found {len(files)} result files')

# Collect all sweep data
data = {'skip_sweep': {}, 'audio_guidance_sweep': {}, 'text_guidance_sweep': {},
        'full_audio': {}, 'comparison': {}}

for f in files:
    name = os.path.basename(f).replace('.json', '')
    with open(f) as fh:
        content = json.load(fh)

    if 'comparison__' in name:
        parts = name.split('__')
        key = f'{parts[1]}__{parts[2]}__{parts[3]}'
        data['comparison'][key] = content
        continue

    parts = name.split('__')
    if len(parts) < 4:
        continue
    model, music, prompt, sweep_type = parts[0], parts[1], parts[2], parts[3]

    if model not in data.get(sweep_type, {}):
        if sweep_type.endswith('_sweep') or sweep_type == 'full_audio':
            if sweep_type not in data:
                data[sweep_type] = {}
            data[sweep_type].setdefault(model, [])

    if sweep_type == 'full_audio' and 'RhythmScore_mean' in content:
        data['full_audio'].setdefault(model, []).append({
            'music': music, 'prompt': prompt, **{k: content[k]
            for k in content if k.endswith('_mean') or k == 'n_samples'}
        })
    elif 'sweep' in content:
        for val, res in content['sweep'].items():
            if 'RhythmScore_mean' in res:
                data[sweep_type].setdefault(model, []).append({
                    'music': music, 'prompt': prompt, 'value': val,
                    **{k: res[k] for k in res if k.endswith('_mean')}
                })

# Print summary tables
def print_model_sweep(sweep_data, sweep_name, value_key='value'):
    print(f'\n{\"=\" * 115}')
    print(f'{sweep_name.upper()} — AVERAGED ACROSS PROMPTS AND AUDIO TRACKS')
    print(f'{\"=\" * 115}')
    for model in sorted(sweep_data.keys()):
        entries = sweep_data[model]
        if not entries:
            continue
        # Group by sweep value
        by_val = {}
        for e in entries:
            v = str(e.get(value_key, '?'))
            by_val.setdefault(v, []).append(e)

        print(f'\n  {model}:')
        print(f'  {\"Value\":>8} {\"RScore\":>8} {\"RDS\":>7} {\"BAS\":>7} {\"RRBA\":>7} '
              f'{\"Gap\":>7} {\"BDS\":>7} {\"DEC\":>7} {\"OMA\":>7} {\"TF\":>8} (n)')

        for v in sorted(by_val.keys(), key=lambda x: float(x)):
            es = by_val[v]
            n = len(es)
            def avg(k): return np.mean([e.get(k, 0) for e in es])
            print(f'  {v:>8} {avg(\"RhythmScore_mean\"):>8.4f} {avg(\"RDS_mean\"):>7.3f} '
                  f'{avg(\"BAS_mean\"):>7.3f} {avg(\"RRBA_mean\"):>7.3f} '
                  f'{avg(\"RRBA_BAS_gap_mean\"):>7.3f} {avg(\"BDS_mean\"):>7.3f} '
                  f'{avg(\"DEC_mean\"):>7.3f} {avg(\"OMA_mean\"):>7.3f} '
                  f'{avg(\"TF_mean\"):>8.4f} ({n})')

for stype in ['skip_sweep', 'audio_guidance_sweep', 'text_guidance_sweep']:
    if stype in data and data[stype]:
        print_model_sweep(data[stype], stype.replace('_', ' '))

# Best config per model
print(f'\n{\"=\" * 80}')
print('BEST CONFIGURATION PER MODEL (by RhythmScore)')
print(f'{\"=\" * 80}')

for stype in ['skip_sweep', 'audio_guidance_sweep', 'text_guidance_sweep']:
    if stype not in data:
        continue
    for model, entries in sorted(data[stype].items()):
        if not entries:
            continue
        best = max(entries, key=lambda x: x.get('RhythmScore_mean', 0))
        print(f'{model} ({stype}): value={best.get(\"value\",\"?\")} '
              f'[{best[\"music\"]}/{best[\"prompt\"]}] '
              f'→ RhythmScore={best.get(\"RhythmScore_mean\",0):.4f}')

# Save
agg_path = os.path.join(results_dir, 'aggregate_results.json')
with open(agg_path, 'w') as f:
    json.dump(data, f, indent=2, default=str)
print(f'\nAggregate saved to {agg_path}')
"

echo ""
echo "###############################################################################"
echo "# ABLATION STUDY COMPLETE"
echo "###############################################################################"
echo ""
echo "Results: ${RESULTS_DIR}/"
echo ""
echo "Key outputs:"
echo "  Skip sweep:      *__skip_sweep.json"
echo "  Audio guidance:   *__audio_guidance_sweep.json"
echo "  Text guidance:    *__text_guidance_sweep.json"
echo "  Comparisons:      comparison__*.json"
echo "  Summary:          aggregate_results.json"
echo ""

####################################################################################[end]
####################################################################################[end]