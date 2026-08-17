#!/bin/bash
# Evaluation pipeline — iterates over experiment folders, runs fb_eval_greedy.py
# for each experiment, and saves named CSV + coco_instances.json to a central output directory.
#
# Syntax check:  bash -n run_evaluation_pipeline.sh && echo "Syntax OK"
#
# Paths below are the evaluation host's; override any of them from the environment,
# e.g.  DATA_ROOT=/mnt/data bash run_evaluation_pipeline.sh

set -uo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_ROOT="${DATA_ROOT:-/data/sam3}"
# The matcher (fb_eval_greedy.py) ships next to this script.
SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
GT_JSON="${GT_JSON:-$DATA_ROOT/fb_yolo/insects/labels/val/instances_default.json}"
IMG_DIR="${IMG_DIR:-$DATA_ROOT/fb_yolo/insects/images/val}"

# All outputs collected here — one named CSV + JSON per experiment
OUTPUT_BASE="${OUTPUT_BASE:-$DATA_ROOT/final_experiments_results_for_thesis}"

# ── Experiment sources ─────────────────────────────────────────────────────────
# 1. Single experiment folder (contains coco_instances.json directly)
SINGLE_EXP="${SINGLE_EXP:-$DATA_ROOT/output/fine_tuned_sam3_with_lora_checkpoint_10_mask_5_score_005}"

# 2. Multi-experiment parent: each sub-folder is one experiment
MULTI_EXP_DIR="${MULTI_EXP_DIR:-$DATA_ROOT/output/results/rerun_evaluation_using_fb_eval_greedy_refactored}"

# ── Setup ──────────────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_BASE"
ERRORS=0

# ── Helper: evaluate one experiment ───────────────────────────────────────────
# Usage: run_eval <path/to/coco_instances.json> <experiment_name>
run_eval() {
    local coco_json="$1"
    local exp_name="$2"

    local tmp_out="$OUTPUT_BASE/.tmp_${exp_name}"
    local log_file="$OUTPUT_BASE/${exp_name}_eval.log"
    local out_csv="$OUTPUT_BASE/${exp_name}_combined_results.csv"
    local out_json="$OUTPUT_BASE/${exp_name}_coco_instances.json"

    echo ""
    echo "=============================="
    echo "Experiment : $exp_name"
    echo "Input      : $coco_json"
    echo "=============================="

    # Skip if both outputs already exist
    if [[ -f "$out_csv" && -f "$out_json" ]]; then
        echo "  [SKIP] Outputs already exist. Delete them to re-run."
        return
    fi

    mkdir -p "$tmp_out"

    if python3 -u "$SCRIPT_DIR/fb_eval_greedy.py" \
            -p "$coco_json" \
            -g "$GT_JSON" \
            -I "$IMG_DIR" \
            -o "$tmp_out" \
            -c --combine --coco-matching \
            > "$log_file" 2>&1; then

        cp "$tmp_out/combined_results.csv" "$out_csv"
        cp "$coco_json"                    "$out_json"
        rm -rf "$tmp_out"
        echo "  [OK]    CSV : $(basename "$out_csv")"
        echo "  [OK]    JSON: $(basename "$out_json")"
        echo "  [OK]    Log : $(basename "$log_file")"
    else
        rm -rf "$tmp_out"
        ERRORS=$((ERRORS + 1))
        echo "  [ERROR] Evaluation failed for: $exp_name"
        echo "          See log: $log_file"
    fi
}

# ── 1. Single experiment ───────────────────────────────────────────────────────
echo ">>> Single experiment: $SINGLE_EXP"
if [[ -f "$SINGLE_EXP/coco_instances.json" ]]; then
    run_eval "$SINGLE_EXP/coco_instances.json" "$(basename "$SINGLE_EXP")"
else
    echo "  [WARNING] No coco_instances.json found in $SINGLE_EXP — skipping"
    ERRORS=$((ERRORS + 1))
fi

# ── 2. Sub-experiments loop ────────────────────────────────────────────────────
echo ""
echo ">>> Multi-experiment directory: $MULTI_EXP_DIR"
for exp_dir in "$MULTI_EXP_DIR"/*/; do
    [[ -d "$exp_dir" ]] || continue
    exp_name="$(basename "${exp_dir%/}")"
    if [[ -f "$exp_dir/coco_instances.json" ]]; then
        run_eval "$exp_dir/coco_instances.json" "$exp_name"
    else
        echo "  [WARNING] No coco_instances.json in $exp_dir — skipping"
    fi
done

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo "=================================="
echo "Pipeline complete."
echo "Results saved to: $OUTPUT_BASE"
if [[ $ERRORS -gt 0 ]]; then
    echo "WARNING: $ERRORS experiment(s) had errors. Check the .log files above."
    exit 1
fi
echo "All experiments evaluated successfully."
echo "=================================="