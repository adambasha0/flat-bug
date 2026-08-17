# Exploratory / superseded analysis scripts

Working scripts written while developing the thesis evaluation. They are kept for
provenance — none of them produces a figure or number that appears in the final
thesis. The scripts that do are in [`../thesis/`](../thesis/).

| Folder | What it was for |
| --- | --- |
| `ap_curves/` | Successive iterations of the PR-curve/AP script — per-model, per-score-threshold and COCO-matching variants. All superseded by `thesis/experiments_1_3/ap_curves_thesis_experiments_refactored.R`, which produces every experiment's curve from one code path. |
| `false_positives/` | Characterising false positives: confidence buckets, FP-to-GT overlap (unlabeled insect vs genuine hallucination), FP trajectories between checkpoints. Fed the Experiment 3 motivation. |
| `thresholds_and_masks/` | Mask-threshold and score-threshold sweeps, plus the mask-dilation estimate. Fixed the shared configuration (mask 0.5, score 0.005). |
| `confidence/` | One-off side-by-side TP-vs-FP confidence figures (base vs fine-tuned, fine-tuned vs LoRA), later folded into `generate_thesis_graphs.py`. |
| `lora_and_planning/` | LoRA feasibility analysis and the phase-planning diagnostic used to decide whether the precision target was reachable at all. |
| `misc/` | IoU counting and CSV row-counting utilities. |
| `reports/` | Written notes on two dead ends: why the SAM3 mask threshold barely moves the masks, and alternatives to post-processing dilation. |

Paths inside these scripts are mostly hard-coded relative to the `flat-bug` repository
root and to CSVs that are no longer committed; expect to fix a path before re-running one.
