# Bachelor's Thesis — Analysis & Evaluation Scripts

Scripts and result data behind

> **Zero-Shot and Fine-Tuned Segmentation of Insects Using a Vision Foundation Model**
> Adam Basha, Bachelor's Thesis, Philipps-Universität Marburg, August 2026.

Everything in this folder is *final* — it produced a figure, a table or a number that
appears in the thesis. Superseded and exploratory scripts were kept out; in the
`flat-bug` working repository they live in `../exploratory/`.

The model and training code (SAM3 inference, decoder-only fine-tune, LoRA adapters,
background-query suppression loss) lives in the `sam3-insect-segmentation` repository.

## Layout

| Folder | Thesis section | Contents |
| --- | --- | --- |
| `experiments_1_3/` | §7.3–7.5 | Unified pipeline for Experiments 1, 2 and 3 (PR curves, AP, FP/FN, confidence, size analysis) |
| `experiment_4_ood/` | §7.6 | Out-of-distribution study on Pest24 / Urban Insects / MassID45, incl. the delivered result data |
| `ap_evaluation/` | §7.1, Appendix B | AP estimators, greedy-vs-optimal matching, AP from a matched CSV, bbox-IoU column |
| `methods/` | §6 | Inference, matching/evaluation and hard-negative mining entry points |
| `visualization/` | §4.3.1, §7.3.4, §7.4.6 | Pyramid-tiling illustration and qualitative FP/FN failure panels |
| `helpers/` | — | Shared R plotting/statistics utilities (`flatbug_init.R` and friends) |

## Experiments 1–3 (`experiments_1_3/`)

Shared configuration: score threshold **0.005**, mask threshold **0.5**, greedy
confidence-sorted matching, minimum object size **sqrt(area) ≥ 32 px**.

Figure naming is unified as `exp{N}_{model}_{type}_score_005.pdf`, with
`model ∈ {sam3, flatbug, ft, lora}` — `sam3` = zero-shot base (Exp 1), `flatbug` =
FlatBug-L baseline (Exp 1), `ft` = decoder-only fine-tune (Exp 2), `lora` =
LoRA + background-query suppression (Exp 3).

```bash
# PR curves + AP50 / AP50-95 per model
Rscript experiments_1_3/ap_curves_thesis_experiments_refactored.R

# FP/FN distributions, TP-vs-FP confidence, error rates by object size
# (also prints the metrics quoted in the experiment chapters)
python3 experiments_1_3/generate_thesis_graphs.py            # all experiments
python3 experiments_1_3/generate_thesis_graphs.py -e 2       # one experiment
```

Both scripts resolve their inputs relative to their own location and expect the
matched-detection CSVs under
`../../data/thesis_experiments_using_fb_eval_refactored/with_bb/`
(one `*_experiment_N.csv` per model). Point them elsewhere with environment
variables — `FB_DATA_DIR`, `FB_GRAPH_ROOT` / `FB_FIGURE_DIR`, `FB_HELPERS_DIR`.

> These CSVs are several tens of MB each and are **not** committed. Regenerate them
> with `methods/fb_eval_greedy.py` (see below) or copy them from the evaluation host.

## Experiment 4 — OOD generalisation (`experiment_4_ood/`)

Self-contained deliverable: per-dataset matched CSVs, computed metrics, PR curves and
the qualitative TP/FP/FN panels for **Pest24**, **Urban Insects** and **MassID45**.

```
experiment_4_ood/
├── OOD_EXPERIMENTS_REPORT.md      # written report of the study
├── _pr_curves_thesis_pdf/         # PR curves as included in the thesis
├── <dataset>/csv/                 # matched detections, per model
├── <dataset>/metrics/             # AP / precision / recall JSON
├── <dataset>/pr_curves/
├── <dataset>/visualizations/      # original qualitative panels
├── <dataset>/visualizations_thesis/  # print-sized labels (what the thesis uses)
└── scripts/
```

Deviations from the shared configuration (§7.6.2): **no** 32 px size floor and **no**
edge/boundary margins, so the very small specimens in these datasets survive evaluation.

```bash
# PR curves for all four dataset × IoU-type panels
Rscript experiment_4_ood/scripts/ood_pr_curves_thesis_final.R

# Rebuild the written report from the metrics JSONs
python3 experiment_4_ood/scripts/gen_final_report.py

# Re-render qualitative panels from source images (needs the raw datasets)
python3 experiment_4_ood/scripts/ood_tp_fp_fn_examples.py

# Relabel existing panels without the raw images (recomputes counts from the CSVs)
python3 experiment_4_ood/scripts/relabel_ood_panels.py --thesis-copy
```

`ood_tp_fp_fn_examples.py` is the only script here that needs the **raw datasets and
prediction COCO JSONs** (`SAM3_ROOT`, default `/data/sam3`), which are not part of this
repository. For label-only changes use `relabel_ood_panels.py`, which recomputes every
count from the shipped CSVs and rebuilds the label bands on the existing PNGs.
Burned-in label size is derived from the LaTeX include width (`latex_frac` in the
`DS` table of `ood_panel_labels.py`) — a fixed font scale prints unreadably small.

## AP evaluation (`ap_evaluation/`)

| Script | Purpose |
| --- | --- |
| `compare_ap_estimators.py` | Reproduces the Appendix B table. Every estimator consumes the *identical* classification and ranking; only the PR-integration rule varies. |
| `compare_greedy_vs_optimal.py` | Greedy confidence-sorted matching vs optimal assignment — justifies the matcher used throughout. |
| `compute_ap_from_csv.py` | AP50, AP50-95, precision and recall (bbox + mask) straight from a matched CSV. |
| `add_iou_bb.py` | Adds the `IoU_bb` (bounding-box IoU) column to a matched CSV, producing the `*_with_bb.csv` inputs the plotting scripts expect. |

## Methods (`methods/`)

Copies of the entry points named in Chapter 6, kept here so the analysis pipeline is
readable end to end. In the `flat-bug` working repository the originals live at
`src/bin/` and `scripts/training/`.

| Script | Thesis section |
| --- | --- |
| `sam3_predict.py` | §6.1 — SAM3 tiled-pyramid inference, checkpoint dispatch, LoRA merging, post-processing |
| `fb_eval_greedy.py` | §7.1 — greedy confidence-sorted GT↔prediction matching (`--coco-matching` for the COCO cross-check) |
| `generate_hard_negatives.py` | §6.3.1 — hard-negative mining for the background-query suppression loss |
| `run_evaluation_pipeline.sh` | glue: predict → match → metrics |

## Visualization (`visualization/`)

| Script | Thesis section |
| --- | --- |
| `visualize_pyramid.py` | §4.3.1 — pyramid-tiling illustration |
| `example_bbox_vs_contour.py` | §2.1 — bounding box vs contour annotation |
| `visualize_failure_examples.py` | §7.3.4 / §7.4.6 — qualitative FP/FN panels from a matched CSV |
| `visualize_false_negatives.py` | missed detections: unmatched FNs and low-IoU FNs |
| `visualize_hyper_confident_fps.py` | the most confident hallucinations, as context crops |

These need the evaluation images (`fb_yolo/insects/images/val` in the FlatBug working
tree); each script takes `--csv` and `--images-dir`.

## Requirements

- Python 3.10+ with `numpy`, `pandas`, `matplotlib`, `opencv-python`, `pycocotools`
- R 4.x with `tidyverse`, `data.table`, `ggplot2`, `mgcv`, `extrafont`
  (`R_LIBS_USER` must point at a library that has them)
