# Out-of-Distribution (OOD) Generalization Study — SAM3-ft vs FlatBug-L
### Experiment report for thesis write-up

> **How to use this document.** It is a structured, fact-dense record of the OOD
> experiments (narrative flow, problems solved, methodology, final metrics, figures,
> and citations). It is written so a second agent can extract clean bullet points from
> it, after which the human author rewrites those bullets into thesis prose. Every
> number here is reproduced from the committed CSVs / metrics files listed in §8; nothing
> is rounded away from its source. Where a claim is a judgement call rather than a
> measured number, it is flagged as **[interpretation]**.

---

## 1. What the experiment is and why we ran it

Two insect detectors were compared **head-to-head, zero-shot, on three datasets that
neither model saw during training**, to answer one question: *which model generalises
better to unseen insect imagery?*

- **SAM3-ft** — Meta's Segment Anything 3 image model, **fine-tuned on the flat-bug
  aggregate** (23 insect datasets); we use **checkpoint / epoch 18** of the
  `flatbug_medium_ft` run. Produces instance masks + boxes + confidence.
- **FlatBug-L** — the published `flat_bug_L.pt` YOLOv8-seg detector, the incumbent
  universal insect model. Produces instance masks + boxes + confidence.

Both models were trained on the same flat-bug aggregate, so the comparison is fair: the
three test datasets below are **out-of-distribution for both**. The original intent was
to use flat-bug's own held-out "prospective" sets (crall2023 / chavez2024 / InsectCV),
but those are **not publicly downloadable** (private S3); three public, densely-annotated
insect datasets were adopted as the substitute (see §7).

### The three OOD datasets

| Dataset | Domain | #images | #GT insects | GT annotation type | Objects / image | Notable property |
|---|---|---|---|---|---|---|
| **Pest24** | Field pest light-traps | 7 600 | 58 195 | **Bounding boxes only** | ~8 | Partially annotated (only 24 pest species labelled) |
| **Urban Insects** | UV sticky-card scans (huge) | 69 | 24 756 | **Bounding boxes only** | ~359 | Ultra-dense; 10200×14039 px scans downscaled to max-dim 4096 |
| **MassID45** | Malaise-trap bulk tray (lab) | 1 029 | 5 852 | **Real polygon instance masks** | ~6 | Only OOD set with true masks; specimens tiny (median ~21 px) |

**[interpretation]** These three span the realistic deployment range: dense field traps
(Pest24), extreme-density scanned cards (Urban Insects), and clean lab bulk samples with
real masks (MassID45). Together they stress detection density, object size, and mask
quality respectively.

---

## 2. Narrative flow of the work (condensed)

1. **Pest24 first run was disappointing** — SAM3-ft AP looked far too low. Investigation
   showed the model *was not emitting sub-32-px detections at all*.
2. **Root cause: a hidden 32-px inference floor.** A size filter deep in the SAM3
   inference code deleted every detection smaller than 32 px *before* the COCO file was
   written — a histogram of box sizes showed a 128× cliff exactly at 32 px. Since insects
   in these datasets are frequently <32 px, this silently destroyed recall.
3. **Fix + re-optimise config** — removed the floor (`MIN_MAX_OBJ_SIZE=[4, 1e8]`),
   dropped edge/boundary margins, lowered the score threshold to 0.005. Validated on a
   100-image sample (AP50 22 %→34 %, small-object recall ~5.5×) *before* committing to the
   full 7 600-image run.
4. **Built a symmetric pipeline** — identical config philosophy for FlatBug-L, same
   evaluation (`fb_eval_greedy.py --coco-matching`), so any AP gap reflects the model, not
   the harness.
5. **Extended to Urban Insects and MassID45** — downloaded, prepped to single-class COCO,
   ran both models, evaluated, and generated PR curves + qualitative panels.
6. **Two diagnostic deep-dives** — (a) Pest24 partial-annotation study (cross-model
   corroboration of "false positives"); (b) recall-ceiling decomposition (localisation vs
   true misses).
7. **This deliverable** — regenerated the PR curves in the enhanced thesis PDF format and
   collected every artifact into one download folder.

---

## 3. Evaluation methodology (and the bbox-vs-mask decision)

### 3.1 Inference
- SAM3-ft: `sam3_predict_with_mask_threshold_checkpoint.py` with the optimised config
  (`MASK_THRESHOLD=0.5`, `SCORE_THRESHOLD=0.005`, `IOU_THRESHOLD=0.2`, no size floor, no
  margins). FlatBug-L: `flat_bug_L.pt` with a matched config.
- Single class ("insect") for every dataset — we evaluate *detection/localisation*, not
  species classification.
- Urban Insects scans are downscaled 10200×14039 → max-dim 4096 **equally for both
  models** so the comparison stays fair on tractable tiles.

### 3.2 Matching & AP
- Ground-truth ↔ prediction matching is done by `fb_eval_greedy.py --coco-matching`:
  predictions are sorted by descending confidence and greedily matched to unused GT
  (COCO-style). This is the *same matching that produces the AP numbers*, and it writes a
  per-pair CSV (`combined_results_with_bb.csv`) with both a **mask IoU** (`IoU`) and a
  **bbox IoU** (`IoU_bb`) column plus the prediction confidence (`conf2`).
- **Two AP computations are reported, and they differ for a specific, understood reason:**
  - **Thesis pipeline (primary):** trapezoidal AP integrated from the greedy-matching CSV,
    **no cap on detections per image**, all object sizes kept. Consistent with every other
    PR figure in the thesis. **These are the numbers to report.**
  - **pycocotools (cross-check):** standard COCO AP, but it **caps detections at 100 per
    image** (`maxDets=100`). On ultra-dense Urban Insects (~359 insects/image) this cap
    throttles the model and depresses AP (Urban SAM3 22 % capped vs 60 % uncapped). Pest24
    and MassID45 have few enough objects/image that the cap barely binds, so the two
    methods agree there. **The winner (SAM3-ft) is identical under both methods on all
    three datasets** — the discrepancy is purely the cap, not a disagreement about ranking.

### 3.3 Why the PR curve is **bbox-based for Pest24 & Urban Insects, but both bbox and mask for MassID45**

This is dictated entirely by **what ground truth each dataset provides**:

| Dataset | GT format | Meaningful IoU for PR? | Reason |
|---|---|---|---|
| **Pest24** | Bounding boxes | **bbox only** | No mask GT exists. Both models *emit* masks, but there is nothing to score them against — the "GT mask" is just the box rectangle, so a real predicted mask scored against a rectangle gives an artificially low, meaningless mask-IoU (matched mask-IoU ≈ 0.57, an artifact). |
| **Urban Insects** | Bounding boxes | **bbox only** | Same as Pest24 — GT is boxes; mask-IoU would compare real masks to rectangles. |
| **MassID45** | Real polygon instance masks | **bbox *and* mask** | Genuine per-instance polygons exist, so mask-IoU is real. We report both: bbox for cross-dataset parity, mask because it is the only set where mask quality can be honestly measured. SAM3-ft matched mask-IoU averages **0.745** here — its masks are genuinely accurate, not just its boxes. |

**[interpretation]** In short: **you can only report a mask PR curve where real mask GT
exists.** For box-only datasets, reporting a mask curve would be misleading, so we
deliberately restrict those to bbox PR. MassID45 is the one place a mask curve is earned.

---

## 4. Problems tackled and solved (lessons for the thesis "methods/challenges" section)

| # | Problem | Symptom | Root cause | Fix / lesson |
|---|---|---|---|---|
| 1 | **Hidden 32-px inference floor** | SAM3 AP collapsed; near-zero small-object recall | A size filter deleted sub-32-px detections *before* the COCO save | Removed floor (`MIN_MAX_OBJ_SIZE=[4,1e8]`); verified via a box-size histogram showing a 128× cliff at exactly 32 px. **Lesson: always histogram your output sizes — a floor can hide upstream of your metric.** |
| 2 | **maxDets=100 cap** | pycocotools AP much lower than the greedy pipeline on dense scans | COCO's default caps detections/image; Urban has ~359/image | Report the uncapped thesis-pipeline AP; keep pycocotools only as a cross-check. **Lesson: when two correct methods disagree, the disagreement itself is the finding.** |
| 3 | **Pest24 partial annotation** | Precision/AP suppressed; many high-confidence "FPs" | Pest24 labels only 24 pest species; other real insects count as FP | Cross-model corroboration study: ~90 % of SAM3's high-confidence "FPs" are also fired on by FlatBug ⇒ real unlabelled insects, not hallucinations. Corrected AP50 lower bound is ~+8–10 pp. **Not a filter bug — it is the dataset.** |
| 4 | **Recall "ceiling"** | SAM3 recall < 100 % even at low threshold | Decomposition: recalled@0.5 / detected-but-mislocalised (IoU 0.1–0.5) / truly missed | The gap is **localisation precision on small objects**, not total misses; growing boxes made recall *worse*, proving boxes aren't too small, just imprecise. |
| 5 | **Python 3.10 vs 3.11** | `ImportError: cannot import name 'Self'`; starred-subscript `SyntaxError` | flat_bug needs Py≥3.11 syntax | `typing.Self` shim (`fb_predict_compat.py`); patched `iou[*m.T]`→`iou[tuple(m.T)]`. |
| 6 | **ultralytics too new** | `'dict2attr' has no attribute 'compile'` | flat_bug requires ultralytics ≤8.3.124 | Pinned `ultralytics==8.3.124`. |
| 7 | **Urban viz double-offset** | Almost no boxes visible in Urban qualitative panels | Boxes offset by crop origin *and then* image cropped ⇒ boxes pushed out of view | Crop the image **first**, then draw with a single offset. |
| 8 | **MassID45 COCO layout** | Top-level JSON had no `file_name` | Usable COCO (with `file_name`) is *inside* the image zip under `tiled_bulk_images/{train2017,val2017}/` | Always inspect a COCO file's keys/`file_name`/bbox format before trusting it. |

---

## 5. Final results

### 5.1 Primary — thesis-pipeline AP (greedy `--coco-matching`, **no maxDets cap**, all sizes)

Computed at the 0.5:0.95 IoU sweep (step 0.01) directly from the committed CSVs.
**Bold = winner.**

| Dataset | IoU basis | SAM3-ft AP50 | SAM3-ft AP50-95 | FlatBug-L AP50 | FlatBug-L AP50-95 |
|---|---|---|---|---|---|
| Urban Insects | bbox | **60.35 %** | **21.72 %** | 44.67 % | 17.33 % |
| MassID45 | bbox | **66.99 %** | **47.32 %** | 47.58 % | 32.92 % |
| MassID45 | **mask** | **67.28 %** | **36.49 %** | 45.52 % | 19.30 % |
| Pest24 | bbox | **39.53 %** | **20.20 %** | 11.28 % | 5.15 % |

> These are the numbers plotted in the thesis PR curves (§6). AP50 values match
> `FINAL_OOD_REPORT.md`; small AP50-95 differences vs that report come from the finer
> 0.01 IoU step used here (the report used 0.05).

### 5.2 Cross-check — pycocotools bbox AP (`maxDets` raised, single class)

From the committed `ood_*_metrics.json`:

| Dataset | Model | #pred | AP50 | AP[.5:.95] | AP_small | AP_med | AR@100 |
|---|---|---|---|---|---|---|---|
| Urban Insects | SAM3-ft | 95 742 | 22.1 % | 11.0 % | 3.4 % | 13.9 % | 13.4 % |
| Urban Insects | FlatBug-L | 15 588 | 18.7 % | 8.9 % | 0.2 % | 12.4 % | 10.3 % |
| MassID45 | SAM3-ft | 23 529 | 61.9 % | 37.0 % | 27.3 % | 81.7 % | 43.1 % |
| MassID45 | FlatBug-L | 3 321 | 26.3 % | 10.4 % | 3.9 % | 42.0 % | 13.2 % |
| Pest24 | SAM3-ft | 394 010 | 39.8 % | 20.5 % | 13.0 % | 29.3 % | 44.7 % |
| Pest24 | FlatBug-L | 38 186 | 7.6 % | 2.3 % | 0.5 % | 4.5 % | 6.3 % |

(Urban Insects is where the two methods diverge most — the 22 % here vs 60 % above is the
`maxDets=100` cap acting on a ~359-insect/image scan. Ranking is unchanged.)

### 5.3 MassID45 — bbox vs mask (the only set where mask AP is real)

| Model | bbox AP50 / AP50-95 | mask AP50 / AP50-95 | matched mask-IoU (mean) |
|---|---|---|---|
| SAM3-ft | 66.99 % / 47.32 % | 67.28 % / 36.49 % | **0.745** |
| FlatBug-L | 47.58 % / 32.92 % | 45.52 % / 19.30 % | 0.702 |

**[interpretation]** SAM3-ft's mask AP50 ≈ its bbox AP50 and its matched mask-IoU is
0.745 — the masks are as trustworthy as the boxes. FlatBug's mask AP50-95 drops much more
(19.3 %), i.e. its masks are looser at high IoU thresholds.

### 5.4 Operating-point / recall summary (from the CSVs, matched at bbox-IoU ≥ 0.5)

| Dataset | Model | recall@IoU_bb0.5 | mean matched bbox-IoU | #predictions |
|---|---|---|---|---|
| Pest24 | SAM3-ft | **81.4 %** | 0.742 | 393 981 |
| Pest24 | FlatBug-L | 25.9 % | 0.675 | 38 186 |
| Urban Insects | SAM3-ft | **72.6 %** | 0.635 | 95 740 |
| Urban Insects | FlatBug-L | 49.3 % | 0.675 | 15 588 |
| MassID45 | SAM3-ft | **77.3 %** | 0.808 | 22 826 |
| MassID45 | FlatBug-L | 48.8 % | 0.822 | 3 278 |

**[interpretation]** A consistent pattern across all three: **SAM3-ft recalls far more
insects** (it predicts ~6–10× more objects and finds many more true ones), while FlatBug-L
is more conservative — slightly higher *mean matched IoU* on some sets (it only commits to
easy, well-localised objects) but it **misses most of the small specimens**. On Pest24 the
huge SAM3 prediction count is inflated by the dataset's partial annotation (many predicted
insects are real but unlabelled — see problem #3).

### 5.5 Verdict

**SAM3-ft wins all 3 datasets on every headline metric (AP50, AP50-95, recall), under both
AP computations.** Fine-tuning SAM3 transfers better to unseen insect imagery than the
incumbent FlatBug-L, especially on small objects and dense scenes.

---

## 6. Figures and visualizations produced

### 6.1 PR curves — **thesis format (vector PDF)**, regenerated for this deliverable

Same enhanced format as `ap_curves_thesis_experiments_refactored_final.R` (cairo_pdf,
4-sided border, inside ticks, Courier legend with per-model AP50 / AP50-95 block), but each
panel **overlays the two models** (SAM3-ft in blue `#0072B2`, FlatBug-L in orange
`#D55E00`). Script: `_scripts/ood_pr_curves_thesis_final.R`. A `.png` sibling accompanies
each `.pdf` for previewing.

| File (in each dataset's `pr_curves/`) | Dataset | IoU basis |
|---|---|---|
| `ood_pest24_pr_curve.pdf` | Pest24 | bbox |
| `ood_urban_insects_pr_curve.pdf` | Urban Insects | bbox |
| `ood_massid45_bbox_pr_curve.pdf` | MassID45 | bbox |
| `ood_massid45_mask_pr_curve.pdf` | MassID45 | **mask (real GT)** |

All four PDFs are also collected flat in `_pr_curves_thesis_pdf/`.

### 6.2 PR curves — COCO-interpolated (pycocotools, maxDets=100), secondary cross-check
`coco_interp_pr_<dataset>_bbox.png` in each dataset's `pr_curves/`.

### 6.3 Qualitative TP/FP/FN panels
Two side-by-side panels (SAM3-ft | FlatBug-L) per image, boxes colour-coded at the conf≥0.5
operating point: **green = TP, yellow = FN (missed GT), red = FP**. Banner shows real
counts and P/R. Urban Insects is cropped to its densest region (the full 4096-px tile is
too dense to read). MassID45 additionally has **mask panels** (polygon fills). Script:
`_scripts/ood_tp_fp_fn_examples.py`.
- `pest24/visualizations/` — 3 bbox panels
- `urban_insects/visualizations/` — 3 bbox panels (densest-crop)
- `massid45/visualizations/` — 3 bbox + 3 mask panels

---

## 7. Data sources and citations (for the thesis)

**Download origins (verified):**

| Dataset | Origin | License | Download |
|---|---|---|---|
| **Pest24** | Kaggle `boatshuai/pest24`; original paper: Wang et al. 2020, *Computers and Electronics in Agriculture* **175**, 105585 | see Kaggle page | https://www.kaggle.com/datasets/boatshuai/pest24 |
| **Urban Insects** | Figshare article `28280792` (**Ong & Lim**, 2025); paper: Lim, Chan & Ong 2025, *Data in Brief* **60**, 111673 | CC BY 4.0 | https://doi.org/10.6084/m9.figshare.28280792.v1 — 5 site zips via `ndownloader.figshare.com/files/{51931694,51931703,51931697,51931700,51931706}` |
| **MassID45** | Zenodo concept DOI `10.5281/zenodo.15479861` (download record `18963816`); paper: Orsholm et al. 2026, *Scientific Data* **13**, 630 | CC BY 4.0 | https://zenodo.org/records/18963816 — file `tiled_bulk_images.zip` |

> **Attribution correction (verified 2026-07-11):** an earlier draft credited Urban Insects
> to "Parraga-Alava et al." — that is **wrong**. The dataset authors are **Song-Quan Ong &
> Min-Hui Lim**. Pest24's exact title is "…for **multi-target detection**" (not
> "monitoring"). The MassID45 specific version record `18963816` is what we downloaded from,
> but cite the **concept DOI** `10.5281/zenodo.15479861` (it always resolves to the latest
> version).

**Models:** SAM3 (Meta, "Segment Anything with Concepts", arXiv:2511.16719, 2025);
FlatBug / flat-bug (Svenning et al. 2026, *Methods in Ecology and Evolution*).

### 7.1 BibTeX (verified; ready to paste)

```bibtex
@article{wang2020pest24,
  title   = {Pest24: A large-scale very small object data set of agricultural pests for multi-target detection},
  author  = {Wang, Qi-Jin and Zhang, Sheng-Yu and Dong, Shi-Feng and Zhang, Guang-Cai and Yang, Jin and Li, Rui and Wang, Hong-Qiang},
  journal = {Computers and Electronics in Agriculture},
  volume  = {175},
  pages   = {105585},
  year    = {2020},
  doi     = {10.1016/j.compag.2020.105585}
}

@misc{ong2025urbaninsects_figshare,
  title     = {An annotated image dataset of urban insects for computer vision and deep learning},
  author    = {Ong, Song-Quan and Lim, Min-Hui},
  year      = {2025},
  publisher = {figshare},
  doi       = {10.6084/m9.figshare.28280792.v1},
  note      = {CC BY 4.0},
  url       = {https://doi.org/10.6084/m9.figshare.28280792.v1}
}

@article{lim2025urbaninsects,
  title   = {An annotated image dataset of urban insects for the development of computer vision and deep learning models with detection tasks},
  author  = {Lim, Min Hui and Chan, Hiang Hao and Ong, Song-Quan},
  journal = {Data in Brief},
  volume  = {60},
  pages   = {111673},
  year    = {2025},
  doi     = {10.1016/j.dib.2025.111673}
}

@article{orsholm2026massid45,
  title   = {A multi-modal dataset for insect biodiversity with imagery and DNA at the trap and individual level},
  author  = {Orsholm, Johanna and Quinto, John and Autto, Hannu and Banelyte, Gaia and Chazot, Nicolas and deWaard, Jeremy and deWaard, Stephanie and Farrell, Arielle and Furneaux, Brendan and Hardwick, Bess and Ito, Nao and Kar, Amlan and Kalttop{\"a}{\"a}, Oula and Kerdraon, Deirdre and Kristensen, Erik and McKeown, Jaclyn and Mononen, Tommi and Nein, Ellen and Rogers, Hanna and Roslin, Tomas and Schmitz, Paula and Sones, Jayme and Sujala, Maija and Thompson, Amy and Zakharov, Evgeny V. and Zarubiieva, Iuliia and Gupta, Akshita and Lowe, Scott C. and Taylor, Graham W.},
  journal = {Scientific Data},
  volume  = {13},
  pages   = {630},
  year    = {2026},
  doi     = {10.1038/s41597-026-07251-x}
}

@dataset{massid45_zenodo,
  title     = {MassID45: A multi-modal dataset for insect biodiversity with imagery and DNA at the trap and individual level},
  author    = {Orsholm, Johanna and Quinto, John and Autto, Hannu and others},
  publisher = {Zenodo},
  year      = {2025},
  doi       = {10.5281/zenodo.15479861},
  note      = {Concept DOI (resolves to latest version); downloaded from record 18963816},
  url       = {https://doi.org/10.5281/zenodo.15479861}
}

@article{svenning2026flatbug,
  title   = {A general method for detection and segmentation of terrestrial arthropods in images},
  author  = {Svenning, Asger and Mougeot, Guillaume and Alison, Jamie and Chevalier, Daphne and Chavez Molina, Nisa and Ong, Song-Quan and Bjerge, Kim and Carrillo, Juli and H{\o}ye, Toke Thomas and Geissmann, Quentin},
  journal = {Methods in Ecology and Evolution},
  volume  = {17},
  number  = {3},
  year    = {2026},
  doi     = {10.1111/2041-210X.70249},
  note    = {Software: https://github.com/darsa-group/flat-bug; training data: 10.5281/zenodo.14761447}
}

@article{carion2025sam3,
  title   = {SAM 3: Segment Anything with Concepts},
  author  = {Carion, Nicolas and Gustafson, Laura and Hu, Yuan-Ting and Debnath, Shoubhik and Hu, Ronghang and Suris, Didac and Ryali, Chaitanya and Alwala, Kalyan Vasudev and Khedr, Haitham and Huang, Andrew and others},
  journal = {arXiv preprint arXiv:2511.16719},
  year    = {2025},
  doi     = {10.48550/arXiv.2511.16719}
}

@article{ravi2024sam2,
  title   = {SAM 2: Segment Anything in Images and Videos},
  author  = {Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal = {arXiv preprint arXiv:2408.00714},
  year    = {2024},
  doi     = {10.48550/arXiv.2408.00714}
}

@inproceedings{kirillov2023segment,
  title     = {Segment Anything},
  author    = {Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2023},
  pages     = {4015--4026}
}
```

> **Verification status:** all entries checked against Crossref / arXiv / Nature / PMC /
> Figshare v2 API. UNVERIFIED: MassID45 Zenodo *version* record number `18963816` (concept
> DOI is confirmed). SAM 2 / SAM 3 use `arXiv` DOIs because the ICLR proceedings are not yet
> published.

---

## 8. Deliverable folder contents (`OOD_THESIS_DELIVERABLE/`)

```
OOD_THESIS_DELIVERABLE/
├── OOD_EXPERIMENTS_REPORT.md          ← this document
├── _pr_curves_thesis_pdf/             ← all 4 new thesis PR curves (PDF), flat
├── _scripts/                          ← scripts that produced the artifacts + FINAL_OOD_REPORT
├── pest24/
│   ├── csv/           pest24_{sam3,flatbug}.csv        (greedy-matching, IoU + IoU_bb + conf)
│   ├── pr_curves/     ood_pest24_pr_curve.{pdf,png}, coco_interp_pr_pest24_bbox.png
│   ├── visualizations/ 3 × TP/FP/FN bbox panels
│   └── metrics/       ood_pest24_metrics.json          (pycocotools head-to-head)
├── urban_insects/     (same structure; bbox only)
└── massid45/          (same structure; PR curves for BOTH bbox and mask; bbox+mask panels)
```

- **CSV columns:** `image; idx_1; idx_2; bbox_1; bbox_2; contourArea_1; contourArea_2;
  IoU(mask); conf1; conf2(pred conf); IoU_bb`. `idx_1=-1` ⇒ false positive; `idx_2=-1` ⇒
  false negative; both set ⇒ matched pair.
- All files are **copies**; originals in `manuscript/statistics/` and `output/results/`
  are untouched.

---

## 9. Caveats (carry into the thesis)

- **Urban Insects** downscaled 10200×14039 → max-dim 4096 (equally for both models). Its
  scanned-card modality partially resembles flat-bug's `ubc-scanned-sticky-cards`, so it is
  the *least* purely-OOD of the three. **[interpretation]**
- **MassID45** is the cleanest novel domain and the only one with real masks; specimens are
  tiny (median ~21 px), so it doubles as a small-object stress test.
- **Pest24** is partially annotated (24 species only) ⇒ precision/AP are suppressed by
  ≥8–10 AP-pts because real unlabelled insects are scored as false positives. Reported
  numbers use the floor-removed optimised predictions, all sizes.
- The ideal clean test — flat-bug's own held-out prospective sets — was unavailable
  (private S3); these three public dense datasets are the defensible substitute.
