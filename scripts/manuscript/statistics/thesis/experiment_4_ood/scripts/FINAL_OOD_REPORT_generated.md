# Final report — zero-shot OOD generalization: SAM3-ft vs FlatBug-L

Both models were trained on the flat-bug aggregate (23 datasets); none of the sets below were in either model's training. All numbers are **standard COCO bbox AP** (pycocotools, single 'insect' class, no size filter) on densely-annotated data.

## Head-to-head (bbox)

| Dataset | Domain | Model | #pred | AP50 | AP[.5:.95] | AP_small | AP_med | AR@100 |
|---|---|---|---|---|---|---|---|---|
| Urban Insects | UV sticky-card scans (downscaled) | SAM3 ft. ep18 | 95742 | 22.1% | 11.0% | 3.4% | 13.9% | 13.4% |
| Urban Insects | UV sticky-card scans (downscaled) | FlatBug L | 15588 | 18.7% | 8.9% | 0.2% | 12.4% | 10.3% |
| MassID45 | Malaise-trap bulk tray (lab, real masks) | SAM3 ft. ep18 | 23529 | 61.9% | 37.0% | 27.3% | 81.7% | 43.1% |
| MassID45 | Malaise-trap bulk tray (lab, real masks) | FlatBug L | 3321 | 26.3% | 10.4% | 3.9% | 42.0% | 13.2% |
| Pest24 | field pest traps (24-species labels) | SAM3 ft. ep18 | 394010 | 39.8% | 20.5% | 13.0% | 29.3% | 44.7% |
| Pest24 | field pest traps (24-species labels) | FlatBug L | 38186 | 7.6% | 2.3% | 0.5% | 4.5% | 6.3% |

## Thesis-pipeline AP (fb_eval_greedy `--coco-matching` + R integration; **no maxDets cap**) — use these for the thesis

| Dataset | SAM3-ft  AP50 / AP50-95 | FlatBug-L  AP50 / AP50-95 |
|---|---|---|
| Urban Insects | **60.4% / 22.4%** | 44.7% / 17.8% |
| MassID45 | **67.0% / 46.4%** | 47.6% / 32.2% |
| Pest24 | **40.0% / 20.4%** | 11.3% / 5.2% |

> **Why two AP tables?** The pycocotools table above caps detections at **100 per image** (COCO `maxDets`), which throttles ultra-dense scans (Urban Insects ~359 insects/image) and depresses their AP (e.g. Urban SAM3 22% vs 60% here). The thesis pipeline has **no such cap** and is consistent with every other thesis PRC figure, so **report the thesis-pipeline numbers in the manuscript**. Pest24 agrees between methods (~39.5%) because it has few objects/image so the cap never binds. The verdict (SAM3-ft > FlatBug-L on all three) is identical under both.


## MassID45 — bbox vs **mask** AP (the only OOD set with real GT masks)

| Model | bbox AP50 / AP50-95 | mask AP50 / AP50-95 |
|---|---|---|
| SAM3 ft. ep18 | 67.0% / 46.4% | 67.3% / 36.2% |
| FlatBug L | 47.6% / 32.2% | 45.5% / 19.6% |

> Both models save full polygon segmentation, so mask AP is computed the same way as bbox AP (just the mask-IoU column). SAM3's mask AP (~67%) ≈ its bbox AP, and matched mask-IoU averages 0.75 — i.e. SAM3's masks are genuinely accurate here. (The poor Pest24 mask numbers were purely an artifact of Pest24's rectangular pseudo-masks, not mask quality.)

## Verdict

- **Urban Insects**: winner = **SAM3 ft. ep18** (AP50 22.1%).
- **MassID45**: winner = **SAM3 ft. ep18** (AP50 61.9%).
- **Pest24**: winner = **SAM3 ft. ep18** (AP50 39.8%).

**Overall: SAM3-ft wins 3/3 datasets.** Consistent with the Pest24 deep-dive: SAM3-ft transfers better to unseen insect imagery; FlatBug is more conservative and misses more (esp. small specimens).

## Figures

**Thesis-style PR curves (SAM3-ft vs FlatBug-L overlaid, adaptive axes):**
- Urban Insects (bbox): `urban_insects/pr_curves/ood_urban_insects_pr_curve.png`
- Pest24 (bbox): `pest24/pr_curves/ood_pest24_pr_curve.png`
- MassID45 (**masks**, real GT): `massid45/pr_curves/ood_massid45_mask_pr_curve.png`

**COCO-interpolated PR curves (pycocotools, maxDets=100):**
- Urban Insects: `urban_insects/pr_curves/coco_interp_pr_urban_insects_bbox.png`
- MassID45: `massid45/pr_curves/coco_interp_pr_massid45_bbox.png`
- Pest24: `pest24/pr_curves/coco_interp_pr_pest24_bbox.png`

## Caveats

- **Urban Insects** downscaled 10200x14039 -> max-dim 4096 (equally for both models; tractable tiling). Modality (scanned sticky card) partially resembles flat-bug's `ubc-scanned-sticky-cards`.
- **MassID45** (Malaise bulk tray) is the cleanest novel domain and has **real masks**; specimens are tiny (median ~21px).
- **Pest24** is partially annotated (24 species only) -> precision/AP suppressed by >=8 AP-pts (real unlabeled insects counted as FP); see partial-annotation study. Numbers here use the floor-removed optimized predictions, all sizes.
- Flat-bug held-out 'prospective' sets (crall2023/chavez2024/InsectCV) were the ideal clean test but are NOT publicly downloadable (private S3); these public dense datasets are the substitute.

## Data sources (download origins)

| Dataset | Origin (authors, license) | Download |
|---|---|---|
| **Urban Insects** | Figshare, article `28280792` (Parraga-Alava et al.; CC BY 4.0) | https://figshare.com/articles/28280792 — 5 site zips via `https://ndownloader.figshare.com/files/{51931694,51931703,51931697,51931700,51931706}` |
| **MassID45** | Zenodo record `18963816` (concept DOI `10.5281/zenodo.15479861`; U. Guelph MLRG; CC BY 4.0) | https://zenodo.org/records/18963816 — file `tiled_bulk_images.zip` |
| **Pest24** | Kaggle `boatshuai/pest24` (original: Wang et al. 2020, *Comput. Electron. Agric.*) | https://www.kaggle.com/datasets/boatshuai/pest24 |
