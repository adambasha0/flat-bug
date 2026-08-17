#!/usr/bin/env python3
"""Compare AP estimators on the thesis evaluation CSVs.

Reproduces the table in the thesis appendix "On the Choice of AP Estimator".

The point of this script is that every estimator consumes the IDENTICAL
classification and ranking -- the one implemented by classify_at_threshold() in
ap_curves_thesis_experiments_refactored.R -- so the only thing that varies is
the rule used to integrate the resulting precision-recall curve:

  UT     upper trapezoid, Boyd et al. (2013) Eq. (4).  This is what
         integrate_curve() in the R scripts computes: collapse duplicate recall
         values to their max precision, then trapezoid.  It is the primary
         metric reported throughout the thesis.
  LT     lower trapezoid, Boyd et al. (2013) Eq. (3).
  VOC    all-point interpolated, PASCAL VOC 2010+:  sum (r_n - r_{n-1}) p_interp(r_n)
  C101   COCO 101-point interpolated.

Note that compute_ap_from_csv.py in this directory also implements C101, but it
applies different row filters (no sqrt(area) >= 32 filter on ground truth, plus
a contourArea_2 >= 1024 filter on predictions), so its numbers are NOT
comparable to the thesis figures and cannot be used to isolate the estimator.
Run with --decompose to see how much each of those filters is worth.

Usage:
    python3 compare_ap_estimators.py              # appendix table
    python3 compare_ap_estimators.py --decompose  # attribute the gap vs compute_ap_from_csv.py
"""

import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
_THESIS_ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
_STATS_ROOT = os.path.abspath(os.path.join(_THESIS_ROOT, os.pardir))

# Same CSVs the Experiment 1-3 figures are built from. Standalone layout keeps them
# next to those scripts; embedded in the evaluation working tree they sit two levels
# up. Override with FB_DATA_DIR.
DATA_DIR = os.environ.get(
    "FB_DATA_DIR",
    os.path.join(_STATS_ROOT, "data", "thesis_experiments_using_fb_eval_refactored", "with_bb")
    if os.path.isdir(os.path.join(_STATS_ROOT, "helpers"))
    else os.path.join(_THESIS_ROOT, "experiments_1_3", "data"))

# Thesis sweep (FlatBug protocol) and COCO's own sweep.
GRID46 = np.round(np.arange(0.50, 0.9501, 0.01), 2)
GRID10 = np.round(np.linspace(0.50, 0.95, 10), 2)

MIN_SIZE = 32.0     # R load_data(): filter(size >= 32), size = sqrt(area)
MIN_AREA = 1024.0   # compute_ap_from_csv.py: MIN_AREA on contourArea_2

MODELS = [
    ("FlatBug-L (Exp. 1)", "fb_score_005_using_fb_eval_greedy_corrected_combined_results_with_bb_experiment_1.csv"),
    ("SAM3 base (Exp. 1)", "sam3_results_score_005_combined_results_with_bb_experiment_1.csv"),
    ("SAM3-ft (Exp. 2)", "sam3_ft_round4_ep18_predictions_mask_05_score_005_combined_results_with_bb_experiment_2.csv"),
    ("SAM3-ft + LoRA (Exp. 3)", "fine_tuned_sam3_with_lora_checkpoint_10_mask_5_score_005_combined_results_with_bb_experiment_3.csv"),
]


# ── Loading ───────────────────────────────────────────────────────────────────

def load(path, size_filter=True):
    df = pd.read_csv(path, sep=';', low_memory=False)
    df.columns = df.columns.str.strip()
    for c in ('idx_1', 'idx_2', 'conf2', 'IoU', 'IoU_bb', 'contourArea_1', 'contourArea_2'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df['IoU'] = df['IoU'].fillna(0.0)
    df['IoU_bb'] = df.get('IoU_bb', pd.Series(0.0, index=df.index)).fillna(0.0)
    df['conf'] = df['conf2'].fillna(0.0)

    area = np.where(df['idx_1'] != -1, df['contourArea_1'], df['contourArea_2'])
    df['size'] = np.sqrt(np.nan_to_num(area.astype(float)))
    if size_filter:
        df = df[df['size'] >= MIN_SIZE].copy()

    df['matched'] = (df['idx_1'] != -1) & (df['idx_2'] != -1)
    df['un_pred'] = (df['idx_1'] == -1) & (df['idx_2'] != -1)
    df['un_gt'] = (df['idx_1'] != -1) & (df['idx_2'] == -1)
    return df


# ── The curve, built exactly as classify_at_threshold() + make_pr_curve() do ───

def curve(df, iou_col, t):
    """Ranked (recall, precision) at IoU threshold t.

    A matched pair below t contributes BOTH a false positive (the prediction)
    and a false negative (the ground truth, at conf 0), matching the R code.
    """
    m = df['matched'].values
    iou = df[iou_col].values
    conf = df['conf'].values
    tp_m = m & (iou >= t)
    bad_m = m & (iou < t)
    up = df['un_pred'].values
    ug = df['un_gt'].values

    n_bad = int(bad_m.sum())
    conf_all = np.concatenate([conf[tp_m], conf[up], conf[bad_m],
                               np.zeros(n_bad), conf[ug]])
    is_tp = np.concatenate([np.ones(tp_m.sum()), np.zeros(up.sum()), np.zeros(n_bad),
                            np.zeros(n_bad), np.zeros(ug.sum())])
    is_fp = np.concatenate([np.zeros(tp_m.sum()), np.ones(up.sum()), np.ones(n_bad),
                            np.zeros(n_bad), np.zeros(ug.sum())])

    n_tp = int(tp_m.sum())
    n_fn = n_bad + int(ug.sum())
    if n_tp == 0:
        return None, None

    order = np.argsort(-conf_all, kind='stable')
    tpc = np.cumsum(is_tp[order])
    fpc = np.cumsum(is_fp[order])
    denom = tpc + fpc
    keep = denom > 0
    return tpc[keep] / (n_tp + n_fn), tpc[keep] / denom[keep]


def collapse(rec, prec):
    """Per distinct recall value: (r, p_min, p_max)."""
    d = pd.DataFrame({'r': rec, 'p': prec}).groupby('r')['p'].agg(['min', 'max']).sort_index()
    return d.index.values, d['min'].values, d['max'].values


# ── Estimators ────────────────────────────────────────────────────────────────

def estimators(df, iou_col, t):
    rec, prec = curve(df, iou_col, t)
    if rec is None:
        return dict(UT=0.0, LT=0.0, VOC=0.0, C101=0.0)
    r, pmin, pmax = collapse(rec, prec)
    dr = np.diff(r)

    ut = float(np.sum(dr * (pmax[:-1] + pmax[1:]) / 2))   # Boyd Eq. (4)
    lt = float(np.sum(dr * (pmin[:-1] + pmax[1:]) / 2))   # Boyd Eq. (3)

    env = np.maximum.accumulate(pmax[::-1])[::-1]         # p_interp over distinct points
    r_prev = np.concatenate([[0.0], r[:-1]])
    voc = float(np.sum((r - r_prev) * env))

    grid = np.linspace(0.0, 1.0, 101)
    idx = np.searchsorted(r, grid, side='left')
    inside = idx < len(env)
    c101 = float(np.sum(np.where(inside, env[np.minimum(idx, len(env) - 1)], 0.0)) / 101.0)

    return dict(UT=ut, LT=lt, VOC=voc, C101=c101)


# ── Appendix table ────────────────────────────────────────────────────────────

def appendix_table():
    rows = []
    hdr = (f"{'Model':<24} {'IoU':<5} | {'AP50':>24} | {'AP50-95 (46 thr.)':>24} | {'(10 thr.)':>9}")
    print(hdr)
    print(f"{'':<24} {'':<5} | {'trapez.':>7} {'interp.':>7} {'delta':>7} | "
          f"{'trapez.':>7} {'interp.':>7} {'delta':>7} | {'interp.':>9}")
    print("-" * len(hdr))

    for name, fname in MODELS:
        df = load(os.path.join(DATA_DIR, fname))
        for iou_col, nm in [('IoU', 'mask'), ('IoU_bb', 'bbox')]:
            per_t = {t: estimators(df, iou_col, t) for t in GRID46}
            ut50 = per_t[0.50]['UT'] * 100
            c50 = per_t[0.50]['C101'] * 100
            ut46 = np.mean([per_t[t]['UT'] for t in GRID46]) * 100
            c46 = np.mean([per_t[t]['C101'] for t in GRID46]) * 100
            c10 = np.mean([per_t[t]['C101'] for t in GRID10]) * 100
            rows.append(dict(model=name, iou=nm, UT_50=ut50, C101_50=c50,
                             UT_5095_46=ut46, C101_5095_46=c46, C101_5095_10=c10))
            print(f"{name:<24} {nm:<5} | {ut50:7.2f} {c50:7.2f} {ut50 - c50:+7.2f} | "
                  f"{ut46:7.2f} {c46:7.2f} {ut46 - c46:+7.2f} | {c10:9.2f}", flush=True)

    out = pd.DataFrame(rows)
    d50 = out['UT_50'] - out['C101_50']
    d5095 = out['UT_5095_46'] - out['C101_5095_46']
    grid = out['C101_5095_46'] - out['C101_5095_10']
    print("-" * len(hdr))
    print(f"{'Mean delta':<24} {'':<5} | {'':>7} {'':>7} {d50.mean():+7.2f} | "
          f"{'':>7} {'':>7} {d5095.mean():+7.2f} |")
    print(f"\nEstimator effect  AP50      : {d50.mean():+.2f} pp "
          f"(worst {d50.abs().max():.2f} pp)")
    print(f"Estimator effect  AP50-95    : {d5095.mean():+.2f} pp "
          f"(worst {d5095.abs().max():.2f} pp, same 46-threshold sweep)")
    print(f"Threshold-grid effect AP50-95: {grid.mean():+.2f} pp (46 thr. vs COCO's 10)")
    return out


# ── Gap attribution vs compute_ap_from_csv.py ─────────────────────────────────

def decompose(fname=None, iou_col='IoU', t=0.50):
    """Show that the gap to compute_ap_from_csv.py is row selection, not the estimator."""
    fname = fname or MODELS[0][1]
    path = os.path.join(DATA_DIR, fname)
    print(f"{os.path.basename(path)}  --  {'mask' if iou_col == 'IoU' else 'bbox'} AP50\n")
    print(f"{'rows kept':<32} {'pred. area filter':<18} {'n_gt':>8} {'trapez.':>8} {'101-pt':>8}")

    for label, size_filter in [("sqrt(area) >= 32  (R pipeline)", True),
                               ("all rows          (py script)", False)]:
        df_base = load(path, size_filter=size_filter)
        for filt in (False, True):
            df = df_base
            if filt:
                # compute_ap_from_csv.py drops predictions below MIN_AREA but keeps
                # every GT row in the recall denominator.
                drop = (df['matched'] | df['un_pred']) & (df['contourArea_2'].fillna(0.0) < MIN_AREA)
                df = df[~drop].copy()
                df['un_gt'] = df['un_gt'] | (df['idx_1'] != -1) & (df['idx_2'] == -1)
            n_gt = int((df_base['idx_1'] != -1).sum())
            rec, prec = curve(df, iou_col, t)
            if rec is None:
                continue
            r, _, pmax = collapse(rec, prec)
            # rescale recall onto the unfiltered GT count so the denominators match
            scale = int((df['idx_1'] != -1).sum()) / n_gt
            r = r * scale
            env = np.maximum.accumulate(pmax[::-1])[::-1]
            ut = float(np.sum(np.diff(r) * (pmax[:-1] + pmax[1:]) / 2))
            grid = np.linspace(0.0, 1.0, 101)
            idx = np.searchsorted(r, grid, side='left')
            c101 = float(np.sum(np.where(idx < len(env),
                                         env[np.minimum(idx, len(env) - 1)], 0.0)) / 101.0)
            print(f"{label:<32} {str(filt):<18} {n_gt:>8,} {ut * 100:>8.2f} {c101 * 100:>8.2f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--decompose', action='store_true',
                    help="attribute the gap against compute_ap_from_csv.py to row filters")
    ap.add_argument('--csv', default=None, help="CSV filename inside the data dir (--decompose only)")
    args = ap.parse_args()

    if args.decompose:
        decompose(args.csv)
    else:
        appendix_table()
