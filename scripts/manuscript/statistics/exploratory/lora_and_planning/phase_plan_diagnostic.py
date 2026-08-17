#!/usr/bin/env python3
"""
Phase-planning diagnostic: what exactly needs to change to hit
  Precision ≥ 70%,  Recall ≥ 94%,  AP50 ≥ 0.90

Analyses:
  A. TP score distribution — how many TPs are "stranded" below threshold
  B. FN breakdown — per-dataset miss rate and score gaps
  C. High-confidence FP anatomy — what the model is most confident about wrongly
  D. AP50 gap decomposition — where the PR curve loses area
  E. Score-shift simulation — what FP-median shift would be needed
  F. Per-dataset recall breakdown

Usage:
    python3 scripts/manuscript/statistics/phase_plan_diagnostic.py \
        --csv anti_hallucination_lora_v3_ckpt6...with_bb.csv \
        --label "v3 ckpt6" \
        --op-threshold 0.0856
"""

import argparse, re, sys, time
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

MIN_AREA      = 1024.0   # 32×32 px²
COCO_IOU_T    = np.linspace(0.50, 0.95, 10)


# ── Loading ───────────────────────────────────────────────────────────────────

def load_csv(path):
    df = pd.read_csv(path, sep=';', low_memory=False)
    df.columns = df.columns.str.strip()
    for c in ('idx_1','idx_2','conf2','IoU','IoU_bb','contourArea_1','contourArea_2'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df['IoU_bb']         = df['IoU_bb'].fillna(0.0)
    df['IoU']            = df['IoU'].fillna(0.0)
    df['contourArea_2']  = df.get('contourArea_2', 0.0).fillna(0.0)
    df['contourArea_1']  = df.get('contourArea_1', 0.0).fillna(0.0)
    return df


def split(df):
    large = df['contourArea_2'] >= MIN_AREA
    is_tp = (df['idx_1'] != -1) & df['idx_1'].notna() & (df['idx_2'] != -1) & df['idx_2'].notna() & large
    is_fp = (df['idx_1'] == -1) & (df['idx_2'] != -1) & df['idx_2'].notna() & large
    is_fn = (df['idx_1'] != -1) & df['idx_1'].notna() & ((df['idx_2'] == -1) | df['idx_2'].isna())
    return df[is_tp].copy(), df[is_fp].copy(), df[is_fn].copy()


# ── AP (101-pt interpolation) ─────────────────────────────────────────────────

def _pr_to_ap101(rec, prec):
    """101-point COCO interpolation using np.interp (fast)."""
    interp_rec = np.linspace(0, 1, 101)
    # COCO uses the precision envelope (max-right)
    prec_env = np.maximum.accumulate(prec[::-1])[::-1]
    interp_prec = np.interp(interp_rec, rec, prec_env, left=0.0, right=0.0)
    return float(np.mean(interp_prec))


def compute_ap50(tp_df, fp_df, n_gt, iou_col='IoU_bb'):
    # Use numpy directly to avoid pandas sort overhead on large frames
    tp_conf = tp_df['conf2'].values.astype(float)
    tp_iou  = tp_df[iou_col].values.astype(float)
    fp_conf = fp_df['conf2'].values.astype(float) if len(fp_df) > 0 else np.array([])
    all_conf  = np.concatenate([tp_conf, fp_conf])
    all_match = np.concatenate([
        (tp_iou >= 0.50).astype(int),
        np.zeros(len(fp_conf), dtype=int),
    ])
    order  = np.argsort(-all_conf, kind='stable')
    tp_c   = np.cumsum(all_match[order])
    fp_c   = np.cumsum(1 - all_match[order])
    rec    = tp_c / n_gt
    prec   = tp_c / (tp_c + fp_c)
    return _pr_to_ap101(rec, prec)


def compute_ap_sweep(tp_df, fp_df, n_gt, iou_col='IoU_bb'):
    return {float(t): _ap_at(tp_df, fp_df, n_gt, iou_col, t) for t in COCO_IOU_T}


def _ap_at(tp_df, fp_df, n_gt, iou_col, iou_thresh):
    tp_conf = tp_df['conf2'].values.astype(float)
    tp_iou  = tp_df[iou_col].values.astype(float)
    fp_conf = fp_df['conf2'].values.astype(float) if len(fp_df) > 0 else np.array([])
    all_conf  = np.concatenate([tp_conf, fp_conf])
    all_match = np.concatenate([
        (tp_iou >= iou_thresh).astype(int),
        np.zeros(len(fp_conf), dtype=int),
    ])
    order = np.argsort(-all_conf, kind='stable')
    tp_c  = np.cumsum(all_match[order])
    fp_c  = np.cumsum(1 - all_match[order])
    rec   = tp_c / n_gt
    prec  = tp_c / (tp_c + fp_c)
    return _pr_to_ap101(rec, prec)


# ── Dataset tag from image name ────────────────────────────────────────────────

def dataset_tag(name: str) -> str:
    name = str(name)
    # Known prefixes in FlatBug
    for prefix in ('abram2023', 'pinoy2023', 'ubc-scanned', 'ALUS', 'cao2022',
                   'ramos', 'danish', 'drews', 'liu', 'nabel', 'paul',
                   'sack', 'schirmel', 'scott', 'wiedmann'):
        if name.lower().startswith(prefix.lower()):
            return prefix
    # Fallback: first alphanumeric token
    m = re.match(r'^([A-Za-z0-9]+)', name)
    return m.group(1) if m else 'unknown'


# ── Main ──────────────────────────────────────────────────────────────────────

SEP = "=" * 78

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")


def run(df, label, op_thresh):
    tp, fp, fn = split(df)
    n_gt = len(tp) + len(fn)

    print(f"\n{'#'*78}")
    print(f"  PHASE PLAN DIAGNOSTIC — {label}")
    print(f"  Operating threshold: {op_thresh}  |  GT: {n_gt:,}  TP: {len(tp):,}  FP: {len(fp):,}  FN: {len(fn):,}")
    print(f"{'#'*78}")

    # ── A. TP score distribution & stranded TPs ───────────────────────────
    section("A. TP SCORE DISTRIBUTION — how many TPs are below the operating threshold")
    tp_s = tp['conf2'].values
    fp_s = fp['conf2'].values
    bins = [(0.005, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, op_thresh),
            (op_thresh, 0.20), (0.20, 0.50), (0.50, 1.01)]
    print(f"  {'Bin':<22s}  {'TPs':>8s}  {'TP%':>6s}  {'FPs':>8s}  {'FP%':>6s}  {'TP:FP':>8s}  Note")
    print("  " + "-" * 76)
    stranded = 0
    for lo, hi in bins:
        n_tp_b = int(((tp_s >= lo) & (tp_s < hi)).sum())
        n_fp_b = int(((fp_s >= lo) & (fp_s < hi)).sum())
        tp_pct = 100.0 * n_tp_b / len(tp_s) if len(tp_s) > 0 else 0
        fp_pct = 100.0 * n_fp_b / len(fp_s) if len(fp_s) > 0 else 0
        ratio  = n_tp_b / n_fp_b if n_fp_b > 0 else float('inf')
        note = ""
        if hi <= op_thresh:
            stranded += n_tp_b
            note = f"  ← STRANDED (below threshold)"
        elif lo == op_thresh:
            note = f"  ← operating threshold boundary"
        print(f"  [{lo:.4f}, {hi:.4f})   {n_tp_b:>8,}  {tp_pct:>5.1f}%  "
              f"{n_fp_b:>8,}  {fp_pct:>5.1f}%  {ratio:>8.3f}{note}")
    total_tp_above = int((tp_s >= op_thresh).sum())
    recall_at_thresh = total_tp_above / n_gt
    print(f"\n  TPs stranded below threshold ({op_thresh}): {stranded:,}")
    print(f"  → These are real insects the model found but scores too low.")
    print(f"  → Recall at operating threshold: {recall_at_thresh:.4f}  ({100*recall_at_thresh:.1f}%)")
    print(f"  → To reach 94% recall at this threshold, need {int(0.94*n_gt - total_tp_above):+,} more TPs above it")

    # ── B. FN analysis — per dataset ──────────────────────────────────────
    section("B. FALSE NEGATIVE BREAKDOWN — which datasets miss most insects")
    tp['_ds']  = tp['image'].apply(dataset_tag)
    fn['_ds']  = fn['image'].apply(dataset_tag)
    # FNs are GT-only rows (no prediction matched); also count TPs per dataset
    ds_tp  = Counter(tp['_ds'])
    ds_fn  = Counter(fn['_ds'])
    all_ds = sorted(set(ds_tp) | set(ds_fn))
    print(f"  {'Dataset':<24s}  {'GT':>7s}  {'TP':>7s}  {'FN':>7s}  {'Recall':>8s}  {'Miss%':>7s}")
    print("  " + "-" * 66)
    for ds in sorted(all_ds, key=lambda d: -(ds_fn.get(d,0))):
        n_tp_d = ds_tp.get(ds, 0)
        n_fn_d = ds_fn.get(ds, 0)
        n_gt_d = n_tp_d + n_fn_d
        rec_d  = n_tp_d / n_gt_d if n_gt_d > 0 else 0.0
        miss_pct = 100.0 * n_fn_d / len(fn) if len(fn) > 0 else 0.0
        flag = "  ← worst" if rec_d < 0.70 else ("  ← low" if rec_d < 0.85 else "")
        print(f"  {ds:<24s}  {n_gt_d:>7,}  {n_tp_d:>7,}  {n_fn_d:>7,}  {rec_d:>8.4f}  {miss_pct:>6.1f}%{flag}")

    # ── B2. Score distribution of TPs that ARE matched (include low scorers) ─
    section("B2. STRANDED TP DETAILS — real insects scored below operating threshold")
    stranded_tp = tp[tp['conf2'] < op_thresh].copy()
    print(f"  Count: {len(stranded_tp):,}  ({100*len(stranded_tp)/len(tp):.1f}% of all TPs)")
    if len(stranded_tp) > 0:
        stranded_tp['_ds'] = stranded_tp['image'].apply(dataset_tag)
        ds_strand = Counter(stranded_tp['_ds'])
        print(f"  Mean score: {stranded_tp['conf2'].mean():.4f}   Median: {stranded_tp['conf2'].median():.4f}")
        print(f"  IoU_bb mean: {stranded_tp['IoU_bb'].mean():.4f}  (quality of actual match)")
        print(f"\n  By dataset:")
        print(f"  {'Dataset':<24s}  {'Stranded TPs':>14s}  {'% of dataset TPs':>18s}")
        print("  " + "-" * 60)
        for ds, cnt in ds_strand.most_common():
            ds_total_tp = ds_tp.get(ds, 0)
            pct = 100.0 * cnt / ds_total_tp if ds_total_tp > 0 else 0
            print(f"  {ds:<24s}  {cnt:>14,}  {pct:>17.1f}%")

    # ── C. High-confidence FP anatomy ────────────────────────────────────
    section(f"C. HIGH-CONFIDENCE FP ANATOMY — FPs at score ≥ {op_thresh}")
    hc_fp = fp[fp['conf2'] >= op_thresh].copy()
    print(f"  Count: {len(hc_fp):,}  ({100*len(hc_fp)/len(fp):.2f}% of all FPs)")
    if len(hc_fp) > 0:
        hc_fp['_ds']   = hc_fp['image'].apply(dataset_tag)
        hc_fp['_area'] = np.sqrt(hc_fp['contourArea_2'].values.astype(float))

        print(f"  Conf: mean={hc_fp['conf2'].mean():.4f}  median={hc_fp['conf2'].median():.4f}  max={hc_fp['conf2'].max():.4f}")
        print(f"  Size (√area): mean={hc_fp['_area'].mean():.1f}px  median={hc_fp['_area'].median():.1f}px")

        print(f"\n  By dataset (top-10 contributors):")
        print(f"  {'Dataset':<24s}  {'HC FPs':>9s}  {'% of HC FP total':>18s}  {'Avg conf':>9s}")
        print("  " + "-" * 66)
        for ds, grp in sorted(hc_fp.groupby('_ds'), key=lambda x: -len(x[1])):
            pct = 100.0 * len(grp) / len(hc_fp)
            print(f"  {ds:<24s}  {len(grp):>9,}  {pct:>17.1f}%  {grp['conf2'].mean():>9.4f}")

        print(f"\n  Confidence sub-breakdown:")
        for lo, hi in [(op_thresh, 0.15), (0.15, 0.30), (0.30, 0.50), (0.50, 1.01)]:
            cnt = int(((hc_fp['conf2'] >= lo) & (hc_fp['conf2'] < hi)).sum())
            pct = 100.0 * cnt / len(hc_fp)
            print(f"  [{lo:.4f}, {hi:.4f}): {cnt:>8,}  {pct:.1f}%")

    # ── D. AP50 gap decomposition ─────────────────────────────────────────
    section("D. AP50 GAP DECOMPOSITION — where does the PR curve lose area vs 0.90 target?")
    # Full PR curve at threshold=0.005 (all predictions) — use pure numpy for speed
    tp_conf_d = tp['conf2'].values.astype(float)
    fp_conf_d = fp['conf2'].values.astype(float)
    tp_ioubb  = tp['IoU_bb'].values.astype(float)
    tp_ioumk  = tp['IoU'].values.astype(float)
    all_conf_d = np.concatenate([tp_conf_d, fp_conf_d])
    order_d    = np.argsort(-all_conf_d, kind='stable')
    n_total_d  = len(all_conf_d)
    # BBox track
    match_bb   = np.concatenate([(tp_ioubb >= 0.50).astype(int),
                                  np.zeros(len(fp_conf_d), dtype=int)])
    sorted_bb  = match_bb[order_d]
    tp_cum_bb  = np.cumsum(sorted_bb)
    fp_cum_bb  = np.cumsum(1 - sorted_bb)
    rec_bb     = tp_cum_bb / n_gt
    prec_bb    = tp_cum_bb / (tp_cum_bb + fp_cum_bb)
    # Mask track
    match_mk   = np.concatenate([(tp_ioumk >= 0.50).astype(int),
                                  np.zeros(len(fp_conf_d), dtype=int)])
    sorted_mk  = match_mk[order_d]
    tp_cum_mk  = np.cumsum(sorted_mk)
    fp_cum_mk  = np.cumsum(1 - sorted_mk)
    rec_mk     = tp_cum_mk / n_gt
    prec_mk    = tp_cum_mk / (tp_cum_mk + fp_cum_mk)

    # Sample the PR curve at recall checkpoints
    print(f"  PR curve at recall checkpoints (BBox / Mask):")
    print(f"  {'Recall':>8s}  {'BBox Prec':>10s}  {'Mask Prec':>10s}  {'Gap to 0.90 target':>20s}")
    print("  " + "-" * 58)
    for r_target in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.92, 0.94, 0.96]:
        sys.stdout.flush()
        m_bb = rec_bb >= r_target
        m_mk = rec_mk >= r_target
        p_bb = float(prec_bb[m_bb].max()) if m_bb.any() else 0.0
        m_mk_any = m_mk.any()
        p_mk = float(prec_mk[m_mk].max()) if m_mk_any else 0.0
        gap  = 0.90 - r_target * p_bb   # rough area contribution
        print(f"  {r_target:>8.2f}  {p_bb:>10.4f}  {p_mk:>10.4f}  {'(precision at this recall)':>20s}")
        sys.stdout.flush()
    print(f"\n  → For AP50 = area under PR curve to reach 0.90:")
    print(f"    The curve needs substantially higher precision at ALL recall levels.")
    print(f"    Current: at recall=0.85, BBox precision ≈ {float(prec_bb[rec_bb>=0.85].max()) if (rec_bb>=0.85).any() else 0.0:.3f}")
    print(f"    Needed:  at recall=0.85, precision ≈ 0.90+  (to lift curve area to 0.90)")

    # ── E. Score-shift simulation ─────────────────────────────────────────
    section("E. SCORE-SHIFT SIMULATION — what FP suppression is needed for each target")
    print(f"  Simulating: if v4 LoRA pushes FP scores DOWN by a factor, what do we get?")
    print(f"  (TP scores held constant; FP scores multiplied by shift_factor)\n")
    print(f"  {'FP shift':>10s}  {'New FP med':>10s}  {'BBox AP50':>10s}  {'t=0.05 Prec':>12s}  "
          f"{'t=0.05 Rec':>11s}  {'t=0.085 Prec':>13s}  {'t=0.085 Rec':>12s}")
    print("  " + "-" * 92)

    # Pre-compute once outside the loop
    tp_iou_flags = (tp['IoU_bb'].values >= 0.50)
    tp_s_cached  = tp_s  # already computed above

    for factor in [1.0, 0.75, 0.50, 0.40, 0.30, 0.20, 0.10]:
        fp_shifted = fp_s * factor
        new_med    = float(np.median(fp_shifted))

        # Recompute AP50 with shifted FP scores (pure numpy, fast)
        all_conf_s   = np.concatenate([tp_s_cached, fp_shifted])
        match_iou_s  = np.concatenate([tp_iou_flags.astype(int),
                                        np.zeros(len(fp_shifted), dtype=int)])
        sort_idx     = np.argsort(-all_conf_s, kind='stable')
        sorted_match = match_iou_s[sort_idx]
        tc  = np.cumsum(sorted_match)
        fc  = np.cumsum(1 - sorted_match)
        rec_s  = tc / n_gt
        prec_s = tc / (tc + fc)
        ap50_s = _pr_to_ap101(rec_s, prec_s)

        # At t=0.05
        fp05   = int((fp_shifted >= 0.05).sum())
        tp05   = int((tp_s >= 0.05).sum())
        p05    = tp05/(tp05+fp05) if (tp05+fp05)>0 else 0.0
        r05    = tp05/n_gt if n_gt>0 else 0.0
        # At op_thresh
        fp_op  = int((fp_shifted >= op_thresh).sum())
        tp_op  = int((tp_s >= op_thresh).sum())
        p_op   = tp_op/(tp_op+fp_op) if (tp_op+fp_op)>0 else 0.0
        r_op   = tp_op/n_gt if n_gt>0 else 0.0

        print(f"  {factor:>10.2f}  {new_med:>10.4f}  {ap50_s:>10.4f}  "
              f"{p05:>12.4f}  {r05:>11.4f}  {p_op:>13.4f}  {r_op:>12.4f}")

    # ── F. What AP50 gains would come from each source ────────────────────
    section("F. IMPROVEMENT LEVERS — marginal AP50 gain per intervention")
    current_ap50 = compute_ap50(tp, fp, n_gt, 'IoU_bb')
    print(f"  Current BBox AP50: {current_ap50:.4f}")
    print(f"\n  Simulation: effect of removing specific FP subsets")
    print(f"  {'Intervention':<40s}  {'FPs removed':>12s}  {'New AP50':>9s}  {'ΔAP50':>8s}")
    print("  " + "-" * 76)

    hc_fp_idx = fp['conf2'] >= op_thresh
    med_fp_idx = (fp['conf2'] >= 0.02) & (fp['conf2'] < op_thresh)

    interventions = [
        ("Remove HC FPs (score ≥ op_thresh)",        hc_fp_idx),
        ("Remove mid FPs (0.02 ≤ score < op_thresh)", med_fp_idx),
        ("Remove HC + mid FPs (score ≥ 0.02)",         fp['conf2'] >= 0.02),
        ("Remove ALL FPs (oracle)",                    pd.Series(True, index=fp.index)),
    ]
    for name, mask in interventions:
        fp_kept = fp[~mask]
        n_removed = int(mask.sum())
        ap_new = compute_ap50(tp, fp_kept, n_gt, 'IoU_bb')
        print(f"  {name:<40s}  {n_removed:>12,}  {ap_new:>9.4f}  {ap_new-current_ap50:>+8.4f}")

    print(f"\n  Simulation: effect of rescoring stranded TPs above threshold")
    stranded_tp_idx = tp['conf2'] < op_thresh
    n_stranded = int(stranded_tp_idx.sum())
    tp_boosted = tp.copy()
    tp_boosted.loc[stranded_tp_idx, 'conf2'] = op_thresh + 0.001  # just above
    ap_boosted = compute_ap50(tp_boosted, fp, n_gt, 'IoU_bb')
    print(f"  {'Boost stranded TPs to threshold+ε':<40s}  {'n/a':>12s}  "
          f"{ap_boosted:>9.4f}  {ap_boosted-current_ap50:>+8.4f}")
    tp_boosted2 = tp.copy()
    tp_boosted2.loc[stranded_tp_idx, 'conf2'] = 0.50  # to high confidence
    ap_boosted2 = compute_ap50(tp_boosted2, fp, n_gt, 'IoU_bb')
    print(f"  {'Boost stranded TPs to 0.5+':<40s}  {'n/a':>12s}  "
          f"{ap_boosted2:>9.4f}  {ap_boosted2-current_ap50:>+8.4f}")
    print(f"  → {n_stranded:,} stranded TPs exist; boosting them would add up to {ap_boosted2-current_ap50:+.4f} AP50")

    # ── G. Concrete targets ───────────────────────────────────────────────
    section("G. TARGETS & FEASIBILITY VERDICT")
    prec_at_op  = float(len(tp[tp['conf2']>=op_thresh]) /
                        (len(tp[tp['conf2']>=op_thresh]) + len(hc_fp)))
    rec_at_op   = float(len(tp[tp['conf2']>=op_thresh])) / n_gt

    print(f"  Current  @t={op_thresh}: Prec={prec_at_op:.3f}  Rec={rec_at_op:.3f}  AP50={current_ap50:.4f}")
    print(f"  Targets:                  Prec=0.700   Rec=0.940   AP50=0.900")
    print()

    # Recall target: need to recover stranded TPs (by lowering threshold or boosting scores)
    target_rec = 0.94
    target_tp  = int(target_rec * n_gt)
    extra_tp_needed = target_tp - int((tp_s >= op_thresh).sum())
    print(f"  ── Recall gap ──")
    print(f"  Need {extra_tp_needed:+,} more TPs above threshold to hit 94% recall")
    print(f"  {n_stranded:,} TPs currently stranded below threshold — "
          f"{'ENOUGH' if n_stranded >= extra_tp_needed else 'NOT ENOUGH'} to close gap by rescoring")
    # What threshold gives 94% recall?
    # Better: sort TP scores and find the 94th-percentile cutoff
    tp_sorted = np.sort(tp_s)
    thresh_for_94 = float(np.percentile(tp_s, (1 - target_rec) * 100))
    tp_above  = int((tp_s >= thresh_for_94).sum())
    rec_check = tp_above / n_gt
    print(f"  Threshold that includes 94% of TPs: {thresh_for_94:.5f}  "
          f"(actual recall={rec_check:.4f})")
    fp_at_94_thresh = int((fp_s >= thresh_for_94).sum())
    prec_at_94      = tp_above / (tp_above + fp_at_94_thresh) if (tp_above+fp_at_94_thresh)>0 else 0.0
    print(f"  At that threshold: FPs={fp_at_94_thresh:,}  Prec={prec_at_94:.4f}")
    print()

    # Precision target
    print(f"  ── Precision gap ──")
    target_prec = 0.70
    tp_at_op = int((tp_s >= op_thresh).sum())
    needed_fp = int(tp_at_op * (1 - target_prec) / target_prec)
    print(f"  At op_thresh={op_thresh}, to hit 70% prec with {tp_at_op:,} TPs:")
    print(f"    Need ≤ {needed_fp:,} FPs  (currently {len(hc_fp):,}, delta={needed_fp-len(hc_fp):+,})")
    print()

    # AP50 target
    print(f"  ── AP50 gap ──")
    print(f"  AP50 = 0.90 requires high precision at ALL recall levels.")
    print(f"  Currently: the curve drops below 0.50 precision at recall ≈ 0.85+")
    print(f"  To reach AP50=0.90: need precision ≥ 0.90 for recall ≤ 0.90,")
    print(f"    and a graceful drop at high recall.")
    print(f"  This requires eliminating the vast majority of FPs across ALL confidence levels,")
    print(f"    OR significantly boosting TP scores so they dominate early in the ranked list.")
    print()

    print(f"  ── Summary of what each training intervention addresses ──")
    print(f"""
  INTERVENTION                     ADDRESSES          EXPECTED GAIN
  ─────────────────────────────────────────────────────────────────────
  A. Hard Negative Mining          Precision ↑       HC FPs ↓ → prec +3–8pp
     (top-10 worst images as neg)  AP50 ↑
  B. TP Score Boosting             Recall ↑          Stranded TPs → rec +3–5pp
     (LoRA loss term for low-conf  AP50 ↑ most
      TP predictions)
  C. Combined A+B in v4 training   All 3 targets     Best path to targets
  D. Raise threshold to 0.12       Prec ↑ fast       Prec 60→70%  TODAY
     (no training needed)          Rec drops ~1pp
  E. Full decoder fine-tune        AP50 ↑            Only option for AP50→0.90
     (not LoRA)                                      Requires > 32-rank LoRA
  ─────────────────────────────────────────────────────────────────────
  Realistic with LoRA v4:   Prec ~68%, Rec ~91%, AP50 ~0.83–0.86
  Achievable TODAY:         Prec 70%+, Rec 85%, AP50 0.80  (t=0.12)
  Hard ceiling (LoRA):      AP50 ~0.87 (FP suppression only helps so much)
  AP50=0.90 likely needs:   Full decoder unfreeze or larger rank + more epochs
""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',          required=True)
    parser.add_argument('--label',        default='experiment')
    parser.add_argument('--op-threshold', type=float, default=0.0856,
                        help="Operating threshold to analyse around (default: 0.0856)")
    args = parser.parse_args()

    df = load_csv(args.csv)
    run(df, args.label, args.op_threshold)


if __name__ == '__main__':
    main()
