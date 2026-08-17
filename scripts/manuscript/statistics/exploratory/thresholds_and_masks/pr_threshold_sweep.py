#!/usr/bin/env python3
"""
Precision vs Recall — Confidence Threshold Sweep.

Sweeps confidence thresholds from 0.005 to 0.60 (step 0.005) and computes
precision and recall at each threshold.  Three separate analyses are produced:

  Part A — Compare the two fine-tuned models side-by-side:
      • anti_hallucination_lora_v3_ckpt6  (mask=0.5, score=0.005, greedy)
      • sam3_ft_ckpt18                    (mask=0.5, score=0.005)

  Part B — Compare the winner (from Part A) against Flatbug:
      • winner model (from Part A)
      • fb_score_005_with_bb.csv  (Flatbug)
    Flatbug sweep starts at 0.005 (its file already contains preds >= 0.005).

  Part C — Sanity check: do score_005 and score_35 agree for thresh >= 0.35?
      • sam3_ft_ckpt18  score_005  (predictions with conf2 >= 0.005)
      • sam3_ft_ckpt18  score_35   (predictions with conf2 >= 0.35)
    After applying a confidence filter >= 0.35, both files SHOULD give
    identical TP / FP / FN counts if the greedy matching is deterministic
    and independent.  Any discrepancy reveals threshold-induced matching
    differences (a known artefact of greedy per-image matching).

Outputs (written to  scripts/manuscript/statistics/figures/):
    pr_threshold_sweep_part_A.png
    pr_threshold_sweep_part_B.png
    pr_threshold_sweep_part_C_sanity.png

Usage:
    cd /home/dolma/repo/flat-bug
    python3 scripts/manuscript/statistics/pr_threshold_sweep.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_AREA  = 1024.0   # 32 × 32 px²  (same as all other scripts)
IOU_MATCH = 0.50     # IoU threshold that defines a TP

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR / "data" / "thesis_experiments_using_fb_eval_refactored"
FIG_DIR    = SCRIPT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── File paths ─────────────────────────────────────────────────────────────────
FILES = {
    # Part A — two fine-tuned models (score_005, greedy eval)
    "lora_v3_ckpt6": DATA_DIR / "fine_tuned_sam3_with_lora_checkpoint_10_mask_5_score_005_combined_results_with_bb.csv",
    "ft_ckpt18_s005": DATA_DIR / "sam3_ft_round4_ep18_predictions_mask_05_score_005_combined_results_with_bb.csv",

    # Part B — Flatbug baseline
    "flatbug": DATA_DIR / "fb_score_005_using_fb_eval_greedy_corrected_with_bb.csv",

    # Part C — sanity check: score_005 vs score_35 for ckpt18
    "ft_ckpt18_s035": DATA_DIR / "sam3_ft_round3_ep18_predictions_mask_5_score_35_combined_results_with_bb.csv",
}

# Display labels
LABELS = {
    "lora_v3_ckpt6":  "LoRA v3 ckpt-6  (score≥0.005)",
    "ft_ckpt18_s005": "SAM3-FT ckpt-18  (score≥0.005)",
    "flatbug":        "Flatbug  (score≥0.005)",
    "ft_ckpt18_s035": "SAM3-FT ckpt-18  (score≥0.35)",
}

COLORS = {
    "lora_v3_ckpt6":  "#E69F00",   # amber
    "ft_ckpt18_s005": "#0072B2",   # deep blue
    "flatbug":        "#009E73",   # teal-green
    "ft_ckpt18_s035": "#CC79A7",   # pink/magenta
}

# ── Data loading ───────────────────────────────────────────────────────────────

def _bbox_area(s) -> float:
    """Parse '[x1, y1, x2, y2]' string and return (x2-x1)*(y2-y1), or 0.0."""
    try:
        coords = [float(x) for x in str(s).strip("[]").split(",")]
        if len(coords) == 4:
            return max(0.0, (coords[2] - coords[0]) * (coords[3] - coords[1]))
    except Exception:
        pass
    return 0.0


def load_csv(path: Path) -> pd.DataFrame:
    """Load semicolon-separated greedy-eval CSV, coerce numeric columns."""
    df = pd.read_csv(path, sep=";", low_memory=False)
    df.columns = df.columns.str.strip()
    for col in ("idx_1", "idx_2", "conf2", "IoU", "IoU_bb",
                "contourArea_1", "contourArea_2"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["IoU"]    = df["IoU"].fillna(0.0)
    df["IoU_bb"] = df.get("IoU_bb", pd.Series(0.0, index=df.index)).fillna(0.0)
    df["contourArea_1"] = df.get("contourArea_1",
                                  pd.Series(0.0, index=df.index)).fillna(0.0)
    df["contourArea_2"] = df.get("contourArea_2",
                                  pd.Series(0.0, index=df.index)).fillna(0.0)
    # Some files (e.g. Flatbug) store 0 for contourArea_2 but have bbox_2.
    # Fall back to bounding-box area so the size filter works correctly.
    if "bbox_2" in df.columns:
        zero_mask = df["contourArea_2"] == 0.0
        if zero_mask.any():
            df.loc[zero_mask, "contourArea_2"] = (
                df.loc[zero_mask, "bbox_2"].apply(_bbox_area)
            )
    return df


def prepare(df: pd.DataFrame):
    """
    Split into predictions pool and GT count.

    Predictions: rows where idx_2 != -1 AND contourArea_2 >= MIN_AREA
    n_gt:        all rows where idx_1 != -1  (GT side — no size filter,
                 consistent with compute_ap_from_csv.py / COCO convention)

    Each prediction row carries:
        conf2   — confidence score
        IoU     — mask IoU with its matched GT (0.0 if unmatched)
        IoU_bb  — bbox  IoU with its matched GT (0.0 if unmatched)
        matched — True iff idx_1 != -1  (this prediction was paired to a GT)
    """
    is_pred    = (df["idx_2"].notna()) & (df["idx_2"] != -1)
    large_pred = df["contourArea_2"] >= MIN_AREA
    large_gt   = df["contourArea_1"] >= MIN_AREA
    is_gt      = (df["idx_1"].notna()) & (df["idx_1"] != -1)

    # Keep only predictions with pred-area >= MIN_AREA that are NOT matched to a
    # small GT.  A prediction paired with a small GT is excluded entirely (not a
    # TP, not a FP) — consistent with the R evaluation scripts which filter rows
    # by  area = ifelse(idx_1 != -1, contourArea_1, contourArea_2)  before
    # classification.
    matched_small_gt = ((df["idx_1"].notna()) & (df["idx_1"] != -1) &
                        (df["idx_2"].notna()) & (df["idx_2"] != -1) &
                        (~large_gt))
    preds = df[is_pred & large_pred & ~matched_small_gt].copy()

    # A prediction is a valid TP candidate only when its matched GT is also large
    # (the filter above already removed small-GT matches, so this is just the
    # standard matched flag — kept explicit for clarity).
    preds["matched"] = (preds["idx_1"] != -1) & preds["idx_1"].notna()

    # GT count: only annotations large enough to be detectable
    n_gt = int((is_gt & large_gt).sum())

    return preds, n_gt


# ── PR sweep ──────────────────────────────────────────────────────────────────

def pr_sweep(preds: pd.DataFrame, n_gt: int,
             thresholds: np.ndarray,
             iou_col: str = "IoU",
             iou_match: float = IOU_MATCH) -> pd.DataFrame:
    """
    For each confidence threshold t, compute:
        precision = TP / (TP + FP)
        recall    = TP / n_gt
        F1        = harmonic mean of precision and recall

    TP = predictions with conf2 >= t  AND  matched=True  AND  iou_col >= iou_match
    FP = predictions with conf2 >= t  that are NOT TP
    FN = n_gt - TP  (unmatched GTs + matched-but-filtered predictions)
    """
    rows = []
    for t in thresholds:
        kept   = preds[preds["conf2"].fillna(0.0) >= t]
        n_tp   = int((kept["matched"] & (kept[iou_col] >= iou_match)).sum())
        n_fp   = len(kept) - n_tp
        n_fn   = n_gt - n_tp
        prec   = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0.0
        rec    = n_tp / n_gt           if n_gt > 0           else 0.0
        f1     = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        rows.append(dict(threshold=t,
                         tp=n_tp, fp=n_fp, fn=n_fn,
                         precision=prec, recall=rec, f1=f1))
    return pd.DataFrame(rows)


# ── AP computation ────────────────────────────────────────────────────────────

_IOU_THRESHOLDS_50_90 = np.round(np.arange(0.50, 1.00, 0.05), 2)  # 0.50…0.95


def compute_ap(curve: pd.DataFrame) -> float:
    """Area under the PR curve — COCO-style 101-point interpolated AP."""
    rec  = curve["recall"].values.copy()
    prec = curve["precision"].values.copy()
    # Prepend (recall=0, precision=1) sentinel and append (recall=1, precision=0)
    rec  = np.concatenate([[0.0], rec,  [1.0]])
    prec = np.concatenate([[1.0], prec, [0.0]])
    # Monotone envelope: max precision at or to the right of each recall level
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    ap = 0.0
    for r in np.linspace(0.0, 1.0, 101):
        p_arr = prec[rec >= r]
        ap += p_arr.max() if len(p_arr) > 0 else 0.0
    return float(ap / 101)


def compute_ap_metrics(preds: pd.DataFrame, n_gt: int,
                       thresholds: np.ndarray,
                       iou_col: str = "IoU") -> tuple:
    """
    Return (AP@50, AP@50:90).
      AP@50    — AUC with TP criterion IoU ≥ 0.50
      AP@50:90 — mean AP over IoU match thresholds 0.50, 0.55, …, 0.95
    """
    aps = [
        compute_ap(pr_sweep(preds, n_gt, thresholds,
                            iou_col=iou_col, iou_match=float(t)))
        for t in _IOU_THRESHOLDS_50_90
    ]
    return aps[0], float(np.mean(aps))   # AP@50 is first (iou=0.50)


def _ap_label(base_label: str, ap50: float, ap5090: float) -> str:
    """Append AP metrics to a legend label string."""
    return f"{base_label}\nAP@50={ap50:.3f}  AP@50:90={ap5090:.3f}"


# ── Axis-limit helper ────────────────────────────────────────────────────────

def tight_lim(series_list, pad=0.05, lo_clamp=0.0, hi_clamp=1.0):
    """
    Return (lo, hi) tight axis limits computed from a list of 1-D arrays,
    with fractional padding on each side and optional clamping to [0, 1].
    """
    all_vals = np.concatenate([np.asarray(s).ravel() for s in series_list
                               if len(np.asarray(s).ravel()) > 0])
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals) == 0:
        return lo_clamp, hi_clamp
    lo, hi = float(all_vals.min()), float(all_vals.max())
    span = max(hi - lo, 1e-4)
    return max(lo_clamp, lo - pad * span), min(hi_clamp, hi + pad * span)


# ── Text table helper ───────────────────────────────────────────────────────

def print_sweep_table(title: str, label_curve_pairs: list,
                      stride: int = 1, include_counts: bool = True):
    """
    Print a formatted text table of PR sweep results for one or more models.

    Parameters
    ----------
    title             : Table title / section heading
    label_curve_pairs : list of (label_str, curve_DataFrame)
    stride            : Print every Nth row (1 = all rows)
    include_counts    : Include TP / FP / FN integer columns
    """
    print(f"\n{'─' * 72}")
    print(f"  {title}")
    print(f"{'─' * 72}")
    for i, (lbl, _) in enumerate(label_curve_pairs, 1):
        print(f"  M{i} = {lbl}")
    print()

    # Build merged DataFrame — one row per threshold, columns per model
    merged = None
    for i, (_, curve) in enumerate(label_curve_pairs, 1):
        cols = ["threshold", "precision", "recall", "f1"]
        if include_counts:
            cols += ["tp", "fp", "fn"]
        sub = curve[cols].copy()
        rename = {
            "precision": f"M{i}_P",
            "recall":    f"M{i}_R",
            "f1":        f"M{i}_F1",
        }
        if include_counts:
            rename.update({"tp": f"M{i}_TP", "fp": f"M{i}_FP", "fn": f"M{i}_FN"})
        sub = sub.rename(columns=rename)
        merged = sub if merged is None else merged.merge(
            sub, on="threshold", how="outer")

    merged = merged.sort_values("threshold").reset_index(drop=True)
    if stride > 1:
        merged = merged.iloc[::stride].reset_index(drop=True)

    prec_fmt   = lambda v: f"{v:.4f}"
    int_fmt    = lambda v: f"{int(v):>8,}"
    thresh_fmt = lambda v: f"{v:.3f}"
    fmt = {"threshold": thresh_fmt}
    for col in merged.columns:
        if col.endswith(("_P", "_R", "_F1")):
            fmt[col] = prec_fmt
        elif col.endswith(("_TP", "_FP", "_FN")):
            fmt[col] = int_fmt

    print(merged.to_string(index=False, formatters=fmt))
    print()


# ── Figures ───────────────────────────────────────────────────────────────────

def _style_axes(ax, xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1)):
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=11)


def plot_pr_curve_annotated(ax, curve: pd.DataFrame, label: str,
                            color: str, annotate_every: float = 0.10):
    """
    Draw a PR curve as a solid line in `color` and annotate selected
    threshold values as small text markers along the curve.
    """
    xs = curve["recall"].values
    ys = curve["precision"].values
    ts = curve["threshold"].values

    ax.plot(xs, ys, color=color, linewidth=2.5, label=label)

    # Annotate key threshold values along the curve
    last_annotated = -np.inf
    for i, t in enumerate(ts):
        if t - last_annotated >= annotate_every - 1e-9:
            ax.annotate(f"{t:.2f}",
                        xy=(xs[i], ys[i]),
                        fontsize=7,
                        xytext=(4, 2),
                        textcoords="offset points",
                        color=color,
                        alpha=0.85)
            last_annotated = t


def plot_threshold_lines(ax_prec, ax_rec,
                         curve: pd.DataFrame, label: str, color: str):
    """Plot precision-vs-threshold and recall-vs-threshold on two axes."""
    ax_prec.plot(curve["threshold"], curve["precision"],
                 color=color, linewidth=2, label=label)
    ax_rec.plot(curve["threshold"], curve["recall"],
                color=color, linewidth=2, linestyle="--", label=label)


# ── Part A: two fine-tuned models ─────────────────────────────────────────────

def part_a(thresholds: np.ndarray):
    print("\n" + "=" * 72)
    print("PART A — Fine-tuned model comparison (LoRA v3 vs FT ckpt-18)")
    print("=" * 72)

    curves = {}
    ap_metrics = {}     # key → (ap50, ap50_90)
    for key in ("lora_v3_ckpt6", "ft_ckpt18_s005"):
        path = FILES[key]
        if not path.exists():
            print(f"  [SKIP] File not found: {path}")
            continue
        print(f"\n  Loading: {path.name}")
        df = load_csv(path)
        preds, n_gt = prepare(df)
        print(f"    Predictions (size≥32): {len(preds):,}   GT: {n_gt:,}")
        curve = pr_sweep(preds, n_gt, thresholds, iou_col="IoU")
        curves[key] = curve
        ap50, ap5090 = compute_ap_metrics(preds, n_gt, thresholds)
        ap_metrics[key] = (ap50, ap5090)
        print(f"    AP@50={ap50:.4f}  AP@50:90={ap5090:.4f}")

        # Report operating point at threshold = 0.005
        row0 = curve[curve["threshold"] == thresholds[0]].iloc[0]
        print(f"    @ thresh={thresholds[0]:.3f}:  "
              f"P={row0.precision:.4f}  R={row0.recall:.4f}  "
              f"F1={row0.f1:.4f}  "
              f"TP={row0.tp:,}  FP={row0.fp:,}  FN={row0.fn:,}")

        # Find best F1 operating point
        best = curve.loc[curve["f1"].idxmax()]
        print(f"    Best F1={best.f1:.4f}  @ thresh={best.threshold:.3f}:  "
              f"P={best.precision:.4f}  R={best.recall:.4f}")

    if len(curves) < 2:
        print("  Not enough data for Part A plot — skipping.")
        return curves, ap_metrics

    ap_labels = {k: _ap_label(LABELS[k], *ap_metrics[k]) for k in curves}

    # ── Figure ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Part A — Fine-tuned Models: Precision vs Recall Sweep",
                 fontsize=14, fontweight="bold")

    ax_pr, ax_p, ax_r = axes

    # Left: PR curve with threshold annotations
    for key, curve in curves.items():
        plot_pr_curve_annotated(ax_pr, curve, ap_labels[key],
                                COLORS[key], annotate_every=0.10)
    _xlim_pr = tight_lim([c["recall"].values    for c in curves.values()])
    _ylim_pr = tight_lim([c["precision"].values for c in curves.values()])
    _style_axes(ax_pr, xlabel="Recall", ylabel="Precision",
                xlim=_xlim_pr, ylim=_ylim_pr)
    ax_pr.set_title("PR Curve  (colour = threshold value)")
    ax_pr.legend(fontsize=9, loc="lower left")

    # Middle: Precision vs Threshold
    for key, curve in curves.items():
        ax_p.plot(curve["threshold"], curve["precision"],
                  color=COLORS[key], linewidth=2, label=LABELS[key])
    _style_axes(ax_p, xlabel="Confidence Threshold", ylabel="Precision",
                xlim=(thresholds[0], thresholds[-1]),
                ylim=tight_lim([c["precision"].values for c in curves.values()]))
    ax_p.set_title("Precision vs Threshold")
    ax_p.legend(fontsize=9, loc="upper left")

    # Right: Recall vs Threshold
    for key, curve in curves.items():
        ax_r.plot(curve["threshold"], curve["recall"],
                  color=COLORS[key], linewidth=2, label=LABELS[key])
    _style_axes(ax_r, xlabel="Confidence Threshold", ylabel="Recall",
                xlim=(thresholds[0], thresholds[-1]),
                ylim=tight_lim([c["recall"].values for c in curves.values()]))
    ax_r.set_title("Recall vs Threshold")
    ax_r.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    out = FIG_DIR / "pr_threshold_sweep_part_A_final.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out}")

    print_sweep_table(
        "Part A — PR Sweep: threshold × precision / recall / F1",
        [(LABELS[k], curves[k]) for k in curves],
    )

    return curves, ap_metrics


# ── Part B: winner vs Flatbug ─────────────────────────────────────────────────

def part_b(curves_a: dict, ap_metrics_a: dict, thresholds: np.ndarray):
    print("\n" + "=" * 72)
    print("PART B — Winner vs Flatbug")
    print("=" * 72)

    # Determine winner from Part A by best F1 at any threshold
    winner_key = None
    best_f1_so_far = -1.0
    for key, curve in curves_a.items():
        best_f1 = curve["f1"].max()
        print(f"  {LABELS[key]}: best F1 = {best_f1:.4f}")
        if best_f1 > best_f1_so_far:
            best_f1_so_far = best_f1
            winner_key = key

    if winner_key is None:
        print("  No Part A curves available — skipping Part B.")
        return

    print(f"\n  Winner: {LABELS[winner_key]}")

    # Load Flatbug
    fb_path = FILES["flatbug"]
    if not fb_path.exists():
        print(f"  [SKIP] Flatbug file not found: {fb_path}")
        return

    print(f"  Loading Flatbug: {fb_path.name}")
    df_fb = load_csv(fb_path)
    preds_fb, n_gt_fb = prepare(df_fb)
    print(f"    Predictions (size≥32): {len(preds_fb):,}   GT: {n_gt_fb:,}")
    curve_fb = pr_sweep(preds_fb, n_gt_fb, thresholds, iou_col="IoU")

    row0_fb = curve_fb[curve_fb["threshold"] == thresholds[0]].iloc[0]
    print(f"    @ thresh={thresholds[0]:.3f}:  "
          f"P={row0_fb.precision:.4f}  R={row0_fb.recall:.4f}  "
          f"F1={row0_fb.f1:.4f}  "
          f"TP={row0_fb.tp:,}  FP={row0_fb.fp:,}  FN={row0_fb.fn:,}")
    best_fb = curve_fb.loc[curve_fb["f1"].idxmax()]
    print(f"    Best F1={best_fb.f1:.4f}  @ thresh={best_fb.threshold:.3f}:  "
          f"P={best_fb.precision:.4f}  R={best_fb.recall:.4f}")

    ap50_fb, ap5090_fb = compute_ap_metrics(preds_fb, n_gt_fb, thresholds)
    print(f"    AP@50={ap50_fb:.4f}  AP@50:90={ap5090_fb:.4f}")

    winner_curve = curves_a[winner_key]
    winner_ap    = ap_metrics_a.get(winner_key, (0.0, 0.0))
    winner_lbl   = _ap_label(LABELS[winner_key], *winner_ap)
    fb_lbl       = _ap_label(LABELS["flatbug"], ap50_fb, ap5090_fb)

    # ── Figure ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Part B — {LABELS[winner_key]}  vs  Flatbug",
                 fontsize=14, fontweight="bold")

    ax_pr, ax_p, ax_r = axes

    _b_curves = [winner_curve, curve_fb]
    for key, curve, pr_lbl, plain_lbl in [
        (winner_key, winner_curve, winner_lbl, LABELS[winner_key]),
        ("flatbug",  curve_fb,     fb_lbl,     LABELS["flatbug"]),
    ]:
        ax_pr.plot(curve["recall"], curve["precision"],
                   color=COLORS[key], linewidth=2.5, label=pr_lbl)
        ax_p.plot(curve["threshold"], curve["precision"],
                  color=COLORS[key], linewidth=2.5, label=plain_lbl)
        ax_r.plot(curve["threshold"], curve["recall"],
                  color=COLORS[key], linewidth=2.5, label=plain_lbl)

    _style_axes(ax_pr, xlabel="Recall", ylabel="Precision",
                xlim=tight_lim([c["recall"].values    for c in _b_curves]),
                ylim=tight_lim([c["precision"].values for c in _b_curves]))
    ax_pr.set_title("PR Curve")
    ax_pr.legend(fontsize=9, loc="lower left")

    _style_axes(ax_p, xlabel="Confidence Threshold", ylabel="Precision",
                xlim=(thresholds[0], thresholds[-1]),
                ylim=tight_lim([c["precision"].values for c in _b_curves]))
    ax_p.set_title("Precision vs Threshold")
    ax_p.legend(fontsize=9, loc="upper left")

    _style_axes(ax_r, xlabel="Confidence Threshold", ylabel="Recall",
                xlim=(thresholds[0], thresholds[-1]),
                ylim=tight_lim([c["recall"].values for c in _b_curves]))
    ax_r.set_title("Recall vs Threshold")
    ax_r.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    out = FIG_DIR / "pr_threshold_sweep_part_B_final.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out}")

    print_sweep_table(
        "Part B — PR Sweep: threshold × precision / recall / F1",
        [(LABELS[winner_key], winner_curve), (LABELS["flatbug"], curve_fb)],
    )


# ── Part C: sanity check ──────────────────────────────────────────────────────

def part_c(thresholds_full: np.ndarray):
    """
    Compare score_005 and score_35 results when both are filtered to >= 0.35.
    If the greedy matching is order-independent, the two files should agree.
    Divergence means the greedy matcher made different choices when more
    low-confidence predictions were present (a known greedy-matching artefact).
    """
    print("\n" + "=" * 72)
    print("PART C — Sanity check: score_005 vs score_35  (thresh ≥ 0.35)")
    print("=" * 72)

    sanity_keys = ("ft_ckpt18_s005", "ft_ckpt18_s035")
    curves_c   = {}

    # Use thresholds >= 0.35 only
    t_start = 0.35
    thresholds_c = thresholds_full[thresholds_full >= t_start - 1e-9]

    for key in sanity_keys:
        path = FILES[key]
        if key == "ft_ckpt18_s035":
            path = FILES["ft_ckpt18_s035"]
        if not path.exists():
            print(f"  [SKIP] File not found: {path}")
            continue

        print(f"\n  Loading: {path.name}")
        df = load_csv(path)
        preds, n_gt = prepare(df)
        print(f"    Predictions (size≥32): {len(preds):,}   GT: {n_gt:,}")
        curve = pr_sweep(preds, n_gt, thresholds_c, iou_col="IoU")
        curves_c[key] = (curve, n_gt)

        row_start = curve[curve["threshold"] >= t_start - 1e-9].iloc[0]
        print(f"    @ thresh={row_start.threshold:.3f}:  "
              f"P={row_start.precision:.4f}  R={row_start.recall:.4f}  "
              f"TP={row_start.tp:,}  FP={row_start.fp:,}  FN={row_start.fn:,}")

    if len(curves_c) < 2:
        print("  Not enough data for Part C plot — skipping.")
        return

    c1, _ = curves_c["ft_ckpt18_s005"]
    c2, _ = curves_c["ft_ckpt18_s035"]

    # Align on common thresholds
    merged = c1.merge(c2, on="threshold", suffixes=("_s005", "_s035"))
    merged["delta_prec"] = (merged["precision_s005"] - merged["precision_s035"]).abs()
    merged["delta_rec"]  = (merged["recall_s005"]    - merged["recall_s035"]).abs()
    max_dp = merged["delta_prec"].max()
    max_dr = merged["delta_rec"].max()

    print(f"\n  Max |ΔPrecision|: {max_dp:.6f}")
    print(f"  Max |ΔRecall|:    {max_dr:.6f}")
    if max_dp < 1e-4 and max_dr < 1e-4:
        print("  ✅ MATCH: both files produce identical results for thresh ≥ 0.35.")
    else:
        print("  ⚠️  DIVERGENCE detected — greedy matching is order-sensitive.")
        print("     Rows where |ΔPrecision| > 0.001:")
        diff_rows = merged[merged["delta_prec"] > 0.001][
            ["threshold", "precision_s005", "precision_s035",
             "recall_s005", "recall_s035", "delta_prec", "delta_rec"]]
        if len(diff_rows):
            print(diff_rows.to_string(index=False))
        else:
            print("     None above 0.001 threshold (differences are negligible).")

    # ── Figure ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Part C — Sanity Check: score_005 vs score_35 (SAM3-FT ckpt-18)",
                 fontsize=14, fontweight="bold")

    ax_pr, ax_p, ax_r = axes

    key_map = {
        "ft_ckpt18_s005": ("SAM3-FT ckpt-18  (file score≥0.005, filtered≥0.35)",
                           COLORS["ft_ckpt18_s005"], c1),
        "ft_ckpt18_s035": ("SAM3-FT ckpt-18  (file score≥0.35, filtered≥0.35)",
                           COLORS["ft_ckpt18_s035"], c2),
    }

    _c_curves = [c1, c2]
    for key, (lbl, col, curve) in key_map.items():
        ax_pr.plot(curve["recall"], curve["precision"],
                   color=col, linewidth=2.5, label=lbl)
        ax_p.plot(curve["threshold"], curve["precision"],
                  color=col, linewidth=2.5, label=lbl)
        ax_r.plot(curve["threshold"], curve["recall"],
                  color=col, linewidth=2.5, label=lbl)

    _style_axes(ax_pr, xlabel="Recall", ylabel="Precision",
                xlim=tight_lim([c["recall"].values    for c in _c_curves]),
                ylim=tight_lim([c["precision"].values for c in _c_curves]))
    ax_pr.set_title("PR Curve  (thresh 0.35 → 0.60)")
    ax_pr.legend(fontsize=8, loc="lower left")

    _style_axes(ax_p, xlabel="Confidence Threshold", ylabel="Precision",
                xlim=(t_start, thresholds_c[-1]),
                ylim=tight_lim([c["precision"].values for c in _c_curves]))
    ax_p.set_title("Precision vs Threshold")
    ax_p.legend(fontsize=8, loc="upper left")

    _style_axes(ax_r, xlabel="Confidence Threshold", ylabel="Recall",
                xlim=(t_start, thresholds_c[-1]),
                ylim=tight_lim([c["recall"].values for c in _c_curves]))
    ax_r.set_title("Recall vs Threshold")
    ax_r.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    out = FIG_DIR / "pr_threshold_sweep_part_C_sanity_final.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out}")

    print_sweep_table(
        "Part C — Sanity Check: threshold × precision / recall / F1",
        [("SAM3-FT ckpt-18  (file score≥0.005, filtered≥0.35)", c1),
         ("SAM3-FT ckpt-18  (file score≥0.35, filtered≥0.35)",  c2)],
    )


# ── Also produce a combined 4-panel overview ──────────────────────────────────

def combined_overview(curves_a: dict, ap_metrics_a: dict, thresholds: np.ndarray):
    """
    Single figure with:
      (top-left)  PR curve — all models (Part A + winner vs Flatbug)
      (top-right) Precision vs Threshold — all models
      (bottom-left)  Recall vs Threshold — all models
      (bottom-right) F1 vs Threshold — all models
    """
    fb_path = FILES["flatbug"]
    if not fb_path.exists():
        return
    df_fb = load_csv(fb_path)
    preds_fb, n_gt_fb = prepare(df_fb)
    curve_fb = pr_sweep(preds_fb, n_gt_fb, thresholds, iou_col="IoU")
    ap50_fb, ap5090_fb = compute_ap_metrics(preds_fb, n_gt_fb, thresholds)

    all_curves = {**curves_a, "flatbug": curve_fb}
    all_ap     = {**ap_metrics_a, "flatbug": (ap50_fb, ap5090_fb)}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Precision–Recall Threshold Sweep: All Models Overview",
                 fontsize=15, fontweight="bold")

    ax_pr = axes[0, 0]
    ax_p  = axes[0, 1]
    ax_r  = axes[1, 0]
    ax_f1 = axes[1, 1]

    for key, curve in all_curves.items():
        base = LABELS.get(key, key)
        lbl  = _ap_label(base, *all_ap.get(key, (0.0, 0.0)))
        col  = COLORS.get(key, "grey")
        ax_pr.plot(curve["recall"], curve["precision"],
                   color=col, linewidth=2, label=lbl)
        ax_p.plot(curve["threshold"], curve["precision"],
                  color=col, linewidth=2, label=base)
        ax_r.plot(curve["threshold"], curve["recall"],
                  color=col, linewidth=2, label=base)
        ax_f1.plot(curve["threshold"], curve["f1"],
                   color=col, linewidth=2, label=base)

    _ov_prec = [c["precision"].values for c in all_curves.values()]
    _ov_rec  = [c["recall"].values    for c in all_curves.values()]
    _ov_f1   = [c["f1"].values        for c in all_curves.values()]

    _style_axes(ax_pr, "Recall", "Precision",
                xlim=tight_lim(_ov_rec),
                ylim=tight_lim(_ov_prec))
    ax_pr.set_title("PR Curve", fontsize=12)
    ax_pr.legend(fontsize=8, loc="lower left")

    _style_axes(ax_p, "Confidence Threshold", "Precision",
                xlim=(thresholds[0], thresholds[-1]),
                ylim=tight_lim(_ov_prec))
    ax_p.set_title("Precision vs Threshold", fontsize=12)
    ax_p.legend(fontsize=8, loc="upper left")

    _style_axes(ax_r, "Confidence Threshold", "Recall",
                xlim=(thresholds[0], thresholds[-1]),
                ylim=tight_lim(_ov_rec))
    ax_r.set_title("Recall vs Threshold", fontsize=12)
    ax_r.legend(fontsize=8, loc="upper right")

    _style_axes(ax_f1, "Confidence Threshold", "F1 Score",
                xlim=(thresholds[0], thresholds[-1]),
                ylim=tight_lim(_ov_f1))
    ax_f1.set_title("F1 Score vs Threshold", fontsize=12)
    ax_f1.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    out = FIG_DIR / "pr_threshold_sweep_overview_final.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out}")

    # TP/FP/FN omitted — too wide for 3 models side-by-side
    print_sweep_table(
        "Combined Overview — PR Sweep: threshold × precision / recall / F1",
        [(LABELS.get(k, k), c) for k, c in all_curves.items()],
        include_counts=False,
    )


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    # Threshold grid: 0.005 to 1.00 in steps of 0.005
    # (200 points — fine enough for a smooth curve without heavy memory use)
    thresholds = np.round(np.arange(0.005, 1.005, 0.005), 6)

    # Verify files
    print("Checking input files …")
    missing = []
    for key, path in FILES.items():
        status = "✓" if path.exists() else "✗ MISSING"
        print(f"  [{status}] {key}: {path.name}")
        if not path.exists():
            missing.append(key)
    if missing:
        print(f"\nWarning: {len(missing)} file(s) missing — those parts will be skipped.")

    # ── Parts A, B, C ──────────────────────────────────────────────────────
    curves_a, ap_metrics_a = part_a(thresholds)
    part_b(curves_a, ap_metrics_a, thresholds)
    part_c(thresholds)

    # ── Combined overview (all models on one figure) ────────────────────────
    print("\n" + "=" * 72)
    print("COMBINED OVERVIEW")
    print("=" * 72)
    combined_overview(curves_a, ap_metrics_a, thresholds)

    print("\nDone.  Figures written to:", FIG_DIR)


if __name__ == "__main__":
    main()
