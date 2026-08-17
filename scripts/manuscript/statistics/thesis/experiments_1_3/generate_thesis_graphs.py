#!/usr/bin/env python3
"""
Thesis Graph Generator for FP/FN Analysis.

Produces publication-quality graphs analyzing False Positives, False Negatives,
confidence distributions, and size-based error analysis.

Output structure:
    <thesis root>/_output/graphs/
        confidence/       - Confidence percentile graphs (TP vs FP)
        fp_fn_breakdown/  - FP and FN distribution charts
        size_analysis/    - Size-based error analysis

Usage (run from anywhere; paths resolve relative to this file):
    python3 generate_thesis_graphs.py                      # all CSVs
    python3 generate_thesis_graphs.py sam3_005             # one key
    python3 generate_thesis_graphs.py --csv path/to/file.csv   # custom file

Set FB_DATA_DIR / FB_GRAPH_ROOT to point at CSVs / outputs outside this tree.
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Configuration ────────────────────────────────────────────────────────────
# Paths resolve relative to this file, so the script runs from any working
# directory and from either checkout. Override with environment variables:
#   FB_DATA_DIR    directory holding the *_experiment_N with-bbox CSVs
#   FB_GRAPH_ROOT  output root for the generated figures
#
# Two supported layouts. Standalone (default): this folder is self-contained —
# CSVs in `experiments_1_3/data/` (unpack `data/*.zip` first), output under
# `<thesis root>/_output/`. Embedded: this folder sits inside the larger
# evaluation working tree, which keeps its own `statistics/{data,graphs}`
# directories two levels up; detected by the upstream `helpers/` beside them.
_HERE = os.path.dirname(os.path.abspath(__file__))
_THESIS_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
_STATS_ROOT = os.path.abspath(os.path.join(_THESIS_ROOT, os.pardir))
_EMBEDDED = os.path.isdir(os.path.join(_STATS_ROOT, "helpers"))

DATA_DIR = os.environ.get(
    "FB_DATA_DIR",
    os.path.join(_STATS_ROOT, "data",
                 "thesis_experiments_using_fb_eval_refactored", "with_bb")
    if _EMBEDDED else os.path.join(_HERE, "data"))

# One entry per *model* (unique CSV). Keys are the short model tokens used in the
# unified figure names: sam3 = zero-shot base, flatbug = flatbug model,
# ft = full decoder fine-tune (Exp 2 / Phase 1), lora = LoRA + background-query suppression
# (Exp 3). CSVs were renamed with an _experiment_N suffix by the user.
CSV = {
    "sam3":    f"{DATA_DIR}/sam3_results_score_005_combined_results_with_bb_experiment_1.csv",
    "flatbug": f"{DATA_DIR}/fb_score_005_using_fb_eval_greedy_corrected_combined_results_with_bb_experiment_1.csv",
    "ft":      f"{DATA_DIR}/sam3_ft_round4_ep18_predictions_mask_05_score_005_combined_results_with_bb_experiment_2.csv",
    "lora":    f"{DATA_DIR}/fine_tuned_sam3_with_lora_checkpoint_10_mask_5_score_005_combined_results_with_bb_experiment_3.csv",
}

# Figure jobs: (experiment number, model token in the figure name, CSV key).
# A model can appear under several experiments because each experiment compares a
# model against a baseline (Exp 2: ft vs base sam3; Exp 3: lora vs Phase-1 ft).
# Same CSV → identical figure content, but named per experiment for easy tracking.
JOBS = [
    (1, "sam3",    "sam3"),
    (1, "flatbug", "flatbug"),
    (2, "sam3",    "sam3"),     # base SAM3 = comparison baseline for the fine-tune
    (2, "ft",      "ft"),
    (3, "ft",      "ft"),       # Phase-1 fine-tune = comparison baseline for LoRA
    (3, "lora",    "lora"),
]

# Score threshold token embedded in every unified figure name.
SCORE_TOKEN = "score_005"

IOU_THRESHOLD = 0.5
LOW_IOU_MIN   = 0.2        # minimum IoU to count as "low_iou" (vs unmatched)
SIZE_THRESHOLD = 32         # sqrt(area) minimum filter

# Size category boundaries (sqrt of mask area in pixels)
SIZE_SMALL_MAX  = 64
SIZE_MEDIUM_MAX = 128

GRAPH_ROOT = os.environ.get(
    "FB_GRAPH_ROOT",
    os.path.join(_STATS_ROOT, "graphs") if _EMBEDDED
    else os.path.join(_THESIS_ROOT, "_output", "graphs"))
DPI = 300

# Output a vector PDF by default (like the R scripts' cairo_pdf): bars, lines and
# text stay razor-sharp at any \includegraphics size — never pixelated on print.
# Override with --format png for a quick raster preview.
FIG_FORMAT = "pdf"

# Side-by-side figures in the thesis are displayed at ~0.485\linewidth ≈ 2.77 in,
# but matplotlib draws them 7–10 in wide, so \includegraphics down-scales the text
# to ~0.29–0.36× on paper (a 15 pt tick label prints at only ~4–5 pt). We counteract
# that by enlarging the in-figure point sizes. Two tiers, because the wider
# size-analysis figures shrink the most. Raise these if the print still looks small.
FONT_SCALE      = 1.20      # confidence / distribution graphs  (displayed ~0.36×)
FONT_SCALE_SIZE = 1.25      # size-analysis graphs              (displayed ~0.29×)

# Base point sizes, before the per-figure scale above is applied.
BASE_TITLE = 15
BASE_LABEL = 16
BASE_TICK  = 15
BASE_LEG   = 12


def font_rc(scale):
    """rcParams dict scaling every text element by `scale`. Applied globally at
    FONT_SCALE; the size-analysis figures re-apply it at FONT_SCALE_SIZE."""
    return {
        "font.size":       13 * scale,
        "axes.titlesize":  BASE_TITLE * scale,
        "axes.labelsize":  BASE_LABEL * scale,
        "xtick.labelsize": BASE_TICK * scale,
        "ytick.labelsize": BASE_TICK * scale,
        "legend.fontsize": BASE_LEG * scale,
    }


# Thesis‑friendly style (non-font rcParams; font sizes come from font_rc below)
plt.rcParams.update({
    "font.family":      "serif",
    "axes.labelweight": "bold",         # bold x/y axis titles
    "axes.labelpad":    10,             # more space between axis title and axis
    # Ticks point inward, matching the R PR-curve style (axis.ticks.length < 0)
    "xtick.direction":  "in",
    "ytick.direction":  "in",
    "xtick.major.size":  5,             # visible inward tick length
    "ytick.major.size":  5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "figure.titlesize": 16,
    "figure.dpi":       DPI,
    "savefig.dpi":      DPI,            # only affects raster (png) exports
    "savefig.bbox":     "tight",        # crop to content, like R's bbox_inches="tight"
    "savefig.pad_inches": 0.02,
    # Embed real TrueType fonts (type 42) instead of Type-3 outlines, so text
    # stays sharp/selectable and passes thesis PDF/A font-embedding checks.
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
    "axes.grid":        True,
    "grid.alpha":       0.3,
})
plt.rcParams.update(font_rc(FONT_SCALE))   # default scale for non-size figures


# ── Helpers ──────────────────────────────────────────────────────────────────

def classify_row(row, iou_thresh=IOU_THRESHOLD, low_iou_min=LOW_IOU_MIN):
    """Return one of: TP, FP_unmatched, FP_low_iou, FN_unmatched, FN_low_iou, OTHER."""
    idx_1 = row["idx_1"]   # GT index
    idx_2 = row["idx_2"]   # Prediction index
    iou   = row["IoU"] if pd.notna(row["IoU"]) else 0.0

    if idx_1 != -1 and idx_2 == -1:
        return "FN_unmatched"
    if idx_1 == -1 and idx_2 != -1:
        return "FP_unmatched"
    if idx_1 != -1 and idx_2 != -1:
        if iou >= iou_thresh:
            return "TP"
        elif iou >= low_iou_min:
            return "FP_low_iou"      # matched but insufficient IoU
        else:
            return "FP_unmatched"    # IoU so low it is effectively unmatched
    return "OTHER"


def add_fn_low_iou(df):
    """Mark the GT side of low‑IoU matches as FN_low_iou (mirror of FP_low_iou)."""
    # For every FP_low_iou row there is also a missed GT → FN_low_iou.
    # We track the count rather than duplicating rows.
    return df


def size_category(sqrt_area):
    if sqrt_area < SIZE_SMALL_MAX:
        return "Small"
    elif sqrt_area < SIZE_MEDIUM_MAX:
        return "Medium"
    return "Large"


def detect_sep(csv_path):
    """Return the delimiter used by this CSV (comma or semicolon)."""
    with open(csv_path, "r") as fh:
        first = fh.readline()
    return "," if "," in first else ";"


def load_and_prepare(csv_path):
    """Load CSV, apply size filter, classify rows, compute derived columns."""
    df = pd.read_csv(csv_path, sep=detect_sep(csv_path))

    # Compute object size from mask area (use GT area if available, else pred)
    df["area"] = df.apply(
        lambda r: r["contourArea_1"] if r["idx_1"] != -1 else r["contourArea_2"],
        axis=1,
    )
    df["size"] = df["area"] ** 0.5
    df = df[df["size"] >= SIZE_THRESHOLD].copy()

    df["classification"] = df.apply(classify_row, axis=1)
    df["conf"] = df["conf2"]
    df["size_cat"] = df["size"].apply(size_category)

    return df


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def csv_tag(csv_path):
    """Short tag derived from csv filename (used in output filenames)."""
    return os.path.splitext(os.path.basename(csv_path))[0]


def out_file(out_dir, stem):
    """Output path for a figure, with the configured (vector by default) format."""
    return os.path.join(out_dir, f"{stem}.{FIG_FORMAT}")


def cap_yticks(ax, cap, fmt="{:g}"):
    """Drop y ticks (tooth + gridline + label) above `cap`, but keep the padded
    y-limit so the headroom for bar labels / the legend remains. Used because a
    score (≤ 1.0) or a rate (≤ 100 %) cannot exceed `cap` — the extra step above
    it is just blank space, so it should carry neither a tick mark nor a number.
    Call after set_ylim() so the auto tick locations reflect the padded range."""
    ticks = [t for t in ax.get_yticks() if t <= cap + 1e-9]
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: fmt.format(v)))


# ── Graph 1: Confidence percentile comparison (TP vs FP) ────────────────────

def plot_confidence_percentiles(df, csv_name, out_dir):
    """
    Two side‑by‑side bar charts showing 25th / 75th / 90th / 95th percentile
    of confidence scores for TPs and FPs respectively.
    """
    ensure_dir(out_dir)

    percentiles = [25, 75, 90, 95]

    tp_conf = df.loc[df["classification"] == "TP", "conf"].dropna()
    fp_conf = df.loc[df["classification"].isin(["FP_unmatched", "FP_low_iou"]), "conf"].dropna()

    tp_vals = [np.percentile(tp_conf, p) if len(tp_conf) else 0 for p in percentiles]
    fp_vals = [np.percentile(fp_conf, p) if len(fp_conf) else 0 for p in percentiles]

    # ── Log summary ──
    print()
    print("  ┌─ Graph 1: Confidence Score Percentiles ")
    print("  │  Confidence scores reflect how certain the model is about each prediction.")
    print("  │  A well-calibrated model should show higher confidence for TPs than FPs.")
    print("  │")
    print(f"  │  True Positives  (n={len(tp_conf):,}):")
    for p, v in zip(percentiles, tp_vals):
        print(f"  │    {p:2d}th pct: {v:.4f}")
    print(f"  │  False Positives (n={len(fp_conf):,}):")
    for p, v in zip(percentiles, fp_vals):
        print(f"  │    {p:2d}th pct: {v:.4f}")
    # Separability check: if FP median < TP 10th pct, distributions are separable
    fp_median = np.median(fp_conf) if len(fp_conf) else 0
    tp_10 = np.percentile(tp_conf, 10) if len(tp_conf) else 0
    if fp_median < tp_10:
        print(f"  │  Separability: GOOD — FP median ({fp_median:.4f}) < TP 10th pct ({tp_10:.4f})")
        print("  │  Raising the confidence threshold can remove FPs with little TP loss.")
    else:
        print(f"  │  Separability: OVERLAP — FP median ({fp_median:.4f}) >= TP 10th pct ({tp_10:.4f})")
        print("  │  Distributions overlap significantly; threshold tuning will damage recall.")
    print("  └" + "─" * 60)

    x = np.arange(len(percentiles))
    width = 0.35

    # ── Individual TP graph ──
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(x, tp_vals, width, color="#2ca02c", edgecolor="black", linewidth=0.6)
    ax.set_xlabel("Percentile")
    ax.set_ylabel("Confidence Score")
    ax.set_title("True Positive Confidence Percentiles")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}th" for p in percentiles])
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, tp_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11 * FONT_SCALE)
    fig.tight_layout()
    fname = out_file(out_dir, f"{csv_name}_tp_confidence_{SCORE_TOKEN}")
    fig.savefig(fname)
    plt.close(fig)
    print(f"  Saved: {fname}")

    # ── Individual FP graph ──
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(x, fp_vals, width, color="#d62728", edgecolor="black", linewidth=0.6)
    ax.set_xlabel("Percentile")
    ax.set_ylabel("Confidence Score")
    ax.set_title("False Positive Confidence Percentiles")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}th" for p in percentiles])
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, fp_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11 * FONT_SCALE)
    fig.tight_layout()
    fname = out_file(out_dir, f"{csv_name}_fp_confidence_{SCORE_TOKEN}")
    fig.savefig(fname)
    plt.close(fig)
    print(f"  Saved: {fname}")

    # ── Combined TP vs FP graph ──
    fig, ax = plt.subplots(figsize=(8, 5))
    bars_tp = ax.bar(x - width / 2, tp_vals, width, label="True Positives (TP)",
                     color="#2ca02c", edgecolor="black", linewidth=0.6)
    bars_fp = ax.bar(x + width / 2, fp_vals, width, label="False Positives (FP)",
                     color="#d62728", edgecolor="black", linewidth=0.6)

    ax.set_xlabel("Percentile")
    ax.set_ylabel("Confidence Score")
    ax.set_title("TP vs FP Confidence Score Percentiles")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}th" for p in percentiles])
    ax.set_ylim(0, 1.25)              # headroom for the legend + bar labels
    cap_yticks(ax, 1.0, "{:.1f}")     # confidence is 0–1: no tick/label above 1.0
    ax.legend(loc="upper left")

    for bar, val in zip(bars_tp, tp_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11 * FONT_SCALE, color="#2ca02c")
    for bar, val in zip(bars_fp, fp_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11 * FONT_SCALE, color="#d62728")

    fig.tight_layout()
    fname = out_file(out_dir, f"{csv_name}_confidence_{SCORE_TOKEN}")
    fig.savefig(fname)
    plt.close(fig)
    print(f"  Saved: {fname}")


# ── Graph 2: FP distribution (unmatched vs low_iou) ─────────────────────────

def plot_fp_distribution(df, csv_name, out_dir):
    """Stacked / grouped bar chart of FP categories."""
    ensure_dir(out_dir)

    fp_unmatched = (df["classification"] == "FP_unmatched").sum()
    fp_low_iou   = (df["classification"] == "FP_low_iou").sum()
    total_fp     = fp_unmatched + fp_low_iou

    pct_u  = 100 * fp_unmatched / total_fp if total_fp > 0 else 0
    pct_li = 100 * fp_low_iou   / total_fp if total_fp > 0 else 0

    # ── Log summary ──
    print()
    print("  ┌─ Graph 2: False Positive Distribution ")
    print("  │  FPs are predictions that do not correspond to a valid ground-truth object.")
    print("  │  Two types are distinguished:")
    print("  │    • Unmatched (IoU = 0): prediction has no overlap with any GT at all.")
    print(f"  │    • Low-IoU matches: prediction overlaps a GT (IoU ≥ {LOW_IOU_MIN}) but falls")
    print(f"  │      below the IoU threshold ({IOU_THRESHOLD}), so it is rejected as a TP.")
    print("  │")
    print(f"  │  Total FP:           {total_fp:>10,}")
    print(f"  │  Unmatched:          {fp_unmatched:>10,}  ({pct_u:.1f}%)")
    print(f"  │  Low-IoU:            {fp_low_iou:>10,}  ({pct_li:.1f}%)")
    print("  └" + "─" * 60)

    categories = ["Unmatched\n(IoU = 0)", f"Low IoU\n(IoU ≥ {LOW_IOU_MIN} & < {IOU_THRESHOLD})"]
    values = [fp_unmatched, fp_low_iou]
    colors = ["#d62728", "#ff7f0e"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(categories, values, color=colors, edgecolor="black", linewidth=0.6, width=0.5)

    for bar, val in zip(bars, values):
        pct = 100 * val / total_fp if total_fp > 0 else 0
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.009,
                f"{val:,} ({pct:.1f}%)", ha="center", va="bottom", fontsize=12 * FONT_SCALE, fontweight="bold")

    ax.set_ylabel("Count")
    ax.set_title(f"False Positive Distribution (Total FP: {total_fp:,})")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_ylim(0, max(values) * 1.40)

    fig.tight_layout()
    fname = out_file(out_dir, f"{csv_name}_fp_distribution_{SCORE_TOKEN}")
    fig.savefig(fname)
    plt.close(fig)
    print(f"  Saved: {fname}")


# ── Graph 3: FN distribution (unmatched vs low_iou) ─────────────────────────

def plot_fn_distribution(df, csv_name, out_dir):
    """Bar chart of FN categories. FN_low_iou mirrors FP_low_iou count."""
    ensure_dir(out_dir)

    fn_unmatched = (df["classification"] == "FN_unmatched").sum()
    # Low‑IoU matches are both FP *and* FN (the GT was not adequately detected)
    fn_low_iou   = (df["classification"] == "FP_low_iou").sum()
    total_fn     = fn_unmatched + fn_low_iou

    pct_u  = 100 * fn_unmatched / total_fn if total_fn > 0 else 0
    pct_li = 100 * fn_low_iou   / total_fn if total_fn > 0 else 0

    # ── Log summary ──
    print()
    print("  ┌─ Graph 3: False Negative Distribution ")
    print("  │  FNs are ground-truth objects that the model failed to detect correctly.")
    print("  │  Two types are distinguished:")
    print("  │    • Unmatched (Missed GT): no prediction overlaps this GT at all.")
    print(f"  │    • Low-IoU: a prediction was made (IoU ≥ {LOW_IOU_MIN}) but quality was")
    print(f"  │      too low to count as TP (IoU < {IOU_THRESHOLD}). Counted as both FP and FN.")
    print("  │")
    print(f"  │  Total FN:           {total_fn:>10,}")
    print(f"  │  Unmatched:          {fn_unmatched:>10,}  ({pct_u:.1f}%)")
    print(f"  │  Low-IoU (mirrored): {fn_low_iou:>10,}  ({pct_li:.1f}%)")
    print("  └" + "─" * 60)

    categories = ["Unmatched\n(Missed GT)", f"Low IoU\n(IoU ≥ {LOW_IOU_MIN} & < {IOU_THRESHOLD})"]
    values = [fn_unmatched, fn_low_iou]
    colors = ["#1f77b4", "#9467bd"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(categories, values, color=colors, edgecolor="black", linewidth=0.6, width=0.5)

    for bar, val in zip(bars, values):
        pct = 100 * val / total_fn if total_fn > 0 else 0
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.009,
                f"{val:,} ({pct:.1f}%)", ha="center", va="bottom", fontsize=12 * FONT_SCALE, fontweight="bold")

    ax.set_ylabel("Count")
    ax.set_title(f"False Negative Distribution (Total FN: {total_fn:,})")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_ylim(0, max(values) * 1.40)

    fig.tight_layout()
    fname = out_file(out_dir, f"{csv_name}_fn_distribution_{SCORE_TOKEN}")
    fig.savefig(fname)
    plt.close(fig)
    print(f"  Saved: {fname}")


# ── Graph 4: FP & FN by object size category ────────────────────────────────

def plot_size_analysis(df, csv_name, out_dir):
    """Grouped bar chart of FP and FN counts by object size category."""
    ensure_dir(out_dir)

    # These figures are the widest (9–10 in) and thus shrink the most on the page,
    # so bump their fonts more than the default. Restored at the end of the function.
    plt.rcParams.update(font_rc(FONT_SCALE_SIZE))

    size_order = ["Small", "Medium", "Large"]
    size_labels = [
        f"Small\n({SIZE_THRESHOLD} \u2264 $\\sqrt{{\\mathrm{{area}}}}$ < {SIZE_SMALL_MAX})",
        f"Medium\n({SIZE_SMALL_MAX} \u2264 $\\sqrt{{\\mathrm{{area}}}}$ < {SIZE_MEDIUM_MAX})",
        f"Large\n($\\sqrt{{\\mathrm{{area}}}}$ \u2265 {SIZE_MEDIUM_MAX})",
    ]

    # Counts per size category
    fp_counts = []
    fn_counts = []
    tp_counts = []

    for cat in size_order:
        sub = df[df["size_cat"] == cat]
        fp = sub["classification"].isin(["FP_unmatched", "FP_low_iou"]).sum()
        fn_unmatched = (sub["classification"] == "FN_unmatched").sum()
        fn_low_iou   = (sub["classification"] == "FP_low_iou").sum()  # mirror
        fn = fn_unmatched + fn_low_iou
        tp = (sub["classification"] == "TP").sum()
        fp_counts.append(fp)
        fn_counts.append(fn)
        tp_counts.append(tp)

    x = np.arange(len(size_order))
    width = 0.25

    # ── Log summary ──
    print()
    print("  ┌─ Graph 4: TP / FP / FN and Rates by Object Size ")
    print("  │  Objects are grouped by the square root of their mask area (in pixels):")
    print(f"  │    Small : {SIZE_THRESHOLD} – {SIZE_SMALL_MAX} px  |  "
          f"Medium: {SIZE_SMALL_MAX} – {SIZE_MEDIUM_MAX} px  |  Large: ≥ {SIZE_MEDIUM_MAX} px")
    print("  │")
    print(f"  │  {'Category':<10}  {'TP':>8}  {'FP':>10}  {'FN':>8}  "
          f"{'FDR':>8}  {'Miss Rate':>10}")
    print("  │  " + "-" * 58)
    for cat, tp, fp, fn in zip(size_order, tp_counts, fp_counts, fn_counts):
        fdr  = 100 * fp / (tp + fp) if (tp + fp) > 0 else 0
        miss = 100 * fn / (tp + fn) if (tp + fn) > 0 else 0
        print(f"  │  {cat:<10}  {tp:>8,}  {fp:>10,}  {fn:>8,}  "
              f"{fdr:>7.1f}%  {miss:>9.1f}%")
    print("  │")
    print("  │  False Discovery Rate (FDR) = FP / (TP+FP):")
    print("  │    Fraction of all predictions that are wrong.")
    print("  │  Miss Rate = FN / (TP+FN):")
    print("  │    Fraction of all ground-truth objects not detected (= 1 - Recall).")
    print("  │  Note: FDR and Miss Rate use different denominators and")
    print("  │        are independent — they do not sum to 100%.")
    print("  └" + "─" * 60)
    # Wider canvas (vs the rate figure's 10 in) so the long multi-line √area
    # x-labels get more room per group and don't collide at the enlarged font.
    fig, ax = plt.subplots(figsize=(12, 5.5))
    bars_tp = ax.bar(x - width, tp_counts, width, label="TP", color="#2ca02c",
                     edgecolor="black", linewidth=0.6)
    bars_fp = ax.bar(x, fp_counts, width, label="FP", color="#d62728",
                     edgecolor="black", linewidth=0.6)
    bars_fn = ax.bar(x + width, fn_counts, width, label="FN", color="#1f77b4",
                     edgecolor="black", linewidth=0.6)

    ax.set_xlabel("Object Size Category")
    ax.set_ylabel("Count")
    ax.set_title("TP / FP / FN by Object Size")
    ax.set_xticks(x)
    ax.set_xticklabels(size_labels)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    all_heights = fp_counts + fn_counts + tp_counts
    max_count = max(all_heights) if all_heights else 1
    label_offset = max_count * 0.01
    for bars in [bars_tp, bars_fp, bars_fn]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + label_offset,
                        f"{int(h):,}", ha="center", va="bottom", fontsize=11 * FONT_SCALE_SIZE)

    # Expand y-axis to fit bar labels + 3-entry legend inside the plot
    ax.set_ylim(0, max_count * 1.35)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fname = out_file(out_dir, f"{csv_name}_size_counts_{SCORE_TOKEN}")
    fig.savefig(fname)
    plt.close(fig)
    print(f"  Saved: {fname}")

    # ── Percentage (error rate) view ──
    fig, ax = plt.subplots(figsize=(10, 5.5))

    fp_pct = []
    fn_pct = []
    for tp, fp, fn in zip(tp_counts, fp_counts, fn_counts):
        total_pred = tp + fp
        total_gt   = tp + fn
        fp_pct.append(100 * fp / total_pred if total_pred > 0 else 0)
        fn_pct.append(100 * fn / total_gt if total_gt > 0 else 0)

    bars_fp = ax.bar(x - width / 2, fp_pct, width,
                     label="False Discovery Rate: FP / (TP+FP)",
                     color="#d62728", edgecolor="black", linewidth=0.6)
    bars_fn = ax.bar(x + width / 2, fn_pct, width,
                     label="Miss Rate: FN / (TP+FN)",
                     color="#1f77b4", edgecolor="black", linewidth=0.6)

    ax.set_xlabel("Object Size Category")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("False Discovery Rate and Miss Rate by Object Size")
    ax.set_xticks(x)
    ax.set_xticklabels(size_labels)

    all_pct = fp_pct + fn_pct
    max_pct = max(all_pct) if any(v > 0 for v in all_pct) else 1
    for bars in [bars_fp, bars_fn]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + max_pct * 0.01,
                        f"{h:.1f}%", ha="center", va="bottom", fontsize=12 * FONT_SCALE_SIZE)

    # Expand y-axis to fit bar labels + 2-entry legend inside the plot
    ax.set_ylim(0, max_pct * 1.35)
    cap_yticks(ax, 100)              # a rate can't exceed 100 %: no tick/label above
    ax.legend(loc="upper right")

    fig.tight_layout()
    fname = out_file(out_dir, f"{csv_name}_rate_by_size_{SCORE_TOKEN}")
    fig.savefig(fname)
    plt.close(fig)
    print(f"  Saved: {fname}")

    plt.rcParams.update(font_rc(FONT_SCALE))   # restore default scale for other figures


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate thesis FP/FN analysis graphs "
                    "(unified expN_model_type_score naming).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 %(prog)s                 # all experiments (1, 2, 3)
  python3 %(prog)s --experiment 2  # only experiment-2 figures
  python3 %(prog)s --format png    # quick raster preview
""",
    )
    parser.add_argument(
        "--experiment", "-e", type=int, choices=[1, 2, 3], default=None,
        help="Only generate figures for this experiment (default: all).",
    )
    parser.add_argument(
        "--format", dest="fmt", choices=["pdf", "png"], default=FIG_FORMAT,
        help="Output figure format (default: pdf — vector, crisp on print).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    global FIG_FORMAT
    FIG_FORMAT = args.fmt

    jobs = [j for j in JOBS if args.experiment is None or j[0] == args.experiment]

    print("=" * 80)
    print("THESIS GRAPH GENERATOR  (unified expN_model_type_score naming)")
    print("=" * 80)

    # Group jobs by CSV so each (large) file is loaded/classified only once; a
    # single model's printed metrics then apply to every experiment reusing it.
    from collections import OrderedDict
    by_csv = OrderedDict()
    for exp, model, key in jobs:
        by_csv.setdefault(key, []).append((exp, model))

    for key, targets in by_csv.items():
        csv_path = CSV[key]
        if not os.path.isfile(csv_path):
            print(f"\n⚠  Skipping model '{key}': file not found → {csv_path}")
            continue

        label = ", ".join(f"exp{e}:{m}" for e, m in targets)
        print(f"\n{'─' * 80}")
        print(f"Model '{key}'  →  {label}")
        print(f"  CSV: {csv_path}")
        print(f"{'─' * 80}")

        df = load_and_prepare(csv_path)
        print(f"  Rows after size filter: {len(df):,}")

        counts = df["classification"].value_counts()
        tp      = counts.get("TP", 0)
        fp_u    = counts.get("FP_unmatched", 0)
        fp_li   = counts.get("FP_low_iou", 0)
        fn_u    = counts.get("FN_unmatched", 0)
        fn_li   = fp_li  # low-IoU matches also count as FN
        print(f"    TP:           {tp:,}")
        print(f"    FP_unmatched: {fp_u:,}")
        print(f"    FP_low_iou:   {fp_li:,}")
        print(f"    FP total:     {fp_u + fp_li:,}")
        print(f"    FN_unmatched: {fn_u:,}")
        print(f"    FN_low_iou:   {fn_li:,}  (mirrored from FP_low_iou)")
        print(f"    FN total:     {fn_u + fn_li:,}")

        # Same df → render once per (experiment, model) with the unified prefix.
        for exp, model in targets:
            prefix   = f"exp{exp}_{model}"
            conf_dir = os.path.join(GRAPH_ROOT, "confidence",      f"experiment-{exp}")
            fpfn_dir = os.path.join(GRAPH_ROOT, "fp_fn_breakdown", f"experiment-{exp}")
            size_dir = os.path.join(GRAPH_ROOT, "size_analysis",   f"experiment-{exp}")
            print(f"\n  ══ {prefix}  (→ experiment-{exp}) ══")
            print("  [1/4] Confidence percentile graphs …")
            plot_confidence_percentiles(df, prefix, conf_dir)
            print("  [2/4] FP distribution graph …")
            plot_fp_distribution(df, prefix, fpfn_dir)
            print("  [3/4] FN distribution graph …")
            plot_fn_distribution(df, prefix, fpfn_dir)
            print("  [4/4] Size analysis graphs …")
            plot_size_analysis(df, prefix, size_dir)

    print(f"\n{'=' * 80}")
    print(f"All graphs saved under: {GRAPH_ROOT}/<type>/experiment-N/")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
