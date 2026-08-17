"""
Diagnostic script: compare PR curves from fb_eval.py (optimal matching)
vs fb_eval_greedy.py (greedy confidence-sorted matching).

Outputs
-------
figures/pr_curve_greedy_vs_optimal.png
    Side-by-side / overlaid PR curves with the difference region highlighted.
figures/bad_match_analysis.png
    Distribution of IoU values for "matched" pairs from each method.
stdout: per-region P/R/F1 differences and bad-match counts.

Run from anywhere; paths resolve relative to this file:
    python compare_greedy_vs_optimal.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── paths ─────────────────────────────────────────────────────────────────────
# Needs the FlatBug-L detections matched twice — once by fb_eval.py (optimal
# assignment) and once by fb_eval_greedy.py (confidence-sorted greedy). Standalone
# layout keeps both next to the Experiment 1-3 scripts; embedded in the evaluation
# working tree they sit in its own data directory. Override with FB_DATA_DIR.
_HERE = os.path.dirname(os.path.abspath(__file__))
_THESIS_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
_STATS_ROOT = os.path.abspath(os.path.join(_THESIS_ROOT, os.pardir))
_EMBEDDED = os.path.isdir(os.path.join(_STATS_ROOT, "helpers"))

DATA_DIR = os.environ.get(
    "FB_DATA_DIR",
    os.path.join(_STATS_ROOT, "data") if _EMBEDDED
    else os.path.join(_THESIS_ROOT, "experiments_1_3", "data"))
FIG_DIR = os.environ.get(
    "FB_FIGURE_DIR",
    os.path.join(_STATS_ROOT, "figures") if _EMBEDDED
    else os.path.join(_THESIS_ROOT, "_output", "ap_evaluation"))
os.makedirs(FIG_DIR, exist_ok=True)

OPTIMAL_CSV = os.path.join(DATA_DIR, "fb_score_005_using_fb_eval_with_bb.csv")
GREEDY_CSV  = os.path.join(DATA_DIR, "fb_score_005_using_fb_eval_greedy_with_bb.csv")

IOU_THRESH  = 0.5   # standard COCO-style evaluation threshold
MIN_SIZE    = 32    # same filter as R script (size = sqrt(area) >= 32)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_and_classify(csv_path, iou_col="IoU", iou_thresh=IOU_THRESH):
    """Load a per-pair CSV (from fb_eval.py or fb_eval_greedy.py) and return a
    flat DataFrame of (conf, result) pairs — one row per TP/FP/FN entry —
    suitable for building a PR curve by sorting on conf DESC.

    match_type derivation (mirrors R script):
        idx_1 != -1 & idx_2 == -1  →  unmatched_gt   (FN)
        idx_1 == -1 & idx_2 != -1  →  unmatched_pred  (FP)
        idx_1 != -1 & idx_2 != -1  →  matched          (TP or bad match)

    Bad matches (matched but IoU < iou_thresh) become:
        one FP entry with their actual confidence
        one FN entry with conf = 0
    This mirrors classify_at_threshold() in the R script exactly.
    """
    df = pd.read_csv(csv_path, sep=";")

    # Derive match_type
    df["match_type"] = np.where(
        (df["idx_1"] != -1) & (df["idx_2"] == -1), "unmatched_gt",
        np.where(
            (df["idx_1"] == -1) & (df["idx_2"] != -1), "unmatched_pred",
            "matched"
        )
    )
    df["conf"] = df.get("conf2", pd.Series(0.0, index=df.index)).fillna(0.0)

    # Size filter  (mirrors R: filter(size >= 32))
    area_col = np.where(df["idx_1"] != -1, df["contourArea_1"], df["contourArea_2"])
    df = df[np.sqrt(area_col) >= MIN_SIZE].copy()

    if iou_col not in df.columns:
        raise KeyError(f"Column '{iou_col}' not found in {csv_path}. "
                       f"Available: {list(df.columns)}")

    rows = []
    for _, row in df.iterrows():
        mt = row["match_type"]
        if mt == "unmatched_gt":
            rows.append({"conf": 0.0, "result": "FN", "IoU": np.nan})
        elif mt == "unmatched_pred":
            rows.append({"conf": row["conf"], "result": "FP", "IoU": np.nan})
        else:  # matched
            iou_val = row[iou_col]
            if pd.isna(iou_val) or iou_val < iou_thresh:
                # Bad match → FP + FN (conf=0)
                rows.append({"conf": row["conf"], "result": "FP", "IoU": iou_val})
                rows.append({"conf": 0.0,         "result": "FN", "IoU": iou_val})
            else:
                rows.append({"conf": row["conf"], "result": "TP", "IoU": iou_val})

    result_df = pd.DataFrame(rows)
    return result_df, df   # second return is the raw classified df for analysis


def build_pr_curve(classified_df):
    """Sort by confidence DESC, compute cumulative TP/FP, return (recall, precision)."""
    df = classified_df.sort_values("conf", ascending=False).copy()
    n_fn_total = (df["result"] == "FN").sum()
    n_tp_total = (df["result"] == "TP").sum()
    total_pos  = n_tp_total + n_fn_total   # all true positives (GT objects)

    if total_pos == 0:
        return np.array([0.0, 1.0]), np.array([1.0, 1.0])

    tp_cumsum  = (df["result"] == "TP").cumsum().values
    fp_cumsum  = (df["result"] == "FP").cumsum().values

    recall    = tp_cumsum  / total_pos
    denom     = tp_cumsum + fp_cumsum
    precision = np.where(denom > 0, tp_cumsum / denom, 1.0)

    return recall, precision


def integrate_pr(recall, precision):
    """Area under PR curve (11-point interpolation, same as VOC/COCO AP)."""
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        prec_at_thr = precision[recall >= thr]
        ap += np.max(prec_at_thr) if len(prec_at_thr) > 0 else 0.0
    return ap / 11.0


def ap_at_threshold(classified_df, iou_thresh):
    """Recompute AP for a single IoU threshold from a raw (not yet classified) df."""
    pass   # see compute_ap_range below


def compute_ap_range(raw_df, iou_col="IoU", lo=0.5, hi=0.95, step=0.05):
    """AP averaged over IoU thresholds [lo, hi] with given step (COCO-style)."""
    aps = []
    for t in np.arange(lo, hi + step / 2, step):
        cl, _ = load_and_classify.__wrapped__(raw_df, iou_col=iou_col, iou_thresh=t)
        r, p  = build_pr_curve(cl)
        aps.append(integrate_pr(r, p))
    return float(np.mean(aps))


# ── main analysis ─────────────────────────────────────────────────────────────

def analyse(label, csv_path, iou_col="IoU"):
    classified, raw = load_and_classify(csv_path, iou_col=iou_col)
    n_tp  = (classified["result"] == "TP").sum()
    n_fp  = (classified["result"] == "FP").sum()
    n_fn  = (classified["result"] == "FN").sum()

    matched  = raw[raw["match_type"] == "matched"]
    n_bad    = (matched[iou_col].fillna(0) < IOU_THRESH).sum()
    n_good   = (matched[iou_col].fillna(0) >= IOU_THRESH).sum()

    r, p = build_pr_curve(classified)
    ap50 = integrate_pr(r, p)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Rows total (after size filter): {len(raw)}")
    print(f"  Matched pairs:  {len(matched)}")
    print(f"    Good (IoU >= {IOU_THRESH}): {n_good}")
    print(f"    Bad  (IoU <  {IOU_THRESH}): {n_bad}  ← causes FP+FN oscillations")
    print(f"  TP={n_tp}  FP={n_fp}  FN={n_fn}")
    print(f"  AP@50 (11-point): {ap50:.4f}")
    return dict(label=label, classified=classified, raw=raw,
                r=r, p=p, ap50=ap50, n_bad=n_bad, n_good=n_good, iou_col=iou_col)


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_pr_comparison(results, out_path):
    colours = {"Optimal (fb_eval)": "#2563EB", "Greedy (fb_eval_greedy)": "#DC2626"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: overlaid PR curves ──────────────────────────────────────────────
    ax = axes[0]
    for res in results:
        c = colours.get(res["label"], "grey")
        ax.plot(res["r"], res["p"], color=c, linewidth=1.5,
                label=f"{res['label']}  AP50={res['ap50']:.3f}")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("PR Curve Comparison (IoU ≥ 0.5)", fontsize=13)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # ── Right: |Precision difference| along recall axis ───────────────────────
    ax2 = axes[1]
    if len(results) == 2:
        # Interpolate both onto common recall grid
        recall_grid = np.linspace(0, 1, 500)
        prec_interp = []
        for res in results:
            # Reverse-monotone interpolation (max-precision envelope)
            r, p = res["r"], res["p"]
            p_env = np.maximum.accumulate(p[::-1])[::-1]
            pi = np.interp(recall_grid, r, p_env, left=p_env[0], right=0.0)
            prec_interp.append(pi)

        diff = prec_interp[0] - prec_interp[1]
        ax2.fill_between(recall_grid, diff, 0,
                         where=diff >= 0, alpha=0.4, color="#2563EB",
                         label="Optimal better")
        ax2.fill_between(recall_grid, diff, 0,
                         where=diff < 0, alpha=0.4, color="#DC2626",
                         label="Greedy better")
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_xlabel("Recall", fontsize=12)
        ax2.set_ylabel("Precision difference (Optimal − Greedy)", fontsize=11)
        ax2.set_title("Precision Difference Along Recall Axis", fontsize=13)
        ax2.set_xlim(0, 1)
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)

        # Print where biggest differences are
        peak_idx = np.argmax(np.abs(diff))
        print(f"\nBiggest absolute difference: {diff[peak_idx]:+.4f} "
              f"at recall ≈ {recall_grid[peak_idx]:.3f}")
        # High / mid / low recall thirds
        for lo, hi, name in [(0.0, 0.33, "low recall [0.0–0.33]"),
                             (0.33, 0.67, "mid recall [0.33–0.67]"),
                             (0.67, 1.0,  "high recall [0.67–1.0]")]:
            mask = (recall_grid >= lo) & (recall_grid < hi)
            mean_diff = diff[mask].mean()
            print(f"  Mean diff in {name}: {mean_diff:+.4f}")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPR comparison saved → {out_path}")


def plot_iou_distributions(results, out_path):
    """Histogram of IoU values for matched pairs, both methods."""
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 4))
    if len(results) == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        matched = res["raw"][res["raw"]["match_type"] == "matched"]
        iou_vals = matched[res["iou_col"]].dropna()
        ax.hist(iou_vals, bins=50, color="#4B5563", edgecolor="white", linewidth=0.4)
        ax.axvline(IOU_THRESH, color="red", linestyle="--", linewidth=1.2,
                   label=f"IoU threshold = {IOU_THRESH}")
        n_bad = (iou_vals < IOU_THRESH).sum()
        ax.set_title(
            f"{res['label']}\n"
            f"Matched pairs: {len(iou_vals)}, bad (< {IOU_THRESH}): {n_bad} "
            f"({100*n_bad/max(len(iou_vals),1):.1f}%)",
            fontsize=11
        )
        ax.set_xlabel("IoU", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"IoU distribution saved  → {out_path}")


def plot_precision_at_recall_thresholds(results, out_path):
    """Bar chart: precision at fixed recall thresholds {0.5, 0.6, ..., 0.9}."""
    recall_points = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    colours_list  = ["#2563EB", "#DC2626"]
    bar_width     = 0.35
    x             = np.arange(len(recall_points))

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, res in enumerate(results):
        r, p = res["r"], res["p"]
        p_env = np.maximum.accumulate(p[::-1])[::-1]
        prec_at = [np.interp(rc, r, p_env, left=p_env[0], right=0.0)
                   for rc in recall_points]
        offset = (i - 0.5) * bar_width
        bars = ax.bar(x + offset, prec_at, bar_width, label=res["label"],
                      color=colours_list[i % len(colours_list)], alpha=0.8)
        for bar, val in zip(bars, prec_at):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"R={rc}" for rc in recall_points])
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_title("Precision at Fixed Recall Points", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Precision-at-recall chart saved → {out_path}")


# ── entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Patch load_and_classify so that compute_ap_range can call it with a df directly
    import functools
    _orig = load_and_classify

    @functools.wraps(_orig)
    def _patched(csv_or_df, iou_col="IoU", iou_thresh=IOU_THRESH):
        if isinstance(csv_or_df, pd.DataFrame):
            # called from compute_ap_range with raw df already loaded
            return _orig.__wrapped__(csv_or_df, iou_col=iou_col, iou_thresh=iou_thresh)
        return _orig(csv_or_df, iou_col=iou_col, iou_thresh=iou_thresh)

    # Attach __wrapped__ for introspection (compute_ap_range uses it)
    _orig.__wrapped__ = lambda df, iou_col, iou_thresh: (
        load_and_classify.__wrapped__(df, iou_col, iou_thresh)
        if hasattr(load_and_classify, "__wrapped__")
        else _orig(df, iou_col=iou_col, iou_thresh=iou_thresh)
    )

    print("Loading and classifying data …")
    results = []

    # Check which IoU column the greedy CSV actually has
    greedy_cols = pd.read_csv(GREEDY_CSV, sep=";", nrows=0).columns.tolist()
    greedy_iou_col = "IoU_bb" if "IoU_bb" in greedy_cols else "IoU"

    for label, path, iou_col in [
        ("Optimal (fb_eval)",        OPTIMAL_CSV, "IoU"),
        ("Greedy (fb_eval_greedy)",  GREEDY_CSV,  greedy_iou_col),
    ]:
        try:
            res = analyse(label, path, iou_col=iou_col)
            results.append(res)
        except FileNotFoundError:
            print(f"  [SKIP] File not found: {path}")

    if len(results) < 2:
        print("\nNeed both CSV files present to run comparison. Exiting.")
        raise SystemExit(1)

    # ── Root cause summary ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  ROOT CAUSE SUMMARY")
    print("="*60)
    for res in results:
        print(f"  {res['label']}")
        print(f"    Bad matches (IoU < {IOU_THRESH}): {res['n_bad']}")
        print(f"    Each bad match contributes 1 FP + 1 FN in R classify_at_threshold()")
        print(f"    → {res['n_bad']} extra FP + {res['n_bad']} extra FN rows in curve data")
    print()
    print("  The extra FP rows from bad matches appear at the confidence level of")
    print("  their prediction (could be anywhere in the sorted curve), causing")
    print("  sudden drops in precision not tied to any real low-confidence prediction.")
    print("  The extra FN rows have conf=0 so they always add to the denominator,")
    print("  depressing recall at every point above that confidence.")
    print()
    print("  FIX: use --iou-threshold 0.5 in fb_eval_greedy.py / fb_eval.py --greedy")
    print("  This eliminates all bad matches by construction → smoother PR curves.")

    # ── Generate figures ──────────────────────────────────────────────────────
    plot_pr_comparison(
        results,
        os.path.join(FIG_DIR, "pr_curve_greedy_vs_optimal.png")
    )
    plot_iou_distributions(
        results,
        os.path.join(FIG_DIR, "bad_match_analysis.png")
    )
    plot_precision_at_recall_thresholds(
        results,
        os.path.join(FIG_DIR, "precision_at_recall_comparison.png")
    )
    print("\nDone.")
