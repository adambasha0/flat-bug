#!/usr/bin/env python3
"""
Generate a single side-by-side TP vs FP confidence percentile figure
comparing base SAM3 (score 0.005, mask 0.50) vs fine-tuned SAM3
(checkpoint 18, score 0.005, mask 0.50).

Produces:
  figures/sam3_base_vs_ft_confidence_score005.png

Run from the flat-bug root:
  python3 scripts/manuscript/statistics/compare_confidence_finetuned.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_BASE = os.path.join(
    SCRIPT_DIR, "data",
    "combined_results_sam3_005_with_bb.csv"
)
CSV_FT = os.path.join(
    SCRIPT_DIR, "data", "finetuned",
    "combined_results_sam3_ft_mask_5_checkpoint18_score_005_with_bb.csv"
)
OUT_DIR  = os.path.join(SCRIPT_DIR, "figures")
OUT_FILE = os.path.join(OUT_DIR, "sam3_base_vs_ft_confidence_score005.png")

IOU_THRESHOLD = 0.5
LOW_IOU_MIN   = 0.2
SIZE_THRESHOLD = 32

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def detect_sep(csv_path):
    with open(csv_path, "r") as fh:
        first = fh.readline()
    return "," if "," in first else ";"


def load_and_classify(csv_path):
    df = pd.read_csv(csv_path, sep=detect_sep(csv_path))
    df["area"] = df.apply(
        lambda r: r["contourArea_1"] if r["idx_1"] != -1 else r["contourArea_2"], axis=1
    )
    df["size"] = df["area"] ** 0.5
    df = df[df["size"] >= SIZE_THRESHOLD].copy()
    df["conf"] = df["conf2"]

    def classify(row):
        idx_1, idx_2 = row["idx_1"], row["idx_2"]
        iou = row["IoU"] if pd.notna(row["IoU"]) else 0.0
        if idx_1 != -1 and idx_2 == -1:
            return "FN"
        if idx_1 == -1 and idx_2 != -1:
            return "FP"
        if idx_1 != -1 and idx_2 != -1:
            if iou >= IOU_THRESHOLD:
                return "TP"
            elif iou >= LOW_IOU_MIN:
                return "FP"
            else:
                return "FP"
        return "OTHER"

    df["cls"] = df.apply(classify, axis=1)
    return df


def confidence_percentiles(df):
    percentiles = [25, 75, 90, 95]
    tp_conf = df.loc[df["cls"] == "TP", "conf"].dropna()
    fp_conf = df.loc[df["cls"] == "FP", "conf"].dropna()
    tp_vals = [np.percentile(tp_conf, p) if len(tp_conf) else 0 for p in percentiles]
    fp_vals = [np.percentile(fp_conf, p) if len(fp_conf) else 0 for p in percentiles]
    return percentiles, tp_vals, fp_vals, len(tp_conf), len(fp_conf)


def draw_panel(ax, percentiles, tp_vals, fp_vals, n_tp, n_fp, title):
    x = np.arange(len(percentiles))
    width = 0.35

    bars_tp = ax.bar(x - width / 2, tp_vals, width, label=f"True Positives (n={n_tp:,})",
                     color="#2ca02c", edgecolor="black", linewidth=0.6)
    bars_fp = ax.bar(x + width / 2, fp_vals, width, label=f"False Positives (n={n_fp:,})",
                     color="#d62728", edgecolor="black", linewidth=0.6)

    ax.set_xlabel("Percentile", fontsize=12)
    ax.set_ylabel("Confidence Score", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}th" for p in percentiles], fontsize=10)
    ax.set_ylim(0, 1.25)
    ax.tick_params(axis="y", labelsize=10)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    for bar, val in zip(bars_tp, tp_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8, color="#2ca02c")
    for bar, val in zip(bars_fp, fp_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8, color="#d62728")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df_base = load_and_classify(CSV_BASE)
    df_ft   = load_and_classify(CSV_FT)

    pct_base, tp_base, fp_base, n_tp_base, n_fp_base = confidence_percentiles(df_base)
    pct_ft,   tp_ft,   fp_ft,   n_tp_ft,   n_fp_ft   = confidence_percentiles(df_ft)

    # One figure, two axes — sharey ensures identical y-axis
    fig, (ax_l, ax_r) = plt.subplots(
        1, 2,
        figsize=(14, 5),
        sharey=True
    )

    draw_panel(ax_l, pct_base, tp_base, fp_base, n_tp_base, n_fp_base,
               "Base SAM3 (score 0.005)")
    draw_panel(ax_r, pct_ft,   tp_ft,   fp_ft,   n_tp_ft,   n_fp_ft,
               "Fine-tuned SAM3 (score 0.005)")

    ax_r.set_ylabel("")

    fig.tight_layout(pad=2.0)
    fig.savefig(OUT_FILE, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_FILE}")

    print("\nBase SAM3  — TP pcts:", [f"{v:.3f}" for v in tp_base],
          " FP pcts:", [f"{v:.3f}" for v in fp_base])
    print("Fine-tuned — TP pcts:", [f"{v:.3f}" for v in tp_ft],
          " FP pcts:", [f"{v:.3f}" for v in fp_ft])


if __name__ == "__main__":
    main()
