#!/usr/bin/env python3
"""Side-by-side TP vs FP confidence comparison: Phase-1 FT SAM3 vs LoRA SAM3, both score 0.005."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FT  = os.path.join(SCRIPT_DIR,"data","finetuned","combined_results_sam3_ft_mask_5_checkpoint18_score_005_with_bb.csv")
CSV_LORA= os.path.join(SCRIPT_DIR,"data","finetuned","lora","fine_tuned_sam3_with_lora_checkpoint_10_mask_5_score_005_with_bb.csv")
OUT_DIR = os.path.join(SCRIPT_DIR,"figures")
OUT_FILE= os.path.join(OUT_DIR,"sam3_ft_vs_lora_confidence_score005.png")

def detect_sep(path):
    with open(path) as f: h=f.readline()
    return ";" if h.count(";")>h.count(",") else ","

def load_and_classify(path):
    df=pd.read_csv(path, sep=detect_sep(path))
    def get_area(r):
        return r["contourArea_1"] if r["idx_1"]!=-1 else r["contourArea_2"]
    df["area"]=df.apply(get_area,axis=1)
    df["size"]=np.sqrt(df["area"].clip(lower=0))
    df=df[df["size"]>=32]
    tp=df[(df["idx_1"]!=-1)&(df["idx_2"]!=-1)&(df["IoU"]>=0.5)].copy(); tp["label"]="TP"
    fp=df[(df["idx_1"]==-1)|(((df["idx_1"]!=-1)&(df["idx_2"]!=-1)&(df["IoU"]<0.5)))].copy(); fp["label"]="FP"
    return pd.concat([tp,fp])

def confidence_percentiles(df):
    pct=[10,25,50,75,90,95]
    tp=df[df["label"]=="TP"]["conf2"].dropna()
    fp=df[df["label"]=="FP"]["conf2"].dropna()
    return pct, [tp.quantile(p/100) for p in pct], [fp.quantile(p/100) for p in pct], len(tp), len(fp)

def draw_panel(ax, pct, tp_vals, fp_vals, n_tp, n_fp, title):
    x=np.arange(len(pct))
    w=0.35
    ax.bar(x-w/2, tp_vals, w, label=f"TP (n={n_tp:,})", color="#2196F3", alpha=0.85)
    ax.bar(x+w/2, fp_vals, w, label=f"FP (n={n_fp:,})", color="#F44336", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([f"{p}th" for p in pct], fontsize=10)
    ax.set_xlabel("Percentile", fontsize=11); ax.set_ylabel("Confidence Score", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold"); ax.set_ylim(0,1)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

os.makedirs(OUT_DIR, exist_ok=True)
df_ft  = load_and_classify(CSV_FT)
df_lora= load_and_classify(CSV_LORA)
pct_ft,   tp_ft,   fp_ft,   n_tp_ft,   n_fp_ft   = confidence_percentiles(df_ft)
pct_lora, tp_lora, fp_lora, n_tp_lora, n_fp_lora = confidence_percentiles(df_lora)
fig,(ax_l,ax_r)=plt.subplots(1,2,figsize=(14,5),sharey=True)
draw_panel(ax_l,pct_ft,  tp_ft,  fp_ft,  n_tp_ft,  n_fp_ft,  "Phase-1 Fine-tuned SAM3 (score 0.005)")
draw_panel(ax_r,pct_lora,tp_lora,fp_lora,n_tp_lora,n_fp_lora,"LoRA Fine-tuned SAM3 (score 0.005)")
ax_r.set_ylabel("")
fig.tight_layout(pad=2.0)
fig.savefig(OUT_FILE, dpi=300, bbox_inches="tight"); plt.close(fig)
print(f"Saved: {OUT_FILE}")
print("FT   TP:", [f"{v:.3f}" for v in tp_ft],   " FP:", [f"{v:.3f}" for v in fp_ft])
print("LoRA TP:", [f"{v:.3f}" for v in tp_lora], " FP:", [f"{v:.3f}" for v in fp_lora])
