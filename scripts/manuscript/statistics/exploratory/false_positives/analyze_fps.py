# python3 scripts/manuscript/statistics/analyze_fps.py \

import pandas as pd

# 1. Load your specific LoRA results CSV
csv_path = "scripts/manuscript/statistics/data/finetuned/lora/anti_hallucination_lora_v3_checkpoint_6_mask_5_score_005_using_greed_eval_with_bb.csv"

""" 
csv_path = "scripts/manuscript/statistics/data/combined_results_sam3_ft_mask_05_checkpoint18_score_001_greedy.csv"
Total False Positives analyzed: 853104
-------------------------------------------------------
Confidence Range   | FP Count     | % of Total FPs
-------------------------------------------------------
0.001 to 0.005     | 141405       | 16.58%  (Cumulative: 16.58%)
0.005 to 0.01      | 174629       | 20.47%  (Cumulative: 37.05%)
0.01 to 0.02       | 293612       | 34.42%  (Cumulative: 71.46%)
0.02 to 0.05       | 200954       | 23.56%  (Cumulative: 95.02%)
0.05 to 0.10       | 30134        |  3.53%  (Cumulative: 98.55%)
0.10 to 0.50       | 10063        |  1.18%  (Cumulative: 99.73%)
0.50 to 1.00       | 2307         |  0.27%  (Cumulative: 100.00%)
-------------------------------------------------------

csv_path = "scripts/manuscript/statistics/data/combined_results_sam3_fine_tuning_with_lora_mask_05_score_001_using_greedy_eval_with_bb.csv"

-------------------------------------------------------
Confidence Range   | FP Count     | % of Total FPs
-------------------------------------------------------
0.001 to 0.005     | 72023        | 14.66%  (Cumulative: 14.66%)
0.005 to 0.01      | 122693       | 24.97%  (Cumulative: 39.63%)
0.01 to 0.02       | 190757       | 38.83%  (Cumulative: 78.46%)
0.02 to 0.05       | 80473        | 16.38%  (Cumulative: 94.84%)
0.05 to 0.10       | 16007        |  3.26%  (Cumulative: 98.10%)
0.10 to 0.50       | 7220         |  1.47%  (Cumulative: 99.56%)
0.50 to 1.00       | 2043         |  0.42%  (Cumulative: 99.98%) 
-------------------------------------------------------
"""


print(f"Loading data from {csv_path}...")

# Let Pandas automatically figure out if it's separated by tabs (\t) or commas (,), 
# and strip any accidental whitespace from the column headers
df = pd.read_csv(csv_path, sep=None, engine='python')
df.columns = df.columns.str.strip()

print(f"Successfully loaded. Found columns: {df.columns.tolist()}\n")

# 2. Isolate the False Positives
# (idx_1 == -1 means it's an unmatched prediction, which is a pure FP)
fps = df[df['idx_1'] == -1].copy()

# 3. Define the exact confidence bins we want to investigate
bins = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0]
labels = [
    "0.001 to 0.005", 
    "0.005 to 0.01", 
    "0.01 to 0.02", 
    "0.02 to 0.05", 
    "0.05 to 0.10", 
    "0.10 to 0.50", 
    "0.50 to 1.00"
]

# 4. Sort the FPs into the bins
fps['conf_bin'] = pd.cut(fps['conf2'], bins=bins, labels=labels, include_lowest=True)
distribution = fps['conf_bin'].value_counts().sort_index()

# 5. Print the report
print(f"Total False Positives analyzed: {len(fps)}")
print("-" * 55)
print(f"{'Confidence Range':<18} | {'FP Count':<12} | {'% of Total FPs'}")
print("-" * 55)

cumulative_percent = 0
for index, count in distribution.items():
    percent = (count / len(fps)) * 100 if len(fps) > 0 else 0
    cumulative_percent += percent
    print(f"{index:<18} | {count:<12} | {percent:>5.2f}%  (Cumulative: {cumulative_percent:>5.2f}%)")
print("-" * 55)