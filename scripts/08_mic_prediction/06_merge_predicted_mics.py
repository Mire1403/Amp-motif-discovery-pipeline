"""
Merge APEX predicted MICs with motif combination metadata.
Keeps only E. coli ATCC11775 as target MIC column.
"""

import pandas as pd
from pathlib import Path

# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

comb_file = (
    PROJECT_ROOT
    / "results/13_motif_combinations/01b_combinations_3cat/01_combinations.xlsx"
)
mic_file = (
    PROJECT_ROOT
    / "results/13_motif_combinations/01b_combinations_3cat/Predicted_MICs_combinations_3cat.csv"
)
output_file = (
    PROJECT_ROOT
    / "results/13_motif_combinations/01b_combinations_3cat/04_merged_predicted_MICs.csv"
)

TARGET_COL = "E. coli ATCC11775"

# =========================================================
# LOAD
# =========================================================

print("Loading combination metadata...")
comb_df = pd.read_excel(comb_file)
comb_df.columns = comb_df.columns.str.strip()

print("Loading APEX predictions...")
mic_df = pd.read_csv(mic_file)
mic_df.columns = mic_df.columns.str.strip()

# ensure first column is called Sequence
if "Sequence" not in mic_df.columns:
    mic_df = mic_df.rename(columns={mic_df.columns[0]: "Sequence"})

print(f"Combinations: {len(comb_df)}")
print(f"Predictions:  {len(mic_df)}")

# =========================================================
# MERGE
# =========================================================

merged = comb_df.merge(
    mic_df[["Sequence", TARGET_COL]],
    on="Sequence",
    how="left"
).rename(columns={TARGET_COL: "MIC_combined"})

# =========================================================
# KEEP ONLY RELEVANT COLUMNS
# =========================================================

keep_cols = [
    "Combination_ID", "Motif_1", "Mechanism_1",
    "Motif_2", "Mechanism_2", "Combination_type",
    "Order", "Sequence", "Seq_len",
    "MIC_motif1", "MIC_motif2", "MIC_combined"
]
merged = merged[[c for c in keep_cols if c in merged.columns]]

# =========================================================
# ADD DERIVED COLUMNS
# =========================================================

merged["MIC_mean_individual"] = (merged["MIC_motif1"] + merged["MIC_motif2"]) / 2
merged["MIC_delta"]           = merged["MIC_combined"] - merged["MIC_mean_individual"]

# =========================================================
# STATS
# =========================================================

print(f"\nMerged: {len(merged)} combinations")
print(f"MIC_combined range: {merged['MIC_combined'].min():.1f} – "
      f"{merged['MIC_combined'].max():.1f} µmol/L")
print(f"Missing MIC_combined: {merged['MIC_combined'].isna().sum()}")

print(f"\n=== MIC by combination type ===")
for ctype in ["Carpet+Carpet", "Carpet+Pore", "Pore+Pore"]:
    sub = merged[merged["Combination_type"] == ctype]["MIC_combined"].dropna()
    print(f"  {ctype:<20} n={len(sub):>5}  "
          f"mean={sub.mean():.1f}  median={sub.median():.1f}")

# =========================================================
# SAVE
# =========================================================

merged.to_csv(output_file, index=False)

print(f"\n✅ Saved: {output_file}")
print(f"   Rows: {len(merged)}")