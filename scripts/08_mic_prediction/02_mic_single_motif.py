"""
Clean APEX predictions and compute per-motif MIC statistics.

- Merges APEX predictions with motif mapping (Peptide_ID, Motif_representative, etc.)
- Keeps E. coli ATCC11775 as the target column
- Computes per-motif MIC stats (mean, median, std, n)
- Flags outliers (IQR method) per motif before computing mean
"""

from pathlib import Path
import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_FILE = (
    PROJECT_ROOT / "results/12_apex_prediction/01_input"
    / "Predicted_MICs_single_motifs.csv"
)
MAPPING_FILE = (
    PROJECT_ROOT / "results/12_apex_prediction/01_input"
    / "apex_sequence_mapping.csv"
)
MECHANISM_FILE = (
    PROJECT_ROOT / "results/11_mechanism_analysis/3_motif_mechanism"
    / "01_peptide_analysis.xlsx"
)

OUT_DIR = PROJECT_ROOT / "results/12_apex_prediction/02_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "E. coli ATCC11775"

# =========================
# LOAD
# =========================

pred = pd.read_csv(PREDICTIONS_FILE, index_col=0)
pred.index.name = "Sequence_window"
pred = pred.reset_index()

mapping = pd.read_csv(MAPPING_FILE)

print(f"Predictions: {len(pred)} rows")
print(f"Mapping:     {len(mapping)} rows")
print(f"Target:      {TARGET_COL}")

# =========================
# MERGE predictions + mapping
# APEX preserves row order → merge on position (APEX_row)
# =========================

pred["APEX_row"] = pred.index
df = mapping.merge(pred[["APEX_row", "Sequence_window", TARGET_COL]],
                   on="APEX_row", how="left", suffixes=("_map", "_pred"))

# use Sequence_window from mapping (source of truth)
if "Sequence_window_map" in df.columns:
    df["Sequence_window"] = df["Sequence_window_map"]
    df = df.drop(columns=["Sequence_window_map", "Sequence_window_pred"],
                 errors="ignore")

df = df.rename(columns={TARGET_COL: "MIC_ecoli"})

print(f"\nMerged: {len(df)} peptides")
print(f"MIC range: {df['MIC_ecoli'].min():.1f} – {df['MIC_ecoli'].max():.1f} µmol/L")

# =========================
# MERGE mechanism labels
# =========================

try:
    mech = pd.read_excel(MECHANISM_FILE, sheet_name="Peptide_data")[
        ["Peptide_ID", "Mechanism_weighted", "Mechanism_weighted_reduced",
         "Consensus_detail", "Consensus_reduced",
         "Mechanism_FMAP", "Mechanism_PPM"]
    ]
    df = df.merge(mech, on="Peptide_ID", how="left")
    print(f"Mechanism labels merged for {df['Mechanism_weighted'].notna().sum()} peptides")
except Exception as e:
    print(f"WARNING: could not merge mechanism labels: {e}")

# =========================
# OUTLIER FLAG (IQR per motif)
# Flag peptides whose MIC is outside Q1-1.5*IQR / Q3+1.5*IQR
# within their motif group — excluded from mean but kept in data
# =========================

def get_outlier_mask(group_mic):
    q1, q3 = group_mic.quantile(0.25), group_mic.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    return ~group_mic.between(lo, hi)

df["MIC_outlier"] = df.groupby("Motif_representative")["MIC_ecoli"].transform(
    get_outlier_mask
)

n_outliers = df["MIC_outlier"].sum()
print(f"\nOutliers flagged (IQR per motif): {n_outliers} "
      f"({100*n_outliers/len(df):.1f}%) — kept in data but excluded from mean")

# =========================
# PER-MOTIF STATS
# =========================

def motif_stats(group):
    clean = group[~group["MIC_outlier"]]
    n_total = len(group)
    n_clean = len(clean)
    mic     = clean["MIC_ecoli"]
    return pd.Series({
        "n_peptides":       n_total,
        "n_clean":          n_clean,
        "n_outliers":       n_total - n_clean,
        "MIC_mean":         round(mic.mean(), 2) if n_clean > 0 else np.nan,
        "MIC_median":       round(mic.median(), 2) if n_clean > 0 else np.nan,
        "MIC_std":          round(mic.std(), 2) if n_clean > 1 else np.nan,
        "MIC_min":          round(mic.min(), 2) if n_clean > 0 else np.nan,
        "MIC_max":          round(mic.max(), 2) if n_clean > 0 else np.nan,
        "Family_ID":        group["Family_ID"].iloc[0],
        "Dominant_mechanism": group["Mechanism_weighted"].value_counts().idxmax()
                              if "Mechanism_weighted" in group.columns and
                                 group["Mechanism_weighted"].notna().any()
                              else "Unknown",
        "Dominant_reduced": group["Mechanism_weighted_reduced"].value_counts().idxmax()
                             if "Mechanism_weighted_reduced" in group.columns and
                                group["Mechanism_weighted_reduced"].notna().any()
                             else "Unknown",
        "Consensus_rate":   round(group["Consensus_reduced"].mean() * 100, 1)
                            if "Consensus_reduced" in group.columns else np.nan,
    })

motif_summary = (
    df.groupby("Motif_representative")
    .apply(motif_stats)
    .reset_index()
    .sort_values("MIC_mean")
)

print(f"\n=== Per-motif MIC summary (sorted by mean MIC, lower = more active) ===")
print(motif_summary[["Motif_representative", "n_peptides", "n_clean",
                      "MIC_mean", "MIC_std", "Dominant_reduced"]
                     ].head(15).to_string(index=False))

# =========================
# SAVE
# =========================

out_peptide = OUT_DIR / "01_peptide_mic.csv"
out_motif   = OUT_DIR / "02_motif_mic_summary.csv"
out_excel   = OUT_DIR / "03_mic_results.xlsx"

df.to_csv(out_peptide, index=False)
motif_summary.to_csv(out_motif, index=False)

with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Peptide_MIC", index=False)
    motif_summary.to_excel(writer, sheet_name="Motif_MIC_summary", index=False)

    # sheet: one row per peptide, columns = Motif | MIC | Outlier_excluded
    raw_sheet = df[[
        "Motif_representative", "Sequence_window",
        "Peptide_ID", "MIC_ecoli", "MIC_outlier"
    ]].copy()
    raw_sheet = raw_sheet.rename(columns={
        "Motif_representative": "Motif",
        "MIC_ecoli":            "MIC_E_coli_ATCC11775",
        "MIC_outlier":          "Outlier_excluded",
    })
    raw_sheet = raw_sheet.sort_values(["Motif", "MIC_E_coli_ATCC11775"])
    raw_sheet.to_excel(writer, sheet_name="Motif_MIC_raw", index=False)

    # sheets per mechanism
    if "Mechanism_weighted_reduced" in df.columns:
        for mech in df["Mechanism_weighted_reduced"].dropna().unique():
            sub = df[df["Mechanism_weighted_reduced"] == mech]
            sub.to_excel(writer, sheet_name=f"MIC_{mech[:12]}", index=False)

print(f"\n✅ Saved:")
print(f"  {out_peptide}")
print(f"  {out_motif}")
print(f"  {out_excel}")