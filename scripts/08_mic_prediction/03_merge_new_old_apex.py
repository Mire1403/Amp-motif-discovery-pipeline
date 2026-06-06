from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# original batch
PRED_OLD     = PROJECT_ROOT / "results/12_apex_prediction/01_input/Predicted_MICs_single_motifs.csv"
MAPPING_OLD  = PROJECT_ROOT / "results/12_apex_prediction/01_input/apex_sequence_mapping.csv"

# new batch
PRED_NEW     = PROJECT_ROOT / "results/12_apex_prediction/01b_input/Predicted_MICs_new.csv"
MAPPING_NEW  = PROJECT_ROOT / "results/12_apex_prediction/01b_input/apex_sequence_mapping_new.csv"

# updated TOMTOM families
TOMTOM_FAMILIES = (
    PROJECT_ROOT / "results/07_motif_families/01_tomtom"
    / "motif_families_master.xlsx"
)

# mechanism labels
MECHANISM_FILE = (
    PROJECT_ROOT / "results/11_mechanism_analysis/3_motif_mechanism"
    / "01_peptide_analysis.xlsx"
)

OUT_DIR = PROJECT_ROOT / "results/12_apex_prediction/02_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "E. coli ATCC11775"

# =========================
# LOAD UPDATED TOMTOM FAMILIES
# =========================

print("Loading updated TOMTOM families...")
new_fam = pd.read_excel(
    TOMTOM_FAMILIES, sheet_name="All_motifs_with_family"
)[["Motif","Family_ID"]].rename(columns={"Motif": "Motif_representative"})
new_fam["Motif_representative"] = new_fam["Motif_representative"].str.upper().str.strip()
family_map = new_fam.drop_duplicates("Motif_representative")
print(f"  Families loaded: {family_map['Family_ID'].nunique()} unique families "
      f"for {len(family_map)} motifs")

# =========================
# LOAD & HOMOGENISE OLD BATCH
# =========================

pred_old    = pd.read_csv(PRED_OLD, index_col=0).reset_index()
pred_old.index.name = None
pred_old["APEX_row"] = pred_old.index
mapping_old = pd.read_csv(MAPPING_OLD)

df_old = mapping_old.merge(
    pred_old[["APEX_row", TARGET_COL]], on="APEX_row", how="left"
)
df_old = df_old.rename(columns={TARGET_COL: "MIC_ecoli"})

# update Family_ID with new TOMTOM families
df_old["Motif_representative"] = df_old["Motif_representative"].str.upper().str.strip()
df_old = df_old.drop(columns=["Family_ID"], errors="ignore")
df_old = df_old.merge(family_map, on="Motif_representative", how="left")

print(f"Old batch: {len(df_old)} peptides  "
      f"(Family_ID updated: {df_old['Family_ID'].notna().sum()}/{len(df_old)})")

# =========================
# LOAD & HOMOGENISE NEW BATCH
# =========================

pred_new    = pd.read_csv(PRED_NEW, index_col=0).reset_index()
pred_new.index.name = None
pred_new["APEX_row_new"] = pred_new.index
mapping_new = pd.read_csv(MAPPING_NEW)

# rename to match old schema
mapping_new = mapping_new.rename(columns={
    "Sequence": "Sequence_window",
    "Motif":    "Motif_representative",
})
mapping_new["Motif_representative"] = (
    mapping_new["Motif_representative"].str.upper().str.strip()
)

df_new = mapping_new.merge(
    pred_new[["APEX_row_new", TARGET_COL]], on="APEX_row_new", how="left"
)
df_new = df_new.rename(columns={TARGET_COL: "MIC_ecoli"})

# recover Family_ID from updated TOMTOM families
df_new = df_new.merge(family_map, on="Motif_representative", how="left")

print(f"New batch: {len(df_new)} peptides  "
      f"(Family_ID recovered: {df_new['Family_ID'].notna().sum()}/{len(df_new)})")

# =========================
# CONCATENATE + DEDUPLICATE
# =========================

df = pd.concat([df_old, df_new], ignore_index=True)
print(f"\nCombined before dedup: {len(df)} peptides")

df = df.drop_duplicates(
    subset=["Motif_representative", "Sequence_window"], keep="first"
).reset_index(drop=True)

print(f"After deduplication:   {len(df)} peptides "
      f"({len(df_old) + len(df_new) - len(df)} duplicates removed)")

if "seq_len" in df.columns:
    df["seq_len"] = df["seq_len"].fillna(df["Sequence_window"].str.len())

print(f"MIC range: {df['MIC_ecoli'].min():.1f} – {df['MIC_ecoli'].max():.1f} µmol/L")

# =========================
# MERGE MECHANISM LABELS
# Primary: motif analysis file (original 630)
# Fallback: extended classified files (new 219)
# =========================

try:
    mech = pd.read_excel(MECHANISM_FILE, sheet_name="Peptide_data")[[
        "Peptide_ID", "Mechanism_weighted", "Mechanism_weighted_reduced",
        "Consensus_detail", "Consensus_reduced",
        "Mechanism_FMAP", "Mechanism_PPM"
    ]]
    df = df.merge(mech, on="Peptide_ID", how="left")
    print(f"Mechanism labels from motif analysis: "
          f"{df['Mechanism_weighted'].notna().sum()}/{len(df)}")
except Exception as e:
    print(f"WARNING: could not merge mechanism labels: {e}")

# fallback for new peptides not in motif analysis
missing_mask = df["Mechanism_weighted"].isna()
if missing_mask.sum() > 0:
    fmap_ext = pd.read_csv(
        PROJECT_ROOT / "results/11_mechanism_analysis"
        / "1_motif_activity_analysis/fmap2/01_fmap_classified_extended.csv"
    )[["Peptide_ID", "Mechanism"]].rename(columns={"Mechanism": "Mechanism_FMAP"})

    ppm_ext = pd.read_csv(
        PROJECT_ROOT / "results/11_mechanism_analysis"
        / "1_motif_activity_analysis/ppm2/01_ppm_classified_extended.csv"
    )[["Peptide_ID", "Mechanism"]].rename(columns={"Mechanism": "Mechanism_PPM"})

    fallback = fmap_ext.merge(ppm_ext, on="Peptide_ID", how="outer")

    def weighted_mech(row):
        votes = defaultdict(float)
        if pd.notna(row.get("Mechanism_FMAP")) and row["Mechanism_FMAP"] != "Ambiguous":
            votes[row["Mechanism_FMAP"]] += 1.0
        if pd.notna(row.get("Mechanism_PPM")) and row["Mechanism_PPM"] != "Ambiguous":
            votes[row["Mechanism_PPM"]] += 1.5
        return max(votes, key=votes.get) if votes else "Ambiguous"

    def reduce_mech(m):
        return "Pore" if m in ("Barrel_stave", "Toroidal_pore") else m

    fallback["Mechanism_weighted"]         = fallback.apply(weighted_mech, axis=1)
    fallback["Mechanism_weighted_reduced"] = fallback["Mechanism_weighted"].apply(reduce_mech)
    fallback["Consensus_detail"]  = fallback["Mechanism_FMAP"] == fallback["Mechanism_PPM"]
    fallback["Consensus_reduced"] = (
        fallback["Mechanism_weighted_reduced"] ==
        fallback["Mechanism_PPM"].apply(reduce_mech)
    )

    missing_ids  = df.loc[missing_mask, "Peptide_ID"]
    fallback_sub = fallback[fallback["Peptide_ID"].isin(missing_ids)]

    for col in ["Mechanism_weighted", "Mechanism_weighted_reduced",
                "Mechanism_FMAP", "Mechanism_PPM",
                "Consensus_detail", "Consensus_reduced"]:
        if col in fallback_sub.columns:
            df.loc[missing_mask, col] = df.loc[missing_mask, "Peptide_ID"].map(
                fallback_sub.set_index("Peptide_ID")[col]
            )

    print(f"Mechanism labels after fallback: "
          f"{df['Mechanism_weighted'].notna().sum()}/{len(df)}")

# =========================
# OUTLIER FLAG (IQR per motif)
# =========================

def get_outlier_mask(group_mic):
    q1, q3 = group_mic.quantile(0.25), group_mic.quantile(0.75)
    iqr    = q3 - q1
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    return ~group_mic.between(lo, hi)

df["MIC_outlier"] = df.groupby("Motif_representative")["MIC_ecoli"].transform(
    get_outlier_mask
)

n_out = df["MIC_outlier"].sum()
print(f"Outliers flagged (IQR): {n_out} ({100*n_out/len(df):.1f}%)")

# =========================
# PER-MOTIF STATS
# =========================

def motif_stats(group):
    clean   = group[~group["MIC_outlier"]]
    n_total = len(group)
    n_clean = len(clean)
    mic     = clean["MIC_ecoli"]
    return pd.Series({
        "n_peptides":         n_total,
        "n_clean":            n_clean,
        "n_outliers":         n_total - n_clean,
        "MIC_mean":           round(mic.mean(),   2) if n_clean > 0 else np.nan,
        "MIC_median":         round(mic.median(), 2) if n_clean > 0 else np.nan,
        "MIC_std":            round(mic.std(),    2) if n_clean > 1 else np.nan,
        "MIC_min":            round(mic.min(),    2) if n_clean > 0 else np.nan,
        "MIC_max":            round(mic.max(),    2) if n_clean > 0 else np.nan,
        "Family_ID":          group["Family_ID"].iloc[0],
        "Dominant_mechanism": group["Mechanism_weighted"].value_counts().idxmax()
                              if "Mechanism_weighted" in group.columns and
                                 group["Mechanism_weighted"].notna().any()
                              else "Unknown",
        "Dominant_reduced":   group["Mechanism_weighted_reduced"].value_counts().idxmax()
                              if "Mechanism_weighted_reduced" in group.columns and
                                 group["Mechanism_weighted_reduced"].notna().any()
                              else "Unknown",
        "Consensus_rate":     round(group["Consensus_reduced"].mean() * 100, 1)
                              if "Consensus_reduced" in group.columns else np.nan,
    })

motif_summary = (
    df.groupby("Motif_representative", group_keys=False)
    .apply(motif_stats)
    .reset_index()
    .sort_values("MIC_mean")
)

print(f"\n=== Per-motif MIC summary (top 15, lower MIC = more active) ===")
print(motif_summary[[
    "Motif_representative", "n_peptides", "n_clean",
    "MIC_mean", "MIC_std", "Family_ID", "Dominant_reduced"
]].head(15).to_string(index=False))

# family coverage check
n_with_fam = motif_summary["Family_ID"].notna().sum()
print(f"\nMotifs with Family_ID (updated TOMTOM): {n_with_fam}/{len(motif_summary)}")

# =========================
# SAVE
# =========================

out_peptide = OUT_DIR / "01_peptide_mic_merged.csv"
out_motif   = OUT_DIR / "02_motif_mic_summary_merged.csv"
out_excel   = OUT_DIR / "03_mic_results_merged.xlsx"

df.to_csv(out_peptide, index=False)
motif_summary.to_csv(out_motif, index=False)

with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
    df.to_excel(writer,            sheet_name="Peptide_MIC",    index=False)
    motif_summary.to_excel(writer, sheet_name="Motif_summary",  index=False)

    raw_sheet = df[[
        "Motif_representative", "Sequence_window",
        "Peptide_ID", "MIC_ecoli", "MIC_outlier"
    ]].copy().rename(columns={
        "Motif_representative": "Motif",
        "MIC_ecoli":            "MIC_E_coli_ATCC11775",
        "MIC_outlier":          "Outlier_excluded",
    }).sort_values(["Motif", "MIC_E_coli_ATCC11775"])
    raw_sheet.to_excel(writer, sheet_name="Motif_MIC_raw", index=False)

    if "Mechanism_weighted_reduced" in df.columns:
        for mech in df["Mechanism_weighted_reduced"].dropna().unique():
            sub = df[df["Mechanism_weighted_reduced"] == mech]
            sub.to_excel(writer, sheet_name=f"MIC_{mech[:12]}", index=False)

print(f"\n✅ Saved:")
print(f"  {out_peptide}")
print(f"  {out_motif}")
print(f"  {out_excel}")