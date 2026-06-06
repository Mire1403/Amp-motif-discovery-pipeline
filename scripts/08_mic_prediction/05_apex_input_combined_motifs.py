"""
Motif consensus mechanism selection for combination design.

For each motif (Motif_representative):
  - Count how many sequences each tool assigns to each mechanism
  - Weight: PPM=1.5, FMAP=1.0
  - Motif mechanism = mechanism with highest weighted vote count
  - Keep only motifs with clear consensus (no Ambiguous winner,
    winner score >= MIN_CONSENSUS_PCT of total weighted votes)

Output:
  - 01_consensus_motifs_3cat.xlsx  (Carpet / Toroidal_pore / Barrel_stave)
  - 02_consensus_motifs_reduced.xlsx (Carpet / Pore)
  Both contain:
    Sheet "Motif_summary"  — one row per motif with scores and dominant mechanism
    Sheet "Peptide_detail" — one row per sequence with mechanism + MIC (outliers removed)
"""

from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PEPTIDE_FILE = (
    PROJECT_ROOT / "results/12_apex_prediction/02_results"
    / "01_peptide_mic_merged.csv"
)

OUT_DIR = PROJECT_ROOT / "results/13_motif_combinations/00_consensus_motifs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WEIGHT_PPM   = 1.5
WEIGHT_FMAP  = 1.0
MIN_CONSENSUS_PCT = 60.0 

MECHS_3CAT   = ["Carpet", "Toroidal_pore", "Barrel_stave"]
MECHS_REDUCED = ["Carpet", "Pore"]

# =========================
# LOAD
# =========================

df = pd.read_csv(PEPTIDE_FILE, sep=";")
df["Mechanism_FMAP"] = df["Mechanism_FMAP"].fillna("Ambiguous")
df["Mechanism_PPM"]  = df["Mechanism_PPM"].fillna("Ambiguous")
df["MIC_ecoli"]      = pd.to_numeric(df["MIC_ecoli"], errors="coerce")

def reduce_mech(m):
    return "Pore" if m in ("Toroidal_pore", "Barrel_stave") else m

df["Mech_FMAP_red"] = df["Mechanism_FMAP"].apply(reduce_mech)
df["Mech_PPM_red"]  = df["Mechanism_PPM"].apply(reduce_mech)

print(f"Loaded {len(df)} peptides, {df['Motif_representative'].nunique()} motifs")

# =========================
# OUTLIER REMOVAL (IQR per motif) — on raw MIC
# =========================

def get_outlier_mask(group_mic):
    q1, q3 = group_mic.quantile(0.25), group_mic.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    return ~group_mic.between(lo, hi)

df["MIC_outlier"] = df.groupby("Motif_representative")["MIC_ecoli"].transform(
    get_outlier_mask
)
print(f"Outliers flagged (IQR): {df['MIC_outlier'].sum()} "
      f"({100*df['MIC_outlier'].mean():.1f}%)")

# =========================
# PER-MOTIF WEIGHTED VOTE AGGREGATION
# =========================

def aggregate_motif(group, fmap_col, ppm_col, valid_mechs):
    """
    For a group of peptides belonging to one motif:
    - Count how many sequences each tool assigns to each mechanism
    - Weight FMAP × 1.0, PPM × 1.5
    - Return winner, score, confidence%, and per-mechanism breakdown
    """
    n = len(group)
    scores = defaultdict(float)

    for mech in valid_mechs:
        n_fmap = (group[fmap_col] == mech).sum()
        n_ppm  = (group[ppm_col]  == mech).sum()
        scores[mech] = n_fmap * WEIGHT_FMAP + n_ppm * WEIGHT_PPM

    total_score = sum(scores.values())
    if total_score == 0:
        return None

    winner = max(scores, key=scores.get)
    winner_score = scores[winner]
    confidence   = 100 * winner_score / total_score

    # MIC stats on non-outlier sequences
    clean = group[~group["MIC_outlier"]]["MIC_ecoli"].dropna()

    row = {
        "n_peptides":    n,
        "n_clean_mic":   len(clean),
        "Mechanism":     winner,
        "Winner_score":  round(winner_score, 2),
        "Total_score":   round(total_score, 2),
        "Confidence_pct": round(confidence, 1),
        "MIC_mean":      round(clean.mean(),   2) if len(clean) > 0 else np.nan,
        "MIC_median":    round(clean.median(), 2) if len(clean) > 0 else np.nan,
        "MIC_std":       round(clean.std(),    2) if len(clean) > 1 else np.nan,
        "Family_ID":     group["Family_ID"].iloc[0]
                         if "Family_ID" in group.columns else "",
    }
    # per-mechanism breakdown
    for mech in valid_mechs:
        n_fmap = (group[fmap_col] == mech).sum()
        n_ppm  = (group[ppm_col]  == mech).sum()
        row[f"n_FMAP_{mech}"]  = int(n_fmap)
        row[f"n_PPM_{mech}"]   = int(n_ppm)
        row[f"pct_FMAP_{mech}"] = round(100*n_fmap/n, 1)
        row[f"pct_PPM_{mech}"]  = round(100*n_ppm/n,  1)

    return pd.Series(row)

# 3-category aggregation
summary_3cat = (
    df.groupby("Motif_representative")
    .apply(aggregate_motif,
           fmap_col="Mechanism_FMAP", ppm_col="Mechanism_PPM",
           valid_mechs=MECHS_3CAT)
    .reset_index()
    .dropna(subset=["Mechanism"])
)

# reduced aggregation
summary_red = (
    df.groupby("Motif_representative")
    .apply(aggregate_motif,
           fmap_col="Mech_FMAP_red", ppm_col="Mech_PPM_red",
           valid_mechs=MECHS_REDUCED)
    .reset_index()
    .dropna(subset=["Mechanism"])
)

# =========================
# FILTER: clear consensus only
# - Winner is not Ambiguous
# - Confidence >= MIN_CONSENSUS_PCT
# =========================

def filter_consensus(summary):
    mask = (
        (summary["Mechanism"] != "Ambiguous") &
        (summary["Confidence_pct"] >= MIN_CONSENSUS_PCT)
    )
    kept    = summary[mask].copy()
    removed = summary[~mask].copy()
    return kept, removed

kept_3cat,  removed_3cat  = filter_consensus(summary_3cat)
kept_red,   removed_red   = filter_consensus(summary_red)

print(f"\n=== 3-category consensus ===")
print(f"  Motifs with clear consensus: {len(kept_3cat)}/{len(summary_3cat)}")
print(kept_3cat["Mechanism"].value_counts().to_string())

print(f"\n=== Reduced (Carpet/Pore) consensus ===")
print(f"  Motifs with clear consensus: {len(kept_red)}/{len(summary_red)}")
print(kept_red["Mechanism"].value_counts().to_string())

# =========================
# PEPTIDE DETAIL — only for consensus motifs, outliers removed
# =========================

def peptide_detail(kept_summary, fmap_col, ppm_col):
    consensus_motifs = kept_summary["Motif_representative"].tolist()
    detail = df[
        df["Motif_representative"].isin(consensus_motifs) &
        ~df["MIC_outlier"]
    ].copy()
    # add motif-level mechanism
    mech_map = kept_summary.set_index("Motif_representative")["Mechanism"]
    detail["Motif_Mechanism"] = detail["Motif_representative"].map(mech_map)
    detail["Confidence_pct"]  = detail["Motif_representative"].map(
        kept_summary.set_index("Motif_representative")["Confidence_pct"]
    )
    cols = [
        "Peptide_ID", "Family_ID", "Motif_representative",
        "Motif_variant", "Sequence_window",
        fmap_col, ppm_col, "Motif_Mechanism", "Confidence_pct",
        "MIC_ecoli"
    ]
    return detail[[c for c in cols if c in detail.columns]]

detail_3cat = peptide_detail(kept_3cat, "Mechanism_FMAP", "Mechanism_PPM")
detail_red  = peptide_detail(kept_red,  "Mech_FMAP_red",  "Mech_PPM_red")

print(f"\nPeptides in consensus motifs (3cat):    {len(detail_3cat)}")
print(f"Peptides in consensus motifs (reduced): {len(detail_red)}")

# =========================
# SAVE EXCEL
# =========================

def save_excel(path, summary, detail, removed):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary.sort_values("MIC_mean").to_excel(
            writer, sheet_name="Motif_summary", index=False)
        detail.sort_values(["Motif_representative", "MIC_ecoli"]).to_excel(
            writer, sheet_name="Peptide_detail", index=False)
        removed.to_excel(
            writer, sheet_name="Excluded_motifs", index=False)

out_3cat = OUT_DIR / "01_consensus_motifs_3cat.xlsx"
out_red  = OUT_DIR / "02_consensus_motifs_reduced.xlsx"

save_excel(out_3cat, kept_3cat,  detail_3cat, removed_3cat)
save_excel(out_red,  kept_red,   detail_red,  removed_red)

print(f"\n✅ Saved:")
print(f"  {out_3cat}")
print(f"  {out_red}")
print(f"\nThreshold used: confidence >= {MIN_CONSENSUS_PCT}% of weighted votes")
print(f"Weights: PPM={WEIGHT_PPM}, FMAP={WEIGHT_FMAP}")