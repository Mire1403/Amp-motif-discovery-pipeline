"""
FMAP vs PPM comparison script.

Loads the classified outputs from both methods and compares:
  - Mechanism agreement (Carpet / Barrel_stave / Toroidal_pore / Ambiguous)
  - Physical parameter differences (tilt, depth, energy)
  - Per-motif mechanism breakdown from each method
  - Reduced agreement (Carpet vs Pore vs Ambiguous)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# =========================
# PATHS
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FMAP_FILE = (
    PROJECT_ROOT / "results" / "11_mechanism_analysis"
    / "1_motif_activity_analysis" / "fmap2" / "01_fmap_classified.csv"
)
PPM_FILE = (
    PROJECT_ROOT / "results" / "11_mechanism_analysis"
    / "1_motif_activity_analysis" / "ppm2" / "01_ppm_classified.csv"
)
OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "11_mechanism_analysis"
    / "1_motif_activity_analysis" / "comparison2"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_XLSX = OUTPUT_DIR / "01_fmap_ppm_comparison.xlsx"

# =========================
# LOAD
# =========================

fmap = pd.read_csv(FMAP_FILE)
ppm  = pd.read_csv(PPM_FILE)

print(f"FMAP: {len(fmap)} peptides")
print(f"PPM:  {len(ppm)} peptides")
print(f"\nFMAP raw columns: {list(fmap.columns)}")
print(f"PPM  raw columns: {list(ppm.columns)}")

# =========================
# NORMALISE COLUMN NAMES
# Both files should already have Mechanism from their respective
# classify scripts. We rename to _FMAP / _PPM suffixes for merging.
# =========================

fmap = fmap.rename(columns={
    "Mechanism":              "Mechanism_FMAP",
    "Tilt_Angle":             "Tilt_FMAP",
    "Depth_Thickness":        "Depth_FMAP",
    "Membrane_Binding_Energy":"Energy_FMAP",
    "Insertion_Strength":     "Insertion_FMAP",
})

ppm = ppm.rename(columns={
    "Mechanism":    "Mechanism_PPM",
    "Tilt_deg":     "Tilt_PPM",
    "Thickness_A":  "Depth_PPM",
    "Best_Energy":  "Energy_PPM",
})

# keep only what we need from each
FMAP_COLS = ["Peptide_ID", "Mechanism_FMAP", "Tilt_FMAP",
             "Depth_FMAP", "Energy_FMAP", "Insertion_FMAP"]
PPM_COLS  = ["Peptide_ID", "Mechanism_PPM", "Tilt_PPM",
             "Depth_PPM",  "Energy_PPM",  "Membrane_Type"]

# add Motif if present
for col in ["Motif", "Motif_ID"]:
    if col in fmap.columns and "Motif" not in FMAP_COLS:
        fmap = fmap.rename(columns={col: "Motif"}) if col != "Motif" else fmap
        FMAP_COLS.append("Motif")
        break

fmap = fmap[[c for c in FMAP_COLS if c in fmap.columns]]
ppm  = ppm[[c  for c in PPM_COLS  if c in ppm.columns]]

print(f"\nFMAP columns after selection: {list(fmap.columns)}")
print(f"PPM  columns after selection: {list(ppm.columns)}")

for name, df_check in [("FMAP", fmap), ("PPM", ppm)]:
    if "Peptide_ID" not in df_check.columns:
        raise ValueError(
            f"Peptide_ID missing from {name} after column selection.\n"
            f"Available columns: {list(df_check.columns)}"
        )

# =========================
# MERGE (inner: only peptides present in both)
# =========================

merged = pd.merge(fmap, ppm, on="Peptide_ID", how="inner")
print(f"Merged: {len(merged)} peptides in common")

# =========================
# MECHANISM AGREEMENT
# =========================

MECHANISMS = ["Carpet", "Barrel_stave", "Toroidal_pore", "Ambiguous"]

MECHANISM_COLORS = {
    "Carpet":        "#457b9d",
    "Barrel_stave":  "#f4a261",
    "Toroidal_pore": "#2a9d8f",
    "Ambiguous":     "#adb5bd",
}

# pairs where disagreement is biologically understandable
BIO_AMBIGUOUS_PAIRS = {
    frozenset({"Toroidal_pore", "Barrel_stave"}),   # mechanistic continuum
    frozenset({"Toroidal_pore", "Carpet"}),          # oblique insertion ambiguity
    frozenset({"Ambiguous",     "Carpet"}),
    frozenset({"Ambiguous",     "Toroidal_pore"}),
    frozenset({"Ambiguous",     "Barrel_stave"}),
}


def mech_agreement(row):
    m1, m2 = row["Mechanism_FMAP"], row["Mechanism_PPM"]
    if pd.isna(m1) or pd.isna(m2):
        return "Unknown"
    if m1 == m2:
        return "Match"
    if frozenset({m1, m2}) in BIO_AMBIGUOUS_PAIRS:
        return "Biological_ambiguous"
    return "Conflict"


merged["Mechanism_Agreement"] = merged.apply(mech_agreement, axis=1)

# reduced: Carpet vs Pore (barrel+toroidal) vs Ambiguous
def reduce_mech(m):
    if m == "Carpet":
        return "Carpet"
    if m in ("Barrel_stave", "Toroidal_pore"):
        return "Pore"
    return "Ambiguous"

merged["FMAP_reduced"] = merged["Mechanism_FMAP"].apply(reduce_mech)
merged["PPM_reduced"]  = merged["Mechanism_PPM"].apply(reduce_mech)
merged["Reduced_Agreement"] = (merged["FMAP_reduced"] == merged["PPM_reduced"]).map(
    {True: "Match", False: "Different"}
)

# =========================
# PHYSICAL PARAMETER DIFFERENCES
# =========================

merged["Tilt_diff"]   = (merged["Tilt_FMAP"]   - merged["Tilt_PPM"]).abs()
merged["Depth_diff"]  = (merged["Depth_FMAP"]  - merged["Depth_PPM"]).abs()
merged["Energy_diff"] = (merged["Energy_FMAP"] - merged["Energy_PPM"]).abs()

# =========================
# METRICS SUMMARY
# =========================

total = len(merged)

metrics = pd.DataFrame({
    "Metric": [
        "Mechanism Match",
        "Mechanism Bio-ambiguous",
        "Mechanism Conflict",
        "Reduced Match (Carpet/Pore/Ambiguous)",
        "Reduced Different",
        "N peptides compared",
    ],
    "N": [
        (merged["Mechanism_Agreement"] == "Match").sum(),
        (merged["Mechanism_Agreement"] == "Biological_ambiguous").sum(),
        (merged["Mechanism_Agreement"] == "Conflict").sum(),
        (merged["Reduced_Agreement"]   == "Match").sum(),
        (merged["Reduced_Agreement"]   == "Different").sum(),
        total,
    ],
})
metrics["%"] = (metrics["N"] / total * 100).round(1)

# =========================
# PHYSICAL DIFF BY AGREEMENT
# =========================

param_by_agreement = merged.groupby("Mechanism_Agreement")[
    ["Tilt_diff", "Depth_diff", "Energy_diff"]
].agg(["mean", "median"]).round(2)

# =========================
# MOTIF BREAKDOWN (if Motif column present)
# =========================

motif_sheets = {}
if "Motif" in merged.columns:
    for method, col in [("FMAP", "Mechanism_FMAP"), ("PPM", "Mechanism_PPM")]:
        ct = pd.crosstab(merged["Motif"], merged[col])
        ct["Total"] = ct.sum(axis=1)
        ct_norm = pd.crosstab(merged["Motif"], merged[col], normalize="index").round(3)
        ct_norm["Total"] = ct["Total"]
        motif_sheets[f"Motif_{method}_counts"] = ct.sort_values("Total", ascending=False)
        motif_sheets[f"Motif_{method}_pct"]    = ct_norm.loc[motif_sheets[f"Motif_{method}_counts"].index]

    # agreement rate per motif
    motif_agr = merged.groupby("Motif")["Mechanism_Agreement"].value_counts(
        normalize=True
    ).unstack(fill_value=0).round(3)
    motif_agr["N"] = merged.groupby("Motif").size()
    motif_sheets["Motif_Agreement"] = motif_agr.sort_values("N", ascending=False)

# =========================
# PRINT
# =========================

print("\n=== Mechanism agreement ===")
print(metrics.to_string(index=False))

print("\n=== Physical parameter differences by agreement ===")
print(param_by_agreement.to_string())

# =========================
# SAVE XLSX
# =========================

with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
    merged.to_excel(writer, sheet_name="Merged", index=False)
    metrics.to_excel(writer, sheet_name="Metrics", index=False)
    param_by_agreement.to_excel(writer, sheet_name="Params_by_Agreement")
    for sheet_name, df_sheet in motif_sheets.items():
        df_sheet.to_excel(writer, sheet_name=sheet_name[:31])  # Excel 31-char limit

print(f"\n✅ Saved: {OUTPUT_XLSX}")

# =========================
# PLOTS
# =========================

# --- Plot 1: confusion matrix (FMAP vs PPM mechanism) ---
fig, ax = plt.subplots(figsize=(7, 6))
cm = pd.crosstab(merged["Mechanism_FMAP"], merged["Mechanism_PPM"])
cm = cm.reindex(index=MECHANISMS, columns=MECHANISMS, fill_value=0)
im = ax.imshow(cm.values, cmap="Blues")
ax.set_xticks(range(len(MECHANISMS)))
ax.set_yticks(range(len(MECHANISMS)))
ax.set_xticklabels(MECHANISMS, rotation=30, ha="right", fontsize=9)
ax.set_yticklabels(MECHANISMS, fontsize=9)
ax.set_xlabel("PPM mechanism", fontsize=10)
ax.set_ylabel("FMAP mechanism", fontsize=10)
ax.set_title("Mechanism assignment: FMAP vs PPM", fontsize=11)
for i in range(len(MECHANISMS)):
    for j in range(len(MECHANISMS)):
        val = cm.values[i, j]
        ax.text(j, i, str(val), ha="center", va="center",
                fontsize=9, color="white" if val > cm.values.max() * 0.5 else "black")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "01_mechanism_confusion_matrix.png", dpi=300)
plt.close(fig)

# --- Plot 2: tilt scatter FMAP vs PPM, coloured by agreement ---
AGR_COLORS = {"Match": "#2a9d8f", "Biological_ambiguous": "#f4a261", "Conflict": "#e63946", "Unknown": "#adb5bd"}
colors_agr = merged["Mechanism_Agreement"].map(AGR_COLORS).fillna("#adb5bd")

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(merged["Tilt_PPM"], merged["Tilt_FMAP"],
           c=colors_agr, alpha=0.6, s=30, edgecolors="none")
lim = max(merged["Tilt_PPM"].max(), merged["Tilt_FMAP"].max()) + 5
ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.4)
ax.set_xlabel("Tilt PPM (°)", fontsize=10)
ax.set_ylabel("Tilt FMAP (°)", fontsize=10)
ax.set_title("Tilt angle: FMAP vs PPM", fontsize=11)
patches = [mpatches.Patch(color=c, label=l) for l, c in AGR_COLORS.items()
           if l in merged["Mechanism_Agreement"].values]
ax.legend(handles=patches, title="Agreement", fontsize=8, title_fontsize=8)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "02_tilt_fmap_vs_ppm.png", dpi=300)
plt.close(fig)

# --- Plot 3: depth scatter FMAP vs PPM ---
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(merged["Depth_PPM"], merged["Depth_FMAP"],
           c=colors_agr, alpha=0.6, s=30, edgecolors="none")
lim = max(merged["Depth_PPM"].max(), merged["Depth_FMAP"].max()) + 2
ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.4)
ax.set_xlabel("Depth PPM (Å)", fontsize=10)
ax.set_ylabel("Depth FMAP (Å)", fontsize=10)
ax.set_title("Insertion depth: FMAP vs PPM", fontsize=11)
ax.legend(handles=patches, title="Agreement", fontsize=8, title_fontsize=8)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "03_depth_fmap_vs_ppm.png", dpi=300)
plt.close(fig)

# --- Plot 4: agreement distribution bar chart ---
fig, ax = plt.subplots(figsize=(6, 4))
agr_counts = merged["Mechanism_Agreement"].value_counts()
bars = ax.bar(agr_counts.index, agr_counts.values,
              color=[AGR_COLORS.get(x, "#adb5bd") for x in agr_counts.index],
              edgecolor="none")
for bar, val in zip(bars, agr_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1, str(val),
            ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Number of peptides", fontsize=10)
ax.set_title("Mechanism agreement between FMAP and PPM", fontsize=11)
ax.set_xticklabels(agr_counts.index, rotation=15, ha="right", fontsize=9)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "04_agreement_distribution.png", dpi=300)
plt.close(fig)

print("✅ Plots saved to:", OUTPUT_DIR)