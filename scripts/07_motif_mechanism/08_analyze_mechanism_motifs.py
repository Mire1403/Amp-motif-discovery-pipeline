from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# =========================
# CONFIG
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MOTIF_FILE = (
    PROJECT_ROOT / "results/09_motif_mechanism_input/01_fmap_input"
    / "motif_selected_sequences.csv"
)
FMAP_FILE = (
    PROJECT_ROOT / "results/11_mechanism_analysis"
    / "1_motif_activity_analysis/fmap2/01_fmap_classified_extended.csv"
)
PPM_FILE = (
    PROJECT_ROOT / "results/11_mechanism_analysis"
    / "1_motif_activity_analysis/ppm2/01_ppm_classified_extended.csv"
)

OUT_DIR = PROJECT_ROOT / "results/11_mechanism_analysis" / "3_motif_mechanism"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

WEIGHT_PPM  = 1.5
WEIGHT_FMAP = 1.0
MIN_OVERLAP = 2   # minimum aa overlap between helix and motif variant
DPI = 300

MECHS_DETAIL  = ["Carpet", "Toroidal_pore", "Barrel_stave", "Ambiguous"]
MECHS_REDUCED = ["Carpet", "Pore", "Ambiguous"]

PALETTE_DETAIL = {
    "Carpet":        "#E07B54",
    "Toroidal_pore": "#4C72B0",
    "Barrel_stave":  "#3BAF77",
    "Ambiguous":     "#CCCCCC",
}
PALETTE_REDUCED = {
    "Carpet":    "#E07B54",
    "Pore":      "#4C72B0",
    "Ambiguous": "#CCCCCC",
}

sns.set_theme(style="whitegrid", font_scale=1.1)

# =========================
# LOAD
# =========================

motif_df = pd.read_csv(MOTIF_FILE)
motif_df = motif_df.rename(columns={"Motif": "Motif_representative", "Matched_sequence":"Motif_variant"})
fmap_df  = pd.read_csv(FMAP_FILE).rename(columns={
    "Tilt_Angle":              "Tilt_FMAP",
    "Depth_Thickness":         "Depth_FMAP",
    "Membrane_Binding_Energy": "Energy_FMAP",
    "Mechanism":               "Mechanism_FMAP",
    "Helix_Start":             "Helix_Start",
    "Helix_End":               "Helix_End",
})
ppm_df   = pd.read_csv(PPM_FILE).rename(columns={
    "Tilt_deg":    "Tilt_PPM",
    "Thickness_A": "Depth_PPM",
    "Best_Energy": "Energy_PPM",
    "Mechanism":   "Mechanism_PPM",
})

fmap_keep = ["Peptide_ID", "Mechanism_FMAP", "Tilt_FMAP", "Depth_FMAP",
             "Energy_FMAP", "Helix_Start", "Helix_End"]
ppm_keep  = ["Peptide_ID", "Mechanism_PPM", "Tilt_PPM", "Depth_PPM", "Energy_PPM"]

fmap_df = fmap_df[[c for c in fmap_keep if c in fmap_df.columns]]
ppm_df  = ppm_df[[c  for c in ppm_keep  if c in ppm_df.columns]]

print(f"Motif sequences: {len(motif_df)}")
print(f"FMAP classified: {len(fmap_df)}")
print(f"PPM  classified: {len(ppm_df)}")

# =========================
# MERGE
# =========================

df = motif_df.merge(fmap_df, on="Peptide_ID", how="left")
df = df.merge(ppm_df,        on="Peptide_ID", how="left")

print(f"After merge: {len(df)} peptides")

# =========================
# REDUCE MECHANISM
# =========================

def reduce_mech(m):
    if pd.isna(m): return "Ambiguous"
    if m in ("Barrel_stave", "Toroidal_pore"): return "Pore"
    return m

df["Mech_FMAP_reduced"] = df["Mechanism_FMAP"].apply(reduce_mech)
df["Mech_PPM_reduced"]  = df["Mechanism_PPM"].apply(reduce_mech)

# flag: peptide has real data from BOTH tools (not NaN)
df["Both_available"] = (
    df["Mechanism_FMAP"].notna() & (df["Mechanism_FMAP"] != "Ambiguous") &
    df["Mechanism_PPM"].notna()  & (df["Mechanism_PPM"]  != "Ambiguous")
)

# fill NaN mechanisms as Ambiguous
df["Mechanism_FMAP"] = df["Mechanism_FMAP"].fillna("Ambiguous")
df["Mechanism_PPM"]  = df["Mechanism_PPM"].fillna("Ambiguous")

# =========================
# WEIGHTED MECHANISM
# PPM weight=1.5, FMAP weight=1.0
# Votes cast by each tool for its assigned mechanism
# Ambiguous counts as no vote
# =========================

def weighted_mechanism(row, detail=True):
    fmap_col = "Mechanism_FMAP" if detail else "Mech_FMAP_reduced"
    ppm_col  = "Mechanism_PPM"  if detail else "Mech_PPM_reduced"
    votes = defaultdict(float)
    mf = row[fmap_col]
    mp = row[ppm_col]
    if mf != "Ambiguous": votes[mf] += WEIGHT_FMAP
    if mp != "Ambiguous": votes[mp] += WEIGHT_PPM
    if not votes:
        return "Ambiguous"
    return max(votes, key=votes.get)

df["Mechanism_weighted"]         = df.apply(weighted_mechanism,
                                             detail=True, axis=1)
df["Mechanism_weighted_reduced"] = df.apply(weighted_mechanism,
                                             detail=False, axis=1)

# =========================
# CONSENSUS
# =========================

df["Consensus_detail"] = (
    df["Both_available"] &
    (df["Mechanism_FMAP"] == df["Mechanism_PPM"])
)
df["Consensus_reduced"] = (
    df["Both_available"] &
    (df["Mech_FMAP_reduced"] == df["Mech_PPM_reduced"])
)

# =========================
# HELIX-MOTIF OVERLAP
# FMAP helix (Helix_Start/Helix_End) vs position of Motif_variant
# within Sequence_window. Both are 1-based positions in the window.
# =========================

def helix_motif_overlap(row):
    seq     = str(row.get("Sequence_window", ""))
    variant = str(row.get("Motif_variant",   ""))
    h_start = row.get("Helix_Start")
    h_end   = row.get("Helix_End")

    if not seq or not variant or pd.isna(h_start) or pd.isna(h_end):
        return False, 0

    pos = seq.find(variant)
    if pos == -1:
        return False, 0

    m_start = pos + 1          # 1-based
    m_end   = pos + len(variant)

    overlap = max(0, min(int(h_end), m_end) - max(int(h_start), m_start) + 1)
    return overlap >= MIN_OVERLAP, overlap

df[["Helix_Motif_Overlap", "Overlap_AA"]] = pd.DataFrame(
    df.apply(helix_motif_overlap, axis=1).tolist(),
    index=df.index
)

# =========================
# SUMMARY STATS PRINT
# =========================

print(f"\nMechanism_weighted distribution (detailed):")
print(df["Mechanism_weighted"].value_counts().to_string())
print(f"\nMechanism_weighted distribution (reduced):")
print(df["Mechanism_weighted_reduced"].value_counts().to_string())
n_both = df["Both_available"].sum()
print(f"\nPeptides with data from both tools: {n_both}/{len(df)} "
      f"({100*n_both/len(df):.1f}%)")
print(f"Consensus rate (detail,  shared only): "
      f"{df['Consensus_detail'].sum()/n_both*100:.1f}%")
print(f"Consensus rate (reduced, shared only): "
      f"{df['Consensus_reduced'].sum()/n_both*100:.1f}%")
print(f"Helix-motif overlap ≥{MIN_OVERLAP}aa: "
      f"{df['Helix_Motif_Overlap'].mean()*100:.1f}%")

# =========================
# DISAGREEMENT BREAKDOWN
# =========================

disagree = df[
    df["Both_available"] &
    ~df["Consensus_detail"]
].copy()

disagree["Disagreement_pair"] = disagree.apply(
    lambda r: " vs ".join(sorted([r["Mechanism_FMAP"], r["Mechanism_PPM"]])),
    axis=1
)

n_both      = df["Both_available"].sum()
n_consensus = df["Consensus_detail"].sum()
n_disagree  = len(disagree)

print(f"\n=== Disagreement breakdown — detailed (Carpet/Toroidal/Barrel) ===")
print(f"  Shared peptides (both tools):  {n_both}")
print(f"  Consensus:                     {n_consensus} ({100*n_consensus/n_both:.1f}%)")
print(f"  Disagreement:                  {n_disagree}  ({100*n_disagree/n_both:.1f}%)")
print(f"\n  Disagreement pairs:")
pair_counts = disagree["Disagreement_pair"].value_counts()
for pair, count in pair_counts.items():
    pct = 100 * count / n_both
    bio = "  ← biologically ambiguous" if "Toroidal" in pair and "Carpet" in pair \
          else ("  ← borderline" if "Toroidal" in pair and "Barrel" in pair \
          else "  ← problematic")
    print(f"    {pair:<40} {count:>4}  ({pct:.1f}%){bio}")

# reduced breakdown
disagree_red = df[
    df["Both_available"] &
    ~df["Consensus_reduced"]
].copy()
disagree_red["Disagreement_pair_red"] = disagree_red.apply(
    lambda r: " vs ".join(sorted([r["Mech_FMAP_reduced"], r["Mech_PPM_reduced"]])),
    axis=1
)
n_cons_red    = df["Consensus_reduced"].sum()
n_disagree_red = len(disagree_red)

print(f"\n=== Disagreement breakdown — reduced (Carpet/Pore) ===")
print(f"  Consensus:    {n_cons_red} ({100*n_cons_red/n_both:.1f}%)")
print(f"  Disagreement: {n_disagree_red}  ({100*n_disagree_red/n_both:.1f}%)")
print(f"\n  Disagreement pairs:")
for pair, count in disagree_red["Disagreement_pair_red"].value_counts().items():
    print(f"    {pair:<25} {count:>4}  ({100*count/n_both:.1f}%)")

# =========================
# AGGREGATE PER MOTIF
# =========================

# =========================

def motif_agg(group):
    n = len(group)
    row = {"n_peptides": n,
           "Family_ID": group["Family_ID"].iloc[0]
           if "Family_ID" in group.columns else ""}

    # per tool: detailed + reduced
    for tool, mech_col, red_col in [
        ("FMAP", "Mechanism_FMAP", "Mech_FMAP_reduced"),
        ("PPM",  "Mechanism_PPM",  "Mech_PPM_reduced"),
    ]:
        for mech in MECHS_DETAIL:
            row[f"n_{mech}_{tool}"]   = (group[mech_col] == mech).sum()
            row[f"pct_{mech}_{tool}"] = round(100 * row[f"n_{mech}_{tool}"] / n, 1)
        row[f"Dominant_detail_{tool}"] = \
            group[mech_col].value_counts().idxmax() if n > 0 else "Ambiguous"
        for mech in MECHS_REDUCED:
            row[f"n_{mech}_red_{tool}"]   = (group[red_col] == mech).sum()
            row[f"pct_{mech}_red_{tool}"] = round(
                100 * row[f"n_{mech}_red_{tool}"] / n, 1)
        row[f"Dominant_reduced_{tool}"] = \
            group[red_col].value_counts().idxmax() if n > 0 else "Ambiguous"

    # weighted dominant
    row["Dominant_weighted"]         = group["Mechanism_weighted"].value_counts().idxmax()
    row["Dominant_weighted_reduced"] = group["Mechanism_weighted_reduced"].value_counts().idxmax()

    # consensus — denominator = peptides with data from both tools
    n_both = group["Both_available"].sum()
    row["n_both_available"]       = n_both
    row["Consensus_rate_detail"]  = round(
        group["Consensus_detail"].sum() / n_both * 100, 1) if n_both > 0 else 0
    row["Consensus_rate_reduced"] = round(
        group["Consensus_reduced"].sum() / n_both * 100, 1) if n_both > 0 else 0
    row["Helix_overlap_rate"]     = round(group["Helix_Motif_Overlap"].mean() * 100, 1)
    row["Mean_overlap_AA"]        = round(group["Overlap_AA"].mean(), 2)

    # mean physical params per tool
    for tool, t_col, d_col, e_col in [
        ("FMAP", "Tilt_FMAP", "Depth_FMAP", "Energy_FMAP"),
        ("PPM",  "Tilt_PPM",  "Depth_PPM",  "Energy_PPM"),
    ]:
        row[f"Mean_Tilt_{tool}"]   = round(group[t_col].mean(), 2)
        row[f"Mean_Depth_{tool}"]  = round(group[d_col].mean(), 2)
        row[f"Mean_Energy_{tool}"] = round(group[e_col].mean(), 2)

    return pd.Series(row)

motif_summary = df.groupby("Motif_representative").apply(motif_agg).reset_index()
motif_summary = motif_summary.sort_values("n_peptides", ascending=False)

# =========================
# AGGREGATE PER FAMILY
# =========================

def family_agg(group):
    n = len(group)
    row = {"n_peptides": n,
           "n_motifs": group["Motif_representative"].nunique()}
    for tool, mech_col, red_col in [
        ("FMAP", "Mechanism_FMAP", "Mech_FMAP_reduced"),
        ("PPM",  "Mechanism_PPM",  "Mech_PPM_reduced"),
    ]:
        for mech in MECHS_REDUCED:
            row[f"n_{mech}_red_{tool}"]   = (group[red_col] == mech).sum()
            row[f"pct_{mech}_red_{tool}"] = round(
                100 * row[f"n_{mech}_red_{tool}"] / n, 1)
        row[f"Dominant_reduced_{tool}"] = \
            group[red_col].value_counts().idxmax() if n > 0 else "Ambiguous"
    row["Dominant_weighted_reduced"] = \
        group["Mechanism_weighted_reduced"].value_counts().idxmax()
    n_both = group["Both_available"].sum()
    row["n_both_available"]       = n_both
    row["Consensus_rate_detail"]  = round(
        group["Consensus_detail"].sum() / n_both * 100, 1) if n_both > 0 else 0
    row["Consensus_rate_reduced"] = round(
        group["Consensus_reduced"].sum() / n_both * 100, 1) if n_both > 0 else 0
    row["Helix_overlap_rate"]     = round(group["Helix_Motif_Overlap"].mean() * 100, 1)
    return pd.Series(row)

family_summary = df.groupby("Family_ID").apply(family_agg).reset_index()
family_summary = family_summary.sort_values("n_peptides", ascending=False)

print(f"\nMotifs analysed: {len(motif_summary)}")
print(f"Families analysed: {len(family_summary)}")

# =========================
# SAVE EXCEL
# =========================

out_peptide = OUT_DIR / "01_peptide_analysis.xlsx"
out_motif   = OUT_DIR / "02_motif_summary.xlsx"
out_family  = OUT_DIR / "03_family_summary.xlsx"

df["Disagreement_pair"] = df.apply(
    lambda r: " vs ".join(sorted([r["Mechanism_FMAP"], r["Mechanism_PPM"]]))
    if not r["Consensus_detail"] and
       r["Mechanism_FMAP"] != "Ambiguous" and
       r["Mechanism_PPM"]  != "Ambiguous"
    else "", axis=1
)

with pd.ExcelWriter(out_peptide, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Peptide_data", index=False)
    df[df["Disagreement_pair"] != ""].to_excel(
        writer, sheet_name="Disagreements", index=False)

with pd.ExcelWriter(out_motif, engine="openpyxl") as writer:
    motif_summary.to_excel(writer, sheet_name="Motif_summary", index=False)
    for mech in MECHS_DETAIL:
        sub = df[df["Mechanism_weighted"] == mech]
        if not sub.empty:
            sub.to_excel(writer, sheet_name=f"Peptides_{mech[:10]}", index=False)

with pd.ExcelWriter(out_family, engine="openpyxl") as writer:
    family_summary.to_excel(writer, sheet_name="Family_summary", index=False)

print(f"\n✅ Excel saved:")
print(f"  {out_peptide}")
print(f"  {out_motif}")
print(f"  {out_family}")

# =========================
# FIGURES
# =========================

top_motifs   = motif_summary.head(30)
motif_labels = top_motifs["Motif_representative"].tolist()
x            = np.arange(len(top_motifs))

# --- Figure 1: FMAP vs PPM stacked bars per motif (reduced) ---
fig, axes = plt.subplots(2, 1, figsize=(18, 11), sharex=True)
fig.patch.set_facecolor("#fafafa")
for ax, (tool, suffix) in zip(axes, [("FMAP","_FMAP"), ("PPM","_PPM")]):
    ax.set_facecolor("#fafafa")
    bottom = np.zeros(len(top_motifs))
    for mech in MECHS_REDUCED:
        col  = f"pct_{mech}_red{suffix}"
        if col not in top_motifs.columns: continue
        vals = top_motifs[col].fillna(0).values
        ax.bar(x, vals, bottom=bottom,
               color=PALETTE_REDUCED.get(mech, "#ccc"),
               label=mech, edgecolor="white", linewidth=0.4, width=0.72)
        # label bars > 15%
        for xi, (v, b) in enumerate(zip(vals, bottom - vals)):
            if v > 15:
                ax.text(xi, b + vals[xi]/2, f"{v:.0f}%",
                        ha="center", va="center", fontsize=7,
                        color="white", fontweight="bold")
        bottom += vals
    ax.set_ylabel("% peptides", fontsize=10)
    ax.set_ylim(0, 108)
    ax.set_title(f"{tool} — Carpet / Pore / Ambiguous per motif",
                 fontweight="bold", fontsize=11)
    ax.legend(title="Mechanism", fontsize=9,
              bbox_to_anchor=(1.01,1), loc="upper left")
    ax.spines[["top","right"]].set_visible(False)
axes[1].set_xticks(x)
axes[1].set_xticklabels(motif_labels, rotation=45, ha="right", fontsize=8)
fig.suptitle("FMAP vs PPM — mechanism per motif (top 30, reduced: Carpet/Pore)\n"
             f"PPM weight={WEIGHT_PPM} · FMAP weight={WEIGHT_FMAP}",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "01_fmap_vs_ppm_reduced.png",
            dpi=DPI, bbox_inches="tight"); plt.close()

# --- Figure 2: FMAP vs PPM stacked bars per motif (detailed) ---
fig, axes = plt.subplots(2, 1, figsize=(18, 11), sharex=True)
fig.patch.set_facecolor("#fafafa")
for ax, (tool, suffix) in zip(axes, [("FMAP","_FMAP"), ("PPM","_PPM")]):
    ax.set_facecolor("#fafafa")
    bottom = np.zeros(len(top_motifs))
    for mech in MECHS_DETAIL:
        col  = f"pct_{mech}{suffix}"
        if col not in top_motifs.columns: continue
        vals = top_motifs[col].fillna(0).values
        ax.bar(x, vals, bottom=bottom,
               color=PALETTE_DETAIL.get(mech, "#ccc"),
               label=mech, edgecolor="white", linewidth=0.4, width=0.72)
        for xi, (v, b) in enumerate(zip(vals, bottom - vals)):
            if v > 15:
                ax.text(xi, b + vals[xi]/2, f"{v:.0f}%",
                        ha="center", va="center", fontsize=7,
                        color="white", fontweight="bold")
        bottom += vals
    ax.set_ylabel("% peptides", fontsize=10)
    ax.set_ylim(0, 108)
    ax.set_title(f"{tool} — Carpet / Toroidal / Barrel / Ambiguous per motif",
                 fontweight="bold", fontsize=11)
    ax.legend(title="Mechanism", fontsize=9,
              bbox_to_anchor=(1.01,1), loc="upper left")
    ax.spines[["top","right"]].set_visible(False)
axes[1].set_xticks(x)
axes[1].set_xticklabels(motif_labels, rotation=45, ha="right", fontsize=8)
fig.suptitle("FMAP vs PPM — mechanism per motif (top 30, detailed)",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "02_fmap_vs_ppm_detailed.png",
            dpi=DPI, bbox_inches="tight"); plt.close()

# --- Figure 3: Dominant mechanism FMAP vs PPM — confusion heatmap ---
pivot = pd.crosstab(
    motif_summary["Dominant_reduced_FMAP"],
    motif_summary["Dominant_reduced_PPM"],
)
fig, ax = plt.subplots(figsize=(6, 5))
fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd",
            linewidths=0.5, ax=ax, annot_kws={"size": 13, "weight": "bold"})
ax.set_xlabel("PPM dominant mechanism", fontsize=11)
ax.set_ylabel("FMAP dominant mechanism", fontsize=11)
ax.set_title("Dominant mechanism agreement\nacross all motifs",
             fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "03_dominant_confusion.png",
            dpi=DPI, bbox_inches="tight"); plt.close()

# --- Figure 4: Consensus + helix overlap per motif ---
fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
fig.patch.set_facecolor("#fafafa")
for ax, col, title, color in zip(axes,
    ["Consensus_rate_reduced", "Helix_overlap_rate"],
    ["Consensus rate FMAP vs PPM (%)", f"Helix-motif overlap ≥{MIN_OVERLAP} aa (%)"],
    ["#5B8DB8", "#E07B54"]
):
    ax.set_facecolor("#fafafa")
    vals = top_motifs[col].fillna(0)
    bars = ax.bar(x, vals, color=color, alpha=0.85,
                  edgecolor="white", width=0.72)
    ax.axhline(vals.mean(), color="#333", lw=1.2, ls="--",
               label=f"Mean = {vals.mean():.1f}%")
    ax.axhline(50, color="grey", lw=0.7, ls=":", alpha=0.4)
    ax.set_ylabel("%", fontsize=10)
    ax.set_ylim(0, 115)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    for bar, val in zip(bars, vals):
        if val > 8:
            ax.text(bar.get_x() + bar.get_width()/2, val + 1,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=6.5)
axes[1].set_xticks(x)
axes[1].set_xticklabels(motif_labels, rotation=45, ha="right", fontsize=8)
fig.suptitle("Consensus between FMAP and PPM, and helix-motif overlap per motif (top 30)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "04_consensus_and_overlap.png",
            dpi=DPI, bbox_inches="tight"); plt.close()

print(f"\n✅ Figures saved to: {FIG_DIR}")