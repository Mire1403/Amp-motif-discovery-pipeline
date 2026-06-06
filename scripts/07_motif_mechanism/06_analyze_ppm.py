"""
PPM mechanism classification and visualization.

Input:  ppm_results_clean.csv  (output of 04_parse_ppm.py)
Output: 01_ppm_classified.csv + 4 plots

Mechanism classification is based on two PPM parameters:
  - Tilt_deg   : angle between helix axis and membrane normal
                 0°  = perpendicular (transmembrane)
                 90° = parallel to membrane surface (surface/carpet)
  - Thickness_A: hydrophobic thickness (TM proteins) or immersion depth
                 (peripheral proteins), in Angstroms

Literature-grounded thresholds for AMPs
(Shai 1999, Brogden 2005, Wimley 2010):

  Carpet        : tilt > 70°                          (helix parallel to surface, detergent-like)
  Barrel_stave  : tilt < 35°,  depth > 6 Å           (perpendicular, deep insertion, TM-like)
  Toroidal_pore : tilt 35–70°, depth 3–25 Å          (oblique insertion, lipid-lined pore)
  Ambiguous     : everything else (depth < 3 Å, or tilt < 35° with shallow depth)
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# =========================
# PATHS
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "results" / "10_motif_mechanism_output"
    / "02_ppm_output" / "results2"
    / "ppm_results_clean.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results" / "11_mechanism_analysis"
    / "1_motif_activity_analysis" / "ppm2"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# LOAD
# =========================

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df)} rows, {df['Peptide_ID'].nunique()} unique peptides")
print(f"Columns: {list(df.columns)}\n")

# =========================
# UNIFY ENERGY
# Best_Energy: use the one already computed in the parser.
# If re-running from scratch, reconstruct it.
# =========================

if "Best_Energy" not in df.columns:
    # flat  → Transfer_Energy
    # curved → Energy_Curved (already the selected one)
    df["Best_Energy"] = df["Transfer_Energy"].fillna(df["Energy_Curved"])

# =========================
# FLAG TM PEPTIDES
# Peptides with real TM segments identified by PPM (from datasub1)
# =========================

df["Has_TM_Segment"] = df["Segment_Num"].notna()

# =========================
# CLASSIFY MECHANISM
# =========================

MECHANISMS = [
    "Carpet",
    "Barrel_stave",
    "Toroidal_pore",
    "Ambiguous",
]

MECHANISM_COLORS = {
    "Carpet":        "#457b9d",
    "Barrel_stave":  "#f4a261",
    "Toroidal_pore": "#2a9d8f",
    "Ambiguous":     "#adb5bd",
}


def classify_mechanism(row):
    tilt  = row["Tilt_deg"]
    depth = row["Thickness_A"]

    # missing values → ambiguous
    if pd.isna(tilt) or pd.isna(depth):
        return "Ambiguous"

    # Carpet: helix parallel to membrane surface (high tilt)
    if tilt > 70.0:
        return "Carpet"

    # Barrel stave: nearly perpendicular + deep insertion (depth > 6 Å)
    if tilt < 35.0 and depth > 6.0:
        return "Barrel_stave"

    # Toroidal pore: oblique insertion, tilt 35–70°, depth 3–25 Å
    if 35.0 <= tilt <= 70.0 and 3.0 <= depth <= 25.0:
        return "Toroidal_pore"

    # everything else: borderline tilt/depth, shallow insertion
    return "Ambiguous"


df["Mechanism"] = df.apply(classify_mechanism, axis=1)

# =========================
# SANITY PRINT
# =========================

print("=== Mechanism distribution ===")
mech_counts = df["Mechanism"].value_counts()
for m in MECHANISMS:
    n = mech_counts.get(m, 0)
    pct = 100 * n / len(df)
    print(f"  {m:<20} {n:>4}  ({pct:.1f}%)")

print("\n=== Ambiguous breakdown (to assess whether thresholds need adjusting) ===")
amb = df[df["Mechanism"] == "Ambiguous"].copy()
print(f"  Total ambiguous: {len(amb)}")

# tilt/depth summary
print("\n  Tilt_deg and Thickness_A distribution:")
print(amb[["Tilt_deg", "Thickness_A"]].describe().round(2).to_string())

# why are they ambiguous?
cond_shallow   = amb["Thickness_A"] < 3.0
cond_deep_mid  = (amb["Tilt_deg"] >= 35) & (amb["Tilt_deg"] <= 70) & (amb["Thickness_A"] > 25.0)
cond_low_tilt  = (amb["Tilt_deg"] < 35) & (amb["Thickness_A"] <= 6.0)
cond_mid_tilt  = (amb["Tilt_deg"] >= 35) & (amb["Tilt_deg"] <= 70) & (amb["Thickness_A"] < 3.0)

print(f"\n  Reason breakdown:")
print(f"    depth < 3 Å (too shallow for any category):              {cond_shallow.sum()}")
print(f"    tilt 35–70°, depth > 15 Å (deep but oblique):           {cond_deep_mid.sum()}")
print(f"    tilt < 35°, depth ≤ 6 Å (perpendicular but shallow):    {cond_low_tilt.sum()}")
print(f"    tilt 35–70°, depth < 3 Å (oblique but shallow):         {cond_mid_tilt.sum()}")
print(f"\n  → If 'deep but oblique' is large, consider raising Toroidal upper depth limit.")
print(f"  → If 'perpendicular but shallow' is large, consider lowering Barrel_stave depth threshold.")
print(f"  → If 'too shallow' dominates, Ambiguous is correctly catching non-inserting peptides.")

# =========================
# THRESHOLD OPTIMIZER
# Grid search over plausible threshold combinations.
# Reports top 10 by fewest Ambiguous, keeping each named
# category at ≥ 5% of total (so we don't just absorb everything
# into one bucket by pushing thresholds to extremes).
# =========================

print("\n=== Threshold optimizer (minimizing Ambiguous) ===")

import itertools

carpet_tilts   = [65, 70, 75, 80]          # tilt above → Carpet
barrel_depths  = [4, 5, 6, 7, 8]           # depth above → Barrel (given low tilt)
barrel_tilts   = [20, 25, 30]              # tilt below → Barrel
toroidal_lo    = [2, 3, 4]                 # depth lower bound for Toroidal
toroidal_hi    = [6, 7, 8, 10]             # depth upper bound for Toroidal

results = []

for ct, bd, bt, tlo, thi in itertools.product(
    carpet_tilts, barrel_depths, barrel_tilts, toroidal_lo, toroidal_hi
):
    if tlo >= thi:
        continue
    if bt >= ct:
        continue

    def _classify(row, ct=ct, bd=bd, bt=bt, tlo=tlo, thi=thi):
        tilt  = row["Tilt_deg"]
        depth = row["Thickness_A"]
        if pd.isna(tilt) or pd.isna(depth):
            return "Ambiguous"
        if tilt > ct:
            return "Carpet"
        if tilt < bt and depth > bd:
            return "Barrel_stave"
        if bt <= tilt <= ct and tlo <= depth <= thi:
            return "Toroidal_pore"
        return "Ambiguous"

    counts = df.apply(_classify, axis=1).value_counts()
    n      = len(df)
    n_amb  = counts.get("Ambiguous", 0)
    n_car  = counts.get("Carpet", 0)
    n_bar  = counts.get("Barrel_stave", 0)
    n_tor  = counts.get("Toroidal_pore", 0)

    # reject if any named category < 5% (avoids degenerate solutions)
    if any(x / n < 0.05 for x in [n_car, n_tor]):
        continue

    results.append({
        "carpet_tilt":    ct,
        "barrel_tilt":    bt,
        "barrel_depth":   bd,
        "toroidal_depth": f"{tlo}–{thi}",
        "Carpet_%":       round(100 * n_car / n, 1),
        "Barrel_%":       round(100 * n_bar / n, 1),
        "Toroidal_%":     round(100 * n_tor / n, 1),
        "Ambiguous_%":    round(100 * n_amb / n, 1),
        "Ambiguous_n":    n_amb,
    })

results_df = pd.DataFrame(results).sort_values("Ambiguous_n").head(10)
print(results_df.to_string(index=False))
print("\n  Current thresholds: carpet>70°, barrel tilt<35° depth>6Å, toroidal 35–70° depth 3–25Å")

# =========================
# SAVE CSV
# =========================

out_csv = OUTPUT_DIR / "01_ppm_classified.csv"
df.to_csv(out_csv, index=False)
print(f"\n✅ Saved: {out_csv}")

# =========================
# PLOTS
# =========================

def add_mechanism_legend(ax):
    patches = [
        mpatches.Patch(color=MECHANISM_COLORS[m], label=m)
        for m in MECHANISMS
        if m in df["Mechanism"].values
    ]
    ax.legend(handles=patches, title="Mechanism", fontsize=8,
              title_fontsize=8, loc="best", framealpha=0.8)


colors = df["Mechanism"].map(MECHANISM_COLORS).fillna("#cccccc")

# --- Plot 1: Tilt vs Depth (the classification space) ---
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(
    df["Tilt_deg"], df["Thickness_A"],
    c=colors, alpha=0.65, s=40, edgecolors="none"
)
# draw classification boundary lines
ax.axvline(35,  color="gray", lw=0.8, ls="--", alpha=0.5)
ax.axvline(70,  color="gray", lw=0.8, ls="--", alpha=0.5)
ax.axhline(3,   color="gray", lw=0.8, ls=":",  alpha=0.5)
ax.axhline(25,  color="gray", lw=0.8, ls=":",  alpha=0.5)

# annotate regions
for x, y, label in [
    (10, 20, "Barrel\nstave"),
    (50, 13, "Toroidal\npore"),
    (80, 10, "Carpet"),
    (50,  3, "Ambiguous"),
]:
    ax.text(x, y, label, fontsize=7, color="gray",
            ha="center", va="center", style="italic")

ax.set_xlabel("Tilt angle (°)\n[0° = perpendicular to membrane, 90° = parallel]", fontsize=10)
ax.set_ylabel("Immersion depth / hydrophobic thickness (Å)", fontsize=10)
ax.set_title("PPM classification space: Tilt vs Depth", fontsize=11)
add_mechanism_legend(ax)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "01_tilt_vs_depth_mechanism.png", dpi=300)
plt.close(fig)

# --- Plot 2: Tilt vs Energy, coloured by mechanism ---
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(
    df["Tilt_deg"], df["Best_Energy"],
    c=colors, alpha=0.65, s=40, edgecolors="none"
)
ax.axhline(-2,  color="gray", lw=0.8, ls="--", alpha=0.5, label="−2 kcal/mol threshold")
ax.axhline(-8,  color="gray", lw=0.8, ls=":",  alpha=0.5, label="−8 kcal/mol (moderate)")
ax.set_xlabel("Tilt angle (°)\n[0° = perpendicular to membrane, 90° = parallel]", fontsize=10)
ax.set_ylabel("Transfer energy (kcal/mol)", fontsize=10)
ax.set_title("PPM transfer energy vs tilt angle", fontsize=11)
add_mechanism_legend(ax)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "02_tilt_vs_energy_mechanism.png", dpi=300)
plt.close(fig)

# --- Plot 3: Energy distribution per mechanism ---
fig, ax = plt.subplots(figsize=(8, 5))
for mech in MECHANISMS:
    sub = df[df["Mechanism"] == mech]["Best_Energy"].dropna()
    if len(sub) == 0:
        continue
    ax.hist(
        sub, bins=25, alpha=0.6,
        color=MECHANISM_COLORS[mech], label=f"{mech} (n={len(sub)})",
        edgecolor="none"
    )
ax.axvline(-2, color="black", lw=1, ls="--", alpha=0.6)
ax.set_xlabel("Transfer energy (kcal/mol)", fontsize=10)
ax.set_ylabel("Count", fontsize=10)
ax.set_title("Distribution of PPM transfer energy by mechanism", fontsize=11)
ax.legend(fontsize=8, framealpha=0.8)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "03_energy_distribution_by_mechanism.png", dpi=300)
plt.close(fig)

# --- Plot 4: Depth distribution per mechanism ---
fig, ax = plt.subplots(figsize=(8, 5))
mech_order = [m for m in MECHANISMS if m in df["Mechanism"].values]
data_to_plot = [df[df["Mechanism"] == m]["Thickness_A"].dropna().values for m in mech_order]
bp = ax.boxplot(
    data_to_plot,
    patch_artist=True,
    medianprops=dict(color="black", lw=1.5),
    whiskerprops=dict(lw=1),
    capprops=dict(lw=1),
    flierprops=dict(marker=".", markersize=4, alpha=0.5),
)
for patch, mech in zip(bp["boxes"], mech_order):
    patch.set_facecolor(MECHANISM_COLORS[mech])
    patch.set_alpha(0.7)
ax.set_xticks(range(1, len(mech_order) + 1))
ax.set_xticklabels(mech_order, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Immersion depth / hydrophobic thickness (Å)", fontsize=10)
ax.set_title("Membrane insertion depth by mechanism", fontsize=11)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "04_depth_by_mechanism_boxplot.png", dpi=300)
plt.close(fig)

print("\n✅ Plots saved to:", OUTPUT_DIR)