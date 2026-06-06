from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

sns.set_theme(style="whitegrid", font_scale=1.05)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

def _find_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / ".git").exists() or (p / "pyproject.toml").exists():
            return p
    raise RuntimeError("Project root not found.")

ROOT = _find_root()

MOTIF_SUMMARY_FILE = (ROOT / "results/11_mechanism_analysis"
                      / "3_motif_mechanism/02_motif_summary.xlsx")
MIC_FILE           = (ROOT / "results/12_apex_prediction/02_results"
                      / "02_motif_mic_summary_merged.csv")

OUT_DIR = ROOT / "results/16_fmap_mic_correlation"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

DPI = 300

PAL_MECH = {
    "Carpet":        "#E07B54",
    "Pore":          "#4C72B0",
    "Ambiguous":     "#AAAAAA",
    "Toroidal_pore": "#4C72B0",
    "Barrel_stave":  "#3BAF77",
}

# ─────────────────────────────────────────────
# LOAD & MERGE
# ─────────────────────────────────────────────

print("Loading data...")
motif_summary = pd.read_excel(MOTIF_SUMMARY_FILE, sheet_name="Motif_summary")
mic_df        = pd.read_csv(MIC_FILE)

# columnes físiques disponibles al motif_summary
phys_cols = [c for c in motif_summary.columns
             if any(k in c for k in ["Energy","Tilt","Depth","Mean"])]
print(f"Physical columns available: {phys_cols}")

# merge per Motif_representative
df = motif_summary.merge(
    mic_df[["Motif_representative","MIC_mean","MIC_median",
            "Dominant_reduced","Dominant_mechanism"]],
    on="Motif_representative", how="inner"
)
df["MIC_mean"]   = pd.to_numeric(df["MIC_mean"],   errors="coerce")
df["MIC_median"] = pd.to_numeric(df["MIC_median"], errors="coerce")

# filter to active motifs only (MIC <= 128 µmol/L)
df = df[df["MIC_mean"] <= 128].copy()
print(f"Motifs after merge: {len(df)}  (filtered to MIC ≤ 128 µmol/L)")
print(f"MIC range: {df['MIC_mean'].min():.1f} – {df['MIC_mean'].max():.1f} µmol/L")

# numeric cleaning per a totes les columnes físiques
fmap_params = {
    "Mean_Energy_FMAP": "Transfer Energy FMAP (kcal/mol)",
    "Mean_Tilt_FMAP":   "Tilt Angle FMAP (°)",
    "Mean_Depth_FMAP":  "Depth FMAP (Å)",
}
ppm_params = {
    "Mean_Energy_PPM":  "Energy PPM (kcal/mol)",
    "Mean_Tilt_PPM":    "Tilt Angle PPM (°)",
    "Mean_Depth_PPM":   "Depth PPM (Å)",
}

all_params = {**fmap_params, **ppm_params}
for col in all_params:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ─────────────────────────────────────────────
# CORRELACIÓ GLOBAL
# ─────────────────────────────────────────────

print("\n=== Correlació paràmetres físics vs MIC_mean ===")
corr_rows = []
for col, label in all_params.items():
    if col not in df.columns:
        print(f"  {col:<30} NOT FOUND")
        continue
    sub = df[["MIC_mean", col]].dropna()
    if len(sub) < 5:
        print(f"  {col:<30} n={len(sub)} (too few)")
        continue
    rho, p_rho = spearmanr(sub["MIC_mean"], sub[col])
    r,   p_r   = pearsonr(sub["MIC_mean"],  sub[col])
    direction  = "↑ higher MIC" if rho > 0 else "↓ lower MIC"
    print(f"  {col:<30} ρ={rho:+.3f}  p={p_rho:.4f}  r={r:+.3f}  "
          f"n={len(sub)}  {direction}")
    corr_rows.append({
        "Parameter": col, "Label": label,
        "Tool": "FMAP" if "FMAP" in col else "PPM",
        "Spearman_rho": round(rho, 3), "Spearman_p": round(p_rho, 6),
        "Pearson_r":    round(r, 3),   "Pearson_p":  round(p_r, 6),
        "n": len(sub), "direction": direction
    })
corr_df = pd.DataFrame(corr_rows)

# ─────────────────────────────────────────────
# CORRELACIÓ PER MECANISME
# ─────────────────────────────────────────────

print("\n=== Correlació Energy_FMAP vs MIC per mecanisme ===")
mech_corr_rows = []
mech_col = "Dominant_reduced" if "Dominant_reduced" in df.columns \
           else "Dominant_weighted_reduced"

if mech_col in df.columns and "Mean_Energy_FMAP" in df.columns:
    for mech in df[mech_col].dropna().unique():
        sub = df[df[mech_col]==mech][["MIC_mean","Mean_Energy_FMAP"]].dropna()
        if len(sub) < 5:
            print(f"  {mech:<15} n={len(sub)} (too few)")
            continue
        rho, p = spearmanr(sub["MIC_mean"], sub["Mean_Energy_FMAP"])
        print(f"  {mech:<15} ρ={rho:+.3f}  p={p:.4f}  n={len(sub)}")
        mech_corr_rows.append({"Mechanism": mech, "n": len(sub),
                                "Spearman_rho": round(rho,3),
                                "p": round(p,6)})
mech_corr_df = pd.DataFrame(mech_corr_rows)

# ─────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────

# ── Fig 1: scatter grid — tots els paràmetres vs MIC ──
available = [(col, lab) for col, lab in all_params.items()
             if col in df.columns and df[col].notna().sum() >= 5]
ncols = 3
nrows = int(np.ceil(len(available) / ncols))

fig, axes = plt.subplots(nrows, ncols,
                          figsize=(6*ncols, 5*nrows))
fig.patch.set_facecolor("#fafafa")
axes = axes.flatten() if nrows > 1 else axes

for i, (col, lab) in enumerate(available):
    ax = axes[i]
    ax.set_facecolor("#fafafa")
    sub = df[["MIC_mean", col, mech_col]].dropna()
    colors = [PAL_MECH.get(str(m), "#AAAAAA") for m in sub[mech_col]]
    ax.scatter(sub[col], sub["MIC_mean"],
               c=colors, alpha=0.75, s=55,
               edgecolors="white", linewidths=0.4)
    rho_v = corr_df.loc[corr_df["Parameter"]==col, "Spearman_rho"].values
    p_v   = corr_df.loc[corr_df["Parameter"]==col, "Spearman_p"].values
    if len(rho_v) > 0:
        ax.annotate(f"ρ={rho_v[0]:+.3f}  p={p_v[0]:.4f}\nn={len(sub)}",
                    xy=(0.05,0.92), xycoords="axes fraction", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3",fc="white",ec="#ddd"))
    ax.set_xlabel(lab, fontsize=9)
    ax.set_ylabel("MIC individual (µmol/L)", fontsize=9)
    ax.set_title(col.replace("Mean_","").replace("_"," "),
                 fontweight="bold", fontsize=10)
    ax.spines[["top","right"]].set_visible(False)

# amaga eixos buits
for j in range(len(available), len(axes)):
    axes[j].set_visible(False)

# llegenda mecanisme
patches = [mpatches.Patch(color=c, label=m)
           for m, c in PAL_MECH.items()
           if m in df[mech_col].values]
fig.legend(handles=patches, title="Mechanism",
           fontsize=9, loc="lower right", bbox_to_anchor=(0.98, 0.02))
fig.suptitle("Correlació paràmetres físics FMAP/PPM vs MIC (APEX)\nper motiu",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR/"01_fmap_ppm_vs_mic_grid.png",
            dpi=DPI, bbox_inches="tight")
plt.close()
print(f"\n  Fig 1: 01_fmap_ppm_vs_mic_grid.png")

# ── Fig 2: barplot ρ per paràmetre ──
if not corr_df.empty:
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
    corr_sorted = corr_df.sort_values("Spearman_rho")
    colors_bar  = ["#4C72B0" if t=="FMAP" else "#E07B54"
                   for t in corr_sorted["Tool"]]
    bars = ax.barh(range(len(corr_sorted)), corr_sorted["Spearman_rho"],
                   color=colors_bar, alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(corr_sorted)))
    ax.set_yticklabels(corr_sorted["Label"], fontsize=9)
    ax.axvline(0, color="black", lw=1, alpha=0.5)
    for bar, p_val in zip(bars, corr_sorted["Spearman_p"]):
        sig = "***" if p_val<0.001 else "**" if p_val<0.01 \
              else "*" if p_val<0.05 else "ns"
        x = bar.get_width()
        ax.text(x + (0.005 if x >= 0 else -0.005), bar.get_y()+bar.get_height()/2,
                sig, va="center", ha="left" if x >= 0 else "right", fontsize=10)
    ax.set_xlabel("Spearman ρ vs MIC", fontsize=10)
    ax.set_title("Correlació paràmetres físics vs MIC\n(negatiu = paràmetre més baix → MIC més baix → més actiu)",
                 fontweight="bold")
    patches2 = [mpatches.Patch(color="#4C72B0", label="FMAP"),
                mpatches.Patch(color="#E07B54", label="PPM")]
    ax.legend(handles=patches2, fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR/"02_correlation_barplot.png",
                dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Fig 2: 02_correlation_barplot.png")

# ── Fig 3: Energy FMAP vs MIC per mecanisme ──
if "Mean_Energy_FMAP" in df.columns and mech_col in df.columns:
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
    for mech in df[mech_col].dropna().unique():
        sub = df[df[mech_col]==mech][["MIC_mean","Mean_Energy_FMAP",
                                       "Motif_representative"]].dropna()
        if sub.empty: continue
        color = PAL_MECH.get(str(mech), "#AAAAAA")
        ax.scatter(sub["Mean_Energy_FMAP"], sub["MIC_mean"],
                   c=color, alpha=0.75, s=60,
                   edgecolors="white", linewidths=0.4,
                   label=f"{mech} (n={len(sub)})")
    rho_e = corr_df.loc[corr_df["Parameter"]=="Mean_Energy_FMAP",
                         "Spearman_rho"].values
    p_e   = corr_df.loc[corr_df["Parameter"]=="Mean_Energy_FMAP",
                         "Spearman_p"].values
    if len(rho_e) > 0:
        ax.annotate(f"ρ={rho_e[0]:+.3f}  p={p_e[0]:.4f}",
                    xy=(0.05,0.92), xycoords="axes fraction", fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.4",fc="white",ec="#ddd"))
    ax.set_xlabel("Mean Transfer Energy FMAP (kcal/mol)", fontsize=11)
    ax.set_ylabel("MIC individual (µmol/L)", fontsize=11)
    ax.set_title("Energia de transferència FMAP vs MIC\n"
                 "per mecanisme (per motiu)",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR/"03_energy_fmap_vs_mic_by_mech.png",
                dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Fig 3: 03_energy_fmap_vs_mic_by_mech.png")

# ── Fig 4: FMAP vs PPM Energy scatter (consistència entre eines) ──
if "Mean_Energy_FMAP" in df.columns and "Mean_Energy_PPM" in df.columns:
    sub = df[["Mean_Energy_FMAP","Mean_Energy_PPM",
              mech_col,"MIC_mean"]].dropna()
    rho_ff, p_ff = spearmanr(sub["Mean_Energy_FMAP"], sub["Mean_Energy_PPM"])
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
    colors = [PAL_MECH.get(str(m),"#AAAAAA") for m in sub[mech_col]]
    ax.scatter(sub["Mean_Energy_FMAP"], sub["Mean_Energy_PPM"],
               c=colors, alpha=0.75, s=55,
               edgecolors="white", linewidths=0.4)
    lims = [min(sub["Mean_Energy_FMAP"].min(),
                sub["Mean_Energy_PPM"].min())-1,
            max(sub["Mean_Energy_FMAP"].max(),
                sub["Mean_Energy_PPM"].max())+1]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.4, label="y=x")
    ax.annotate(f"ρ={rho_ff:+.3f}  p={p_ff:.4f}\nn={len(sub)}",
                xy=(0.05,0.92), xycoords="axes fraction", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.4",fc="white",ec="#ddd"))
    ax.set_xlabel("Transfer Energy FMAP (kcal/mol)", fontsize=11)
    ax.set_ylabel("Energy PPM (kcal/mol)", fontsize=11)
    ax.set_title("Consistència entre FMAP i PPM\n(energia de transferència)",
                 fontweight="bold")
    patches3 = [mpatches.Patch(color=c, label=m)
                for m, c in PAL_MECH.items()
                if m in sub[mech_col].values]
    ax.legend(handles=patches3, fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR/"04_fmap_vs_ppm_energy.png",
                dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Fig 4: 04_fmap_vs_ppm_energy.png")
    print(f"\n  FMAP vs PPM Energy consistency: ρ={rho_ff:.3f}  p={p_ff:.4f}")

# ─────────────────────────────────────────────
# SAVE EXCEL
# ─────────────────────────────────────────────

out_xl = OUT_DIR / "fmap_ppm_mic_correlation.xlsx"
with pd.ExcelWriter(out_xl, engine="openpyxl") as w:
    df.to_excel(w,            sheet_name="Motif_data",      index=False)
    corr_df.to_excel(w,       sheet_name="Correlation",     index=False)
    mech_corr_df.to_excel(w,  sheet_name="Corr_by_mech",    index=False)
    # GraphPad
    for col, lab in available:
        sub_gp = df[["Motif_representative", "MIC_mean", col,
                      mech_col]].dropna()
        safe   = col.replace("Mean_","").replace("_","")[:25]
        pd.DataFrame({
            "MIC_mean":  sub_gp["MIC_mean"].values,
            col:         sub_gp[col].values,
            "Mechanism": sub_gp[mech_col].values,
        }).to_excel(w, sheet_name=f"GP_{safe}", index=False)

print(f"\n✅ Excel: {out_xl}")
print(f"✅ Figures: {FIG_DIR}")