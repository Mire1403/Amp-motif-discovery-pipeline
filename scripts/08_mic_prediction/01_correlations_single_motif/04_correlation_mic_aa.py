from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import spearmanr, mannwhitneyu

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

MIC_FILE = (ROOT / "results/12_apex_prediction/02_results"
            / "02_motif_mic_summary_merged.csv")

OUT_DIR = ROOT / "results/17_aminoacid_mic"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

DPI      = 300
MIC_THR  = 128.0   
N_EXTREME = 20     

# Kyte-Doolittle hydrophobicity scale
KD = {"A": 1.8, "R":-4.5, "N":-3.5, "D":-3.5, "C": 2.5,
      "Q":-3.5, "E":-3.5, "G":-0.4, "H":-3.2, "I": 4.5,
      "L": 3.8, "K":-3.9, "M": 1.9, "F": 2.8, "P":-1.6,
      "S":-0.8, "T":-0.7, "W":-0.9, "Y":-1.3, "V": 4.2}

# net charge at pH 7 (approximate)
CHARGE = {"K":+1,"R":+1,"H":+0.1,"D":-1,"E":-1}

CANONICAL_AA = list("ACDEFGHIKLMNPQRSTVWY")

PAL = {"top20":"#E07B54", "bottom20":"#5B8DB8"}

# ─────────────────────────────────────────────
# LOAD & FILTER
# ─────────────────────────────────────────────

print("Loading data...")
mic_df = pd.read_csv(MIC_FILE)
mic_df["MIC_mean"] = pd.to_numeric(mic_df["MIC_mean"], errors="coerce")
mic_df = mic_df[mic_df["MIC_mean"] <= MIC_THR].dropna(
    subset=["MIC_mean","Motif_representative"]).copy()
mic_df = mic_df.sort_values("MIC_mean").reset_index(drop=True)

print(f"Active motifs (MIC ≤ {MIC_THR}): {len(mic_df)}")
print(f"MIC range: {mic_df['MIC_mean'].min():.1f} – {mic_df['MIC_mean'].max():.1f} µmol/L")

# clean sequences (keep only canonical AA)
def clean_seq(s):
    return "".join(aa for aa in str(s).upper() if aa in CANONICAL_AA)

mic_df["Seq_clean"] = mic_df["Motif_representative"].apply(clean_seq)
mic_df = mic_df[mic_df["Seq_clean"].str.len() >= 4].copy()

# ─────────────────────────────────────────────
# FEATURE CALCULATION
# ─────────────────────────────────────────────

def calc_features(seq):
    L = len(seq)
    if L == 0:
        return {}
    feats = {}
    # AA frequencies
    for aa in CANONICAL_AA:
        feats[f"f_{aa}"] = seq.count(aa) / L
    # net charge
    feats["net_charge"]    = sum(CHARGE.get(aa,0) for aa in seq)
    feats["charge_density"]= feats["net_charge"] / L
    # hydrophobicity (mean KD)
    feats["hydrophobicity"]= np.mean([KD.get(aa,0) for aa in seq])
    # %KL alternating (K or L in every position)
    feats["pct_KL"]        = (seq.count("K") + seq.count("L")) / L * 100
    feats["pct_cationic"]  = (seq.count("K") + seq.count("R")) / L * 100
    feats["pct_hydrophobic"]= sum(seq.count(a) for a in "ILVMFW") / L * 100
    feats["length"]        = L
    # simple hydrophobic moment (Eisenberg, helix period 100°)
    angle = 100.0  # degrees per residue for alpha helix
    h_cos = sum(KD.get(aa,0) * np.cos(np.radians(i*angle))
                for i, aa in enumerate(seq))
    h_sin = sum(KD.get(aa,0) * np.sin(np.radians(i*angle))
                for i, aa in enumerate(seq))
    feats["hydrophobic_moment"] = np.sqrt(h_cos**2 + h_sin**2) / L
    return feats

feat_df = pd.DataFrame(
    mic_df["Seq_clean"].apply(calc_features).tolist(),
    index=mic_df.index
)
df = pd.concat([mic_df[["Motif_representative","MIC_mean",
                          "Dominant_reduced"]].reset_index(drop=True),
                feat_df.reset_index(drop=True)], axis=1)

print(f"Features calculated: {feat_df.shape[1]}")

# ─────────────────────────────────────────────
# 1. CORRELACIÓ AA FREQ vs MIC
# ─────────────────────────────────────────────

print("\n=== Correlació freqüència AA vs MIC ===")
aa_corr_rows = []
for aa in CANONICAL_AA:
    col = f"f_{aa}"
    sub = df[["MIC_mean",col]].dropna()
    if len(sub) < 5: continue
    rho, p = spearmanr(sub["MIC_mean"], sub[col])
    aa_corr_rows.append({"AA": aa, "Spearman_rho": round(rho,3),
                          "p": round(p,6), "n": len(sub)})
aa_corr_df = pd.DataFrame(aa_corr_rows).sort_values("Spearman_rho")
print(aa_corr_df.to_string(index=False))

# ─────────────────────────────────────────────
# 2. TOP20 vs BOTTOM20
# ─────────────────────────────────────────────

n_ext   = min(N_EXTREME, len(df)//3)
top20   = df.nsmallest(n_ext, "MIC_mean")   # most active (lower MIC)
bot20   = df.nlargest(n_ext,  "MIC_mean")   # least active (higher MIC)

print(f"\n=== Top {n_ext} (MIC≤{top20['MIC_mean'].max():.1f}) "
      f"vs Bottom {n_ext} (MIC≥{bot20['MIC_mean'].min():.1f}) ===")

mw_rows = []
for aa in CANONICAL_AA:
    col = f"f_{aa}"
    g1  = top20[col].dropna().values
    g2  = bot20[col].dropna().values
    if len(g1) < 3 or len(g2) < 3: continue
    _, p = mannwhitneyu(g1, g2, alternative="two-sided")
    diff = g1.mean() - g2.mean()
    mw_rows.append({"AA": aa,
                    "mean_top20": round(g1.mean(),3),
                    "mean_bot20": round(g2.mean(),3),
                    "diff_top_minus_bot": round(diff,3),
                    "p_mw": round(p,6),
                    "sig": "***" if p<0.001 else "**" if p<0.01
                           else "*" if p<0.05 else "ns"})
mw_df = pd.DataFrame(mw_rows).sort_values("diff_top_minus_bot",
                                            ascending=False)
sig_aas = mw_df[mw_df["p_mw"] < 0.05]
print(f"Significant AAs (p<0.05): {len(sig_aas)}")
print(sig_aas[["AA","mean_top20","mean_bot20",
               "diff_top_minus_bot","p_mw","sig"]].to_string(index=False))

# ─────────────────────────────────────────────
# 3. CORRELACIÓ PROPIETATS GLOBALS vs MIC
# ─────────────────────────────────────────────

print("\n=== Correlació propietats globals vs MIC ===")
prop_cols = ["net_charge","charge_density","hydrophobicity",
             "pct_KL","pct_cationic","pct_hydrophobic",
             "length","hydrophobic_moment"]
prop_corr_rows = []
for col in prop_cols:
    sub = df[["MIC_mean",col]].dropna()
    if len(sub) < 5: continue
    rho, p = spearmanr(sub["MIC_mean"], sub[col])
    print(f"  {col:<22} ρ={rho:+.3f}  p={p:.4f}  n={len(sub)}")
    prop_corr_rows.append({"Property": col,
                            "Spearman_rho": round(rho,3),
                            "p": round(p,6), "n": len(sub)})
prop_corr_df = pd.DataFrame(prop_corr_rows).sort_values("Spearman_rho")

# ─────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────

# ── Fig 1: AA freq top20 vs bottom20 — grouped bar ──
fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
x  = np.arange(len(CANONICAL_AA))
w  = 0.35
t_means = [top20[f"f_{aa}"].mean() for aa in CANONICAL_AA]
b_means = [bot20[f"f_{aa}"].mean() for aa in CANONICAL_AA]
ax.bar(x - w/2, t_means, width=w, color=PAL["top20"],
       alpha=0.85, edgecolor="white",
       label=f"Top {n_ext} most active (MIC ≤ {top20['MIC_mean'].max():.0f})")
ax.bar(x + w/2, b_means, width=w, color=PAL["bottom20"],
       alpha=0.85, edgecolor="white",
       label=f"Bottom {n_ext} least active (MIC ≥ {bot20['MIC_mean'].min():.0f})")
# mark significant differences
for i, aa in enumerate(CANONICAL_AA):
    row = mw_df[mw_df["AA"]==aa]
    if row.empty: continue
    sig = row["sig"].values[0]
    if sig != "ns":
        ax.text(i, max(t_means[i], b_means[i]) + 0.01,
                sig, ha="center", fontsize=9, fontweight="bold", color="#333")
ax.set_xticks(x)
ax.set_xticklabels(CANONICAL_AA, fontsize=10)
ax.set_ylabel("Mean AA frequency", fontsize=10)
ax.set_title(f"Amino acid composition: top {n_ext} vs bottom {n_ext} by MIC\n"
             "* p<0.05  ** p<0.01  *** p<0.001 (Mann-Whitney)",
             fontweight="bold")
ax.legend(fontsize=9)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR/"01_aa_top_vs_bottom.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("\n  Fig 1: 01_aa_top_vs_bottom.png")

# ── Fig 2: AA Spearman ρ barplot ──
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
aa_corr_sorted = aa_corr_df.copy()
colors_aa = ["#E07B54" if r < 0 else "#5B8DB8"
             for r in aa_corr_sorted["Spearman_rho"]]
bars = ax.bar(range(len(aa_corr_sorted)), aa_corr_sorted["Spearman_rho"],
              color=colors_aa, alpha=0.85, edgecolor="white")
ax.set_xticks(range(len(aa_corr_sorted)))
ax.set_xticklabels(aa_corr_sorted["AA"], fontsize=10)
ax.axhline(0, color="black", lw=0.8, alpha=0.5)
# mark significant
for i, (_, row) in enumerate(aa_corr_sorted.iterrows()):
    if row["p"] < 0.05:
        ax.text(i, row["Spearman_rho"] + (0.01 if row["Spearman_rho"]>=0 else -0.02),
                "*", ha="center", fontsize=12, fontweight="bold")
ax.set_ylabel("Spearman ρ vs MIC", fontsize=10)
ax.set_title("Correlació freqüència AA vs MIC\n"
             "negatiu = AA més freqüent en motius actius",
             fontweight="bold")
ax.annotate("* p<0.05", xy=(0.98,0.98), xycoords="axes fraction",
            ha="right", va="top", fontsize=9, color="#555")
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR/"02_aa_spearman_barplot.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("  Fig 2: 02_aa_spearman_barplot.png")

# ── Fig 3: propietats globals vs MIC scatter (2×3 grid) ──
plot_props = [
    ("net_charge",        "Càrrega neta"),
    ("charge_density",    "Densitat de càrrega (càrrega/L)"),
    ("hydrophobicity",    "Hidrofobicitat KD (mitjana)"),
    ("pct_KL",            "% K+L"),
    ("pct_cationic",      "% Catiònic (K+R)"),
    ("hydrophobic_moment","Moment hidrofòbic"),
]
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.patch.set_facecolor("#fafafa")
axes = axes.flatten()

for i, (col, lab) in enumerate(plot_props):
    ax = axes[i]; ax.set_facecolor("#fafafa")
    sub = df[["MIC_mean", col, "Dominant_reduced"]].dropna()
    if sub.empty:
        ax.set_visible(False); continue
    colors = ["#E07B54" if m=="Carpet" else "#4C72B0" if m=="Pore"
              else "#AAAAAA" for m in sub["Dominant_reduced"]]
    ax.scatter(sub[col], sub["MIC_mean"],
               c=colors, alpha=0.75, s=55,
               edgecolors="white", linewidths=0.4)
    rho_v = prop_corr_df.loc[prop_corr_df["Property"]==col,
                               "Spearman_rho"].values
    p_v   = prop_corr_df.loc[prop_corr_df["Property"]==col, "p"].values
    if len(rho_v) > 0:
        ax.annotate(f"ρ={rho_v[0]:+.3f}  p={p_v[0]:.4f}\nn={len(sub)}",
                    xy=(0.05,0.92), xycoords="axes fraction", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3",fc="white",ec="#ddd"))
    ax.set_xlabel(lab, fontsize=9)
    ax.set_ylabel("MIC (µmol/L)", fontsize=9)
    ax.set_title(lab, fontweight="bold", fontsize=10)
    ax.spines[["top","right"]].set_visible(False)

patches = [mpatches.Patch(color="#E07B54",label="Carpet"),
           mpatches.Patch(color="#4C72B0",label="Pore"),
           mpatches.Patch(color="#AAAAAA",label="Ambiguous")]
fig.legend(handles=patches, fontsize=9, loc="lower right",
           bbox_to_anchor=(0.98,0.01))
fig.suptitle("Propietats fisicoquímiques vs MIC (motius actius, MIC ≤ 128 µmol/L)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR/"03_properties_vs_mic.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("  Fig 3: 03_properties_vs_mic.png")

# ── Fig 4: heatmap AA freq per motiu (top 30) ──
top30 = df.nsmallest(min(30, len(df)), "MIC_mean")
aa_matrix = top30[[f"f_{aa}" for aa in CANONICAL_AA]].copy()
aa_matrix.columns = CANONICAL_AA
aa_matrix.index   = top30["Motif_representative"].values

fig, ax = plt.subplots(figsize=(14, max(6, len(top30)*0.35)))
fig.patch.set_facecolor("#fafafa")
sns.heatmap(aa_matrix, cmap="YlOrRd", linewidths=0.3, ax=ax,
            cbar_kws={"label":"Frequency","shrink":0.6},
            annot=False)
ax.set_title(f"AA composition heatmap — top {len(top30)} most active motifs",
             fontweight="bold")
ax.set_xlabel("Amino acid", fontsize=10)
ax.set_ylabel("")
fig.tight_layout()
fig.savefig(FIG_DIR/"04_aa_heatmap_top30.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("  Fig 4: 04_aa_heatmap_top30.png")

# ─────────────────────────────────────────────
# SAVE EXCEL
# ─────────────────────────────────────────────

out_xl = OUT_DIR / "aminoacid_mic_analysis.xlsx"
with pd.ExcelWriter(out_xl, engine="openpyxl") as w:
    df.to_excel(w,            sheet_name="Motif_features",  index=False)
    aa_corr_df.to_excel(w,    sheet_name="AA_correlation",  index=False)
    mw_df.to_excel(w,         sheet_name="Top_vs_Bottom",   index=False)
    prop_corr_df.to_excel(w,  sheet_name="Property_corr",   index=False)
    top20.to_excel(w,         sheet_name="Top20_active",    index=False)
    bot20.to_excel(w,         sheet_name="Bottom20",        index=False)

    # GraphPad
    # Fig 1: grouped bar top vs bottom
    gp1 = pd.DataFrame({
        f"top20_{aa}":  [top20[f"f_{aa}"].mean()] for aa in CANONICAL_AA
    } | {
        f"bot20_{aa}":  [bot20[f"f_{aa}"].mean()] for aa in CANONICAL_AA
    })
    gp1.to_excel(w, sheet_name="GP_Fig1_aa_grouped", index=False)

    # Fig 2: ρ per AA
    aa_corr_df[["AA","Spearman_rho","p"]].to_excel(
        w, sheet_name="GP_Fig2_aa_rho", index=False)

    # Fig 3: global properties
    for col, lab in plot_props:
        sub = df[["MIC_mean", col]].dropna()
        pd.DataFrame({"MIC_mean": sub["MIC_mean"].values,
                      col: sub[col].values}).to_excel(
            w, sheet_name=f"GP_{col[:20]}", index=False)

    # Stats summary
    pd.DataFrame([{
        "MIC_threshold":    MIC_THR,
        "n_motifs":         len(df),
        "n_top_bottom":     n_ext,
        "n_sig_AAs":        len(sig_aas),
        "best_AA_corr":     aa_corr_df.iloc[0]["AA"],
        "best_rho":         aa_corr_df.iloc[0]["Spearman_rho"],
        "best_p":           aa_corr_df.iloc[0]["p"],
    }]).to_excel(w, sheet_name="Summary", index=False)

print(f"\n✅ Excel: {out_xl}")
print(f"✅ Figures: {FIG_DIR}")