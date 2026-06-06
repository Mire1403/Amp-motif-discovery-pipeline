"""
Analysis of motif combinations — 2 categories (Carpet / Pore).

Analyses:
  1. ΔMIC: combined vs mean individual (synergy/additive/interference)
  2. Subsampling (800 iter, balanced n per group): Kruskal-Wallis + pairwise MW
  3. Order effect: A+B vs B+A (paired t-test)
  4. Correlation: MIC_combined vs mean(MIC_individual) per combination type

Input: 04_merged_predicted_MICs.csv (2-category combinations)
"""

from pathlib import Path
from itertools import combinations as _comb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu, spearmanr, ttest_rel, pearsonr

# =========================
# CONFIG
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

COMB_FILE = (PROJECT_ROOT / "results/13_motif_combinations/01_combinations"
             / "04_merged_predicted_MICs.csv")
MOTIF_MIC_FILE = (PROJECT_ROOT / "results/12_apex_prediction/02_results"
                  / "02_motif_mic_summary_merged.csv")

OUT_DIR = PROJECT_ROOT / "results/13_motif_combinations/02_analysis_2cat"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

N_ITER   = 800
RNG      = np.random.default_rng(42)
DPI      = 300

CTYPES   = ["Carpet+Carpet", "Carpet+Pore", "Pore+Pore"]
PALETTE  = {"Carpet+Carpet": "#E07B54",
            "Carpet+Pore":   "#5B8DB8",
            "Pore+Pore":     "#3BAF77"}

sns.set_theme(style="whitegrid", font_scale=1.1)

# =========================
# LOAD
# =========================

df = pd.read_csv(COMB_FILE)
df["MIC_combined"]        = pd.to_numeric(df["MIC_combined"],        errors="coerce")
df["MIC_mean_individual"] = pd.to_numeric(df["MIC_mean_individual"], errors="coerce")
df["MIC_delta"]           = pd.to_numeric(df["MIC_delta"],           errors="coerce")

motif_mic = pd.read_csv(MOTIF_MIC_FILE)
mic_map   = motif_mic.set_index("Motif_representative")["MIC_mean"].to_dict()
mech_map  = motif_mic.set_index("Motif_representative")["Dominant_reduced"].to_dict()

# individual MIC reference by mechanism
carpet_ind = motif_mic[motif_mic["Dominant_reduced"]=="Carpet"]["MIC_mean"].dropna()
pore_ind   = motif_mic[motif_mic["Dominant_reduced"]=="Pore"]["MIC_mean"].dropna()

print(f"Loaded {len(df)} combinations")
for ct in CTYPES:
    n = (df["Combination_type"]==ct).sum()
    print(f"  {ct:<20} n={n}")

# add Family_ID for each motif
family_map = motif_mic.set_index("Motif_representative")["Family_ID"].to_dict() \
             if "Family_ID" in motif_mic.columns else {}
df["Family_1"] = df["Motif_1"].map(family_map)
df["Family_2"] = df["Motif_2"].map(family_map)
df["Same_family"] = (df["Family_1"] == df["Family_2"]) & df["Family_1"].notna()
n_same = df["Same_family"].sum()
print(f"  Same-family combinations: {n_same} ({100*n_same/len(df):.1f}%)")

# =========================
# ANALYSIS 1: ΔMIC — combined vs mean individual
# =========================

print(f"\n=== ΔMIC: combined vs mean individual ===")
delta_rows = []
for ct in CTYPES:
    sub = df[df["Combination_type"]==ct]["MIC_delta"].dropna()
    synergy      = (sub < -(df.loc[df["Combination_type"]==ct,
                                   "MIC_mean_individual"].dropna() * 0.10)).sum()
    interference = (sub >  (df.loc[df["Combination_type"]==ct,
                                   "MIC_mean_individual"].dropna() * 0.10)).sum()
    n = len(sub)
    print(f"  {ct:<20} mean_Δ={sub.mean():.2f}  "
          f"synergy={100*synergy/n:.1f}%  interference={100*interference/n:.1f}%")
    delta_rows.append({"Combination_type": ct, "n": n,
                       "mean_delta": round(sub.mean(),3),
                       "median_delta": round(sub.median(),3),
                       "std_delta": round(sub.std(),3),
                       "pct_synergy": round(100*synergy/n,1),
                       "pct_interference": round(100*interference/n,1)})
delta_df = pd.DataFrame(delta_rows)

# =========================
# ANALYSIS 2: SUBSAMPLING — balanced Kruskal-Wallis + pairwise MW
# =========================

groups_all = {ct: df[df["Combination_type"]==ct]["MIC_combined"].dropna().values
              for ct in CTYPES}
n_draw = min(len(v) for v in groups_all.values())
print(f"\n=== Subsampled analysis ({N_ITER} iter, n={n_draw} per group) ===")

kw_stats, kw_pvals = [], []
mw_results = {f"{ct1} vs {ct2}": {"U":[], "p":[], "d":[]}
              for ct1, ct2 in _comb(CTYPES, 2)}

for _ in range(N_ITER):
    samples = {ct: RNG.choice(v, n_draw, replace=False)
               for ct, v in groups_all.items()}
    h, p = kruskal(*samples.values())
    kw_stats.append(h); kw_pvals.append(p)

    for ct1, ct2 in _comb(CTYPES, 2):
        g1, g2 = samples[ct1], samples[ct2]
        u, p_mw = mannwhitneyu(g1, g2, alternative="two-sided")
        d = (g1.mean()-g2.mean()) / np.sqrt((g1.std()**2+g2.std()**2)/2)
        key = f"{ct1} vs {ct2}"
        mw_results[key]["U"].append(u)
        mw_results[key]["p"].append(p_mw)
        mw_results[key]["d"].append(d)

print(f"  Kruskal-Wallis: H={np.mean(kw_stats):.3f}±{np.std(kw_stats):.3f}  "
      f"p={np.mean(kw_pvals):.4f}±{np.std(kw_pvals):.4f}")

sub_rows = []
for key, res in mw_results.items():
    print(f"  {key}:")
    print(f"    Cohen_d={np.mean(res['d']):.3f}±{np.std(res['d']):.3f}  "
          f"p={np.mean(res['p']):.4f}±{np.std(res['p']):.4f}")
    sub_rows.append({
        "Comparison": key,
        "Cohen_d_mean": round(np.mean(res["d"]),3),
        "Cohen_d_std":  round(np.std(res["d"]),3),
        "p_mean":       round(np.mean(res["p"]),6),
        "p_std":        round(np.std(res["p"]),6),
        "n_iter": N_ITER, "n_per_group": n_draw
    })
sub_df = pd.DataFrame(sub_rows)

kw_df = pd.DataFrame({
    "H_mean": [round(np.mean(kw_stats),3)],
    "H_std":  [round(np.std(kw_stats),3)],
    "p_mean": [round(np.mean(kw_pvals),6)],
    "p_std":  [round(np.std(kw_pvals),6)],
    "n_iter": [N_ITER], "n_per_group": [n_draw]
})

# =========================
# ANALYSIS 3: ORDER EFFECT (A+B vs B+A)
# =========================

df["Pair_key"] = df.apply(
    lambda r: "_".join(sorted([str(r["Motif_1"]), str(r["Motif_2"])])), axis=1)
pairs = df.groupby("Pair_key").filter(lambda g: len(g) == 2)
pair_a = pairs.groupby("Pair_key").apply(
    lambda g: g.sort_values("Combination_ID").iloc[0]["MIC_combined"])
pair_b = pairs.groupby("Pair_key").apply(
    lambda g: g.sort_values("Combination_ID").iloc[1]["MIC_combined"])
valid = pair_a.notna() & pair_b.notna()
t_stat, p_order = ttest_rel(pair_a[valid], pair_b[valid])
order_diff = (pair_a[valid] - pair_b[valid]).abs()
print(f"\n=== Order effect (A+B vs B+A) ===")
print(f"  Pairs: {valid.sum()}  t={t_stat:.3f}  p={p_order:.4f}")
print(f"  Mean |ΔMIC order| = {order_diff.mean():.2f} µmol/L")

order_df = pd.DataFrame([{
    "n_pairs": valid.sum(),
    "t_stat": round(t_stat,3), "p": round(p_order,6),
    "mean_abs_delta": round(order_diff.mean(),3)
}])

# =========================
# ANALYSIS 4: CORRELATION combined vs mean individual
# =========================

corr_rows = []
print(f"\n=== Correlation: MIC_combined vs mean(individual) ===")
for ct in CTYPES:
    sub = df[df["Combination_type"]==ct].dropna(
        subset=["MIC_combined","MIC_mean_individual"])
    rho, p_rho = spearmanr(sub["MIC_combined"], sub["MIC_mean_individual"])
    r,   p_r   = pearsonr( sub["MIC_combined"], sub["MIC_mean_individual"])
    print(f"  {ct:<20} ρ={rho:.3f} p={p_rho:.4f}  r={r:.3f} p={p_r:.4f}  n={len(sub)}")
    corr_rows.append({"Combination_type": ct, "n": len(sub),
                      "Spearman_rho": round(rho,3), "Spearman_p": round(p_rho,6),
                      "Pearson_r": round(r,3),      "Pearson_p": round(p_r,6)})
corr_df = pd.DataFrame(corr_rows)

# =========================
# ANALYSIS 5: DIRECT COMPARISON combined vs individual
# Carpet+Carpet vs Carpet_ind, Pore+Pore vs Pore_ind, Carpet+Pore vs both
# =========================

print(f"\n=== Direct comparison: combined vs individual ===")
direct_rows = []
comparisons_direct = [
    ("Carpet+Carpet", carpet_ind.values, "Carpet+Carpet vs Carpet_individual"),
    ("Pore+Pore",     pore_ind.values,   "Pore+Pore vs Pore_individual"),
    ("Carpet+Pore",   carpet_ind.values, "Carpet+Pore vs Carpet_individual"),
    ("Carpet+Pore",   pore_ind.values,   "Carpet+Pore vs Pore_individual"),
]
for ct, ind_vals, label in comparisons_direct:
    g1 = df[df["Combination_type"]==ct]["MIC_combined"].dropna().values
    g2 = ind_vals
    u, p_mw = mannwhitneyu(g1, g2, alternative="two-sided")
    d = (g1.mean()-g2.mean()) / np.sqrt((g1.std()**2+g2.std()**2)/2)
    direction = "↓ lower" if g1.mean() < g2.mean() else "↑ higher"
    print(f"  {label}: p={p_mw:.4f}  d={d:.3f}  {direction} "
          f"({g1.mean():.1f} vs {g2.mean():.1f})")
    direct_rows.append({
        "Comparison": label, "n_combined": len(g1), "n_individual": len(g2),
        "mean_combined": round(g1.mean(),2), "mean_individual": round(g2.mean(),2),
        "U": round(u,1), "p_mw": round(p_mw,6), "Cohen_d": round(d,3),
        "direction": direction
    })
direct_df = pd.DataFrame(direct_rows)

# top 20 most active combinations
top20 = df.nsmallest(20, "MIC_combined")[
    ["Combination_ID","Motif_1","Mechanism_1","Motif_2","Mechanism_2",
     "Combination_type","Sequence","MIC_combined","MIC_mean_individual","MIC_delta"]
]

# =========================
# SAVE ANALYSIS EXCEL
# =========================
# ANALYSIS 6: CONSISTENTLY GOOD MOTIFS
# For each motif, compute mean MIC across all combinations it appears in
# =========================

print(f"\n=== Consistently good motifs (mean MIC in all combinations) ===")

motif_rows = []
all_motifs = pd.concat([df["Motif_1"], df["Motif_2"]]).unique()
for motif in all_motifs:
    mask = (df["Motif_1"]==motif) | (df["Motif_2"]==motif)
    sub  = df[mask]["MIC_combined"].dropna()
    if len(sub) < 3: continue
    motif_rows.append({
        "Motif":           motif,
        "Mechanism":       df[df["Motif_1"]==motif]["Mechanism_1"].iloc[0]
                           if (df["Motif_1"]==motif).any() else
                           df[df["Motif_2"]==motif]["Mechanism_2"].iloc[0],
        "n_combinations":  len(sub),
        "MIC_mean_in_comb":  round(sub.mean(),2),
        "MIC_median_in_comb":round(sub.median(),2),
        "MIC_std_in_comb":   round(sub.std(),2),
        "CV":              round(sub.std()/sub.mean()*100,1) if sub.mean()>0 else np.nan,
        "MIC_individual":  mic_map.get(motif, np.nan),
    })
motif_profile = pd.DataFrame(motif_rows).sort_values("MIC_mean_in_comb")
print(f"  Top 10 most active motifs in combinations:")
print(motif_profile[["Motif","Mechanism","MIC_individual",
                      "MIC_mean_in_comb","CV"]].head(10).to_string(index=False))

# =========================
# ANALYSIS 7: FAMILY-LEVEL MIC
# Group by Family_ID — mean individual + mean in combinations
# =========================

print(f"\n=== Family-level MIC analysis ===")
family_rows = []
for fam in df["Family_ID"].dropna().unique() if "Family_ID" in df.columns else []:
    mask = (df.get("Family_1",pd.Series())==fam) | (df.get("Family_2",pd.Series())==fam)
    sub  = df[mask]["MIC_combined"].dropna()
    if len(sub) < 2: continue
    family_rows.append({
        "Family_ID": fam,
        "n_combinations": len(sub),
        "MIC_mean_comb":   round(sub.mean(),2),
        "MIC_median_comb": round(sub.median(),2),
    })
family_df = pd.DataFrame(family_rows).sort_values("MIC_mean_comb") \
            if family_rows else pd.DataFrame()

# family from motif individual data
motif_mic_fam = motif_mic.copy() if "Family_ID" in motif_mic.columns else pd.DataFrame()
if not motif_mic_fam.empty:
    fam_ind = motif_mic_fam.groupby("Family_ID")["MIC_mean"].agg(
        ["mean","median","std","count"]).round(2).reset_index()
    fam_ind.columns = ["Family_ID","MIC_ind_mean","MIC_ind_median",
                       "MIC_ind_std","n_motifs"]
    fam_ind = fam_ind.sort_values("MIC_ind_mean")
    print(f"  Top 10 families by individual MIC:")
    print(fam_ind.head(10).to_string(index=False))
else:
    fam_ind = pd.DataFrame()

# =========================
# ANALYSIS 8: ORDER PATTERN — which motifs prefer position 1 or 2?
# =========================

print(f"\n=== Order pattern: motifs that prefer position 1 or 2 ===")
order_pattern_rows = []
for motif in all_motifs:
    as_m1 = df[df["Motif_1"]==motif]["MIC_combined"].dropna()
    as_m2 = df[df["Motif_2"]==motif]["MIC_combined"].dropna()
    if len(as_m1) < 3 or len(as_m2) < 3: continue
    delta = as_m1.mean() - as_m2.mean()
    _, p  = mannwhitneyu(as_m1, as_m2, alternative="two-sided")
    order_pattern_rows.append({
        "Motif":      motif,
        "Mechanism":  mech_map.get(motif,""),
        "MIC_as_pos1": round(as_m1.mean(),2),
        "MIC_as_pos2": round(as_m2.mean(),2),
        "Delta_pos1_minus_pos2": round(delta,2),
        "p_mw": round(p,6),
        "Prefers": "Position_2" if delta > 0 else "Position_1",
    })
order_pattern_df = pd.DataFrame(order_pattern_rows)
sig_order = order_pattern_df[order_pattern_df["p_mw"] < 0.05].sort_values("p_mw")
print(f"  Motifs with significant position preference (p<0.05): {len(sig_order)}")
if len(sig_order) > 0:
    print(sig_order[["Motif","Mechanism","MIC_as_pos1",
                      "MIC_as_pos2","Prefers","p_mw"]].head(10).to_string(index=False))

# =========================
# ANALYSIS 9: NOISY MOTIFS FILTER
# Remove motifs with CV > threshold, repeat KW subsampling
# =========================

CV_THRESHOLD = 40.0  # % coefficient of variation
print(f"\n=== Noisy motifs filter (CV > {CV_THRESHOLD}%) ===")

noisy_motifs = set(motif_profile[motif_profile["CV"] > CV_THRESHOLD]["Motif"])
print(f"  Noisy motifs removed: {len(noisy_motifs)}")

df_clean = df[~df["Motif_1"].isin(noisy_motifs) &
              ~df["Motif_2"].isin(noisy_motifs)].copy()
print(f"  Combinations after filter: {len(df_clean)} (was {len(df)})")

# repeat KW subsampling on clean data
groups_clean = {ct: df_clean[df_clean["Combination_type"]==ct]["MIC_combined"].dropna().values
                for ct in CTYPES}
n_draw_clean = min((len(v) for v in groups_clean.values() if len(v)>0), default=0)

if n_draw_clean >= 10:
    kw_clean, kw_pclean = [], []
    for _ in range(N_ITER):
        samp = {ct: RNG.choice(v, n_draw_clean, replace=False)
                for ct, v in groups_clean.items() if len(v)>=n_draw_clean}
        if len(samp) >= 2:
            h, p = kruskal(*samp.values())
            kw_clean.append(h); kw_pclean.append(p)
    print(f"  KW after filter: H={np.mean(kw_clean):.3f}±{np.std(kw_clean):.3f}  "
          f"p={np.mean(kw_pclean):.4f}±{np.std(kw_pclean):.4f}  "
          f"(n={n_draw_clean}/group)")
    filter_kw_df = pd.DataFrame([{
        "CV_threshold": CV_THRESHOLD,
        "n_noisy_removed": len(noisy_motifs),
        "n_combinations_clean": len(df_clean),
        "n_per_group": n_draw_clean,
        "H_mean": round(np.mean(kw_clean),3),
        "H_std":  round(np.std(kw_clean),3),
        "p_mean": round(np.mean(kw_pclean),6),
        "p_std":  round(np.std(kw_pclean),6),
    }])
else:
    filter_kw_df = pd.DataFrame()
    print("  Not enough data after filter for subsampling")

# =========================
# SAVE ANALYSIS EXCEL
# =========================

out_excel = OUT_DIR / "01_analysis_2cat.xlsx"
with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
    df.to_excel(writer,              sheet_name="Combinations_MIC",    index=False)
    delta_df.to_excel(writer,        sheet_name="Delta_MIC_summary",   index=False)
    direct_df.to_excel(writer,       sheet_name="Direct_comparison",   index=False)
    kw_df.to_excel(writer,           sheet_name="KruskalWallis_sub",   index=False)
    sub_df.to_excel(writer,          sheet_name="MannWhitney_sub",     index=False)
    corr_df.to_excel(writer,         sheet_name="Correlation",         index=False)
    order_df.to_excel(writer,        sheet_name="Order_effect",        index=False)
    top20.to_excel(writer,           sheet_name="Top20_active",        index=False)
    motif_profile.to_excel(writer,   sheet_name="Motif_profile",       index=False)
    if not fam_ind.empty:
        fam_ind.to_excel(writer,     sheet_name="Family_MIC_ind",      index=False)
    if not family_df.empty:
        family_df.to_excel(writer,   sheet_name="Family_MIC_comb",     index=False)
    order_pattern_df.sort_values("p_mw").to_excel(
        writer,                      sheet_name="Order_pattern",       index=False)
    if not filter_kw_df.empty:
        filter_kw_df.to_excel(writer,sheet_name="Filter_noisy_KW",    index=False)
        df_clean.to_excel(writer,    sheet_name="Combinations_clean",  index=False)
    pd.DataFrame({"H": kw_stats, "p": kw_pvals}).to_excel(
        writer,                      sheet_name="KW_iterations",       index=False)

print(f"\n✅ Analysis Excel: {out_excel}")

# =========================
# SAVE GRAPHPAD EXCEL
# One sheet per figure, wide format (one column per group)
# =========================

out_gp = OUT_DIR / "02_graphpad_2cat.xlsx"

# helper: pad series to same length
def to_wide(series_dict):
    max_n = max(len(v) for v in series_dict.values())
    return pd.DataFrame({k: pd.Series(v).reset_index(drop=True)
                         for k, v in series_dict.items()})

with pd.ExcelWriter(out_gp, engine="openpyxl") as writer:

    # GP Fig 1 — violin: individual vs combined
    to_wide({
        "Carpet_individual":  carpet_ind.values,
        "Carpet+Carpet":      groups_all["Carpet+Carpet"],
        "Pore_individual":    pore_ind.values,
        "Pore+Pore":          groups_all["Pore+Pore"],
        "Carpet+Pore":        groups_all["Carpet+Pore"],
    }).to_excel(writer, sheet_name="GP_Fig1_violin", index=False)

    # GP Fig 2 — scatter correlation (3 sheets)
    for ct in CTYPES:
        sub = df[df["Combination_type"]==ct].dropna(
            subset=["MIC_combined","MIC_mean_individual"])
        pd.DataFrame({
            "MIC_mean_individual": sub["MIC_mean_individual"].values,
            "MIC_combined":        sub["MIC_combined"].values,
        }).to_excel(writer, sheet_name=f"GP_Fig2_{ct[:12]}", index=False)

    # GP Fig 3 — ΔMIC by group
    to_wide({ct: df[df["Combination_type"]==ct]["MIC_delta"].dropna().values
             for ct in CTYPES}).to_excel(
        writer, sheet_name="GP_Fig3_delta_MIC", index=False)

    # GP Fig 4 — order effect (paired)
    pd.DataFrame({
        "MIC_order_A": pair_a[valid].values,
        "MIC_order_B": pair_b[valid].values,
    }).to_excel(writer, sheet_name="GP_Fig4_order", index=False)

    # GP Fig 5 — Cohen's d subsampling distributions
    pd.DataFrame({k.replace(" vs ", "_vs_"): pd.Series(v["d"])
                  for k, v in mw_results.items()}).to_excel(
        writer, sheet_name="GP_Fig5_cohens_d", index=False)

    # GP Fig 6 — motif profile (mean MIC in combinations)
    motif_profile[["Motif","Mechanism","MIC_individual",
                   "MIC_mean_in_comb","CV"]].to_excel(
        writer, sheet_name="GP_Fig6_motif_profile", index=False)

    # GP Fig 7 — order pattern (significant)
    if len(sig_order) > 0:
        sig_order[["Motif","MIC_as_pos1","MIC_as_pos2"]].to_excel(
            writer, sheet_name="GP_Fig7_order_pattern", index=False)

    # GP Fig 8 — family individual MIC
    if not fam_ind.empty:
        fam_ind.to_excel(writer, sheet_name="GP_Fig8_family_MIC", index=False)

    # GP Stats summary — ready to annotate figures
    direct_df.to_excel(writer, sheet_name="GP_Stats_direct", index=False)
    sub_df.to_excel(writer,    sheet_name="GP_Stats_subsampled", index=False)

print(f"✅ GraphPad Excel: {out_gp}")

# =========================
# FIGURES
# =========================

# Fig 1: Violin — combined vs individual reference
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
plot_rows = []
for v in carpet_ind: plot_rows.append({"Group":"Carpet (ind.)","MIC":v})
for v in pore_ind:   plot_rows.append({"Group":"Pore (ind.)","MIC":v})
for ct in CTYPES:
    for v in groups_all[ct]:
        plot_rows.append({"Group": ct, "MIC": v})
plot_df = pd.DataFrame(plot_rows)
order_v = ["Carpet (ind.)","Carpet+Carpet","Pore (ind.)","Pore+Pore","Carpet+Pore"]
colors_v= ["#E07B54","#E07B54","#3BAF77","#3BAF77","#5B8DB8"]
sns.violinplot(data=plot_df, x="Group", y="MIC", hue="Group", order=order_v,
               palette=dict(zip(order_v,colors_v)), legend=False,
               inner="quart", linewidth=0.9, ax=ax)
ax.set_xlabel(""); ax.set_ylabel("MIC E. coli ATCC11775 (µmol/L)", fontsize=11)
ax.set_title("MIC: individual motifs vs combinations (Carpet/Pore)",fontweight="bold")
ax.set_xticks(range(len(order_v)))
ax.set_xticklabels(order_v, rotation=15, ha="right")
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR/"01_violin_mic.png", dpi=DPI, bbox_inches="tight"); plt.close()

# Fig 2: Scatter correlation per type
fig, axes = plt.subplots(1,3, figsize=(16,5), sharey=True)
fig.patch.set_facecolor("#fafafa")
for ax, ct in zip(axes, CTYPES):
    ax.set_facecolor("#fafafa")
    sub = df[df["Combination_type"]==ct].dropna(
        subset=["MIC_combined","MIC_mean_individual"])
    ax.scatter(sub["MIC_mean_individual"], sub["MIC_combined"],
               c=PALETTE[ct], alpha=0.3, s=15, edgecolors="none")
    lims=[min(sub["MIC_mean_individual"].min(),sub["MIC_combined"].min())-2,
          max(sub["MIC_mean_individual"].max(),sub["MIC_combined"].max())+2]
    ax.plot(lims,lims,"k--",lw=1,alpha=0.4,label="y=x (additive)")
    rho_s = corr_df.loc[corr_df["Combination_type"]==ct,"Spearman_rho"].values[0]
    ax.set_title(f"{ct}\nρ={rho_s:.3f}",fontweight="bold")
    ax.set_xlabel("Mean MIC individual (µmol/L)")
    ax.set_ylabel("MIC combined (µmol/L)")
    ax.spines[["top","right"]].set_visible(False)
fig.suptitle("MIC combined vs mean individual — below y=x = better than expected",
             fontsize=12,fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR/"02_scatter_corr.png", dpi=DPI, bbox_inches="tight"); plt.close()

# Fig 3: ΔMIC KDE
fig, ax = plt.subplots(figsize=(10,5))
fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
for ct in CTYPES:
    sub = df[df["Combination_type"]==ct]["MIC_delta"].dropna()
    sns.kdeplot(sub, fill=True, alpha=0.4, color=PALETTE[ct],
                label=f"{ct} (n={len(sub)})", ax=ax)
ax.axvline(0,color="black",lw=1.2,ls="--",alpha=0.6,label="no change")
ax.set_xlabel("ΔMIC = combined − mean(individual) (µmol/L)")
ax.set_ylabel("Density")
ax.set_title("Effect of combining motifs on MIC\nnegative = synergy, positive = interference",
             fontweight="bold")
ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR/"03_delta_mic_kde.png", dpi=DPI, bbox_inches="tight"); plt.close()

# Fig 4: Subsampling Cohen's d distribution
fig, ax = plt.subplots(figsize=(10,5))
fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
colors_mw = ["#9B59B6","#E67E22","#1ABC9C"]
for (key, res), color in zip(mw_results.items(), colors_mw):
    sns.kdeplot(res["d"], fill=True, alpha=0.4, color=color,
                label=f"{key}  d={np.mean(res['d']):.3f}±{np.std(res['d']):.3f}",
                ax=ax)
ax.axvline(0,color="black",lw=0.8,ls="--",alpha=0.4)
ax.set_xlabel("Cohen's d"); ax.set_ylabel("Density")
ax.set_title(f"Effect size distribution — {N_ITER} balanced subsamples\n"
             f"(n={n_draw} per group)", fontweight="bold")
ax.legend(fontsize=8.5); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR/"04_cohens_d_subsampling.png", dpi=DPI, bbox_inches="tight"); plt.close()

print(f"✅ Figures: {FIG_DIR}")

# =========================
# ANALYSIS 10: SCORE COMBINADO + HEATMAP TOP MOTIVOS
# + PEORES INDIVIDUALES EN COMBINACION
# =========================

# --- Score combinado: penaliza CV alto ---
# Score = MIC_mean_in_comb × (1 + CV/100)
# Menor score = mejor motivo (activo Y consistente)
motif_profile["Score_combined"] = (
    motif_profile["MIC_mean_in_comb"] * (1 + motif_profile["CV"] / 100)
).round(2)
motif_profile = motif_profile.sort_values("Score_combined")

print(f"\n=== Score combinado (activo + consistente) ===")
print(motif_profile[["Motif","Mechanism","MIC_individual",
                       "MIC_mean_in_comb","CV","Score_combined"]].head(15).to_string(index=False))

# --- Correlación individual vs en combinación ---
corr_ind_comb = motif_profile.dropna(subset=["MIC_individual","MIC_mean_in_comb"])
rho_ic, p_ic = spearmanr(corr_ind_comb["MIC_individual"],
                          corr_ind_comb["MIC_mean_in_comb"])
print(f"\n=== Correlación MIC_individual vs MIC_mean_in_comb ===")
print(f"  Spearman ρ={rho_ic:.3f}  p={p_ic:.4f}  n={len(corr_ind_comb)}")
print(f"  {'Alta correlación — individual predice comportamiento en combinación' if rho_ic>0.7 else 'Correlación moderada — hay motivos que cambian de rango al combinarse'}")

# peores individuales: ¿siguen siendo peores en combinación?
n_extreme = 20
best_ind  = motif_profile.nsmallest(n_extreme, "MIC_individual")["Motif"]
worst_ind = motif_profile.nlargest(n_extreme, "MIC_individual")["Motif"]
best_comb  = motif_profile[motif_profile["Motif"].isin(best_ind)]["MIC_mean_in_comb"].mean()
worst_comb = motif_profile[motif_profile["Motif"].isin(worst_ind)]["MIC_mean_in_comb"].mean()
print(f"\n  Top {n_extreme} best individual → mean MIC in comb:  {best_comb:.1f}")
print(f"  Top {n_extreme} worst individual → mean MIC in comb: {worst_comb:.1f}")
_, p_rank = mannwhitneyu(
    motif_profile[motif_profile["Motif"].isin(best_ind)]["MIC_mean_in_comb"].dropna(),
    motif_profile[motif_profile["Motif"].isin(worst_ind)]["MIC_mean_in_comb"].dropna(),
    alternative="two-sided")
print(f"  Mann-Whitney best vs worst in comb: p={p_rank:.4f}")

# update Excel with score
out_excel = OUT_DIR / "01_analysis_2cat.xlsx"
with pd.ExcelWriter(out_excel, engine="openpyxl", mode="a",
                    if_sheet_exists="replace") as writer:
    motif_profile.to_excel(writer, sheet_name="Motif_profile", index=False)
    pd.DataFrame([{"Spearman_rho": round(rho_ic,3),
                   "p": round(p_ic,6), "n": len(corr_ind_comb),
                   "mean_best_ind_in_comb": round(best_comb,2),
                   "mean_worst_ind_in_comb": round(worst_comb,2),
                   "p_best_vs_worst": round(p_rank,6)}
                  ]).to_excel(writer, sheet_name="Ind_vs_Comb_corr", index=False)

# update GraphPad Excel
out_gp = OUT_DIR / "02_graphpad_2cat.xlsx"
with pd.ExcelWriter(out_gp, engine="openpyxl", mode="a",
                    if_sheet_exists="replace") as writer:
    motif_profile[["Motif","Mechanism","MIC_individual",
                   "MIC_mean_in_comb","CV","Score_combined"]].to_excel(
        writer, sheet_name="GP_Fig6_motif_profile", index=False)
    pd.DataFrame({
        "MIC_individual": corr_ind_comb["MIC_individual"].values,
        "MIC_mean_in_comb": corr_ind_comb["MIC_mean_in_comb"].values,
        "Mechanism": corr_ind_comb["Mechanism"].values,
    }).to_excel(writer, sheet_name="GP_Fig9_ind_vs_comb", index=False)

# =========================
# FIGURE 5: Score combinado — bar chart top 30
# =========================

top30_score = motif_profile.head(30)
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
colors_bar = [PALETTE.get("Carpet+Carpet","#E07B54") if m=="Carpet"
              else PALETTE.get("Pore+Pore","#3BAF77")
              for m in top30_score["Mechanism"]]
bars = ax.bar(range(len(top30_score)), top30_score["Score_combined"],
              color=colors_bar, alpha=0.85, edgecolor="white")
ax.set_xticks(range(len(top30_score)))
ax.set_xticklabels(top30_score["Motif"], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Score combinado (lower = better)", fontsize=10)
ax.set_title("Top 30 motivos por score combinado (actividad × consistencia)\n"
             "penaliza CV alto", fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
patches = [mpatches.Patch(color=PALETTE["Carpet+Carpet"], label="Carpet"),
           mpatches.Patch(color=PALETTE["Pore+Pore"],     label="Pore")]
ax.legend(handles=patches, fontsize=9)
fig.tight_layout()
fig.savefig(FIG_DIR/"05_score_combinado_top30.png", dpi=DPI, bbox_inches="tight")
plt.close()

# =========================
# FIGURE 6: Heatmap top 20 motivos — MIC media de cada par
# =========================

top20_motifs = motif_profile.head(20)["Motif"].tolist()
hmap_data = pd.DataFrame(index=top20_motifs, columns=top20_motifs, dtype=float)
for m1 in top20_motifs:
    for m2 in top20_motifs:
        mask = ((df["Motif_1"]==m1) & (df["Motif_2"]==m2)) | \
               ((df["Motif_1"]==m2) & (df["Motif_2"]==m1))
        sub  = df[mask]["MIC_combined"].dropna()
        hmap_data.loc[m1, m2] = sub.mean() if len(sub) > 0 else np.nan

fig, ax = plt.subplots(figsize=(12, 10))
fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
sns.heatmap(hmap_data.astype(float), annot=True, fmt=".0f",
            cmap="YlOrRd_r", linewidths=0.3, ax=ax,
            annot_kws={"size": 7}, cbar_kws={"label":"MIC (µmol/L)"})
ax.set_title("Mean MIC per motif pair — top 20 motivos (score combinado)\n"
             "darker = more active", fontweight="bold")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
fig.tight_layout()
fig.savefig(FIG_DIR/"06_heatmap_top20_motifs.png", dpi=DPI, bbox_inches="tight")
plt.close()

# save heatmap to GraphPad
with pd.ExcelWriter(out_gp, engine="openpyxl", mode="a",
                    if_sheet_exists="replace") as writer:
    hmap_data.astype(float).to_excel(writer, sheet_name="GP_Fig10_heatmap_top20")

# =========================
# FIGURE 7: Individual vs combinación scatter (¿peores siguen siendo peores?)
# =========================

fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
for mech, color in [("Carpet", PALETTE["Carpet+Carpet"]),
                     ("Pore",   PALETTE["Pore+Pore"])]:
    sub = corr_ind_comb[corr_ind_comb["Mechanism"]==mech]
    ax.scatter(sub["MIC_individual"], sub["MIC_mean_in_comb"],
               c=color, alpha=0.7, s=50, edgecolors="white",
               linewidths=0.4, label=mech)
# identity line
lims = [corr_ind_comb[["MIC_individual","MIC_mean_in_comb"]].min().min()-5,
        corr_ind_comb[["MIC_individual","MIC_mean_in_comb"]].max().max()+5]
ax.plot(lims, lims, "k--", lw=1, alpha=0.4, label="y=x")
ax.annotate(f"ρ={rho_ic:.3f}  p={p_ic:.4f}\nn={len(corr_ind_comb)}",
            xy=(0.05,0.92), xycoords="axes fraction", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ddd", alpha=0.9))
ax.set_xlabel("MIC individual (µmol/L)", fontsize=11)
ax.set_ylabel("MIC media en combinaciones (µmol/L)", fontsize=11)
ax.set_title("¿Los mejores individuales son mejores en combinación?\n"
             "below y=x = mejor en combinación que solo",
             fontweight="bold")
ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR/"07_individual_vs_combination.png", dpi=DPI, bbox_inches="tight")
plt.close()

# =========================
# FIGURE 8: Orden — top motivos con preferencia de posición
# =========================

if len(sig_order) >= 5:
    top_order = sig_order.head(20).copy()
    top_order = top_order.sort_values("Delta_pos1_minus_pos2")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
    colors_ord = ["#3BAF77" if d < 0 else "#E07B54"
                  for d in top_order["Delta_pos1_minus_pos2"]]
    ax.barh(range(len(top_order)), top_order["Delta_pos1_minus_pos2"],
            color=colors_ord, alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(top_order)))
    ax.set_yticklabels(top_order["Motif"], fontsize=9)
    ax.axvline(0, color="black", lw=1, alpha=0.5)
    ax.set_xlabel("ΔMIC (pos1 − pos2)\nnegative = better in position 2", fontsize=10)
    ax.set_title("Motifs with significant position preference\n"
                 "green = prefers pos2, orange = prefers pos1",
                 fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR/"08_order_preference.png", dpi=DPI, bbox_inches="tight")
    plt.close()

print(f"\n✅ All figures saved to: {FIG_DIR}")
print(f"✅ Score combinado top motif: {motif_profile.iloc[0]['Motif']} "
      f"(score={motif_profile.iloc[0]['Score_combined']:.1f}  "
      f"CV={motif_profile.iloc[0]['CV']:.1f}%)")

# =========================
# FIGURE 9: Correlación familiar — MIC individual por familia
# =========================

if not fam_ind.empty and len(fam_ind) >= 5:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#fafafa")

    # scatter: MIC_ind_mean vs n_motifs por familia
    ax = axes[0]; ax.set_facecolor("#fafafa")
    ax.scatter(fam_ind["n_motifs"], fam_ind["MIC_ind_mean"],
               c="#5B8DB8", alpha=0.7, s=60, edgecolors="white", linewidths=0.4)
    for _, row in fam_ind[fam_ind["n_motifs"] >= 2].iterrows():
        ax.annotate(row["Family_ID"],
                    (row["n_motifs"], row["MIC_ind_mean"]),
                    fontsize=7, alpha=0.7, ha="left",
                    xytext=(3, 0), textcoords="offset points")
    rho_fn, p_fn = spearmanr(fam_ind["n_motifs"], fam_ind["MIC_ind_mean"])
    ax.annotate(f"ρ={rho_fn:.3f}  p={p_fn:.4f}",
                xy=(0.05, 0.92), xycoords="axes fraction", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ddd", alpha=0.9))
    ax.set_xlabel("Número de motivos en la familia", fontsize=10)
    ax.set_ylabel("MIC media individual (µmol/L)", fontsize=10)
    ax.set_title("¿Familias más grandes son más activas?", fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)

    # bar chart top 20 familias por MIC_ind_mean
    ax = axes[1]; ax.set_facecolor("#fafafa")
    top20_fam = fam_ind.head(20)
    ax.barh(range(len(top20_fam)), top20_fam["MIC_ind_mean"],
            color="#E07B54", alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(top20_fam)))
    ax.set_yticklabels(top20_fam["Family_ID"], fontsize=8.5)
    ax.set_xlabel("MIC media individual (µmol/L)", fontsize=10)
    ax.set_title("Top 20 familias más activas\n(MIC individual)", fontweight="bold")
    ax.axvline(fam_ind["MIC_ind_mean"].mean(), color="#333", lw=1.2, ls="--",
               label=f"Mean = {fam_ind['MIC_ind_mean'].mean():.1f}")
    ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)

    fig.suptitle("Análisis familiar — actividad individual (2cat)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR/"09_family_correlation.png", dpi=DPI, bbox_inches="tight")
    plt.close()

    # update GraphPad
    with pd.ExcelWriter(out_gp, engine="openpyxl", mode="a",
                        if_sheet_exists="replace") as writer:
        fam_ind.to_excel(writer, sheet_name="GP_Fig9_family_corr", index=False)

# =========================
# FIGURE 10: Verify Fig7 — individual vs combinación por mecanismo (enhanced)
# =========================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("#fafafa")

# left: scatter colored by mechanism
ax = axes[0]; ax.set_facecolor("#fafafa")
for mech, color in [("Carpet", PALETTE["Carpet+Carpet"]),
                     ("Pore",   PALETTE["Pore+Pore"])]:
    sub = corr_ind_comb[corr_ind_comb["Mechanism"]==mech]
    ax.scatter(sub["MIC_individual"], sub["MIC_mean_in_comb"],
               c=color, alpha=0.75, s=55, edgecolors="white",
               linewidths=0.4, label=f"{mech} (n={len(sub)})")
lims = [corr_ind_comb[["MIC_individual","MIC_mean_in_comb"]].min().min()-5,
        corr_ind_comb[["MIC_individual","MIC_mean_in_comb"]].max().max()+5]
ax.plot(lims, lims, "k--", lw=1, alpha=0.4, label="y=x")
ax.annotate(f"ρ={rho_ic:.3f}  p={p_ic:.4f}\nn={len(corr_ind_comb)}",
            xy=(0.05,0.92), xycoords="axes fraction", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ddd", alpha=0.9))
ax.set_xlabel("MIC individual (µmol/L)", fontsize=11)
ax.set_ylabel("MIC media en combinaciones (µmol/L)", fontsize=11)
ax.set_title("Individual vs combinación\ncoloreado por mecanismo", fontweight="bold")
ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)

# right: best vs worst boxplot
ax = axes[1]; ax.set_facecolor("#fafafa")
best_vals  = motif_profile[motif_profile["Motif"].isin(best_ind)]["MIC_mean_in_comb"].dropna()
worst_vals = motif_profile[motif_profile["Motif"].isin(worst_ind)]["MIC_mean_in_comb"].dropna()
bp = ax.boxplot([best_vals, worst_vals], patch_artist=True,
                medianprops=dict(color="black", lw=2),
                whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2),
                flierprops=dict(marker="o", markersize=4, alpha=0.4))
for patch, color in zip(bp["boxes"], [PALETTE["Carpet+Carpet"], PALETTE["Pore+Pore"]]):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax.set_xticks([1, 2])
ax.set_xticklabels([f"Top {n_extreme}\nbest individual",
                    f"Top {n_extreme}\nworst individual"], fontsize=10)
ax.set_ylabel("MIC media en combinaciones (µmol/L)", fontsize=10)
ax.set_title(f"Best vs worst individuales en combinación\np={p_rank:.4f}",
             fontweight="bold")
ax.spines[["top","right"]].set_visible(False)

fig.suptitle("¿Los mejores individuales son mejores en combinación? (2cat)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR/"10_individual_vs_comb_full.png", dpi=DPI, bbox_inches="tight")
plt.close()

print(f"✅ Additional figures saved (09, 10)")

# =========================
# ANALYSIS 11: BEST COMBINATIONS OVERALL vs CROSS-FAMILY
# =========================

print(f"\n=== Top 20 combinations overall (lowest MIC) ===")
top_overall = df.nsmallest(20, "MIC_combined")[[
    "Combination_ID","Motif_1","Mechanism_1","Family_1",
    "Motif_2","Mechanism_2","Family_2",
    "Combination_type","Sequence","MIC_combined",
    "MIC_mean_individual","MIC_delta","Same_family"
]]
print(top_overall[["Motif_1","Motif_2","Combination_type",
                    "MIC_combined","MIC_delta","Same_family"]].to_string(index=False))

print(f"\n=== Top 20 cross-family combinations (lowest MIC, different families) ===")
df_cross = df[~df["Same_family"]].copy()
top_cross = df_cross.nsmallest(20, "MIC_combined")[[
    "Combination_ID","Motif_1","Mechanism_1","Family_1",
    "Motif_2","Mechanism_2","Family_2",
    "Combination_type","Sequence","MIC_combined",
    "MIC_mean_individual","MIC_delta"
]]
print(top_cross[["Motif_1","Family_1","Motif_2","Family_2",
                  "Combination_type","MIC_combined","MIC_delta"]].to_string(index=False))

# compare same-family vs cross-family MIC
u_sf, p_sf = mannwhitneyu(
    df[df["Same_family"]]["MIC_combined"].dropna(),
    df[~df["Same_family"]]["MIC_combined"].dropna(),
    alternative="two-sided"
)
d_sf = (df[df["Same_family"]]["MIC_combined"].mean() -
        df[~df["Same_family"]]["MIC_combined"].mean()) / \
       np.sqrt((df[df["Same_family"]]["MIC_combined"].std()**2 +
                df[~df["Same_family"]]["MIC_combined"].std()**2) / 2)
print(f"\n  Same-family mean MIC:  {df[df['Same_family']]['MIC_combined'].mean():.1f}")
print(f"  Cross-family mean MIC: {df[~df['Same_family']]['MIC_combined'].mean():.1f}")
print(f"  Mann-Whitney p={p_sf:.4f}  Cohen_d={d_sf:.3f}")

# update excels
out_excel = OUT_DIR / "01_analysis_2cat.xlsx"
with pd.ExcelWriter(out_excel, engine="openpyxl", mode="a",
                    if_sheet_exists="replace") as writer:
    top_overall.to_excel(writer, sheet_name="Top20_overall",     index=False)
    top_cross.to_excel(writer,   sheet_name="Top20_crossfamily", index=False)
    pd.DataFrame([{
        "n_same_family":  n_same,
        "n_cross_family": len(df) - n_same,
        "mean_MIC_same":  round(df[df["Same_family"]]["MIC_combined"].mean(),2),
        "mean_MIC_cross": round(df[~df["Same_family"]]["MIC_combined"].mean(),2),
        "p_mw": round(p_sf,6), "Cohen_d": round(d_sf,3)
    }]).to_excel(writer, sheet_name="SameVsCross_family", index=False)

out_gp = OUT_DIR / "02_graphpad_2cat.xlsx"
with pd.ExcelWriter(out_gp, engine="openpyxl", mode="a",
                    if_sheet_exists="replace") as writer:
    pd.DataFrame({
        "Same_family":  df[df["Same_family"]]["MIC_combined"].dropna().values,
    }).to_excel(writer, sheet_name="GP_Fig11_samefam", index=False)
    pd.DataFrame({
        "Cross_family": df[~df["Same_family"]]["MIC_combined"].dropna().values,
    }).to_excel(writer, sheet_name="GP_Fig11_crossfam", index=False)
    top_cross.to_excel(writer, sheet_name="GP_Top20_cross", index=False)

# =========================
# FIGURE 11: Same-family vs cross-family MIC
# =========================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#fafafa")

# violin same vs cross
ax = axes[0]; ax.set_facecolor("#fafafa")
plot_sf = pd.DataFrame({
    "Group": ["Same family"]*df[df["Same_family"]]["MIC_combined"].dropna().shape[0] +
             ["Cross family"]*df[~df["Same_family"]]["MIC_combined"].dropna().shape[0],
    "MIC":   pd.concat([df[df["Same_family"]]["MIC_combined"].dropna(),
                        df[~df["Same_family"]]["MIC_combined"].dropna()]).values
})
sns.violinplot(data=plot_sf, x="Group", y="MIC", hue="Group",
               palette={"Same family": "#CCCCCC", "Cross family": "#5B8DB8"},
               inner="quart", linewidth=0.9, legend=False, ax=ax)
ax.set_xlabel("")
ax.set_ylabel("MIC E. coli ATCC11775 (µmol/L)", fontsize=10)
ax.set_title(f"Same-family vs cross-family combinations\np={p_sf:.4f}  d={d_sf:.3f}",
             fontweight="bold")
ax.spines[["top","right"]].set_visible(False)

# top 20 cross-family combinations bar
ax = axes[1]; ax.set_facecolor("#fafafa")
colors_top = [PALETTE.get(ct, "#888888") for ct in top_cross["Combination_type"]]
ax.barh(range(len(top_cross)), top_cross["MIC_combined"],
        color=colors_top, alpha=0.85, edgecolor="white")
ax.set_yticks(range(len(top_cross)))
ax.set_yticklabels([f"{r['Motif_1']}+{r['Motif_2']}"
                    for _, r in top_cross.iterrows()], fontsize=7.5)
ax.set_xlabel("MIC combined (µmol/L)", fontsize=10)
ax.set_title("Top 20 cross-family combinations\n(different families)",
             fontweight="bold")
patches = [mpatches.Patch(color=c, label=m) for m,c in PALETTE.items()]
ax.legend(handles=patches, fontsize=8)
ax.spines[["top","right"]].set_visible(False)

fig.suptitle("Cross-family vs same-family combinations (2cat)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR/"11_crossfamily_analysis.png", dpi=DPI, bbox_inches="tight")
plt.close()

print(f"✅ Cross-family analysis saved (Fig 11)")

# =========================
# ANALYSIS 12: BEST COMBINATIONS EXCLUDING DOMINANT MOTIF
# FAM_106 / KLLKKLLKLL dominates — find diverse candidates without it
# =========================

# identify dominant motifs (top by appearance in top combinations)
top_appearance = pd.concat([top_overall["Motif_1"], top_overall["Motif_2"]]
                           ).value_counts().head(3)
dominant_motifs = top_appearance.index.tolist()
dominant_families = [family_map.get(m) for m in dominant_motifs
                     if family_map.get(m)]

print(f"\n=== Analysis excluding dominant motif(s) ===")
print(f"  Dominant motifs in top20: {dominant_motifs}")
print(f"  Dominant families:        {dominant_families}")

df_nodom = df[
    ~df["Motif_1"].isin(dominant_motifs) &
    ~df["Motif_2"].isin(dominant_motifs)
].copy()
print(f"  Combinations after exclusion: {len(df_nodom)} (was {len(df)})")

print(f"\n  Top 20 cross-family WITHOUT dominant motif:")
top_nodom_cross = df_nodom[~df_nodom["Same_family"]].nsmallest(20, "MIC_combined")[[
    "Combination_ID","Motif_1","Mechanism_1","Family_1",
    "Motif_2","Mechanism_2","Family_2",
    "Combination_type","Sequence","MIC_combined",
    "MIC_mean_individual","MIC_delta"
]]
print(top_nodom_cross[["Motif_1","Family_1","Motif_2","Family_2",
                        "Combination_type","MIC_combined","MIC_delta"]
                       ].to_string(index=False))

# update excel
out_excel = OUT_DIR / "01_analysis_2cat.xlsx"
with pd.ExcelWriter(out_excel, engine="openpyxl", mode="a",
                    if_sheet_exists="replace") as writer:
    top_nodom_cross.to_excel(writer, sheet_name="Top20_nodominant", index=False)

out_gp = OUT_DIR / "02_graphpad_2cat.xlsx"
with pd.ExcelWriter(out_gp, engine="openpyxl", mode="a",
                    if_sheet_exists="replace") as writer:
    top_nodom_cross.to_excel(writer, sheet_name="GP_Top20_nodominant", index=False)

# =========================
# FIGURE 12: Top combinations without dominant motif
# =========================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor("#fafafa")

# bar chart top 20 no-dominant cross-family
ax = axes[0]; ax.set_facecolor("#fafafa")
colors_nd = [PALETTE.get(ct, "#888") for ct in top_nodom_cross["Combination_type"]]
ax.barh(range(len(top_nodom_cross)), top_nodom_cross["MIC_combined"],
        color=colors_nd, alpha=0.85, edgecolor="white")
ax.set_yticks(range(len(top_nodom_cross)))
ax.set_yticklabels([f"{r['Motif_1']}+{r['Motif_2']}"
                    for _, r in top_nodom_cross.iterrows()], fontsize=7.5)
ax.set_xlabel("MIC combined (µmol/L)", fontsize=10)
ax.set_title(f"Top 20 cross-family\nexcluding {dominant_motifs[0]}",
             fontweight="bold")
patches = [mpatches.Patch(color=c, label=m) for m,c in PALETTE.items()]
ax.legend(handles=patches, fontsize=8)
ax.spines[["top","right"]].set_visible(False)

# ΔMIC scatter for no-dominant combinations
ax = axes[1]; ax.set_facecolor("#fafafa")
for ct, color in PALETTE.items():
    sub = df_nodom[df_nodom["Combination_type"]==ct].dropna(
        subset=["MIC_combined","MIC_mean_individual"])
    if len(sub) < 2: continue
    ax.scatter(sub["MIC_mean_individual"], sub["MIC_combined"],
               c=color, alpha=0.25, s=12, edgecolors="none",
               label=f"{ct} (n={len(sub)})")
lims = [df_nodom[["MIC_combined","MIC_mean_individual"]].min().min()-2,
        df_nodom[["MIC_combined","MIC_mean_individual"]].max().max()+2]
ax.plot(lims, lims, "k--", lw=1, alpha=0.4, label="y=x")
rho_nd, p_nd = spearmanr(
    df_nodom.dropna(subset=["MIC_combined","MIC_mean_individual"])["MIC_mean_individual"],
    df_nodom.dropna(subset=["MIC_combined","MIC_mean_individual"])["MIC_combined"]
)
ax.annotate(f"ρ={rho_nd:.3f}  p={p_nd:.4f}",
            xy=(0.05,0.92), xycoords="axes fraction", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ddd", alpha=0.9))
ax.set_xlabel("Mean MIC individual (µmol/L)", fontsize=10)
ax.set_ylabel("MIC combined (µmol/L)", fontsize=10)
ax.set_title(f"Combined vs individual\nexcluding {dominant_motifs[0]}",
             fontweight="bold")
ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)

fig.suptitle(f"Diversity analysis — excluding dominant motif ({dominant_motifs[0]})",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR/"12_nodominant_analysis.png", dpi=DPI, bbox_inches="tight")
plt.close()

print(f"✅ No-dominant analysis saved (Fig 12)")