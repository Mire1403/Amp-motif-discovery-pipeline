"""
Analysis of motif combinations — 3 categories (Carpet / Toroidal_pore / Barrel_stave).

Analyses:
  1. ΔMIC: combined vs mean individual
  2. Subsampling (800 iter, balanced): Kruskal-Wallis + pairwise MW (6 pairs)
  3. Order effect: A+B vs B+A (paired t-test)
  4. Correlation: MIC_combined vs mean(individual) per combination type
  5. 3×3 heatmap matrix of mean MIC per combination pair

Input: 04_merged_predicted_MICs.csv (3-category combinations)
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

COMB_FILE = (PROJECT_ROOT / "results/13_motif_combinations/01b_combinations_3cat"
             / "04_merged_predicted_MICs.csv")
MOTIF_MIC_FILE = (PROJECT_ROOT / "results/12_apex_prediction/02_results"
                  / "02_motif_mic_summary_merged.csv")
CONSENSUS_3CAT = (PROJECT_ROOT / "results/13_motif_combinations/00_consensus_motifs"
                  / "01_consensus_motifs_3cat.xlsx")

OUT_DIR = PROJECT_ROOT / "results/13_motif_combinations/02_analysis_3cat"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

N_ITER  = 800
RNG     = np.random.default_rng(42)
DPI     = 300

MECHS   = ["Carpet", "Toroidal_pore", "Barrel_stave"]
PALETTE_MECH = {
    "Carpet":        "#E07B54",
    "Toroidal_pore": "#4C72B0",
    "Barrel_stave":  "#3BAF77",
}

sns.set_theme(style="whitegrid", font_scale=1.1)

# =========================
# LOAD
# =========================

df = pd.read_csv(COMB_FILE)
df["MIC_combined"]        = pd.to_numeric(df["MIC_combined"],        errors="coerce")
df["MIC_mean_individual"] = pd.to_numeric(df["MIC_mean_individual"], errors="coerce")
df["MIC_delta"]           = pd.to_numeric(df["MIC_delta"],           errors="coerce")

motif_mic  = pd.read_csv(MOTIF_MIC_FILE)

# load 3cat consensus to get mechanism labels
consensus  = pd.read_excel(CONSENSUS_3CAT, sheet_name="Motif_summary")
mech3_map  = dict(zip(consensus["Motif_representative"], consensus["Mechanism"]))

# add 3-cat mechanism to combinations
df["Mech3_1"] = df["Motif_1"].map(mech3_map)
df["Mech3_2"] = df["Motif_2"].map(mech3_map)
df["Comb3_type"] = df.apply(
    lambda r: "+".join(sorted([str(r["Mech3_1"]), str(r["Mech3_2"])])), axis=1
)

ctypes_3 = sorted(df["Comb3_type"].dropna().unique().tolist())
print(f"Loaded {len(df)} combinations")
print(f"Combination types (3cat):")
for ct in ctypes_3:
    n = (df["Comb3_type"]==ct).sum()
    print(f"  {ct:<35} n={n}")

# add Family_ID
family_map = motif_mic.set_index("Motif_representative")["Family_ID"].to_dict() \
             if "Family_ID" in motif_mic.columns else {}
df["Family_1"] = df["Motif_1"].map(family_map)
df["Family_2"] = df["Motif_2"].map(family_map)
df["Same_family"] = (df["Family_1"] == df["Family_2"]) & df["Family_1"].notna()
n_same = df["Same_family"].sum()
print(f"  Same-family combinations: {n_same} ({100*n_same/len(df):.1f}%)")

# individual MIC by mechanism
ind_mic = {}
for mech in MECHS:
    sub = motif_mic[motif_mic["Dominant_reduced"].str.contains(
        mech.split("_")[0], na=False)]["MIC_mean"].dropna()
    ind_mic[mech] = sub

# =========================
# ANALYSIS 1: ΔMIC
# =========================

print(f"\n=== ΔMIC by combination type (3cat) ===")
delta_rows = []
for ct in ctypes_3:
    sub_d = df[df["Comb3_type"]==ct]["MIC_delta"].dropna()
    sub_i = df[df["Comb3_type"]==ct]["MIC_mean_individual"].dropna()
    if len(sub_d) == 0: continue
    syn  = (sub_d < -(sub_i * 0.10)).sum()
    intf = (sub_d >  (sub_i * 0.10)).sum()
    n    = len(sub_d)
    print(f"  {ct:<35} mean_Δ={sub_d.mean():.2f}  "
          f"synergy={100*syn/n:.1f}%  interference={100*intf/n:.1f}%")
    delta_rows.append({"Combination_type": ct, "n": n,
                       "mean_delta": round(sub_d.mean(),3),
                       "median_delta": round(sub_d.median(),3),
                       "std_delta": round(sub_d.std(),3),
                       "pct_synergy": round(100*syn/n,1),
                       "pct_interference": round(100*intf/n,1)})
delta_df = pd.DataFrame(delta_rows)

# =========================
# ANALYSIS 2: SUBSAMPLING — balanced KW + pairwise MW
# =========================

groups_all = {}
for ct in ctypes_3:
    vals = df[df["Comb3_type"]==ct]["MIC_combined"].dropna().values
    if len(vals) > 0:
        groups_all[ct] = vals

ctypes_present = list(groups_all.keys())
n_draw = min(len(v) for v in groups_all.values())
print(f"\n=== Subsampled analysis ({N_ITER} iter, n={n_draw} per group) ===")

kw_stats, kw_pvals = [], []
mw_results = {f"{ct1} vs {ct2}": {"U":[], "p":[], "d":[]}
              for ct1, ct2 in _comb(ctypes_present, 2)}

for _ in range(N_ITER):
    samples = {ct: RNG.choice(v, n_draw, replace=False)
               for ct, v in groups_all.items()}
    if len(samples) >= 2:
        h, p = kruskal(*samples.values())
        kw_stats.append(h); kw_pvals.append(p)

        for ct1, ct2 in _comb(ctypes_present, 2):
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
    if not res["d"]: continue
    print(f"  {key}:")
    print(f"    d={np.mean(res['d']):.3f}±{np.std(res['d']):.3f}  "
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

kw_df = pd.DataFrame([{
    "H_mean": round(np.mean(kw_stats),3),
    "H_std":  round(np.std(kw_stats),3),
    "p_mean": round(np.mean(kw_pvals),6),
    "p_std":  round(np.std(kw_pvals),6),
    "n_iter": N_ITER, "n_per_group": n_draw
}])

# =========================
# ANALYSIS 3: ORDER EFFECT
# =========================

df["Pair_key"] = df.apply(
    lambda r: "_".join(sorted([str(r["Motif_1"]), str(r["Motif_2"])])), axis=1)
pairs = df.groupby("Pair_key").filter(lambda g: len(g)==2)
pair_a = pairs.groupby("Pair_key").apply(
    lambda g: g.sort_values("Combination_ID").iloc[0]["MIC_combined"])
pair_b = pairs.groupby("Pair_key").apply(
    lambda g: g.sort_values("Combination_ID").iloc[1]["MIC_combined"])
valid = pair_a.notna() & pair_b.notna()
t_stat, p_order = ttest_rel(pair_a[valid], pair_b[valid])
order_diff = (pair_a[valid]-pair_b[valid]).abs()
print(f"\n=== Order effect ===  t={t_stat:.3f}  p={p_order:.4f}  "
      f"mean|Δ|={order_diff.mean():.2f}")
order_df = pd.DataFrame([{"n_pairs": valid.sum(),
                           "t_stat": round(t_stat,3), "p": round(p_order,6),
                           "mean_abs_delta": round(order_diff.mean(),3)}])

# =========================
# ANALYSIS 4: CORRELATION
# =========================

corr_rows = []
print(f"\n=== Correlation: combined vs mean individual ===")
for ct in ctypes_present:
    sub = df[df["Comb3_type"]==ct].dropna(
        subset=["MIC_combined","MIC_mean_individual"])
    if len(sub) < 5: continue
    rho, p_rho = spearmanr(sub["MIC_combined"], sub["MIC_mean_individual"])
    r,   p_r   = pearsonr( sub["MIC_combined"], sub["MIC_mean_individual"])
    print(f"  {ct:<35} ρ={rho:.3f} p={p_rho:.4f}  n={len(sub)}")
    corr_rows.append({"Combination_type": ct, "n": len(sub),
                      "Spearman_rho": round(rho,3), "Spearman_p": round(p_rho,6),
                      "Pearson_r": round(r,3),      "Pearson_p": round(p_r,6)})
corr_df = pd.DataFrame(corr_rows)

# =========================
# HEATMAP DATA (computed here, used in figures and GraphPad)
# =========================

mechs_present = [m for m in MECHS if any(m in ct for ct in ctypes_present)]
hmap = pd.DataFrame(index=mechs_present, columns=mechs_present, dtype=float)
for m1 in mechs_present:
    for m2 in mechs_present:
        key = "+".join(sorted([m1, m2]))
        sub = df[df["Comb3_type"]==key]["MIC_combined"].dropna()
        hmap.loc[m1, m2] = sub.mean() if len(sub) > 0 else np.nan

# =========================
# ANALYSIS 5: DIRECT COMPARISON combined vs individual
# =========================

print(f"\n=== Direct comparison: combined vs individual ===")

# individual MIC by 3-category mechanism
ind_mic_3 = {}
for mech in MECHS:
    sub = motif_mic[motif_mic["Dominant_mechanism"]==mech]["MIC_mean"].dropna() \
          if "Dominant_mechanism" in motif_mic.columns \
          else motif_mic[motif_mic["Dominant_reduced"].str.contains(
              mech.split("_")[0], na=False)]["MIC_mean"].dropna()
    ind_mic_3[mech] = sub.values

direct_rows = []
for ct in ctypes_present:
    mechs_in = ct.split("+")
    for mech in set(mechs_in):
        if mech not in ind_mic_3 or len(ind_mic_3[mech]) == 0: continue
        g1 = df[df["Comb3_type"]==ct]["MIC_combined"].dropna().values
        g2 = ind_mic_3[mech]
        u, p_mw = mannwhitneyu(g1, g2, alternative="two-sided")
        d = (g1.mean()-g2.mean()) / np.sqrt((g1.std()**2+g2.std()**2)/2)
        direction = "↓ lower" if g1.mean() < g2.mean() else "↑ higher"
        label = f"{ct} vs {mech}_individual"
        print(f"  {label}: p={p_mw:.4f}  d={d:.3f}  {direction}")
        direct_rows.append({
            "Comparison": label, "n_combined": len(g1), "n_individual": len(g2),
            "mean_combined": round(g1.mean(),2), "mean_individual": round(g2.mean(),2),
            "U": round(u,1), "p_mw": round(p_mw,6), "Cohen_d": round(d,3),
            "direction": direction
        })
direct_df = pd.DataFrame(direct_rows)

# top 20 most active combinations
top20 = df.nsmallest(20, "MIC_combined")[
    ["Combination_ID","Motif_1","Mech3_1","Motif_2","Mech3_2",
     "Comb3_type","Sequence","MIC_combined","MIC_mean_individual","MIC_delta"]
]

# =========================
# ANALYSIS 6: CONSISTENTLY GOOD MOTIFS
# =========================

print(f"\n=== Consistently good motifs (mean MIC in all combinations) ===")

all_motifs = pd.concat([df["Motif_1"], df["Motif_2"]]).unique()
motif_rows = []
for motif in all_motifs:
    mask = (df["Motif_1"]==motif) | (df["Motif_2"]==motif)
    sub  = df[mask]["MIC_combined"].dropna()
    if len(sub) < 3: continue
    mech = mech3_map.get(motif, "")
    ind_mic_val = motif_mic.loc[
        motif_mic["Motif_representative"]==motif, "MIC_mean"].values
    motif_rows.append({
        "Motif":              motif,
        "Mechanism_3cat":     mech,
        "n_combinations":     len(sub),
        "MIC_mean_in_comb":   round(sub.mean(),2),
        "MIC_median_in_comb": round(sub.median(),2),
        "MIC_std_in_comb":    round(sub.std(),2),
        "CV":                 round(sub.std()/sub.mean()*100,1) if sub.mean()>0 else np.nan,
        "MIC_individual":     round(float(ind_mic_val[0]),2) if len(ind_mic_val)>0 else np.nan,
    })
motif_profile = pd.DataFrame(motif_rows).sort_values("MIC_mean_in_comb")
print(f"  Top 10 most active motifs in combinations:")
print(motif_profile[["Motif","Mechanism_3cat","MIC_individual",
                      "MIC_mean_in_comb","CV"]].head(10).to_string(index=False))

# =========================
# ANALYSIS 7: FAMILY-LEVEL MIC
# =========================

print(f"\n=== Family-level MIC analysis ===")
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
# ANALYSIS 8: ORDER PATTERN
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
        "Motif":                motif,
        "Mechanism_3cat":       mech3_map.get(motif,""),
        "MIC_as_pos1":          round(as_m1.mean(),2),
        "MIC_as_pos2":          round(as_m2.mean(),2),
        "Delta_pos1_minus_pos2":round(delta,2),
        "p_mw":                 round(p,6),
        "Prefers":              "Position_2" if delta > 0 else "Position_1",
    })
order_pattern_df = pd.DataFrame(order_pattern_rows)
sig_order = order_pattern_df[order_pattern_df["p_mw"] < 0.05].sort_values("p_mw")
print(f"  Motifs with significant position preference (p<0.05): {len(sig_order)}")
if len(sig_order) > 0:
    print(sig_order[["Motif","Mechanism_3cat","MIC_as_pos1",
                      "MIC_as_pos2","Prefers","p_mw"]].head(10).to_string(index=False))

# =========================
# ANALYSIS 9: NOISY MOTIFS FILTER
# =========================

CV_THRESHOLD = 40.0
print(f"\n=== Noisy motifs filter (CV > {CV_THRESHOLD}%) ===")
noisy_motifs = set(motif_profile[motif_profile["CV"] > CV_THRESHOLD]["Motif"])
print(f"  Noisy motifs removed: {len(noisy_motifs)}")
df_clean = df[~df["Motif_1"].isin(noisy_motifs) &
              ~df["Motif_2"].isin(noisy_motifs)].copy()
print(f"  Combinations after filter: {len(df_clean)} (was {len(df)})")

groups_clean = {ct: df_clean[df_clean["Comb3_type"]==ct]["MIC_combined"].dropna().values
                for ct in ctypes_present}
n_draw_clean = min((len(v) for v in groups_clean.values() if len(v)>0), default=0)

filter_kw_df = pd.DataFrame()
if n_draw_clean >= 10:
    kw_clean, kw_pclean = [], []
    for _ in range(N_ITER):
        samp = {ct: RNG.choice(v, n_draw_clean, replace=False)
                for ct, v in groups_clean.items() if len(v)>=n_draw_clean}
        if len(samp) >= 2:
            h, p = kruskal(*samp.values())
            kw_clean.append(h); kw_pclean.append(p)
    print(f"  KW after filter: H={np.mean(kw_clean):.3f}±{np.std(kw_clean):.3f}  "
          f"p={np.mean(kw_pclean):.4f}  (n={n_draw_clean}/group)")
    filter_kw_df = pd.DataFrame([{
        "CV_threshold": CV_THRESHOLD,
        "n_noisy_removed": len(noisy_motifs),
        "n_combinations_clean": len(df_clean),
        "n_per_group": n_draw_clean,
        "H_mean": round(np.mean(kw_clean),3),
        "p_mean": round(np.mean(kw_pclean),6),
    }])

# =========================
# SAVE ANALYSIS EXCEL
# =========================

out_excel = OUT_DIR / "01_analysis_3cat.xlsx"
with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
    df.to_excel(writer,              sheet_name="Combinations_MIC",  index=False)
    delta_df.to_excel(writer,        sheet_name="Delta_MIC_summary", index=False)
    direct_df.to_excel(writer,       sheet_name="Direct_comparison", index=False)
    kw_df.to_excel(writer,           sheet_name="KruskalWallis_sub", index=False)
    sub_df.to_excel(writer,          sheet_name="MannWhitney_sub",   index=False)
    corr_df.to_excel(writer,         sheet_name="Correlation",       index=False)
    order_df.to_excel(writer,        sheet_name="Order_effect",      index=False)
    top20.to_excel(writer,           sheet_name="Top20_active",      index=False)
    motif_profile.to_excel(writer,   sheet_name="Motif_profile",     index=False)
    if not fam_ind.empty:
        fam_ind.to_excel(writer,     sheet_name="Family_MIC_ind",    index=False)
    order_pattern_df.sort_values("p_mw").to_excel(
        writer,                      sheet_name="Order_pattern",     index=False)
    if not filter_kw_df.empty:
        filter_kw_df.to_excel(writer,sheet_name="Filter_noisy_KW",  index=False)
        df_clean.to_excel(writer,    sheet_name="Combinations_clean",index=False)
    pd.DataFrame({"H": kw_stats, "p": kw_pvals}).to_excel(
        writer,                      sheet_name="KW_iterations",     index=False)

print(f"\n✅ Analysis Excel: {out_excel}")

# =========================
# SAVE GRAPHPAD EXCEL
# =========================

def to_wide(series_dict):
    return pd.DataFrame({k: pd.Series(v).reset_index(drop=True)
                         for k, v in series_dict.items()})

out_gp = OUT_DIR / "02_graphpad_3cat.xlsx"
with pd.ExcelWriter(out_gp, engine="openpyxl") as writer:

    # GP Fig 1 — violin by combination type
    to_wide({ct: groups_all[ct] for ct in ctypes_present}).to_excel(
        writer, sheet_name="GP_Fig1_violin", index=False)

    # GP Fig 2 — scatter correlation per type
    for ct in ctypes_present:
        sub = df[df["Comb3_type"]==ct].dropna(
            subset=["MIC_combined","MIC_mean_individual"])
        if len(sub) < 5: continue
        safe = ("GP2_" + ct.replace("+","_"))[:31]
        pd.DataFrame({
            "MIC_mean_individual": sub["MIC_mean_individual"].values,
            "MIC_combined":        sub["MIC_combined"].values,
        }).to_excel(writer, sheet_name=safe, index=False)

    # GP Fig 3 — ΔMIC by type
    to_wide({ct: df[df["Comb3_type"]==ct]["MIC_delta"].dropna().values
             for ct in ctypes_present}).to_excel(
        writer, sheet_name="GP_Fig3_delta_MIC", index=False)

    # GP Fig 4 — 3×3 heatmap data
    hmap.astype(float).to_excel(writer, sheet_name="GP_Fig4_heatmap_3x3")

    # GP Fig 5 — order effect (paired)
    pd.DataFrame({
        "MIC_order_A": pair_a[valid].values,
        "MIC_order_B": pair_b[valid].values,
    }).to_excel(writer, sheet_name="GP_Fig5_order", index=False)

    # GP Fig 6 — Cohen's d subsampling
    pd.DataFrame({k.replace(" vs ","_vs_")[:28]: pd.Series(v["d"])
                  for k, v in mw_results.items() if v["d"]}).to_excel(
        writer, sheet_name="GP_Fig6_cohens_d", index=False)

    # GP Fig 7 — motif profile
    motif_profile[["Motif","Mechanism_3cat","MIC_individual",
                   "MIC_mean_in_comb","CV"]].to_excel(
        writer, sheet_name="GP_Fig7_motif_profile", index=False)

    # GP Fig 8 — order pattern
    if len(sig_order) > 0:
        sig_order[["Motif","Mechanism_3cat","MIC_as_pos1",
                   "MIC_as_pos2"]].to_excel(
            writer, sheet_name="GP_Fig8_order_pattern", index=False)

    # GP Fig 9 — family MIC
    if not fam_ind.empty:
        fam_ind.to_excel(writer, sheet_name="GP_Fig9_family_MIC", index=False)

    # stats for annotations
    direct_df.to_excel(writer, sheet_name="GP_Stats_direct",     index=False)
    sub_df.to_excel(writer,    sheet_name="GP_Stats_subsampled", index=False)

print(f"✅ GraphPad Excel: {out_gp}")

# =========================
# FIGURES
# =========================

# Fig 1: Violin by 3cat combination type
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
plot_rows = []
for ct in ctypes_present:
    for v in groups_all[ct]:
        plot_rows.append({"Group": ct, "MIC": v})
plot_df = pd.DataFrame(plot_rows)
sns.violinplot(data=plot_df, x="Group", y="MIC", hue="Group",
               order=ctypes_present, legend=False,
               inner="quart", linewidth=0.9, ax=ax)
ax.set_xlabel(""); ax.set_ylabel("MIC E. coli ATCC11775 (µmol/L)", fontsize=11)
ax.set_title("MIC distribution by 3-category combination type",fontweight="bold")
ax.set_xticks(range(len(ctypes_present)))
ax.set_xticklabels(ctypes_present, rotation=25, ha="right", fontsize=8)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR/"01_violin_3cat.png", dpi=DPI, bbox_inches="tight"); plt.close()

# Fig 2: Heatmap 3×3 — mean MIC per mechanism combination
fig, ax = plt.subplots(figsize=(6, 5))
fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
sns.heatmap(hmap.astype(float), annot=True, fmt=".1f", cmap="YlOrRd_r",
            linewidths=0.5, ax=ax, annot_kws={"size":12,"weight":"bold"})
ax.set_title("Mean MIC (µmol/L) — 3×3 mechanism matrix\nlower = more active",
             fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR/"02_heatmap_3x3.png", dpi=DPI, bbox_inches="tight"); plt.close()

# Fig 3: ΔMIC KDE
fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
cmap = plt.cm.tab10
for i, ct in enumerate(ctypes_present):
    sub = df[df["Comb3_type"]==ct]["MIC_delta"].dropna()
    if len(sub) < 5: continue
    sns.kdeplot(sub, fill=True, alpha=0.35, color=cmap(i),
                label=f"{ct} (n={len(sub)})", ax=ax)
ax.axvline(0,color="black",lw=1.2,ls="--",alpha=0.6,label="no change")
ax.set_xlabel("ΔMIC = combined − mean(individual) (µmol/L)")
ax.set_ylabel("Density")
ax.set_title("ΔMIC distribution — 3-category combinations",fontweight="bold")
ax.legend(fontsize=7.5,framealpha=0.8)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR/"03_delta_mic_kde.png", dpi=DPI, bbox_inches="tight"); plt.close()

# Fig 4: Cohen's d subsampling
fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
cmap2 = plt.cm.Set2
for i, (key, res) in enumerate(mw_results.items()):
    if not res["d"]: continue
    sns.kdeplot(res["d"], fill=True, alpha=0.35, color=cmap2(i),
                label=f"{key}  d={np.mean(res['d']):.3f}±{np.std(res['d']):.3f}",
                ax=ax)
ax.axvline(0,color="black",lw=0.8,ls="--",alpha=0.4)
ax.set_xlabel("Cohen's d"); ax.set_ylabel("Density")
ax.set_title(f"Effect size — {N_ITER} subsamples (n={n_draw}/group)",fontweight="bold")
ax.legend(fontsize=7.5); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR/"04_cohens_d_3cat.png", dpi=DPI, bbox_inches="tight"); plt.close()

print(f"✅ Figures: {FIG_DIR}")

# =========================
# ANALYSIS 10 + FIGURES 5-8 (3cat)
# =========================

# --- Score combinado ---
motif_profile["Score_combined"] = (
    motif_profile["MIC_mean_in_comb"] * (1 + motif_profile["CV"] / 100)
).round(2)
motif_profile = motif_profile.sort_values("Score_combined")

print(f"\n=== Score combinado (activo + consistente) ===")
print(motif_profile[["Motif","Mechanism_3cat","MIC_individual",
                       "MIC_mean_in_comb","CV","Score_combined"]].head(15).to_string(index=False))

# --- Correlación individual vs combinación ---
corr_ind_comb = motif_profile.dropna(subset=["MIC_individual","MIC_mean_in_comb"])
rho_ic, p_ic  = spearmanr(corr_ind_comb["MIC_individual"],
                           corr_ind_comb["MIC_mean_in_comb"])
print(f"\n=== Correlación MIC_individual vs MIC_mean_in_comb ===")
print(f"  Spearman ρ={rho_ic:.3f}  p={p_ic:.4f}  n={len(corr_ind_comb)}")

n_extreme  = 20
best_ind   = motif_profile.nsmallest(n_extreme, "MIC_individual")["Motif"]
worst_ind  = motif_profile.nlargest(n_extreme, "MIC_individual")["Motif"]
best_comb  = motif_profile[motif_profile["Motif"].isin(best_ind)]["MIC_mean_in_comb"].mean()
worst_comb = motif_profile[motif_profile["Motif"].isin(worst_ind)]["MIC_mean_in_comb"].mean()
_, p_rank  = mannwhitneyu(
    motif_profile[motif_profile["Motif"].isin(best_ind)]["MIC_mean_in_comb"].dropna(),
    motif_profile[motif_profile["Motif"].isin(worst_ind)]["MIC_mean_in_comb"].dropna(),
    alternative="two-sided")
print(f"  Top {n_extreme} best individual  → mean MIC in comb: {best_comb:.1f}")
print(f"  Top {n_extreme} worst individual → mean MIC in comb: {worst_comb:.1f}")
print(f"  Mann-Whitney best vs worst in comb: p={p_rank:.4f}")

# update Excel
out_excel = OUT_DIR / "01_analysis_3cat.xlsx"
with pd.ExcelWriter(out_excel, engine="openpyxl", mode="a",
                    if_sheet_exists="replace") as writer:
    motif_profile.to_excel(writer, sheet_name="Motif_profile", index=False)
    pd.DataFrame([{"Spearman_rho": round(rho_ic,3), "p": round(p_ic,6),
                   "n": len(corr_ind_comb),
                   "mean_best_ind_in_comb":  round(best_comb,2),
                   "mean_worst_ind_in_comb": round(worst_comb,2),
                   "p_best_vs_worst": round(p_rank,6)}
                  ]).to_excel(writer, sheet_name="Ind_vs_Comb_corr", index=False)

# update GraphPad Excel
out_gp = OUT_DIR / "02_graphpad_3cat.xlsx"
with pd.ExcelWriter(out_gp, engine="openpyxl", mode="a",
                    if_sheet_exists="replace") as writer:
    motif_profile[["Motif","Mechanism_3cat","MIC_individual",
                   "MIC_mean_in_comb","CV","Score_combined"]].to_excel(
        writer, sheet_name="GP_Fig7_motif_profile", index=False)
    pd.DataFrame({
        "MIC_individual":   corr_ind_comb["MIC_individual"].values,
        "MIC_mean_in_comb": corr_ind_comb["MIC_mean_in_comb"].values,
        "Mechanism_3cat":   corr_ind_comb["Mechanism_3cat"].values,
    }).to_excel(writer, sheet_name="GP_Fig10_ind_vs_comb", index=False)

# --- Fig 5: Score combinado top 30 ---
top30_score = motif_profile.head(30)
PALETTE_3 = {"Carpet": "#E07B54", "Toroidal_pore": "#4C72B0",
             "Barrel_stave": "#3BAF77"}
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
colors_bar = [PALETTE_3.get(m,"#cccccc") for m in top30_score["Mechanism_3cat"]]
ax.bar(range(len(top30_score)), top30_score["Score_combined"],
       color=colors_bar, alpha=0.85, edgecolor="white")
ax.set_xticks(range(len(top30_score)))
ax.set_xticklabels(top30_score["Motif"], rotation=45, ha="right", fontsize=7.5)
ax.set_ylabel("Score combinado (lower = better)", fontsize=10)
ax.set_title("Top 30 motivos — score combinado (actividad × consistencia)\n"
             "3-category mechanism", fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
patches = [mpatches.Patch(color=c, label=m) for m,c in PALETTE_3.items()]
ax.legend(handles=patches, fontsize=9)
fig.tight_layout()
fig.savefig(FIG_DIR/"05_score_combinado_top30.png", dpi=DPI, bbox_inches="tight")
plt.close()

# --- Fig 6: Heatmap top 20 motivos ---
top20_motifs = motif_profile.head(20)["Motif"].tolist()
hmap_top = pd.DataFrame(index=top20_motifs, columns=top20_motifs, dtype=float)
for m1 in top20_motifs:
    for m2 in top20_motifs:
        mask = ((df["Motif_1"]==m1) & (df["Motif_2"]==m2)) | \
               ((df["Motif_1"]==m2) & (df["Motif_2"]==m1))
        sub  = df[mask]["MIC_combined"].dropna()
        hmap_top.loc[m1, m2] = sub.mean() if len(sub) > 0 else np.nan

fig, ax = plt.subplots(figsize=(12, 10))
fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
sns.heatmap(hmap_top.astype(float), annot=True, fmt=".0f",
            cmap="YlOrRd_r", linewidths=0.3, ax=ax,
            annot_kws={"size": 7}, cbar_kws={"label":"MIC (µmol/L)"})
ax.set_title("Mean MIC per pair — top 20 motivos (score combinado, 3cat)\n"
             "darker = more active", fontweight="bold")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
fig.tight_layout()
fig.savefig(FIG_DIR/"06_heatmap_top20_motifs.png", dpi=DPI, bbox_inches="tight")
plt.close()

with pd.ExcelWriter(out_gp, engine="openpyxl", mode="a",
                    if_sheet_exists="replace") as writer:
    hmap_top.astype(float).to_excel(writer, sheet_name="GP_Fig11_heatmap_top20")

# --- Fig 7: Individual vs combinación scatter ---
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
for mech, color in PALETTE_3.items():
    sub = corr_ind_comb[corr_ind_comb["Mechanism_3cat"]==mech]
    if sub.empty: continue
    ax.scatter(sub["MIC_individual"], sub["MIC_mean_in_comb"],
               c=color, alpha=0.75, s=50, edgecolors="white",
               linewidths=0.4, label=mech)
lims = [corr_ind_comb[["MIC_individual","MIC_mean_in_comb"]].min().min()-5,
        corr_ind_comb[["MIC_individual","MIC_mean_in_comb"]].max().max()+5]
ax.plot(lims, lims, "k--", lw=1, alpha=0.4, label="y=x")
ax.annotate(f"ρ={rho_ic:.3f}  p={p_ic:.4f}\nn={len(corr_ind_comb)}",
            xy=(0.05,0.92), xycoords="axes fraction", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ddd", alpha=0.9))
ax.set_xlabel("MIC individual (µmol/L)", fontsize=11)
ax.set_ylabel("MIC media en combinaciones (µmol/L)", fontsize=11)
ax.set_title("¿Mejores individuales son mejores en combinación? (3cat)",
             fontweight="bold")
ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR/"07_individual_vs_combination.png", dpi=DPI, bbox_inches="tight")
plt.close()

# --- Fig 8: Orden — top motivos con preferencia de posición ---
if len(sig_order) >= 5:
    top_order = sig_order.head(20).sort_values("Delta_pos1_minus_pos2")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
    colors_ord = [PALETTE_3.get(m,"#cccccc")
                  for m in top_order["Mechanism_3cat"]]
    ax.barh(range(len(top_order)), top_order["Delta_pos1_minus_pos2"],
            color=colors_ord, alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(top_order)))
    ax.set_yticklabels(top_order["Motif"], fontsize=9)
    ax.axvline(0, color="black", lw=1, alpha=0.5)
    ax.set_xlabel("ΔMIC (pos1 − pos2)\nnegative = better in position 2", fontsize=10)
    ax.set_title("Motifs with significant position preference (3cat)\n"
                 "colored by mechanism", fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR/"08_order_preference.png", dpi=DPI, bbox_inches="tight")
    plt.close()

print(f"\n✅ All figures saved to: {FIG_DIR}")
print(f"✅ Score top motif: {motif_profile.iloc[0]['Motif']} "
      f"(score={motif_profile.iloc[0]['Score_combined']:.1f}  "
      f"CV={motif_profile.iloc[0]['CV']:.1f}%)")

# =========================
# FIGURE 9: Correlación familiar
# =========================

if not fam_ind.empty and len(fam_ind) >= 5:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#fafafa")

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
    ax.set_title("¿Familias más grandes son más activas? (3cat)", fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)

    ax = axes[1]; ax.set_facecolor("#fafafa")
    top20_fam = fam_ind.head(20)
    ax.barh(range(len(top20_fam)), top20_fam["MIC_ind_mean"],
            color="#E07B54", alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(top20_fam)))
    ax.set_yticklabels(top20_fam["Family_ID"], fontsize=8.5)
    ax.set_xlabel("MIC media individual (µmol/L)", fontsize=10)
    ax.set_title("Top 20 familias más activas (3cat)", fontweight="bold")
    ax.axvline(fam_ind["MIC_ind_mean"].mean(), color="#333", lw=1.2, ls="--",
               label=f"Mean = {fam_ind['MIC_ind_mean'].mean():.1f}")
    ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)

    fig.suptitle("Análisis familiar — actividad individual (3cat)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR/"09_family_correlation.png", dpi=DPI, bbox_inches="tight")
    plt.close()

    with pd.ExcelWriter(out_gp, engine="openpyxl", mode="a",
                        if_sheet_exists="replace") as writer:
        fam_ind.to_excel(writer, sheet_name="GP_Fig9_family_MIC", index=False)

# =========================
# FIGURE 10: Individual vs combinación por mecanismo + best vs worst
# =========================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("#fafafa")

ax = axes[0]; ax.set_facecolor("#fafafa")
for mech, color in PALETTE_3.items():
    sub = corr_ind_comb[corr_ind_comb["Mechanism_3cat"]==mech]
    if sub.empty: continue
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
ax.set_title("Individual vs combinación\ncoloreado por mecanismo (3cat)",
             fontweight="bold")
ax.legend(fontsize=8.5); ax.spines[["top","right"]].set_visible(False)

ax = axes[1]; ax.set_facecolor("#fafafa")
best_vals  = motif_profile[motif_profile["Motif"].isin(best_ind)]["MIC_mean_in_comb"].dropna()
worst_vals = motif_profile[motif_profile["Motif"].isin(worst_ind)]["MIC_mean_in_comb"].dropna()
bp = ax.boxplot([best_vals, worst_vals], patch_artist=True,
                medianprops=dict(color="black", lw=2),
                whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2),
                flierprops=dict(marker="o", markersize=4, alpha=0.4))
for patch, color in zip(bp["boxes"], [PALETTE_3["Carpet"], PALETTE_3["Barrel_stave"]]):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax.set_xticks([1, 2])
ax.set_xticklabels([f"Top {n_extreme}\nbest individual",
                    f"Top {n_extreme}\nworst individual"], fontsize=10)
ax.set_ylabel("MIC media en combinaciones (µmol/L)", fontsize=10)
ax.set_title(f"Best vs worst individuales en combinación\np={p_rank:.4f}",
             fontweight="bold")
ax.spines[["top","right"]].set_visible(False)

fig.suptitle("¿Los mejores individuales son mejores en combinación? (3cat)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR/"10_individual_vs_comb_full.png", dpi=DPI, bbox_inches="tight")
plt.close()

print(f"✅ Additional figures saved (09, 10)")

# =========================
# ANALYSIS 11: BEST COMBINATIONS OVERALL vs CROSS-FAMILY (3cat)
# =========================

print(f"\n=== Top 20 combinations overall (3cat) ===")
top_overall = df.nsmallest(20, "MIC_combined")[[
    "Combination_ID","Motif_1","Mech3_1","Family_1",
    "Motif_2","Mech3_2","Family_2",
    "Comb3_type","Sequence","MIC_combined",
    "MIC_mean_individual","MIC_delta","Same_family"
]]
print(top_overall[["Motif_1","Motif_2","Comb3_type",
                    "MIC_combined","MIC_delta","Same_family"]].to_string(index=False))

print(f"\n=== Top 20 cross-family combinations (3cat) ===")
df_cross = df[~df["Same_family"]].copy()
top_cross = df_cross.nsmallest(20, "MIC_combined")[[
    "Combination_ID","Motif_1","Mech3_1","Family_1",
    "Motif_2","Mech3_2","Family_2",
    "Comb3_type","Sequence","MIC_combined",
    "MIC_mean_individual","MIC_delta"
]]
print(top_cross[["Motif_1","Family_1","Motif_2","Family_2",
                  "Comb3_type","MIC_combined","MIC_delta"]].to_string(index=False))

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
out_excel = OUT_DIR / "01_analysis_3cat.xlsx"
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

out_gp = OUT_DIR / "02_graphpad_3cat.xlsx"
with pd.ExcelWriter(out_gp, engine="openpyxl", mode="a",
                    if_sheet_exists="replace") as writer:
    pd.DataFrame({
        "Same_family": df[df["Same_family"]]["MIC_combined"].dropna().values,
    }).to_excel(writer, sheet_name="GP_Fig11_samefam", index=False)
    pd.DataFrame({
        "Cross_family": df[~df["Same_family"]]["MIC_combined"].dropna().values,
    }).to_excel(writer, sheet_name="GP_Fig11_crossfam", index=False)
    top_cross.to_excel(writer, sheet_name="GP_Top20_cross", index=False)

# =========================
# FIGURE 11: Same-family vs cross-family (3cat)
# =========================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#fafafa")

ax = axes[0]; ax.set_facecolor("#fafafa")
plot_sf = pd.DataFrame({
    "Group": (["Same family"] * df[df["Same_family"]]["MIC_combined"].dropna().shape[0] +
              ["Cross family"] * df[~df["Same_family"]]["MIC_combined"].dropna().shape[0]),
    "MIC":   pd.concat([df[df["Same_family"]]["MIC_combined"].dropna(),
                        df[~df["Same_family"]]["MIC_combined"].dropna()]).values
})
sns.violinplot(data=plot_sf, x="Group", y="MIC", hue="Group",
               palette={"Same family": "#CCCCCC", "Cross family": "#4C72B0"},
               inner="quart", linewidth=0.9, legend=False, ax=ax)
ax.set_xlabel("")
ax.set_ylabel("MIC E. coli ATCC11775 (µmol/L)", fontsize=10)
ax.set_title(f"Same-family vs cross-family\np={p_sf:.4f}  d={d_sf:.3f}",
             fontweight="bold")
ax.spines[["top","right"]].set_visible(False)

ax = axes[1]; ax.set_facecolor("#fafafa")
colors_top = [PALETTE_3.get(ct.split("+")[0].strip(), "#888")
              for ct in top_cross["Comb3_type"]]
ax.barh(range(len(top_cross)), top_cross["MIC_combined"],
        color=colors_top, alpha=0.85, edgecolor="white")
ax.set_yticks(range(len(top_cross)))
ax.set_yticklabels([f"{r['Motif_1']}+{r['Motif_2']}"
                    for _, r in top_cross.iterrows()], fontsize=7.5)
ax.set_xlabel("MIC combined (µmol/L)", fontsize=10)
ax.set_title("Top 20 cross-family combinations (3cat)",
             fontweight="bold")
patches = [mpatches.Patch(color=c, label=m) for m,c in PALETTE_3.items()]
ax.legend(handles=patches, fontsize=8)
ax.spines[["top","right"]].set_visible(False)

fig.suptitle("Cross-family vs same-family combinations (3cat)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR/"11_crossfamily_analysis.png", dpi=DPI, bbox_inches="tight")
plt.close()

print(f"✅ Cross-family analysis saved (Fig 11)")

# =========================
# ANALYSIS 12: BEST COMBINATIONS EXCLUDING DOMINANT MOTIF (3cat)
# =========================

top_appearance = pd.concat([top_overall["Motif_1"], top_overall["Motif_2"]]
                           ).value_counts().head(3)
dominant_motifs = top_appearance.index.tolist()

print(f"\n=== Analysis excluding dominant motif(s) (3cat) ===")
print(f"  Dominant motifs: {dominant_motifs}")

df_nodom = df[
    ~df["Motif_1"].isin(dominant_motifs) &
    ~df["Motif_2"].isin(dominant_motifs)
].copy()
print(f"  Combinations after exclusion: {len(df_nodom)} (was {len(df)})")

print(f"\n  Top 20 cross-family WITHOUT dominant motif (3cat):")
top_nodom_cross = df_nodom[~df_nodom["Same_family"]].nsmallest(20, "MIC_combined")[[
    "Combination_ID","Motif_1","Mech3_1","Family_1",
    "Motif_2","Mech3_2","Family_2",
    "Comb3_type","Sequence","MIC_combined",
    "MIC_mean_individual","MIC_delta"
]]
print(top_nodom_cross[["Motif_1","Family_1","Motif_2","Family_2",
                        "Comb3_type","MIC_combined","MIC_delta"]
                      ].to_string(index=False))

out_excel = OUT_DIR / "01_analysis_3cat.xlsx"
with pd.ExcelWriter(out_excel, engine="openpyxl", mode="a",
                    if_sheet_exists="replace") as writer:
    top_nodom_cross.to_excel(writer, sheet_name="Top20_nodominant", index=False)

out_gp = OUT_DIR / "02_graphpad_3cat.xlsx"
with pd.ExcelWriter(out_gp, engine="openpyxl", mode="a",
                    if_sheet_exists="replace") as writer:
    top_nodom_cross.to_excel(writer, sheet_name="GP_Top20_nodominant", index=False)

# =========================
# FIGURE 12: No-dominant analysis (3cat)
# =========================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor("#fafafa")

ax = axes[0]; ax.set_facecolor("#fafafa")
colors_nd = [PALETTE_3.get(ct.split("+")[0].strip(), "#888")
             for ct in top_nodom_cross["Comb3_type"]]
ax.barh(range(len(top_nodom_cross)), top_nodom_cross["MIC_combined"],
        color=colors_nd, alpha=0.85, edgecolor="white")
ax.set_yticks(range(len(top_nodom_cross)))
ax.set_yticklabels([f"{r['Motif_1']}+{r['Motif_2']}"
                    for _, r in top_nodom_cross.iterrows()], fontsize=7.5)
ax.set_xlabel("MIC combined (µmol/L)", fontsize=10)
ax.set_title(f"Top 20 cross-family\nexcluding {dominant_motifs[0]} (3cat)",
             fontweight="bold")
patches = [mpatches.Patch(color=c, label=m) for m,c in PALETTE_3.items()]
ax.legend(handles=patches, fontsize=8)
ax.spines[["top","right"]].set_visible(False)

ax = axes[1]; ax.set_facecolor("#fafafa")
for ct in ctypes_present:
    color = plt.cm.tab10(ctypes_present.index(ct) / len(ctypes_present))
    sub = df_nodom[df_nodom["Comb3_type"]==ct].dropna(
        subset=["MIC_combined","MIC_mean_individual"])
    if len(sub) < 2: continue
    ax.scatter(sub["MIC_mean_individual"], sub["MIC_combined"],
               c=[color]*len(sub), alpha=0.25, s=12, edgecolors="none",
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
ax.set_title(f"Combined vs individual\nexcluding {dominant_motifs[0]} (3cat)",
             fontweight="bold")
ax.legend(fontsize=7); ax.spines[["top","right"]].set_visible(False)

fig.suptitle(f"Diversity analysis — excluding dominant motif ({dominant_motifs[0]}, 3cat)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR/"12_nodominant_analysis.png", dpi=DPI, bbox_inches="tight")
plt.close()

print(f"✅ No-dominant analysis saved (Fig 12)")