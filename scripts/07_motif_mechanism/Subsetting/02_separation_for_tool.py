"""
Discrimination analysis using real mechanism labels.

Each tool evaluated independently on its own parameters and its own labels.
No tercile, no cross-contamination.

For each tool:
  - Carpet vs Pore (Toroidal + Barrel combined)
  - Carpet vs Toroidal vs Barrel (all pairs)

Metrics: Cohen's d, AUC, KS, Mann-Whitney p
Output: Excel + same figures as tercile script
"""

from pathlib import Path
from itertools import combinations as _comb

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ks_2samp, spearmanr
from sklearn.metrics import roc_auc_score

# =========================
# CONFIG
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

FMAP_FILE = (
    PROJECT_ROOT / "results/11_mechanism_analysis"
    / "1_motif_activity_analysis/fmap2/01_fmap_classified.csv"
)
PPM_FILE = (
    PROJECT_ROOT / "results/11_mechanism_analysis"
    / "1_motif_activity_analysis/ppm2/01_ppm_classified.csv"
)

OUT_DIR = PROJECT_ROOT / "results/11_mechanism_analysis/2.3_fmap_ppm_comparison"
FIG_DIR = OUT_DIR / "figures" / "real_labels"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300

PALETTE_REDUCED = {"Carpet": "#E07B54", "Pore": "#4C72B0"}
PALETTE_DETAIL  = {
    "Carpet":        "#E07B54",
    "Toroidal_pore": "#4C72B0",
    "Barrel_stave":  "#3BAF77",
}
BAR_PAL = {"FMAP": "#2E86AB", "PPM": "#E84855"}
VAR_PAL = {"Depth": "#5A9FD4", "Tilt": "#D4785A", "Energy": "#3BAF77"}

sns.set_theme(style="whitegrid", font_scale=1.15)

# =========================
# LOAD & NORMALISE
# =========================

fmap = pd.read_csv(FMAP_FILE).rename(columns={
    "Tilt_Angle":              "Tilt",
    "Depth_Thickness":         "Depth",
    "Membrane_Binding_Energy": "Energy",
})
ppm = pd.read_csv(PPM_FILE).rename(columns={
    "Tilt_deg":    "Tilt",
    "Thickness_A": "Depth",
    "Best_Energy": "Energy",
})

PARAMS = ["Tilt", "Depth", "Energy"]

for df in [fmap, ppm]:
    for col in PARAMS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

fmap = fmap.dropna(subset=PARAMS + ["Mechanism"]).reset_index(drop=True)
ppm  = ppm.dropna(subset=PARAMS + ["Mechanism"]).reset_index(drop=True)

def reduce_mech(m):
    return "Pore" if m in ("Barrel_stave", "Toroidal_pore") else m

fmap["Mechanism_reduced"] = fmap["Mechanism"].apply(reduce_mech)
ppm["Mechanism_reduced"]  = ppm["Mechanism"].apply(reduce_mech)

# exclude Ambiguous from all analysis
fmap_clean = fmap[fmap["Mechanism"] != "Ambiguous"].copy()
ppm_clean  = ppm[ppm["Mechanism"]  != "Ambiguous"].copy()

print("=== Dataset sizes (excluding Ambiguous) ===")
for tool, df_tool in [("FMAP", fmap_clean), ("PPM", ppm_clean)]:
    n = len(df_tool)
    counts = df_tool["Mechanism"].value_counts()
    pcts   = (counts / n * 100).round(1)
    print(f"{tool} ({n} peptides):")
    for mech, cnt in counts.items():
        print(f"  {mech:<20} {cnt:>4}  ({pcts[mech]:.1f}%)")

# =========================
# DISCRIMINATION STATS
# =========================

def cohen_d(a, b):
    pooled = np.sqrt((a.std()**2 + b.std()**2) / 2)
    return abs(a.mean() - b.mean()) / pooled if pooled > 0 else 0

def separation_stats(df_tool, label_col, tool_name):
    rows  = []
    mechs = [m for m in df_tool[label_col].unique() if m != "Ambiguous"]
    for m1, m2 in _comb(sorted(mechs), 2):
        for param in PARAMS:
            g1 = df_tool.loc[df_tool[label_col] == m1, param].dropna()
            g2 = df_tool.loc[df_tool[label_col] == m2, param].dropna()
            if len(g1) < 5 or len(g2) < 5:
                continue
            d      = cohen_d(g1, g2)
            _, p   = mannwhitneyu(g1, g2, alternative="two-sided")
            ks, _  = ks_2samp(g1, g2)
            y_true = np.concatenate([np.ones(len(g1)), np.zeros(len(g2))])
            y_sc   = np.concatenate([g1.values, g2.values])
            auc    = max(roc_auc_score(y_true, y_sc),
                         1 - roc_auc_score(y_true, y_sc))
            rows.append({
                "Tool":       tool_name,
                "Comparison": f"{m1} vs {m2}",
                "Variable":   param,
                "Cohen_d":    round(d, 3),
                "AUC":        round(auc, 3),
                "KS":         round(ks, 3),
                "p_mw":       round(p, 6),
                "n1":         len(g1),
                "n2":         len(g2),
            })
    return pd.DataFrame(rows)

# reduced (Carpet vs Pore)
stats_fmap_red = separation_stats(
    fmap_clean[fmap_clean["Mechanism_reduced"] != "Ambiguous"],
    "Mechanism_reduced", "FMAP"
)
stats_ppm_red  = separation_stats(
    ppm_clean[ppm_clean["Mechanism_reduced"] != "Ambiguous"],
    "Mechanism_reduced", "PPM"
)

# detailed (Carpet / Toroidal / Barrel all pairs)
stats_fmap_det = separation_stats(fmap_clean, "Mechanism", "FMAP")
stats_ppm_det  = separation_stats(ppm_clean,  "Mechanism", "PPM")

stats_reduced  = pd.concat([stats_fmap_red, stats_ppm_red],  ignore_index=True)
stats_detailed = pd.concat([stats_fmap_det, stats_ppm_det],  ignore_index=True)

print("\n=== Discrimination — Detailed (Carpet / Toroidal / Barrel — all pairs) ===")
print(stats_detailed[["Tool","Comparison","Variable",
                        "Cohen_d","AUC","KS","p_mw"]].to_string(index=False))

print("\n=== Discrimination — Reduced (Carpet vs Pore) ===")
print(stats_reduced[["Tool","Comparison","Variable",
                       "Cohen_d","AUC","KS","p_mw"]].to_string(index=False))

print("\n=== Effect of merging Toroidal + Barrel → Pore (Δ Cohen's d vs Carpet) ===")
print(f"{'Tool':<6} {'Variable':<8} {'Carpet_vs_Toroidal':>18} {'Carpet_vs_Barrel':>16} "
      f"{'Carpet_vs_Pore':>14} {'Change':>8}")
for tool in ["FMAP", "PPM"]:
    for param in PARAMS:
        def get_d(comp):
            row = stats_detailed[
                (stats_detailed["Tool"]==tool) &
                (stats_detailed["Comparison"]==comp) &
                (stats_detailed["Variable"]==param)
            ]
            return row["Cohen_d"].values[0] if not row.empty else None

        d_tor = get_d("Carpet vs Toroidal_pore")
        d_bar = get_d("Barrel_stave vs Carpet")
        if d_bar is None:
            d_bar = get_d("Carpet vs Barrel_stave")
        row_red = stats_reduced[
            (stats_reduced["Tool"]==tool) &
            (stats_reduced["Variable"]==param)
        ]
        d_pore = row_red["Cohen_d"].values[0] if not row_red.empty else None

        if d_tor and d_bar and d_pore:
            avg_detail = (d_tor + d_bar) / 2
            change = d_pore - avg_detail
            direction = "↑ better" if change > 0 else "↓ worse"
            print(f"{tool:<6} {param:<8} {d_tor:>18.3f} {d_bar:>16.3f} "
                  f"{d_pore:>14.3f} {change:>+7.3f} {direction}")

print("\n=== Which tool separates better? (Carpet vs Pore) ===")
for param in PARAMS:
    fmap_d = stats_reduced.loc[
        (stats_reduced["Tool"]=="FMAP") &
        (stats_reduced["Variable"]==param), "Cohen_d"].values
    ppm_d  = stats_reduced.loc[
        (stats_reduced["Tool"]=="PPM") &
        (stats_reduced["Variable"]==param), "Cohen_d"].values
    if len(fmap_d) and len(ppm_d):
        better = "PPM" if ppm_d[0] > fmap_d[0] else "FMAP"
        print(f"  {param}: {better}  "
              f"(Δd={abs(ppm_d[0]-fmap_d[0]):.3f}  "
              f"FMAP={fmap_d[0]:.3f}  PPM={ppm_d[0]:.3f})")

# =========================
# CORRELATION FMAP vs PPM
# =========================

RNG    = np.random.default_rng(42)
N_ITER = 500
N_DRAW = 20

fmap_corr = fmap_clean[["Peptide_ID"] + PARAMS + ["Mechanism","Mechanism_reduced"]].copy()
ppm_corr  = ppm_clean[["Peptide_ID"]  + PARAMS + ["Mechanism","Mechanism_reduced"]].copy()

fmap_corr = fmap_corr.rename(columns={p: f"{p}_FMAP" for p in PARAMS})
fmap_corr = fmap_corr.rename(columns={"Mechanism": "Mechanism_FMAP",
                                       "Mechanism_reduced": "Mech_red_FMAP"})
ppm_corr  = ppm_corr.rename(columns={p: f"{p}_PPM" for p in PARAMS})
ppm_corr  = ppm_corr.rename(columns={"Mechanism": "Mechanism_PPM",
                                      "Mechanism_reduced": "Mech_red_PPM"})

both = fmap_corr.merge(ppm_corr, on="Peptide_ID", how="inner")

# consensus: both agree on Carpet or Pore
both["Label_consensus"] = np.where(
    both["Mech_red_FMAP"] == both["Mech_red_PPM"],
    both["Mech_red_FMAP"], "Disagree"
)
both_consensus = both[both["Label_consensus"].isin(["Carpet","Pore"])].copy()

# --- A: Full consensus ρ ---
print(f"\n=== Spearman ρ — full consensus (n={len(both_consensus)}) ===")
corr_full_rows = []
for param in PARAMS:
    fc, pc = f"{param}_FMAP", f"{param}_PPM"
    rho, p = spearmanr(both_consensus[fc], both_consensus[pc])
    print(f"  {param}: ρ={rho:.3f}  p={p:.4f}")
    corr_full_rows.append({"Method": "Full_consensus", "Parameter": param,
                           "rho": round(rho,3), "p": round(p,6),
                           "n": len(both_consensus)})
corr_full_df = pd.DataFrame(corr_full_rows)

# --- B: Subsampled ρ (N_DRAW Carpet + N_DRAW Pore per iteration) ---
carpet_idx = both_consensus.index[both_consensus["Label_consensus"]=="Carpet"].tolist()
pore_idx   = both_consensus.index[both_consensus["Label_consensus"]=="Pore"].tolist()
n_draw     = min(N_DRAW, len(carpet_idx), len(pore_idx))

print(f"\n=== Subsampled ρ — {N_ITER} iterations of "
      f"{n_draw} Carpet + {n_draw} Pore ===")

sub_records = []
for _ in range(N_ITER):
    sel  = np.concatenate([RNG.choice(carpet_idx, n_draw, replace=False),
                           RNG.choice(pore_idx,   n_draw, replace=False)])
    samp = both_consensus.loc[sel]
    row  = {}
    for param in PARAMS:
        fc, pc = f"{param}_FMAP", f"{param}_PPM"
        rho, _ = spearmanr(samp[fc], samp[pc])
        row[f"{param}_rho"] = rho
    sub_records.append(row)

sub_df = pd.DataFrame(sub_records)

corr_sub_rows = []
for param in PARAMS:
    mean_rho = sub_df[f"{param}_rho"].mean()
    std_rho  = sub_df[f"{param}_rho"].std()
    print(f"  {param}: ρ={mean_rho:.3f} ± {std_rho:.3f}")
    corr_sub_rows.append({"Method": "Subsampled", "Parameter": param,
                           "rho": round(mean_rho,3), "rho_std": round(std_rho,3),
                           "n_per_iter": n_draw*2, "n_iter": N_ITER})
corr_sub_df = pd.DataFrame(corr_sub_rows)

# compare
print(f"\n=== Comparison: Full consensus vs Subsampled ===")
print(f"{'Parameter':<10} {'Full ρ':>8} {'Sub ρ (mean±std)':>20}")
for param in PARAMS:
    full = corr_full_df.loc[corr_full_df["Parameter"]==param, "rho"].values[0]
    sub  = corr_sub_df.loc[corr_sub_df["Parameter"]==param]
    print(f"  {param:<10} {full:>8.3f} "
          f"{sub['rho'].values[0]:>8.3f} ± {sub['rho_std'].values[0]:.3f}")

# representative subsample (closest to median ρ)
median_rho = sub_df[[f"{p}_rho" for p in PARAMS]].mean(axis=1).median()
rep_idx    = (sub_df[[f"{p}_rho" for p in PARAMS]].mean(axis=1) - median_rho).abs().idxmin()
rep_sel    = np.concatenate([
    RNG.choice(carpet_idx, n_draw, replace=False),
    RNG.choice(pore_idx,   n_draw, replace=False)
])
rep_sub_data = both_consensus.loc[rep_sel].copy()

# =========================
# OUTLIER REMOVAL FOR CONSENSUS
# =========================

both_cons_clean = both_consensus.copy()
for param in PARAMS:
    fc, pc = f"{param}_FMAP", f"{param}_PPM"
    if param == "Depth":
        thr_f = both_cons_clean[fc].quantile(0.95)
        thr_p = both_cons_clean[pc].quantile(0.95)
        both_cons_clean = both_cons_clean[
            (both_cons_clean[fc] <= thr_f) &
            (both_cons_clean[pc] <= thr_p)]
    elif param == "Energy":
        thr_f = both_cons_clean[fc].quantile(0.05)
        thr_p = both_cons_clean[pc].quantile(0.05)
        both_cons_clean = both_cons_clean[
            (both_cons_clean[fc] >= thr_f) &
            (both_cons_clean[pc] >= thr_p)]
both_cons_clean = both_cons_clean.reset_index(drop=True)

corr_clean_rows = []
for param in PARAMS:
    fc, pc = f"{param}_FMAP", f"{param}_PPM"
    rho, p = spearmanr(both_cons_clean[fc], both_cons_clean[pc])
    corr_clean_rows.append({"Method": "Clean_consensus", "Parameter": param,
                             "rho": round(rho,3), "p": round(p,6),
                             "n": len(both_cons_clean)})
corr_clean_df = pd.DataFrame(corr_clean_rows)

# =========================
# SAVE EXCEL — complete
# =========================

out_xlsx = OUT_DIR / "03_stats_real_labels.xlsx"
with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
    stats_reduced.to_excel(writer,    sheet_name="Discrim_reduced",     index=False)
    stats_detailed.to_excel(writer,   sheet_name="Discrim_detailed",    index=False)
    fmap_clean.to_excel(writer,       sheet_name="FMAP_data",           index=False)
    ppm_clean.to_excel(writer,        sheet_name="PPM_data",            index=False)
    both.to_excel(writer,             sheet_name="Both_merged",         index=False)
    both_consensus.to_excel(writer,   sheet_name="Both_consensus",      index=False)
    both_cons_clean.to_excel(writer,  sheet_name="Consensus_clean",     index=False)
    corr_full_df.to_excel(writer,     sheet_name="Spearman_full",       index=False)
    corr_sub_df.to_excel(writer,      sheet_name="Spearman_subsampled", index=False)
    corr_clean_df.to_excel(writer,    sheet_name="Spearman_clean",      index=False)
    sub_df.to_excel(writer,           sheet_name="Subsample_iterations", index=False)

print(f"\n✅ Excel saved: {out_xlsx}")

# =========================
# PLOT 1: Split violin — reduced (Carpet vs Pore)
# =========================

fig, axes = plt.subplots(1, len(PARAMS), figsize=(6.5*len(PARAMS), 5))
if len(PARAMS) == 1: axes = [axes]

for ax, param in zip(axes, PARAMS):
    rows_v = []
    for tool, df_tool in [("FMAP", fmap_clean), ("PPM", ppm_clean)]:
        tmp = df_tool[["Mechanism_reduced", param]].copy()
        tmp = tmp[tmp["Mechanism_reduced"].isin(["Carpet","Pore"])]
        tmp.columns = ["Label", "Value"]
        tmp["Tool"] = tool
        rows_v.append(tmp)
    plot_df = pd.concat(rows_v)

    sns.violinplot(data=plot_df, x="Tool", y="Value", hue="Label",
                   hue_order=["Carpet","Pore"], palette=PALETTE_REDUCED,
                   split=True, inner="quart", linewidth=0.9, ax=ax)

    for i, tool in enumerate(["FMAP","PPM"]):
        row = stats_reduced[
            (stats_reduced["Tool"]==tool) &
            (stats_reduced["Variable"]==param)
        ]
        if row.empty: continue
        row = row.iloc[0]
        p_str = "p<0.001" if row["p_mw"] < 0.001 else f"p={row['p_mw']:.3f}"
        ax.text(i, ax.get_ylim()[1]*0.97,
                f"d={row['Cohen_d']:.2f}\nAUC={row['AUC']:.2f}\n{p_str}",
                ha="center", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    ax.set_title(f"{param}: Carpet vs Pore", fontweight="bold")
    ax.set_xlabel("Tool", fontsize=10)
    ax.set_ylabel(param, fontsize=10)
    ax.legend(title="Mechanism", fontsize=9, framealpha=0.7)

fig.suptitle("Discrimination — Carpet vs Pore (real mechanism labels)\n"
             "each tool on its own parameters and labels",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "01_violin_reduced.png", dpi=DPI)
plt.close()

# =========================
# PLOT 2: Split violin — detailed (3 groups)
# =========================

MECH_ORDER_DET = ["Carpet", "Toroidal_pore", "Barrel_stave"]

fig, axes = plt.subplots(1, len(PARAMS), figsize=(6.5*len(PARAMS), 5))
if len(PARAMS) == 1: axes = [axes]

for ax, param in zip(axes, PARAMS):
    rows_v = []
    for tool, df_tool in [("FMAP", fmap_clean), ("PPM", ppm_clean)]:
        tmp = df_tool[["Mechanism", param]].copy()
        tmp = tmp[tmp["Mechanism"].isin(MECH_ORDER_DET)]
        tmp.columns = ["Label", "Value"]
        tmp["Tool"] = tool
        rows_v.append(tmp)
    plot_df = pd.concat(rows_v)

    sns.violinplot(data=plot_df, x="Tool", y="Value", hue="Label",
                   hue_order=MECH_ORDER_DET, palette=PALETTE_DETAIL,
                   inner="quart", linewidth=0.9, ax=ax)

    ax.set_title(f"{param}: detailed mechanisms", fontweight="bold")
    ax.set_xlabel("Tool", fontsize=10)
    ax.set_ylabel(param, fontsize=10)
    ax.legend(title="Mechanism", fontsize=8, framealpha=0.7)

fig.suptitle("Discrimination — Carpet / Toroidal / Barrel (real mechanism labels)\n"
             "each tool on its own parameters and labels",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "02_violin_detailed.png", dpi=DPI)
plt.close()

# =========================
# PLOT 3: Cohen's d bar chart — reduced
# =========================

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
x = np.arange(len(PARAMS)); w = 0.35

for ax, metric, ylabel, refs in zip(axes,
    ["Cohen_d", "AUC"], ["Cohen's d", "AUC-ROC"],
    [[(0.5,"--","medium"),(0.8,":","large")],
     [(0.7,"--","good"),(0.9,":","excellent")]]):
    for j, tool in enumerate(["FMAP","PPM"]):
        vals = []
        for param in PARAMS:
            row = stats_reduced[
                (stats_reduced["Tool"]==tool) &
                (stats_reduced["Variable"]==param)
            ]
            vals.append(row[metric].values[0] if not row.empty else 0)
        bars = ax.bar(x+j*w, vals, w, label=tool,
                      color=BAR_PAL[tool], alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=10)
    for ref, ls, lbl in refs:
        ax.axhline(ref, color="grey", ls=ls, lw=1, alpha=0.5, label=lbl)
    ax.set_xticks(x+w/2); ax.set_xticklabels(PARAMS)
    ax.set_ylabel(ylabel); ax.set_title(ylabel, fontweight="bold")
    ax.legend(fontsize=9)

fig.suptitle("Effect size — Carpet vs Pore (real labels)\n"
             "higher = better separation", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "03_effect_size_reduced.png", dpi=DPI)
plt.close()

# =========================
# PLOT 4: Cohen's d — detailed (all pairs)
# =========================

comparisons = stats_detailed["Comparison"].unique()
fig, axes   = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(comparisons)); w = 0.35

for ax, metric, ylabel, refs in zip(axes,
    ["Cohen_d","AUC"],["Cohen's d","AUC-ROC"],
    [[(0.5,"--","medium"),(0.8,":","large")],
     [(0.7,"--","good"),(0.9,":","excellent")]]):
    for j, (tool, color) in enumerate(zip(["FMAP","PPM"],
                                          [BAR_PAL["FMAP"],BAR_PAL["PPM"]])):
        # use Tilt as representative parameter for the bar chart
        vals = []
        for comp in comparisons:
            row = stats_detailed[
                (stats_detailed["Tool"]==tool) &
                (stats_detailed["Comparison"]==comp) &
                (stats_detailed["Variable"]=="Tilt")
            ]
            vals.append(row[metric].values[0] if not row.empty else 0)
        bars = ax.bar(x+j*w, vals, w, label=tool,
                      color=color, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            if val > 0.1:
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+0.02,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    for ref, ls, lbl in refs:
        ax.axhline(ref, color="grey", ls=ls, lw=1, alpha=0.5, label=lbl)
    ax.set_xticks(x+w/2)
    ax.set_xticklabels(comparisons, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel(ylabel); ax.set_title(f"{ylabel} — Tilt (all pairs)",
                                         fontweight="bold")
    ax.legend(fontsize=9)

fig.suptitle("Effect size — detailed mechanism pairs (Tilt parameter)\n"
             "real labels, each tool independently",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "04_effect_size_detailed.png", dpi=DPI)
plt.close()

# =========================
# PLOT 5: KDE — reduced
# =========================

fig, axes = plt.subplots(len(PARAMS), 2, figsize=(13, 4.5*len(PARAMS)))
if len(PARAMS) == 1: axes = axes.reshape(1, 2)

for row, param in enumerate(PARAMS):
    for col, (tool, df_tool) in enumerate(zip(["FMAP","PPM"],
                                               [fmap_clean, ppm_clean])):
        ax = axes[row][col]
        for label, color in PALETTE_REDUCED.items():
            grp = df_tool[df_tool["Mechanism_reduced"]==label]
            if len(grp) < 5: continue
            sns.kdeplot(grp[param], fill=True, alpha=0.5, color=color,
                        label=f"{label} (n={len(grp)})", ax=ax)
        row_s = stats_reduced[
            (stats_reduced["Tool"]==tool) &
            (stats_reduced["Variable"]==param)
        ]
        title_extra = ""
        if not row_s.empty:
            r = row_s.iloc[0]
            title_extra = f"\nd={r['Cohen_d']:.2f}  AUC={r['AUC']:.2f}"
        ax.set_title(f"{tool} — {param}{title_extra}", fontweight="bold")
        ax.set_xlabel(param); ax.set_ylabel("Density")
        ax.legend(fontsize=9, framealpha=0.7)

fig.suptitle("Distribution overlap — Carpet vs Pore (real labels)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "05_kde_reduced.png", dpi=DPI)
plt.close()

# =========================
# PLOT 6: Detailed vs Reduced comparison — how does merging change d?
# =========================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, tool in zip(axes, ["FMAP", "PPM"]):
    rows = []
    for param in PARAMS:
        # detailed: Carpet vs Toroidal
        r = stats_detailed[(stats_detailed["Tool"]==tool) &
                            (stats_detailed["Variable"]==param) &
                            (stats_detailed["Comparison"].str.contains("Toroidal"))]
        if not r.empty:
            rows.append({"Parameter": param, "Comparison": "Carpet vs Toroidal",
                         "Cohen_d": r["Cohen_d"].values[0]})
        # detailed: Carpet vs Barrel
        r = stats_detailed[(stats_detailed["Tool"]==tool) &
                            (stats_detailed["Variable"]==param) &
                            (stats_detailed["Comparison"].str.contains("Barrel"))]
        if not r.empty:
            rows.append({"Parameter": param, "Comparison": "Carpet vs Barrel",
                         "Cohen_d": r["Cohen_d"].values[0]})
        # reduced: Carpet vs Pore
        r = stats_reduced[(stats_reduced["Tool"]==tool) &
                           (stats_reduced["Variable"]==param)]
        if not r.empty:
            rows.append({"Parameter": param, "Comparison": "Carpet vs Pore",
                         "Cohen_d": r["Cohen_d"].values[0]})

    plot_df = pd.DataFrame(rows)
    comp_palette = {
        "Carpet vs Toroidal": "#4C72B0",
        "Carpet vs Barrel":   "#3BAF77",
        "Carpet vs Pore":     "#E07B54",
    }
    sns.barplot(data=plot_df, x="Parameter", y="Cohen_d", hue="Comparison",
                palette=comp_palette, alpha=0.85, ax=ax)
    ax.axhline(0.5, color="grey", ls="--", lw=1, alpha=0.5, label="medium (0.5)")
    ax.axhline(0.8, color="grey", ls=":",  lw=1, alpha=0.5, label="large (0.8)")
    ax.set_title(f"{tool} — detailed vs reduced", fontweight="bold")
    ax.set_ylabel("Cohen's d"); ax.set_xlabel("")
    ax.legend(fontsize=8, framealpha=0.8)

fig.suptitle("Detailed (Carpet/Toroidal/Barrel) vs Reduced (Carpet/Pore)\n"
             "does merging Toroidal+Barrel improve or reduce separation?",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "06_detailed_vs_reduced.png", dpi=DPI)
plt.close()

# =========================
# PLOT 7: Summary heatmap
# =========================

fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for ax, metric, title, vmin, vmax in zip(axes,
    ["Cohen_d","AUC"],["Cohen's d","AUC-ROC"],[0,0.5],[3,1.0]):
    pivot = stats_reduced[stats_reduced["Comparison"]=="Carpet vs Pore"].pivot(
        index="Tool", columns="Variable", values=metric
    )
    # ensure column order
    pivot = pivot.reindex(columns=PARAMS)
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd",
                vmin=vmin, vmax=vmax, linewidths=0.5, ax=ax,
                annot_kws={"size":13,"weight":"bold"})
    ax.set_title(title, fontweight="bold"); ax.set_xlabel("")

fig.suptitle("Summary — Carpet vs Pore separation (real labels)\n"
             "higher = better", fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "07_summary_heatmap.png", dpi=DPI)
plt.close()

print(f"\n✅ Figures saved to: {FIG_DIR}")

# =========================
# PLOT STYLE HELPERS
# =========================

C_CARPET  = "#E07B54"
C_PORE    = "#4C72B0"
C_DISAGR  = "#CCCCCC"
C_ANNOT   = "#2d2d2d"

def style_scatter_ax(ax, param, fc, pc, data, label_col,
                     rho, n, title_suffix=""):
    """Polished scatter ax: coloured dots, y=x ref, ρ annotation."""
    for label, color, zorder in [
        ("Pore",    C_PORE,    2),
        ("Carpet",  C_CARPET,  3),
        ("Disagree",C_DISAGR,  1),
    ]:
        grp = data[data[label_col] == label] if label_col in data.columns \
              else data[data["Label_consensus"] == label]
        if grp.empty: continue
        ax.scatter(grp[fc], grp[pc],
                   c=color, alpha=0.7, s=32,
                   edgecolors="white", linewidths=0.3,
                   zorder=zorder,
                   label=f"{label} (n={len(grp)})")

    lo = min(data[fc].min(), data[pc].min()) - 1
    hi = max(data[fc].max(), data[pc].max()) + 1
    ax.plot([lo, hi], [lo, hi], color="#888888", lw=1,
            ls="--", alpha=0.5, zorder=0, label="y = x")

    ax.set_xlabel(f"{param} — FMAP", fontsize=10, color="#444")
    ax.set_ylabel(f"{param} — PPM",  fontsize=10, color="#444")
    ax.set_title(f"{param}{title_suffix}", fontsize=11,
                 fontweight="bold", pad=8)
    ax.annotate(f"ρ = {rho:.3f}\nn = {n}",
                xy=(0.05, 0.93), xycoords="axes fraction",
                fontsize=10, color=C_ANNOT,
                bbox=dict(boxstyle="round,pad=0.4",
                          fc="white", ec="#dddddd", alpha=0.9))
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8, framealpha=0.8, loc="lower right")


# =========================
# PLOT 7A: Full consensus scatter
# =========================

fig, axes = plt.subplots(1, len(PARAMS), figsize=(6*len(PARAMS), 5.5))
if len(PARAMS) == 1: axes = [axes]
fig.patch.set_facecolor("#fafafa")

for ax, param in zip(axes, PARAMS):
    fc, pc = f"{param}_FMAP", f"{param}_PPM"
    rho = corr_full_df.loc[corr_full_df["Parameter"]==param, "rho"].values[0]
    style_scatter_ax(ax, param, fc, pc, both_consensus,
                     "Label_consensus", rho, len(both_consensus))
    ax.set_facecolor("#fafafa")

fig.suptitle(
    f"Correlation FMAP vs PPM — full consensus (n={len(both_consensus)})\n"
    "peptides where both tools agree on Carpet or Pore",
    fontsize=12, fontweight="bold", color=C_ANNOT, y=1.01
)
fig.tight_layout()
fig.savefig(FIG_DIR / "07_scatter_full_consensus.png",
            dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

# =========================
# PLOT 7B: Subsampled scatter (representative subsample)
# =========================

fig, axes = plt.subplots(1, len(PARAMS), figsize=(6*len(PARAMS), 5.5))
if len(PARAMS) == 1: axes = [axes]
fig.patch.set_facecolor("#fafafa")

for ax, param in zip(axes, PARAMS):
    fc, pc  = f"{param}_FMAP", f"{param}_PPM"
    rho_sub = corr_sub_df.loc[corr_sub_df["Parameter"]==param, "rho"].values[0]
    std_sub = corr_sub_df.loc[corr_sub_df["Parameter"]==param, "rho_std"].values[0]
    style_scatter_ax(ax, param, fc, pc, rep_sub_data,
                     "Label_consensus", rho_sub, len(rep_sub_data),
                     title_suffix=f"\n(representative subsample)")
    # override annotation to show ± std
    ax.texts[-1].remove()
    ax.annotate(f"ρ = {rho_sub:.3f} ± {std_sub:.3f}\n"
                f"mean over {N_ITER} subsamples\nn per iter = {n_draw*2}",
                xy=(0.05, 0.93), xycoords="axes fraction",
                fontsize=9, color=C_ANNOT,
                bbox=dict(boxstyle="round,pad=0.4",
                          fc="white", ec="#dddddd", alpha=0.9))
    ax.set_facecolor("#fafafa")

fig.suptitle(
    f"Correlation FMAP vs PPM — subsampled ({N_DRAW} Carpet + {N_DRAW} Pore, "
    f"{N_ITER} iterations)\nshowing representative subsample",
    fontsize=12, fontweight="bold", color=C_ANNOT, y=1.01
)
fig.tight_layout()
fig.savefig(FIG_DIR / "08_scatter_subsampled.png",
            dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

# =========================
# PLOT 7C: ρ distribution across subsamples (robustness)
# =========================

VAR_COLORS = {"Tilt": C_CARPET, "Depth": "#5A9FD4", "Energy": "#3BAF77"}

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
fig.patch.set_facecolor("#fafafa")

# boxplot
ax = axes[0]
ax.set_facecolor("#fafafa")
res_long = pd.concat([
    sub_df[[f"{p}_rho"]].rename(columns={f"{p}_rho": "ρ"}).assign(Parameter=p)
    for p in PARAMS], ignore_index=True)
sns.boxplot(data=res_long, x="Parameter", y="ρ", hue="Parameter",
            palette=VAR_COLORS, width=0.45, legend=False,
            flierprops=dict(marker="o", markersize=3, alpha=0.4), ax=ax)
sns.stripplot(data=res_long, x="Parameter", y="ρ",
              color="#333333", alpha=0.08, size=2.5, jitter=True, ax=ax)
# full consensus ρ as reference lines
for param in PARAMS:
    rho_full = corr_full_df.loc[corr_full_df["Parameter"]==param, "rho"].values[0]
    xi = PARAMS.index(param)
    ax.plot([xi-0.25, xi+0.25], [rho_full, rho_full],
            color=VAR_COLORS[param], lw=2, ls="--", alpha=0.9)
ax.axhline(0, color="grey", lw=0.8, ls=":", alpha=0.5)
ax.set_title(f"ρ distribution — {N_ITER} subsamples\ndashed = full consensus ρ",
             fontweight="bold")
ax.set_ylabel("Spearman ρ"); ax.set_xlabel("")
ax.spines[["top","right"]].set_visible(False)

# KDE
ax = axes[1]
ax.set_facecolor("#fafafa")
for param in PARAMS:
    rho_full = corr_full_df.loc[corr_full_df["Parameter"]==param, "rho"].values[0]
    rho_sub  = corr_sub_df.loc[corr_sub_df["Parameter"]==param, "rho"].values[0]
    std_sub  = corr_sub_df.loc[corr_sub_df["Parameter"]==param, "rho_std"].values[0]
    sns.kdeplot(sub_df[f"{param}_rho"], fill=True, alpha=0.35,
                color=VAR_COLORS[param],
                label=f"{param}  full ρ={rho_full:.3f}  "
                      f"sub ρ={rho_sub:.3f}±{std_sub:.3f}",
                ax=ax)
    ax.axvline(rho_full, color=VAR_COLORS[param], lw=1.5, ls="--", alpha=0.8)
ax.axvline(0, color="grey", lw=0.8, ls=":", alpha=0.5)
ax.set_xlabel("Spearman ρ"); ax.set_ylabel("Density")
ax.set_title("ρ density — full consensus (dashed) vs subsampled",
             fontweight="bold")
ax.legend(fontsize=8.5, framealpha=0.85)
ax.spines[["top","right"]].set_visible(False)

fig.suptitle(
    f"Correlation robustness — {N_ITER} balanced subsamples "
    f"({N_DRAW} Carpet + {N_DRAW} Pore each)",
    fontsize=12, fontweight="bold", color=C_ANNOT
)
fig.tight_layout()
fig.savefig(FIG_DIR / "09_correlation_robustness.png",
            dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

print(f"✅ Correlation figures saved.")