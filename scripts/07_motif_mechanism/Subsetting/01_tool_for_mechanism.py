"""
Statistical comparison of FMAP vs PPM — original version with label_tercile.
Labels derived geometrically (independent of both tools' classifications).

Saves to: figures/tercile/
"""

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu, ks_2samp
from sklearn.metrics import roc_auc_score

# =========================
# CONFIG
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "results/11_mechanism_analysis/1_motif_activity_analysis"
    / "comparison2/01_fmap_ppm_comparison.xlsx"
)

OUT_DIR = PROJECT_ROOT / "results/11_mechanism_analysis/2_fmap_ppm_comparison"
FIG_DIR = OUT_DIR / "figures" / "tercile"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

RNG        = np.random.default_rng(42)
N_ITER     = 200
N_DRAW     = 10
MIN_SAMPLE = 10
DPI        = 300

DEPTH_PCTL  = 95
ENERGY_PCTL = 5

PALETTE = {"Carpet": "#E07B54", "Pore": "#4C72B0"}
VAR_PAL = {"Depth": "#5A9FD4", "Tilt": "#D4785A", "Energy": "#3BAF77"}
BAR_PAL = {"FMAP": "#2E86AB",  "PPM":  "#E84855"}

sns.set_theme(style="whitegrid", font_scale=1.15)

print(f"Config: N_ITER={N_ITER}, N_DRAW={N_DRAW}, MIN_SAMPLE={MIN_SAMPLE}")

# =========================
# LOAD
# =========================

df = pd.read_excel(INPUT_FILE, sheet_name="Merged")
df.columns = df.columns.str.strip()
print(f"Loaded {len(df)} peptides")

TOOL_COLS = {
    "Depth":  ("Depth_FMAP",  "Depth_PPM"),
    "Tilt":   ("Tilt_FMAP",   "Tilt_PPM"),
    "Energy": ("Energy_FMAP", "Energy_PPM"),
}

VARIABLES = [v for v, (fc, pc) in TOOL_COLS.items()
             if fc in df.columns and pc in df.columns]

all_cols = [c for v in VARIABLES for c in TOOL_COLS[v]]
df[all_cols] = df[all_cols].apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=all_cols).reset_index(drop=True)

# =========================
# OUTLIER REMOVAL
# =========================

print(f"Before outlier removal: {len(df)}")
fc_d, pc_d = TOOL_COLS["Depth"]
df = df[(df[fc_d] <= df[fc_d].quantile(DEPTH_PCTL/100)) &
        (df[pc_d] <= df[pc_d].quantile(DEPTH_PCTL/100))]

if "Energy" in VARIABLES:
    fc_e, pc_e = TOOL_COLS["Energy"]
    df = df[(df[fc_e] >= df[fc_e].quantile(ENERGY_PCTL/100)) &
            (df[pc_e] >= df[pc_e].quantile(ENERGY_PCTL/100))]

df = df.reset_index(drop=True)
print(f"After outlier removal: {len(df)}")

# =========================
# LABEL TERCILE
# Geometric labels independent of both tools' classifications.
# score = z(tilt) - z(depth): high score = high tilt + low depth = Carpet
#                              low score  = low tilt  + high depth = Pore
# =========================

def z(s):
    return (s - s.mean()) / s.std()

def label_median(depth_col, tilt_col, df_in):
    """
    Median split: score = z(tilt) - z(depth)
    score >= median → Carpet
    score <  median → Pore
    """
    score  = z(df_in[tilt_col]) - z(df_in[depth_col])
    median = score.median()
    labels = pd.Series("Pore", index=df_in.index)
    labels[score >= median] = "Carpet"
    return labels, score

def label_tercile_3way(depth_col, tilt_col, df_in):
    """
    3-way tercile split: score = z(tilt) - z(depth)
    Top tercile    → Carpet   (high tilt, low depth)
    Bottom tercile → Barrel   (low tilt, high depth)
    Middle tercile → Toroidal (intermediate)
    """
    score = z(df_in[tilt_col]) - z(df_in[depth_col])
    q33, q67 = score.quantile(1/3), score.quantile(2/3)
    labels = pd.Series("Toroidal_pore", index=df_in.index)
    labels[score >= q67] = "Carpet"
    labels[score <= q33] = "Barrel_stave"
    return labels, score

df["Label_FMAP"],    df["Score_FMAP"]    = label_median("Depth_FMAP", "Tilt_FMAP", df)
df["Label_PPM"],     df["Score_PPM"]     = label_median("Depth_PPM",  "Tilt_PPM",  df)
df["Label3_FMAP"],   df["Score3_FMAP"]   = label_tercile_3way("Depth_FMAP", "Tilt_FMAP", df)
df["Label3_PPM"],    df["Score3_PPM"]    = label_tercile_3way("Depth_PPM",  "Tilt_PPM",  df)

# --- consensus reduced (Carpet vs Pore) ---
consensus = df[df["Label_FMAP"] == df["Label_PPM"]].copy()
consensus["Label"] = consensus["Label_FMAP"]

# --- consensus detailed (3-way) ---
consensus3 = df[df["Label3_FMAP"] == df["Label3_PPM"]].copy()
consensus3["Label"] = consensus3["Label3_FMAP"]

n_total = len(df)
print(f"\nFMAP — Carpet: {(df['Label_FMAP']=='Carpet').sum()} "
      f"({100*(df['Label_FMAP']=='Carpet').sum()/n_total:.1f}%)  "
      f"Pore: {(df['Label_FMAP']=='Pore').sum()} "
      f"({100*(df['Label_FMAP']=='Pore').sum()/n_total:.1f}%)")
print(f"PPM  — Carpet: {(df['Label_PPM']=='Carpet').sum()} "
      f"({100*(df['Label_PPM']=='Carpet').sum()/n_total:.1f}%)  "
      f"Pore: {(df['Label_PPM']=='Pore').sum()} "
      f"({100*(df['Label_PPM']=='Pore').sum()/n_total:.1f}%)")
print(f"Consensus reduced:  {len(consensus)}/{n_total} "
      f"({100*len(consensus)/n_total:.1f}%)  "
      f"Carpet: {(consensus['Label']=='Carpet').sum()} "
      f"({100*(consensus['Label']=='Carpet').sum()/n_total:.1f}%)  "
      f"Pore: {(consensus['Label']=='Pore').sum()} "
      f"({100*(consensus['Label']=='Pore').sum()/n_total:.1f}%)")
print(f"\n3-way tercile:")
for tool, col in [("FMAP","Label3_FMAP"),("PPM","Label3_PPM")]:
    cts = df[col].value_counts()
    print(f"  {tool}: " + "  ".join(
        f"{m}: {cts.get(m,0)} ({100*cts.get(m,0)/n_total:.1f}%)"
        for m in ["Carpet","Toroidal_pore","Barrel_stave"]))
print(f"Consensus detailed: {len(consensus3)}/{n_total} "
      f"({100*len(consensus3)/n_total:.1f}%)  " + "  ".join(
      f"{m}: {(consensus3['Label']==m).sum()} "
      f"({100*(consensus3['Label']==m).sum()/n_total:.1f}%)"
      for m in ["Carpet","Toroidal_pore","Barrel_stave"]))

# =========================
# SPEARMAN CORRELATION
# =========================

print(f"\n=== Spearman ρ (n={len(consensus)}) ===")
global_rho = {}
for var in VARIABLES:
    fc, pc = TOOL_COLS[var]
    rho, p = spearmanr(consensus[fc], consensus[pc])
    global_rho[var] = {"rho": round(rho, 3), "p": round(p, 6)}
    print(f"  {var}: ρ={rho:.3f}  p={p:.4f}")

carpet_idx = consensus.index[consensus["Label"] == "Carpet"].tolist()
pore_idx   = consensus.index[consensus["Label"] == "Pore"].tolist()
n_draw     = min(N_DRAW, len(carpet_idx), len(pore_idx))

print(f"\nRobustness: {N_ITER} subsamples of {n_draw} Carpet + {n_draw} Pore")

records, all_subsets = [], []
for _ in range(N_ITER):
    sel  = np.concatenate([RNG.choice(carpet_idx, n_draw, replace=False),
                           RNG.choice(pore_idx,   n_draw, replace=False)])
    samp = consensus.loc[sel].copy()
    if len(samp) < MIN_SAMPLE: continue
    all_subsets.append(samp)
    row = {"n": len(samp)}
    for var in VARIABLES:
        fc, pc = TOOL_COLS[var]
        rho, p = spearmanr(samp[fc], samp[pc])
        row[f"{var}_rho"] = rho
        row[f"{var}_p"]   = p
    records.append(row)

res = pd.DataFrame(records)
print(f"Robustness ρ (mean ± std, {len(res)} subsamples):")
print(res[[f"{v}_rho" for v in VARIABLES]].agg(["mean","std"]).round(3))

rep_idx = (res[[f"{v}_rho" for v in VARIABLES]].mean(axis=1)
           .sub(res[[f"{v}_rho" for v in VARIABLES]].mean(axis=1).median())
           .abs().idxmin())
rep_sub = all_subsets[rep_idx]

# =========================
# DISCRIMINATION
# =========================

def cohen_d(a, b):
    pooled = np.sqrt((a.std()**2 + b.std()**2) / 2)
    return abs(a.mean() - b.mean()) / pooled if pooled > 0 else 0

disc_rows = []
for tool in ["FMAP", "PPM"]:
    for var in VARIABLES:
        col    = TOOL_COLS[var][0 if tool == "FMAP" else 1]
        carpet = consensus.loc[consensus["Label"] == "Carpet", col]
        pore   = consensus.loc[consensus["Label"] == "Pore",   col]
        d      = cohen_d(carpet, pore)
        _, p   = mannwhitneyu(carpet, pore, alternative="two-sided")
        ks, _  = ks_2samp(carpet, pore)
        y_true = np.concatenate([np.ones(len(carpet)), np.zeros(len(pore))])
        y_sc   = np.concatenate([carpet.values, pore.values])
        auc    = max(roc_auc_score(y_true, y_sc), 1-roc_auc_score(y_true, y_sc))
        disc_rows.append({"Tool": tool, "Variable": var,
                          "Cohen_d": round(d,3), "AUC": round(auc,3),
                          "KS": round(ks,3), "p_mw": round(p,6),
                          "n_carpet": len(carpet), "n_pore": len(pore)})

disc = pd.DataFrame(disc_rows)
print("\n=== Discrimination — Reduced (Carpet vs Pore, median split consensus) ===")
print(disc[["Tool","Variable","Cohen_d","AUC","KS","p_mw"]].to_string(index=False))

# --- 3-way discrimination ---
from itertools import combinations as _comb

MECHS3 = ["Carpet", "Toroidal_pore", "Barrel_stave"]
PALETTE3 = {"Carpet": "#E07B54", "Toroidal_pore": "#4C72B0", "Barrel_stave": "#3BAF77"}

disc3_rows = []
for tool in ["FMAP", "PPM"]:
    for m1, m2 in _comb(MECHS3, 2):
        for var in VARIABLES:
            col = TOOL_COLS[var][0 if tool == "FMAP" else 1]
            g1  = consensus3.loc[consensus3["Label"] == m1, col].dropna()
            g2  = consensus3.loc[consensus3["Label"] == m2, col].dropna()
            if len(g1) < 5 or len(g2) < 5: continue
            d      = cohen_d(g1, g2)
            _, p   = mannwhitneyu(g1, g2, alternative="two-sided")
            ks, _  = ks_2samp(g1, g2)
            y_true = np.concatenate([np.ones(len(g1)), np.zeros(len(g2))])
            y_sc   = np.concatenate([g1.values, g2.values])
            auc    = max(roc_auc_score(y_true, y_sc), 1-roc_auc_score(y_true, y_sc))
            disc3_rows.append({"Tool": tool, "Comparison": f"{m1} vs {m2}",
                               "Variable": var, "Cohen_d": round(d,3),
                               "AUC": round(auc,3), "KS": round(ks,3),
                               "p_mw": round(p,6),
                               "n1": len(g1), "n2": len(g2)})

disc3 = pd.DataFrame(disc3_rows)
print("\n=== Discrimination — Detailed (Carpet/Toroidal/Barrel, 3-way tercile consensus) ===")
print(disc3[["Tool","Comparison","Variable","Cohen_d","AUC","KS","p_mw"]].to_string(index=False))

print("\n=== Which tool separates better? (Carpet vs Pore) ===")
for var in VARIABLES:
    fmap_d = disc.loc[(disc["Tool"]=="FMAP")&(disc["Variable"]==var), "Cohen_d"].values[0]
    ppm_d  = disc.loc[(disc["Tool"]=="PPM") &(disc["Variable"]==var), "Cohen_d"].values[0]
    better = "PPM" if ppm_d > fmap_d else "FMAP"
    diff   = abs(ppm_d - fmap_d)
    print(f"  {var}: {better} (Δd={diff:.3f}  FMAP={fmap_d:.3f}  PPM={ppm_d:.3f})")

# =========================
# BLAND-ALTMAN SUMMARY
# =========================

ba_rows = []
for var in VARIABLES:
    fc, pc = TOOL_COLS[var]
    diff_v = df[fc] - df[pc]
    md, sd = diff_v.mean(), diff_v.std()
    ba_rows.append({"Variable": var, "Bias": round(md,3), "SD": round(sd,3),
                    "LoA_upper": round(md+1.96*sd,3),
                    "LoA_lower": round(md-1.96*sd,3), "n": len(df)})
ba_summary = pd.DataFrame(ba_rows)

# =========================
# SAVE EXCEL
# =========================

out_xlsx = OUT_DIR / "02_stats_summary_tercile.xlsx"
with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
    # all peptides with both tools' params + labels
    df.to_excel(writer,          sheet_name="All_data",             index=False)
    # consensus datasets
    consensus.to_excel(writer,   sheet_name="Consensus_reduced",    index=False)
    consensus3.to_excel(writer,  sheet_name="Consensus_3way",       index=False)
    # discrimination stats
    disc.to_excel(writer,        sheet_name="Discrim_reduced",      index=False)
    disc3.to_excel(writer,       sheet_name="Discrim_3way",         index=False)
    # bias and correlation
    ba_summary.to_excel(writer,  sheet_name="Bland_Altman",         index=False)
    res.to_excel(writer,         sheet_name="Spearman_robustness",  index=False)
    pd.DataFrame([{"Variable": v, **global_rho[v]} for v in VARIABLES]
                 ).to_excel(writer, sheet_name="Spearman_global",   index=False)

print(f"\n✅ Excel saved: {out_xlsx}")

# =========================
# PLOTS
# =========================

# --- 01: Bland-Altman ---
fig, axes = plt.subplots(1, len(VARIABLES), figsize=(6.5*len(VARIABLES), 5))
if len(VARIABLES) == 1: axes = [axes]
for ax, var in zip(axes, VARIABLES):
    fc, pc  = TOOL_COLS[var]
    mean_v  = (df[fc] + df[pc]) / 2
    diff_v  = df[fc] - df[pc]
    md, sd  = diff_v.mean(), diff_v.std()
    for label, color in PALETTE.items():
        mask = df["Label_FMAP"] == label
        ax.scatter(mean_v[mask], diff_v[mask], c=color,
                   alpha=0.5, s=25, edgecolors="none", label=label)
    ax.axhline(md,         color="black",   lw=1.5, label=f"Bias={md:.2f}")
    ax.axhline(md+1.96*sd, color="red",     lw=1, ls="--", label=f"+1.96SD={md+1.96*sd:.2f}")
    ax.axhline(md-1.96*sd, color="red",     lw=1, ls="--", label=f"−1.96SD={md-1.96*sd:.2f}")
    ax.axhline(0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel(f"Mean FMAP & PPM ({var})", fontsize=10)
    ax.set_ylabel(f"FMAP − PPM ({var})", fontsize=10)
    ax.set_title(f"Bland-Altman — {var}\nbias={md:.2f} "
                 f"(LoA: {md-1.96*sd:.2f} to {md+1.96*sd:.2f})", fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.8)
fig.suptitle("Section 1 — Bias analysis: FMAP vs PPM", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "01_bland_altman.png", dpi=DPI); plt.close()

# --- 02: Scatter full consensus ---
fig, axes = plt.subplots(1, len(VARIABLES), figsize=(6.5*len(VARIABLES), 5))
if len(VARIABLES) == 1: axes = [axes]
for ax, var in zip(axes, VARIABLES):
    fc, pc = TOOL_COLS[var]
    for label, color in PALETTE.items():
        grp = consensus[consensus["Label"] == label]
        ax.scatter(grp[fc], grp[pc], c=color, alpha=0.75, s=55,
                   edgecolors="none", label=f"{label} (n={len(grp)})")
    lims = [min(consensus[fc].min(), consensus[pc].min())-1,
            max(consensus[fc].max(), consensus[pc].max())+1]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.4, label="y=x")
    sns.regplot(x=consensus[fc], y=consensus[pc], scatter=False, ci=95,
                line_kws=dict(color="black", lw=1.2), ax=ax)
    ax.set_title(f"{var}  ρ={global_rho[var]['rho']:.2f}  "
                 f"p={global_rho[var]['p']:.4f}", fontweight="bold")
    ax.set_xlabel(f"{var} FMAP", fontsize=10)
    ax.set_ylabel(f"{var} PPM", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.7)
fig.suptitle(f"Section 2 — Correlation FMAP vs PPM (consensus n={len(consensus)})",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "02_scatter_correlation_full.png", dpi=DPI); plt.close()

# --- 03: Scatter subsample ---
fig, axes = plt.subplots(1, len(VARIABLES), figsize=(6.5*len(VARIABLES), 5))
if len(VARIABLES) == 1: axes = [axes]
for ax, var in zip(axes, VARIABLES):
    fc, pc = TOOL_COLS[var]
    for label, color in PALETTE.items():
        grp = rep_sub[rep_sub["Label"] == label]
        ax.scatter(grp[fc], grp[pc], c=color, alpha=0.75, s=55,
                   edgecolors="none", label=f"{label} (n={len(grp)})")
    lims = [min(rep_sub[fc].min(), rep_sub[pc].min())-1,
            max(rep_sub[fc].max(), rep_sub[pc].max())+1]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.4, label="y=x")
    sns.regplot(x=rep_sub[fc], y=rep_sub[pc], scatter=False, ci=95,
                line_kws=dict(color="black", lw=1.2), ax=ax)
    ax.set_title(f"{var}  global ρ={global_rho[var]['rho']:.2f}  "
                 f"sub ρ={res[f'{var}_rho'].mean():.2f}±"
                 f"{res[f'{var}_rho'].std():.2f}", fontweight="bold")
    ax.set_xlabel(f"{var} FMAP", fontsize=10)
    ax.set_ylabel(f"{var} PPM", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.7)
fig.suptitle(f"Section 2 — Robustness: representative subsample (n={n_draw*2})",
             fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "03_scatter_correlation_subsample.png", dpi=DPI); plt.close()

# --- 04: Robustness distribution ---
res_long = pd.concat([
    res[[f"{v}_rho"]].rename(columns={f"{v}_rho": "Spearman ρ"}).assign(Variable=v)
    for v in VARIABLES], ignore_index=True)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
ax = axes[0]
sns.boxplot(data=res_long, x="Variable", y="Spearman ρ", hue="Variable",
            palette=VAR_PAL, width=0.45, legend=False,
            flierprops=dict(marker="o", markersize=3, alpha=0.4), ax=ax)
sns.stripplot(data=res_long, x="Variable", y="Spearman ρ",
              color="black", alpha=0.15, size=3, jitter=True, ax=ax)
for v in VARIABLES:
    ax.axhline(global_rho[v]["rho"], color=VAR_PAL[v], lw=1.5, ls="--", alpha=0.8)
ax.axhline(0, color="grey", lw=1, ls=":")
ax.set_title(f"Robustness ρ — {len(res)} subsamples\ndashed=global ρ", fontweight="bold")
ax = axes[1]
for v in VARIABLES:
    sns.kdeplot(res[f"{v}_rho"], fill=True, alpha=0.4, color=VAR_PAL[v],
                label=f"{v} (ρ={global_rho[v]['rho']:.2f})", ax=ax)
ax.axvline(0, color="grey", lw=1, ls="--")
ax.set_xlabel("Spearman ρ"); ax.set_title("ρ distribution", fontweight="bold")
ax.legend()
fig.suptitle(f"Section 2 — Robustness: ρ across {N_ITER} subsamples",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "04_correlation_robustness.png", dpi=DPI); plt.close()

# --- 05: Split violin discrimination (one half Carpet, one half Pore per tool) ---
fig, axes = plt.subplots(1, len(VARIABLES), figsize=(6.5*len(VARIABLES), 5))
if len(VARIABLES) == 1: axes = [axes]

for ax, var in zip(axes, VARIABLES):
    rows_v = []
    for tool in ["FMAP", "PPM"]:
        col = TOOL_COLS[var][0 if tool == "FMAP" else 1]
        tmp = consensus[["Label", col]].copy()
        tmp.columns = ["Label", "Value"]
        tmp["Tool"] = tool
        rows_v.append(tmp)
    plot_df = pd.concat(rows_v)

    sns.violinplot(data=plot_df, x="Tool", y="Value", hue="Label",
                   hue_order=["Carpet", "Pore"], palette=PALETTE,
                   split=True, inner="quart", linewidth=0.9, ax=ax)

    for i, tool in enumerate(["FMAP", "PPM"]):
        row   = disc[(disc["Tool"]==tool) & (disc["Variable"]==var)].iloc[0]
        p_str = "p<0.001" if row["p_mw"] < 0.001 else f"p={row['p_mw']:.3f}"
        ax.text(i, ax.get_ylim()[1]*0.97,
                f"d={row['Cohen_d']:.2f}\nAUC={row['AUC']:.2f}\n{p_str}",
                ha="center", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    ax.set_title(f"{var}: Carpet vs Pore", fontweight="bold")
    ax.set_xlabel("Tool", fontsize=10)
    ax.set_ylabel(f"{var}", fontsize=10)
    ax.legend(title="Mechanism", fontsize=9, framealpha=0.7)

fig.suptitle("Section 3 — Discrimination: Carpet vs Pore (tercile labels)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "05_violin_discrimination.png", dpi=DPI); plt.close()

# --- 06: Effect size bar chart ---
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
x = np.arange(len(VARIABLES)); w = 0.35
for ax, metric, ylabel, refs in zip(axes,
    ["Cohen_d","AUC"],["Cohen's d","AUC-ROC"],
    [[(0.5,"--","medium"),(0.8,":","large")],
     [(0.7,"--","good"),(0.9,":","excellent")]]):
    for j, tool in enumerate(["FMAP","PPM"]):
        vals = [disc.loc[(disc["Tool"]==tool)&(disc["Variable"]==v), metric].values[0]
                for v in VARIABLES]
        bars = ax.bar(x+j*w, vals, w, label=tool,
                      color=BAR_PAL[tool], alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=10)
    for ref, ls, lbl in refs:
        ax.axhline(ref, color="grey", ls=ls, lw=1, alpha=0.5, label=lbl)
    ax.set_xticks(x+w/2); ax.set_xticklabels(VARIABLES)
    ax.set_ylabel(ylabel); ax.set_title(f"{ylabel}", fontweight="bold")
    ax.legend(fontsize=9)
fig.suptitle("Section 3 — Effect size: Carpet vs Pore per tool (tercile labels)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "06_effect_size_summary.png", dpi=DPI); plt.close()

# --- 07: KDE discrimination ---
fig, axes = plt.subplots(len(VARIABLES), 2, figsize=(13, 4.5*len(VARIABLES)))
if len(VARIABLES) == 1: axes = axes.reshape(1, 2)
for row, var in enumerate(VARIABLES):
    for col, tool in enumerate(["FMAP","PPM"]):
        ax   = axes[row][col]
        vcol = TOOL_COLS[var][0 if tool=="FMAP" else 1]
        for label, color in PALETTE.items():
            grp = consensus[consensus["Label"]==label]
            sns.kdeplot(grp[vcol], fill=True, alpha=0.5, color=color,
                        label=f"{label} (n={len(grp)})", ax=ax)
        d_v   = disc.loc[(disc["Tool"]==tool)&(disc["Variable"]==var),"Cohen_d"].values[0]
        auc_v = disc.loc[(disc["Tool"]==tool)&(disc["Variable"]==var),"AUC"].values[0]
        ax.set_title(f"{tool} — {var}\nd={d_v:.2f}  AUC={auc_v:.2f}", fontweight="bold")
        ax.set_xlabel(f"{var}"); ax.set_ylabel("Density")
        ax.legend(fontsize=9, framealpha=0.7)
fig.suptitle("Section 3 — Distribution overlap: Carpet vs Pore (tercile labels)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "07_kde_discrimination.png", dpi=DPI); plt.close()

# --- 08: Summary heatmap ---
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for ax, metric, title, vmin, vmax in zip(axes,
    ["Cohen_d","AUC"],["Cohen's d","AUC-ROC"],[0,0.5],[3,1.0]):
    pivot = disc.pivot(index="Tool", columns="Variable", values=metric)
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd",
                vmin=vmin, vmax=vmax, linewidths=0.5, ax=ax,
                annot_kws={"size":13,"weight":"bold"})
    ax.set_title(title, fontweight="bold"); ax.set_xlabel("")
fig.suptitle("Summary heatmap — separation quality (tercile labels)",
             fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "08_summary_heatmap.png", dpi=DPI); plt.close()

# --- 09: 3-way violin (Carpet / Toroidal / Barrel) ---
fig, axes = plt.subplots(1, len(VARIABLES), figsize=(6.5*len(VARIABLES), 5))
if len(VARIABLES) == 1: axes = [axes]
MECH3_ORDER = ["Carpet", "Toroidal_pore", "Barrel_stave"]

for ax, var in zip(axes, VARIABLES):
    rows_v = []
    for tool in ["FMAP", "PPM"]:
        col = TOOL_COLS[var][0 if tool == "FMAP" else 1]
        tmp = consensus3[["Label", col]].copy()
        tmp.columns = ["Label", "Value"]
        tmp["Tool"] = tool
        rows_v.append(tmp)
    plot_df = pd.concat(rows_v)
    sns.violinplot(data=plot_df, x="Tool", y="Value", hue="Label",
                   hue_order=MECH3_ORDER, palette=PALETTE3,
                   inner="quart", linewidth=0.9, ax=ax)
    ax.set_title(f"{var}: Carpet / Toroidal / Barrel", fontweight="bold")
    ax.set_xlabel("Tool", fontsize=10)
    ax.set_ylabel(var, fontsize=10)
    ax.legend(title="Mechanism", fontsize=8, framealpha=0.7)

fig.suptitle("3-way discrimination: Carpet / Toroidal / Barrel (geometric tercile labels)\n"
             "consensus: both tools agree on the same 3-way category",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "09_violin_3way.png", dpi=DPI); plt.close()

# --- 10: KDE 3-way ---
fig, axes = plt.subplots(len(VARIABLES), 2, figsize=(13, 4.5*len(VARIABLES)))
if len(VARIABLES) == 1: axes = axes.reshape(1, 2)
for row, var in enumerate(VARIABLES):
    for col, tool in enumerate(["FMAP","PPM"]):
        ax   = axes[row][col]
        vcol = TOOL_COLS[var][0 if tool=="FMAP" else 1]
        for mech, color in PALETTE3.items():
            grp = consensus3[consensus3["Label"]==mech]
            if len(grp) < 5: continue
            sns.kdeplot(grp[vcol], fill=True, alpha=0.45, color=color,
                        label=f"{mech} (n={len(grp)})", ax=ax)
        ax.set_title(f"{tool} — {var}", fontweight="bold")
        ax.set_xlabel(var); ax.set_ylabel("Density")
        ax.legend(fontsize=8, framealpha=0.7)

fig.suptitle("Distribution: Carpet / Toroidal / Barrel (geometric tercile labels)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "10_kde_3way.png", dpi=DPI); plt.close()

print(f"\n✅ All figures saved to: {FIG_DIR}")
print(f"✅ Excel saved:          {out_xlsx}")