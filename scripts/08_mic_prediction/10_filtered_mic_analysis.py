"""
Usage:
    python 10_filtered_mic_analysis.py
    python 10_filtered_mic_analysis.py --mic_threshold 120
"""

from __future__ import annotations

import argparse
import logging
import sys
from itertools import combinations as _comb
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import (kruskal, mannwhitneyu, spearmanr,
                          ttest_rel, wilcoxon, friedmanchisquare)

logging.getLogger("matplotlib").setLevel(logging.WARNING)
sns.set_theme(style="whitegrid", font_scale=1.05)

# ─────────────────────────────────────────────
# PROJECT ROOT
# ─────────────────────────────────────────────

def _find_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / ".git").exists() or (p / "pyproject.toml").exists():
            return p
    raise RuntimeError("Project root not found.")

ROOT = _find_root()

MOTIF_MIC_FILE = ROOT / "results/12_apex_prediction/02_results/02_motif_mic_summary_merged.csv"
COMB_2CAT_FILE = ROOT / "results/13_motif_combinations/01_combinations/04_merged_predicted_MICs.csv"
COMB_3CAT_FILE = ROOT / "results/13_motif_combinations/01b_combinations_3cat/04_merged_predicted_MICs.csv"
LINKER_2CAT_XL = ROOT / "results/13_motif_combinations/04_linker_analysis_2cat/01_linker_analysis_2cat.xlsx"
LINKER_3CAT_XL = ROOT / "results/13_motif_combinations/04_linker_analysis_3cat/01_linker_analysis_3cat.xlsx"

# base palettes — missing keys filled dynamically with grey
_PAL2_BASE = {"Carpet+Carpet": "#E07B54", "Carpet+Pore": "#5B8DB8",
              "Pore+Pore": "#3BAF77"}
_PAL3_BASE = {"Carpet+Carpet": "#E07B54", "Carpet+Toroidal_pore": "#5B8DB8",
              "Carpet+Barrel_stave": "#9B59B6",
              "Toroidal_pore+Toroidal_pore": "#3BAF77",
              "Barrel_stave+Barrel_stave": "#F0C040",
              "Barrel_stave+Toroidal_pore": "#1ABC9C",
              "Barrel_stave+Carpet": "#9B59B6"}
PAL_COND = {"No_linker": "#888888", "GGG": "#4C72B0", "AAA": "#E07B54"}
PAL_MECH = {"Carpet": "#E07B54", "Pore": "#3BAF77", "Ambiguous": "#AAAAAA",
            "Toroidal_pore": "#5B8DB8", "Barrel_stave": "#9B59B6"}

# fallback colour cycle for any unseen combination type
_FALLBACK_COLORS = ["#AAAAAA","#E8A838","#5FC4C0","#D45087",
                    "#FF7C43","#A05195","#665191"]

def dynamic_palette(ctypes: list, base: dict) -> dict:
    """Return palette with a fallback colour for any key missing from base."""
    result = {}
    fallback_idx = 0
    for ct in ctypes:
        if ct in base:
            result[ct] = base[ct]
        else:
            result[ct] = _FALLBACK_COLORS[fallback_idx % len(_FALLBACK_COLORS)]
            fallback_idx += 1
    return result

DPI    = 300
N_ITER = 800
MIN_N  = 5          # minimum n per group for subsampling
RNG    = np.random.default_rng(42)

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

def setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "16_filtered_analysis.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def subsampled_kw(groups: dict, n_draw: int, n_iter: int = N_ITER):
    """Returns (H_mean, H_std, p_mean, p_std). Returns nans if n_draw < MIN_N."""
    if n_draw < MIN_N:
        return np.nan, np.nan, np.nan, np.nan
    hs, ps = [], []
    for _ in range(n_iter):
        samp = [RNG.choice(v, n_draw, replace=False)
                for v in groups.values() if len(v) >= n_draw]
        if len(samp) >= 2:
            h, p = kruskal(*samp)
            hs.append(h); ps.append(p)
    if not hs:
        return np.nan, np.nan, np.nan, np.nan
    return (float(np.mean(hs)), float(np.std(hs)),
            float(np.mean(ps)), float(np.std(ps)))


def subsampled_mw(groups: dict, n_draw: int, n_iter: int = N_ITER) -> pd.DataFrame:
    """Returns empty DataFrame if n_draw < MIN_N."""
    if n_draw < MIN_N:
        return pd.DataFrame()
    pairs = list(_comb(groups.keys(), 2))
    res   = {f"{a} vs {b}": {"d": [], "p": []} for a, b in pairs}
    for _ in range(n_iter):
        samp = {k: RNG.choice(v, n_draw, replace=False)
                for k, v in groups.items() if len(v) >= n_draw}
        for a, b in pairs:
            if a not in samp or b not in samp: continue
            _, p = mannwhitneyu(samp[a], samp[b], alternative="two-sided")
            res[f"{a} vs {b}"]["d"].append(cohen_d(samp[a], samp[b]))
            res[f"{a} vs {b}"]["p"].append(p)
    rows = []
    for key, r in res.items():
        if r["d"]:
            rows.append({"Comparison": key,
                         "Cohen_d_mean": round(np.mean(r["d"]), 3),
                         "Cohen_d_std":  round(np.std(r["d"]), 3),
                         "p_mean":       round(np.mean(r["p"]), 6),
                         "p_std":        round(np.std(r["p"]), 6)})
    return pd.DataFrame(rows)


def savefig(fig, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  Figure: %s", path.name)


def to_wide(d: dict) -> pd.DataFrame:
    return pd.DataFrame({k: pd.Series(v).reset_index(drop=True)
                         for k, v in d.items()})

# ─────────────────────────────────────────────
# A. MOTIF-LEVEL ANALYSIS
# ─────────────────────────────────────────────

def analyse_motifs(motif_mic: pd.DataFrame, active: list,
                   comb_2: pd.DataFrame, fig_dir: Path) -> dict:
    log.info("\n=== A. MOTIF-LEVEL ANALYSIS ===")
    col_m = "Motif_representative"

    active_df = motif_mic[motif_mic[col_m].isin(active)].copy()
    log.info("Active motifs: %d / %d total", len(active_df), len(motif_mic))

    if "Dominant_reduced" in motif_mic.columns:
        log.info("Full distribution:\n%s",
                 motif_mic["Dominant_reduced"].value_counts().to_string())
        log.info("Active distribution:\n%s",
                 active_df["Dominant_reduced"].value_counts().to_string())

    # ── Score combinado ──
    motif_rows = []
    for motif in active:
        mask = (comb_2["Motif_1"] == motif) | (comb_2["Motif_2"] == motif)
        sub  = comb_2[mask]["MIC_combined"].dropna()
        if len(sub) < 3: continue
        mic_ind = motif_mic.loc[motif_mic[col_m]==motif, "MIC_mean"].values
        mic_ind = float(mic_ind[0]) if len(mic_ind) > 0 else np.nan
        mech    = motif_mic.loc[motif_mic[col_m]==motif,
                                 "Dominant_reduced"].values
        mech    = str(mech[0]) if len(mech) > 0 else ""
        fam     = motif_mic.loc[motif_mic[col_m]==motif, "Family_ID"].values
        fam     = str(fam[0]) if len(fam) > 0 else ""
        mean_c  = sub.mean()
        cv      = sub.std(ddof=1) / mean_c * 100 if mean_c > 0 else np.nan
        score   = mean_c * (1 + cv / 100) if not np.isnan(cv) else np.nan
        motif_rows.append({"Motif": motif, "Mechanism": mech,
                            "Family_ID": fam,
                            "MIC_individual": round(mic_ind, 2),
                            "MIC_mean_in_comb": round(mean_c, 2),
                            "CV": round(cv, 1),
                            "Score_combined": round(score, 2),
                            "n_combinations": len(sub)})
    score_df = pd.DataFrame(motif_rows).sort_values("Score_combined")
    log.info("Top 10 motifs by score:\n%s",
             score_df[["Motif", "Mechanism", "MIC_individual",
                        "MIC_mean_in_comb", "CV", "Score_combined"]
                      ].head(10).to_string(index=False))

    # ── Correlation individual vs in-combination ──
    valid = score_df.dropna(subset=["MIC_individual", "MIC_mean_in_comb"])
    rho, p_rho = spearmanr(valid["MIC_individual"], valid["MIC_mean_in_comb"])
    log.info("Spearman MIC_individual vs MIC_mean_in_comb: ρ=%.3f  p=%.4f  n=%d",
             rho, p_rho, len(valid))

    n_ext = min(20, len(score_df) // 2)
    best_ind  = score_df.nsmallest(n_ext, "MIC_individual")["Motif"]
    worst_ind = score_df.nlargest(n_ext,  "MIC_individual")["Motif"]
    bv = score_df[score_df["Motif"].isin(best_ind)]["MIC_mean_in_comb"].dropna()
    wv = score_df[score_df["Motif"].isin(worst_ind)]["MIC_mean_in_comb"].dropna()
    p_bw = np.nan
    if len(bv) >= 3 and len(wv) >= 3:
        _, p_bw = mannwhitneyu(bv, wv, alternative="two-sided")
        log.info("Top %d best  individual → mean MIC in comb: %.1f", n_ext, bv.mean())
        log.info("Top %d worst individual → mean MIC in comb: %.1f", n_ext, wv.mean())
        log.info("MW best vs worst in comb: p=%.4f", p_bw)

    # ── Position preference ──
    pos_rows = []
    for motif in active:
        as_m1 = comb_2[comb_2["Motif_1"] == motif]["MIC_combined"].dropna()
        as_m2 = comb_2[comb_2["Motif_2"] == motif]["MIC_combined"].dropna()
        if len(as_m1) < 3 or len(as_m2) < 3: continue
        delta = as_m1.mean() - as_m2.mean()
        _, p  = mannwhitneyu(as_m1, as_m2, alternative="two-sided")
        mech_v = score_df.loc[score_df["Motif"]==motif, "Mechanism"].values
        pos_rows.append({"Motif": motif,
                         "Mechanism": mech_v[0] if len(mech_v) > 0 else "",
                         "MIC_as_pos1": round(as_m1.mean(), 2),
                         "MIC_as_pos2": round(as_m2.mean(), 2),
                         "Delta_pos1_minus_pos2": round(delta, 2),
                         "p_mw": round(p, 6),
                         "Prefers": "Position_1" if delta < 0 else "Position_2"})
    pos_df = pd.DataFrame(pos_rows).sort_values("p_mw")
    sig_pos = pos_df[pos_df["p_mw"] < 0.05]
    log.info("Motifs with significant position preference: %d", len(sig_pos))

    # ── Figures ──
    # Fig A1: Score combinado top 30
    top_n = min(30, len(score_df))
    top30  = score_df.head(top_n)
    fig, ax = plt.subplots(figsize=(max(10, top_n * 0.5), 6))
    fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
    colors = [PAL_MECH.get(m, "#ccc") for m in top30["Mechanism"]]
    ax.bar(range(len(top30)), top30["Score_combined"],
           color=colors, alpha=0.85, edgecolor="white")
    ax.set_xticks(range(len(top30)))
    ax.set_xticklabels(top30["Motif"], rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("Score combinado (lower = better)", fontsize=10)
    ax.set_title(f"A1. Score combinado — top {top_n} motivos filtrados\n"
                 "(actividad × consistencia)", fontweight="bold")
    patches = [mpatches.Patch(color=c, label=m) for m, c in PAL_MECH.items()
               if m in top30["Mechanism"].values]
    ax.legend(handles=patches, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig(fig, fig_dir / "A1_score_combinado.png")

    # Fig A2: Scatter individual vs en combinacion
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
    for mech, color in PAL_MECH.items():
        sub = valid[valid["Mechanism"] == mech]
        if sub.empty: continue
        ax.scatter(sub["MIC_individual"], sub["MIC_mean_in_comb"],
                   c=color, alpha=0.75, s=55, edgecolors="white",
                   linewidths=0.4, label=f"{mech} (n={len(sub)})")
    lims = [valid[["MIC_individual", "MIC_mean_in_comb"]].min().min() - 5,
            valid[["MIC_individual", "MIC_mean_in_comb"]].max().max() + 5]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.4, label="y=x")
    ax.annotate(f"ρ={rho:.3f}  p={p_rho:.4f}\nn={len(valid)}",
                xy=(0.05, 0.92), xycoords="axes fraction", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ddd"))
    ax.set_xlabel("MIC individual (µmol/L)", fontsize=11)
    ax.set_ylabel("MIC media en combinaciones (µmol/L)", fontsize=11)
    ax.set_title("A2. Individual predice comportamiento en combinación\n"
                 "(motivos filtrados)", fontweight="bold")
    ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig(fig, fig_dir / "A2_individual_vs_comb.png")

    # Fig A3: Position preference
    if len(sig_pos) >= 3:
        top_pos = sig_pos.head(min(20, len(sig_pos))).sort_values(
            "Delta_pos1_minus_pos2")
        fig, ax = plt.subplots(figsize=(10, max(4, len(top_pos) * 0.4)))
        fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
        colors_p = ["#3BAF77" if d < 0 else "#E07B54"
                    for d in top_pos["Delta_pos1_minus_pos2"]]
        ax.barh(range(len(top_pos)), top_pos["Delta_pos1_minus_pos2"],
                color=colors_p, alpha=0.85, edgecolor="white")
        ax.set_yticks(range(len(top_pos)))
        ax.set_yticklabels(top_pos["Motif"], fontsize=9)
        ax.axvline(0, color="black", lw=1, alpha=0.5)
        ax.set_xlabel("ΔMIC (pos1 − pos2)\nnegative = prefers N-term (pos1)", fontsize=10)
        ax.set_title("A3. Preferencia de posición (p<0.05)\n"
                     "verde = mejor en C-term, naranja = mejor en N-term",
                     fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        savefig(fig, fig_dir / "A3_position_preference.png")

    return {
        "score": score_df, "pos": pos_df,
        "corr": pd.DataFrame([{
            "Spearman_rho": round(rho, 3), "p": round(p_rho, 6),
            "n": len(valid),
            "mean_best_in_comb":  round(float(bv.mean()), 2) if len(bv) > 0 else np.nan,
            "mean_worst_in_comb": round(float(wv.mean()), 2) if len(wv) > 0 else np.nan,
            "p_bw": round(float(p_bw), 6) if not np.isnan(p_bw) else np.nan,
        }])
    }

# ─────────────────────────────────────────────
# B/C. COMBINATION ANALYSIS
# ─────────────────────────────────────────────

def analyse_combinations(df: pd.DataFrame, ctype_col: str, label: str,
                          fig_dir: Path, base_palette: dict,
                          motif_mic: pd.DataFrame) -> dict:
    log.info("\n=== %s ===", label)
    ctypes = sorted(df[ctype_col].dropna().unique())
    mechs  = sorted({m for ct in ctypes for m in ct.split("+")})
    pal    = dynamic_palette(ctypes, base_palette)

    # ── ΔMIC ──
    log.info("--- ΔMIC ---")
    delta_rows = []
    for ct in ctypes:
        sub = df[df[ctype_col] == ct]
        d   = sub["MIC_delta"].dropna()
        ind = sub["MIC_mean_individual"].dropna()
        if len(d) == 0: continue
        ind_aligned = ind.reindex(d.index)
        syn  = (d < -(ind_aligned * 0.10)).sum()
        intf = (d >  (ind_aligned * 0.10)).sum()
        n    = len(d)
        log.info("  %-40s ΔMIC=%.2f  syn=%.1f%%  int=%.1f%%  n=%d",
                 ct, d.mean(), 100 * syn / n, 100 * intf / n, n)
        delta_rows.append({
            "Combination_type": ct, "n": n,
            "mean_delta":       round(d.mean(), 3),
            "median_delta":     round(d.median(), 3),
            "std_delta":        round(d.std(ddof=1), 3),
            "pct_synergy":      round(100 * syn / n, 1),
            "pct_interference": round(100 * intf / n, 1),
        })
    delta_df = pd.DataFrame(delta_rows)

    # ── KW subsampled ──
    groups  = {ct: df[df[ctype_col] == ct]["MIC_combined"].dropna().values
               for ct in ctypes}
    groups  = {k: v for k, v in groups.items() if len(v) >= MIN_N}
    n_draw  = min((len(v) for v in groups.values()), default=0)

    if n_draw < MIN_N:
        log.warning("n_draw=%d < %d — skipping KW subsampling for %s",
                    n_draw, MIN_N, label)
        h_m = h_s = p_m = p_s = np.nan
        mw_df = pd.DataFrame()
        kw_str = f"KW skipped (n_draw={n_draw} < {MIN_N})"
    else:
        h_m, h_s, p_m, p_s = subsampled_kw(groups, n_draw)
        mw_df = subsampled_mw(groups, n_draw)
        kw_str = (f"KW H={h_m:.2f}±{h_s:.2f}  p={p_m:.4f}±{p_s:.4f}  "
                  f"n={n_draw}/group  {N_ITER} iter")
        log.info("KW subsampled: %s", kw_str)
        if not mw_df.empty:
            log.info("\n%s", mw_df.to_string(index=False))

    # ── Correlation ──
    corr_rows = []
    for ct in ctypes:
        sub = df[df[ctype_col] == ct].dropna(
            subset=["MIC_combined", "MIC_mean_individual"])
        if len(sub) < 5: continue
        rho, p = spearmanr(sub["MIC_combined"], sub["MIC_mean_individual"])
        corr_rows.append({"Combination_type": ct, "n": len(sub),
                           "Spearman_rho": round(rho, 3), "p": round(p, 6)})
    corr_df = pd.DataFrame(corr_rows)

    # ── Direct comparison combined vs individual reference ──
    col_check = ("Dominant_reduced" if "Dominant_reduced" in motif_mic.columns
                 else "Dominant_mechanism")
    mic_by_mech = {}
    for mech in mechs:
        sub = motif_mic[motif_mic[col_check].str.contains(
            mech.split("_")[0], na=False)]["MIC_mean"].dropna()
        mic_by_mech[mech] = sub.values

    direct_rows = []
    for ct in ctypes:
        for mech in set(ct.split("+")):
            if mech not in mic_by_mech or len(mic_by_mech[mech]) == 0: continue
            g1 = df[df[ctype_col] == ct]["MIC_combined"].dropna().values
            g2 = mic_by_mech[mech]
            if len(g1) < 3 or len(g2) < 3: continue
            _, p = mannwhitneyu(g1, g2, alternative="two-sided")
            d    = cohen_d(g1, g2)
            dirs = "↓ lower" if g1.mean() < g2.mean() else "↑ higher"
            log.info("  %s vs %s_ind: p=%.4f  d=%.3f  %s",
                     ct, mech, p, d, dirs)
            direct_rows.append({
                "Comparison": f"{ct} vs {mech}_individual",
                "mean_combined":    round(g1.mean(), 2),
                "mean_individual":  round(g2.mean(), 2),
                "p_mw": round(p, 6), "Cohen_d": round(d, 3),
                "direction": dirs,
            })
    direct_df = pd.DataFrame(direct_rows)

    # ── Order effect ──
    df2 = df.copy()
    df2["Pair_key"] = df2.apply(
        lambda r: "_".join(sorted([str(r["Motif_1"]), str(r["Motif_2"])])),
        axis=1)
    pairs_g = df2.groupby("Pair_key").filter(lambda g: len(g) == 2)
    order_df = pd.DataFrame()
    if len(pairs_g) >= 10:
        pa = pairs_g.groupby("Pair_key").apply(
            lambda g: g.sort_values("Combination_ID").iloc[0]["MIC_combined"])
        pb = pairs_g.groupby("Pair_key").apply(
            lambda g: g.sort_values("Combination_ID").iloc[1]["MIC_combined"])
        valid_o = pa.notna() & pb.notna()
        if valid_o.sum() >= 5:
            t, p_ord = ttest_rel(pa[valid_o], pb[valid_o])
            diff = (pa[valid_o] - pb[valid_o]).abs().mean()
            log.info("Order effect: t=%.3f  p=%.4f  mean|Δ|=%.2f  n=%d",
                     t, p_ord, diff, valid_o.sum())
            order_df = pd.DataFrame([{
                "n_pairs": int(valid_o.sum()),
                "t_stat": round(t, 3), "p": round(p_ord, 6),
                "mean_abs_delta": round(diff, 3),
            }])

    # ── Same-family vs cross-family ──
    fam_df = pd.DataFrame()
    if "Family_1" in df.columns and "Family_2" in df.columns:
        same_f = (df["Family_1"] == df["Family_2"]) & df["Family_1"].notna()
        s_mic  = df[same_f]["MIC_combined"].dropna()
        c_mic  = df[~same_f]["MIC_combined"].dropna()
        if len(s_mic) >= 3 and len(c_mic) >= 3:
            _, p_sf = mannwhitneyu(s_mic, c_mic, alternative="two-sided")
            d_sf    = cohen_d(s_mic.values, c_mic.values)
            log.info("Same-family %.1f  Cross-family %.1f  p=%.4f  d=%.3f",
                     s_mic.mean(), c_mic.mean(), p_sf, d_sf)
            fam_df = pd.DataFrame([{
                "n_same": len(s_mic), "n_cross": len(c_mic),
                "mean_same":  round(s_mic.mean(), 2),
                "mean_cross": round(c_mic.mean(), 2),
                "p_mw": round(p_sf, 6), "Cohen_d": round(d_sf, 3),
            }])
    else:
        same_f = pd.Series(False, index=df.index)

    # ── Top combinations ──
    top20       = df.nsmallest(20, "MIC_combined")
    top20_cross = df[~same_f].nsmallest(20, "MIC_combined")
    dom_motifs  = (pd.concat([top20["Motif_1"], top20["Motif_2"]])
                   .value_counts().head(3).index.tolist())
    top20_nodom = df[~same_f &
                     ~df["Motif_1"].isin(dom_motifs) &
                     ~df["Motif_2"].isin(dom_motifs)
                     ].nsmallest(20, "MIC_combined")
    log.info("Top combination: %s + %s  MIC=%.2f  ΔMIC=%.2f",
             top20.iloc[0]["Motif_1"], top20.iloc[0]["Motif_2"],
             top20.iloc[0]["MIC_combined"], top20.iloc[0]["MIC_delta"])

    # ── FIGURES ──

    # Fig 1: Violin ΔMIC
    fig, ax = plt.subplots(figsize=(max(8, len(ctypes) * 2.5), 5))
    fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
    rows_p = [{"Type": ct, "MIC_delta": v}
              for ct in ctypes
              for v in df[df[ctype_col] == ct]["MIC_delta"].dropna()]
    if rows_p:
        sns.violinplot(data=pd.DataFrame(rows_p), x="Type", y="MIC_delta",
                       hue="Type", palette=pal, inner="quart",
                       linewidth=0.9, legend=False, ax=ax)
    ax.axhline(0, color="black", lw=1.2, ls="--", alpha=0.6)
    ax.set_xlabel("")
    ax.set_ylabel("ΔMIC = combined − mean(individual) (µmol/L)")
    ax.set_title(f"{label} — ΔMIC by combination type\n{kw_str}",
                 fontweight="bold", fontsize=9)
    ticks = ax.get_xticklabels()
    if ticks:
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels([t.get_text() for t in ticks],
                           rotation=25, ha="right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig(fig, fig_dir / f"{label}_01_delta_mic.png")

    # Fig 2: Heatmap mean MIC per mechanism pair
    hmap = pd.DataFrame(index=mechs, columns=mechs, dtype=float)
    for m1 in mechs:
        for m2 in mechs:
            key = "+".join(sorted([m1, m2]))
            sub = df[df[ctype_col] == key]["MIC_combined"].dropna()
            hmap.loc[m1, m2] = sub.mean() if len(sub) > 0 else np.nan
    fig, ax = plt.subplots(figsize=(max(5, len(mechs) * 2),
                                     max(4, len(mechs) * 1.8)))
    fig.patch.set_facecolor("#fafafa")
    sns.heatmap(hmap.astype(float), annot=True, fmt=".1f",
                cmap="YlOrRd_r", linewidths=0.5, ax=ax,
                annot_kws={"size": 11, "weight": "bold"},
                cbar_kws={"label": "Mean MIC (µmol/L)"})
    ax.set_title(f"{label} — Mean MIC per mechanism combination\n"
                 "lower = more active", fontweight="bold")
    fig.tight_layout()
    savefig(fig, fig_dir / f"{label}_02_heatmap_mechs.png")

    # Fig 3: Scatter combined vs individual (up to 3 types)
    ncols = min(len(ctypes), 3)
    if ncols >= 1:
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5), sharey=True)
        if ncols == 1: axes = [axes]
        fig.patch.set_facecolor("#fafafa")
        for ax, ct in zip(axes, ctypes[:3]):
            ax.set_facecolor("#fafafa")
            sub = df[df[ctype_col] == ct].dropna(
                subset=["MIC_combined", "MIC_mean_individual"])
            color = pal.get(ct, "#888")
            ax.scatter(sub["MIC_mean_individual"], sub["MIC_combined"],
                       c=color, alpha=0.3, s=10, edgecolors="none")
            if len(sub) >= 2:
                lims = [min(sub["MIC_mean_individual"].min(),
                            sub["MIC_combined"].min()) - 2,
                        max(sub["MIC_mean_individual"].max(),
                            sub["MIC_combined"].max()) + 2]
                ax.plot(lims, lims, "k--", lw=1, alpha=0.4)
            rho_v = corr_df.loc[corr_df["Combination_type"]==ct,
                                  "Spearman_rho"].values
            ax.set_title(f"{ct}\nρ={rho_v[0]:.3f}" if len(rho_v) > 0
                         else ct, fontweight="bold", fontsize=9)
            ax.set_xlabel("Mean MIC individual (µmol/L)", fontsize=8)
            ax.set_ylabel("MIC combined (µmol/L)" if ax == axes[0] else "",
                          fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"{label} — Correlation combined vs individual",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        savefig(fig, fig_dir / f"{label}_03_scatter_corr.png")

    # Fig 4: Top 20 combinations by ΔMIC (synergy)
    if len(df) >= 5:
        top20_delta = df.nsmallest(min(20, len(df)), "MIC_delta")
        fig, ax = plt.subplots(figsize=(12, max(4, len(top20_delta) * 0.4)))
        fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
        colors_d = [pal.get(ct, "#888") for ct in top20_delta[ctype_col]]
        ax.barh(range(len(top20_delta)), top20_delta["MIC_delta"],
                color=colors_d, alpha=0.85, edgecolor="white")
        ax.set_yticks(range(len(top20_delta)))
        ax.set_yticklabels([f"{r['Motif_1']}+{r['Motif_2']}"
                            for _, r in top20_delta.iterrows()], fontsize=7.5)
        ax.axvline(0, color="black", lw=1, alpha=0.5)
        ax.set_xlabel("ΔMIC (combined − mean individual)", fontsize=10)
        ax.set_title(f"{label} — Top combinations by ΔMIC (best synergy)",
                     fontweight="bold")
        patches = [mpatches.Patch(color=c, label=m)
                   for m, c in pal.items() if m in top20_delta[ctype_col].values]
        if patches: ax.legend(handles=patches, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        savefig(fig, fig_dir / f"{label}_04_top20_delta.png")

    # Fig 5: Same-family vs cross-family
    if not fam_df.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
        s_vals = df[same_f]["MIC_combined"].dropna()
        c_vals = df[~same_f]["MIC_combined"].dropna()
        plot_sf = pd.DataFrame({
            "Group": (["Same family"] * len(s_vals) +
                      ["Cross family"] * len(c_vals)),
            "MIC": pd.concat([s_vals, c_vals]).values,
        })
        sns.violinplot(data=plot_sf, x="Group", y="MIC", hue="Group",
                       palette={"Same family": "#CCCCCC",
                                "Cross family": "#5B8DB8"},
                       inner="quart", linewidth=0.9, legend=False, ax=ax)
        d_sf = fam_df["Cohen_d"].values[0]
        p_sf = fam_df["p_mw"].values[0]
        ax.set_title(f"{label} — Same-family vs cross-family\n"
                     f"p={p_sf:.4f}  Cohen_d={d_sf:.3f}", fontweight="bold")
        ax.set_xlabel(""); ax.set_ylabel("MIC combined (µmol/L)")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        savefig(fig, fig_dir / f"{label}_05_samefam_vs_cross.png")

    # Fig 6: Top 20 cross-family no-dominant
    if len(top20_nodom) >= 3:
        fig, ax = plt.subplots(figsize=(12, max(4, len(top20_nodom) * 0.4)))
        fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
        colors_nd = [pal.get(ct, "#888") for ct in top20_nodom[ctype_col]]
        ax.barh(range(len(top20_nodom)), top20_nodom["MIC_combined"],
                color=colors_nd, alpha=0.85, edgecolor="white")
        ax.set_yticks(range(len(top20_nodom)))
        ax.set_yticklabels([f"{r['Motif_1']}+{r['Motif_2']}"
                            for _, r in top20_nodom.iterrows()], fontsize=7.5)
        ax.set_xlabel("MIC combined (µmol/L)", fontsize=10)
        ax.set_title(f"{label} — Top 20 cross-family\n"
                     f"(excl. {', '.join(dom_motifs[:2])})",
                     fontweight="bold")
        patches = [mpatches.Patch(color=c, label=m)
                   for m, c in pal.items() if m in top20_nodom[ctype_col].values]
        if patches: ax.legend(handles=patches, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        savefig(fig, fig_dir / f"{label}_06_top20_nodom.png")

    return {
        "delta": delta_df,
        "kw": pd.DataFrame([{"H_mean": round(h_m, 3) if not np.isnan(h_m) else np.nan,
                               "H_std":  round(h_s, 3) if not np.isnan(h_s) else np.nan,
                               "p_mean": round(p_m, 6) if not np.isnan(p_m) else np.nan,
                               "p_std":  round(p_s, 6) if not np.isnan(p_s) else np.nan,
                               "n_draw": n_draw, "n_iter": N_ITER}]),
        "mw": mw_df, "corr": corr_df, "direct": direct_df,
        "order": order_df, "family": fam_df,
        "top20": top20, "top20_cross": top20_cross, "top20_nodom": top20_nodom,
        "heatmap": hmap,
    }

# ─────────────────────────────────────────────
# D. LINKER ANALYSIS
# ─────────────────────────────────────────────

def analyse_linker(all_data: pd.DataFrame, active: list,
                   label: str, fig_dir: Path) -> dict:
    log.info("\n=== D. LINKER ANALYSIS — %s ===", label)

    # filter using boolean mask (avoids pandas bool dtype bug)
    def filt(df: pd.DataFrame) -> pd.DataFrame:
        m1_ok = df["Motif_1"].isin(active)
        m2_ok = df["Motif_2"].isin(active)
        return df[m1_ok & m2_ok].copy()

    df_none = filt(all_data[all_data["Condition"] == "No_linker"])
    df_ggg  = filt(all_data[all_data["Condition"] == "GGG"])
    df_aaa  = filt(all_data[all_data["Condition"] == "AAA"])
    log.info("n per condition — None:%d  GGG:%d  AAA:%d",
             len(df_none), len(df_ggg), len(df_aaa))

    if len(df_none) == 0:
        log.warning("No data after filter for %s, skipping linker.", label)
        return {}

    groups_l = {
        "No_linker": df_none["MIC_combined"].dropna().values,
        "GGG":       df_ggg["MIC_combined"].dropna().values,
        "AAA":       df_aaa["MIC_combined"].dropna().values,
    }
    groups_l = {k: v for k, v in groups_l.items() if len(v) > 0}

    # KW
    h, p_kw = (np.nan, np.nan)
    if len(groups_l) >= 2:
        h, p_kw = kruskal(*groups_l.values())
    log.info("KW overall: H=%.3f  p=%.4f", h, p_kw)

    mw_rows = []
    for c1, c2 in _comb(list(groups_l.keys()), 2):
        g1, g2 = groups_l[c1], groups_l[c2]
        if len(g1) < 3 or len(g2) < 3: continue
        _, p = mannwhitneyu(g1, g2, alternative="two-sided")
        d    = cohen_d(g1, g2)
        log.info("  %s vs %s: p=%.4f  d=%.3f  means=(%.1f vs %.1f)",
                 c1, c2, p, d, g1.mean(), g2.mean())
        mw_rows.append({"Comparison": f"{c1} vs {c2}",
                         "p": round(p, 6), "Cohen_d": round(d, 3),
                         "mean_c1": round(g1.mean(), 2),
                         "mean_c2": round(g2.mean(), 2)})
    mw_df = pd.DataFrame(mw_rows)

    # Paired Wilcoxon
    pk = ["Motif_1", "Motif_2"]
    merged = (df_none[pk + ["MIC_combined"]]
              .merge(df_ggg[pk + ["MIC_combined"]].rename(
                  columns={"MIC_combined": "MIC_GGG"}), on=pk, how="inner")
              .merge(df_aaa[pk + ["MIC_combined"]].rename(
                  columns={"MIC_combined": "MIC_AAA"}), on=pk, how="inner")
              .rename(columns={"MIC_combined": "MIC_None"}))
    log.info("Matched pairs: %d", len(merged))

    wil_rows = []
    for ca, cb, cola, colb in [
        ("No_linker", "GGG", "MIC_None", "MIC_GGG"),
        ("No_linker", "AAA", "MIC_None", "MIC_AAA"),
        ("GGG",       "AAA", "MIC_GGG",  "MIC_AAA"),
    ]:
        v = merged[[cola, colb]].dropna()
        if len(v) < 5: continue
        stat, p = wilcoxon(v[cola], v[colb])
        delta   = (v[colb] - v[cola]).mean()
        better  = cb if delta < 0 else ca
        log.info("  Wilcoxon %s vs %s: p=%.4f  Δ=%.2f  better=%s",
                 ca, cb, p, delta, better)
        wil_rows.append({"Comparison": f"{ca} vs {cb}",
                          "p": round(p, 6), "mean_delta": round(delta, 3),
                          "better": better, "n": len(v)})
    wil_df = pd.DataFrame(wil_rows)

    # Friedman
    fri_df = pd.DataFrame()
    vf = merged[["MIC_None", "MIC_GGG", "MIC_AAA"]].dropna()
    if len(vf) >= 5:
        stat_f, p_f = friedmanchisquare(
            vf["MIC_None"], vf["MIC_GGG"], vf["MIC_AAA"])
        log.info("Friedman: χ²=%.3f  p=%.4f  n=%d", stat_f, p_f, len(vf))
        fri_df = pd.DataFrame([{"chi2": round(stat_f, 3),
                                 "p": round(p_f, 6), "n": len(vf)}])

    # ΔMIC per condition
    delta_cond_rows = []
    for cond, dfx in [("No_linker", df_none), ("GGG", df_ggg), ("AAA", df_aaa)]:
        if "MIC_delta" not in dfx.columns: continue
        d = dfx["MIC_delta"].dropna()
        if len(d) == 0: continue
        log.info("  ΔMIC %s: mean=%.2f  syn=%.1f%%",
                 cond, d.mean(), 100 * (d < 0).mean())
        delta_cond_rows.append({"Condition": cond,
                                  "mean_delta": round(d.mean(), 3),
                                  "pct_synergy": round(100 * (d < 0).mean(), 1)})
    delta_cond_df = pd.DataFrame(delta_cond_rows)

    summary_df = pd.DataFrame([
        {"Condition": c, "mean": round(v.mean(), 2),
         "median": round(float(np.median(v)), 2), "n": len(v)}
        for c, v in groups_l.items()
    ])

    # ── Figures ──
    # Fig D1: Violin overall
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
    rows_p = [{"Condition": c, "MIC": float(v)}
              for c, vals in groups_l.items() for v in vals]
    pal_d  = dynamic_palette(list(groups_l.keys()), PAL_COND)
    sns.violinplot(data=pd.DataFrame(rows_p), x="Condition", y="MIC",
                   hue="Condition", palette=pal_d,
                   inner="quart", linewidth=0.9, legend=False, ax=ax)
    ax.set_title(f"D. Linker effect — {label} (filtered motifs)\n"
                 f"KW H={h:.3f}  p={p_kw:.4f}", fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel("MIC combined (µmol/L)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig(fig, fig_dir / f"D_{label}_01_violin.png")

    # Fig D2: Paired scatter
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#fafafa")
    for ax, (col_x, col_y, lbl_) in zip(axes, [
        ("MIC_None", "MIC_GGG", "No linker vs GGG"),
        ("MIC_None", "MIC_AAA", "No linker vs AAA"),
    ]):
        ax.set_facecolor("#fafafa")
        v = merged[[col_x, col_y]].dropna()
        if len(v) < 3:
            ax.set_title(f"{lbl_}\n(insufficient data)", fontsize=9)
            continue
        ax.scatter(v[col_x], v[col_y], c="#5B8DB8", alpha=0.3,
                   s=10, edgecolors="none")
        lims = [v[[col_x, col_y]].min().min() - 2,
                v[[col_x, col_y]].max().max() + 2]
        ax.plot(lims, lims, "k--", lw=1, alpha=0.4)
        rho_v, _ = spearmanr(v[col_x], v[col_y])
        ax.annotate(f"ρ={rho_v:.3f}  n={len(v)}",
                    xy=(0.05, 0.92), xycoords="axes fraction", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ddd"))
        ax.set_title(lbl_, fontweight="bold")
        ax.set_xlabel(col_x.replace("MIC_", "MIC "), fontsize=9)
        ax.set_ylabel(col_y.replace("MIC_", "MIC "), fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"D. Paired linker effect — {label}", fontsize=11,
                 fontweight="bold")
    fig.tight_layout()
    savefig(fig, fig_dir / f"D_{label}_02_paired_scatter.png")

    # Fig D3: Heatmap type × condition
    ctype_col_d = "Combination_type"
    type_cond = []
    for cond, dfx in [("No_linker", df_none), ("GGG", df_ggg), ("AAA", df_aaa)]:
        if ctype_col_d not in dfx.columns: continue
        for ct in dfx[ctype_col_d].dropna().unique():
            sub = dfx[dfx[ctype_col_d] == ct]["MIC_combined"].dropna()
            if len(sub) == 0: continue
            type_cond.append({"Combination_type": ct, "Condition": cond,
                               "mean_MIC": round(sub.mean(), 2)})
    if type_cond:
        tc_df  = pd.DataFrame(type_cond)
        pivot  = (tc_df.pivot(index="Combination_type", columns="Condition",
                               values="mean_MIC")
                  .reindex(columns=["No_linker", "GGG", "AAA"]))
        fig, ax = plt.subplots(figsize=(7, max(3, len(pivot) * 0.8)))
        fig.patch.set_facecolor("#fafafa")
        sns.heatmap(pivot.astype(float), annot=True, fmt=".1f",
                    cmap="YlOrRd_r", linewidths=0.5, ax=ax,
                    annot_kws={"size": 10, "weight": "bold"})
        ax.set_title(f"D. Mean MIC — combination type × linker\n{label}",
                     fontweight="bold")
        ax.set_xlabel(""); ax.set_ylabel("")
        fig.tight_layout()
        savefig(fig, fig_dir / f"D_{label}_03_heatmap_type_cond.png")

    return {
        "summary": summary_df, "mw": mw_df, "wilcoxon": wil_df,
        "friedman": fri_df, "delta_cond": delta_cond_df,
        "matched_pairs": merged,
        "kw": pd.DataFrame([{"H": round(h, 3) if not np.isnan(h) else np.nan,
                               "p": round(p_kw, 6) if not np.isnan(p_kw) else np.nan}]),
    }

# ─────────────────────────────────────────────
# E. SENSITIVITY
# ─────────────────────────────────────────────

def sensitivity_analysis(df_full: pd.DataFrame, df_filt: pd.DataFrame,
                          ctype_col: str, fig_dir: Path, label: str) -> pd.DataFrame:
    log.info("\n=== E. SENSITIVITY — %s ===", label)
    rows = []
    for ct in sorted(df_full[ctype_col].dropna().unique()):
        for dataset, tag in [(df_full, "Full"), (df_filt, "Filtered")]:
            sub = dataset[dataset[ctype_col] == ct]["MIC_delta"].dropna()
            if len(sub) == 0: continue
            rows.append({
                "Combination_type": ct, "Dataset": tag, "n": len(sub),
                "mean_delta":        round(sub.mean(), 3),
                "pct_synergy":       round(100 * (sub < 0).mean(), 1),
                "pct_interference":  round(100 * (sub > 0).mean(), 1),
            })
    df_s = pd.DataFrame(rows)
    log.info("\n%s", df_s.to_string(index=False))

    ctypes_u = df_s["Combination_type"].unique()
    if len(ctypes_u) >= 1 and len(df_s) >= 2:
        x = np.arange(len(ctypes_u))
        w = 0.35
        fig, ax = plt.subplots(figsize=(max(8, len(ctypes_u) * 2), 5))
        fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
        for i, (tag, color) in enumerate([("Full", "#AAAAAA"),
                                           ("Filtered", "#5B8DB8")]):
            vals = []
            for ct in ctypes_u:
                v = df_s.loc[(df_s["Combination_type"] == ct) &
                              (df_s["Dataset"] == tag), "mean_delta"].values
                vals.append(v[0] if len(v) > 0 else np.nan)
            ax.bar(x + i * w, vals, width=w, color=color, alpha=0.85,
                   edgecolor="white", label=tag)
        ax.axhline(0, color="black", lw=1, ls="--", alpha=0.5)
        ax.set_xticks(x + w / 2)
        ax.set_xticklabels(ctypes_u, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Mean ΔMIC (µmol/L)")
        ax.set_title(f"E. Sensitivity — Full vs Filtered\n"
                     "ΔMIC by combination type", fontweight="bold")
        ax.legend(fontsize=9); ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        savefig(fig, fig_dir / f"E_{label}_sensitivity_delta.png")

    return df_s

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mic_threshold", type=float, default=100.0,
                   help="Max MIC_individual to include (default 100)")
    return p.parse_args()


def main():
    args    = parse_args()
    thr     = args.mic_threshold
    out_dir = ROOT / "results/15_filtered_analysis"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir)
    log.info("=== 16_filtered_mic_analysis_v3 | threshold=%.0f µmol/L ===", thr)

    # ── LOAD ──
    motif_mic = pd.read_csv(MOTIF_MIC_FILE)
    motif_mic["MIC_mean"] = pd.to_numeric(motif_mic["MIC_mean"], errors="coerce")
    col_m  = "Motif_representative"
    active = motif_mic[motif_mic["MIC_mean"] <= thr][col_m].tolist()
    log.info("Active motifs: %d / %d", len(active), len(motif_mic))

    comb_2 = pd.read_csv(COMB_2CAT_FILE)
    comb_3 = pd.read_csv(COMB_3CAT_FILE)
    for df in [comb_2, comb_3]:
        for col in ["MIC_combined", "MIC_mean_individual",
                    "MIC_delta", "MIC_motif1", "MIC_motif2"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if ("MIC_mean_individual" not in df.columns and
                "MIC_motif1" in df.columns):
            df["MIC_mean_individual"] = (
                (df["MIC_motif1"] + df["MIC_motif2"]) / 2)
        if ("MIC_delta" not in df.columns and
                "MIC_mean_individual" in df.columns):
            df["MIC_delta"] = df["MIC_combined"] - df["MIC_mean_individual"]

    def filt(df: pd.DataFrame) -> pd.DataFrame:
        m1_ok = df["Motif_1"].isin(active)
        m2_ok = df["Motif_2"].isin(active)
        return df[m1_ok & m2_ok].copy()

    comb_2_f = filt(comb_2)
    comb_3_f = filt(comb_3)
    log.info("2cat: %d → %d  (%.1f%%)",
             len(comb_2), len(comb_2_f),
             100 * len(comb_2_f) / max(len(comb_2), 1))
    log.info("3cat: %d → %d  (%.1f%%)",
             len(comb_3), len(comb_3_f),
             100 * len(comb_3_f) / max(len(comb_3), 1))

    # add family columns
    fam_map = (motif_mic.set_index(col_m)["Family_ID"].to_dict()
               if "Family_ID" in motif_mic.columns else {})
    for df in [comb_2_f, comb_3_f]:
        if "Family_1" not in df.columns:
            df["Family_1"] = df["Motif_1"].map(fam_map)
        if "Family_2" not in df.columns:
            df["Family_2"] = df["Motif_2"].map(fam_map)
        df["Same_family"] = ((df["Family_1"] == df["Family_2"]) &
                              df["Family_1"].notna())

    # ── A ──
    res_motif = analyse_motifs(motif_mic, active, comb_2_f, fig_dir)

    # ── B ──
    res_2 = analyse_combinations(comb_2_f, "Combination_type",
                                  "B_2cat", fig_dir, _PAL2_BASE, motif_mic)

    # ── C ──
    ctype_3 = ("Comb3_type" if "Comb3_type" in comb_3_f.columns
               else "Combination_type")
    res_3 = analyse_combinations(comb_3_f, ctype_3,
                                  "C_3cat", fig_dir, _PAL3_BASE, motif_mic)

    # ── D ──
    linker_res = {}
    for cat, xl_path in [("2cat", LINKER_2CAT_XL), ("3cat", LINKER_3CAT_XL)]:
        if xl_path.exists():
            try:
                all_data = pd.read_excel(xl_path, sheet_name="All_data")
                all_data["MIC_combined"] = pd.to_numeric(
                    all_data["MIC_combined"], errors="coerce")
                if ("MIC_delta" not in all_data.columns and
                        "MIC_mean_individual" in all_data.columns):
                    all_data["MIC_delta"] = (all_data["MIC_combined"] -
                                              all_data["MIC_mean_individual"])
                linker_res[cat] = analyse_linker(
                    all_data, active, cat, fig_dir)
            except Exception as e:
                log.warning("Linker %s: %s", cat, e)

    # ── E ──
    sens_2 = sensitivity_analysis(
        comb_2, comb_2_f, "Combination_type",
        fig_dir, f"2cat_thr{int(thr)}")
    ctype_3_full = ("Comb3_type" if "Comb3_type" in comb_3.columns
                    else "Combination_type")
    sens_3 = sensitivity_analysis(
        comb_3, comb_3_f, ctype_3_full,
        fig_dir, f"3cat_thr{int(thr)}")

    # ── SAVE EXCEL ──
    out_xl = out_dir / "filtered_analysis.xlsx"
    log.info("\nSaving Excel: %s", out_xl)

    with pd.ExcelWriter(out_xl, engine="openpyxl") as w:

        pd.DataFrame([{
            "mic_threshold":     thr,
            "n_active_motifs":   len(active),
            "n_total_motifs":    len(motif_mic),
            "n_comb_2cat_full":  len(comb_2),
            "n_comb_2cat_filt":  len(comb_2_f),
            "n_comb_3cat_full":  len(comb_3),
            "n_comb_3cat_filt":  len(comb_3_f),
        }]).to_excel(w, sheet_name="Filter_metadata", index=False)

        motif_mic[motif_mic[col_m].isin(active)].to_excel(
            w, sheet_name="Active_motifs", index=False)
        res_motif["score"].to_excel(w, sheet_name="Score_combinado", index=False)
        res_motif["corr"].to_excel(w,  sheet_name="Corr_ind_vs_comb", index=False)
        res_motif["pos"].to_excel(w,   sheet_name="Position_preference", index=False)

        for pfx, res in [("2cat", res_2), ("3cat", res_3)]:
            res["delta"].to_excel(w,  sheet_name=f"{pfx}_delta",      index=False)
            res["kw"].to_excel(w,     sheet_name=f"{pfx}_KW",         index=False)
            if not res["mw"].empty:
                res["mw"].to_excel(w, sheet_name=f"{pfx}_MW",         index=False)
            res["corr"].to_excel(w,   sheet_name=f"{pfx}_corr",       index=False)
            res["direct"].to_excel(w, sheet_name=f"{pfx}_direct_comp",index=False)
            res["top20"].to_excel(w,  sheet_name=f"{pfx}_top20",      index=False)
            res["top20_cross"].to_excel(w,sheet_name=f"{pfx}_top20_cross",index=False)
            res["top20_nodom"].to_excel(w,sheet_name=f"{pfx}_top20_nodom",index=False)
            if not res["order"].empty:
                res["order"].to_excel(w,sheet_name=f"{pfx}_order",    index=False)
            if not res["family"].empty:
                res["family"].to_excel(w,sheet_name=f"{pfx}_samefam", index=False)
            res["heatmap"].to_excel(w, sheet_name=f"{pfx}_heatmap_mech")

        sens_2.to_excel(w, sheet_name="Sens_2cat", index=False)
        sens_3.to_excel(w, sheet_name="Sens_3cat", index=False)

        for cat, lr in linker_res.items():
            if not lr: continue
            lr["summary"].to_excel(w,  sheet_name=f"Link_{cat}_summary",index=False)
            lr["kw"].to_excel(w,       sheet_name=f"Link_{cat}_KW",     index=False)
            if not lr["mw"].empty:
                lr["mw"].to_excel(w,   sheet_name=f"Link_{cat}_MW",     index=False)
            if not lr["wilcoxon"].empty:
                lr["wilcoxon"].to_excel(w,sheet_name=f"Link_{cat}_Wilcoxon",
                                         index=False)
            if not lr["friedman"].empty:
                lr["friedman"].to_excel(w,sheet_name=f"Link_{cat}_Friedman",
                                         index=False)
            if not lr["delta_cond"].empty:
                lr["delta_cond"].to_excel(w,sheet_name=f"Link_{cat}_delta",
                                            index=False)
            lr["matched_pairs"].to_excel(w,sheet_name=f"Link_{cat}_pairs",
                                          index=False)

        # GraphPad sheets
        to_wide({ct: comb_2_f[comb_2_f["Combination_type"]==ct
                               ]["MIC_combined"].dropna().values
                 for ct in sorted(comb_2_f["Combination_type"].dropna().unique())
                 }).to_excel(w, sheet_name="GP_2cat_violin", index=False)

        to_wide({ct: comb_3_f[comb_3_f[ctype_3]==ct
                               ]["MIC_combined"].dropna().values
                 for ct in sorted(comb_3_f[ctype_3].dropna().unique())
                 }).to_excel(w, sheet_name="GP_3cat_violin", index=False)

        to_wide({ct: comb_2_f[comb_2_f["Combination_type"]==ct
                               ]["MIC_delta"].dropna().values
                 for ct in sorted(comb_2_f["Combination_type"].dropna().unique())
                 }).to_excel(w, sheet_name="GP_2cat_delta", index=False)

        sc = res_motif["score"].dropna(
            subset=["MIC_individual", "MIC_mean_in_comb"])
        res_motif["score"][["Motif", "Mechanism", "MIC_individual",
                             "MIC_mean_in_comb", "CV", "Score_combined"]
                            ].to_excel(w, sheet_name="GP_score", index=False)
        pd.DataFrame({
            "MIC_individual":   sc["MIC_individual"].values,
            "MIC_mean_in_comb": sc["MIC_mean_in_comb"].values,
            "Mechanism":        sc["Mechanism"].values,
        }).to_excel(w, sheet_name="GP_ind_vs_comb", index=False)

        res_2["top20_cross"].to_excel(w, sheet_name="GP_top20_cross", index=False)
        res_2["top20_nodom"].to_excel(w, sheet_name="GP_top20_nodom", index=False)
        res_2["heatmap"].astype(float).to_excel(w, sheet_name="GP_heatmap_2cat")

        # stats summary for paper
        paper_rows = []
        for pfx, res in [("2cat", res_2), ("3cat", res_3)]:
            kw = res["kw"].iloc[0]
            paper_rows.append({
                "Analysis": f"KW subsampled {pfx}",
                "H_mean": kw["H_mean"], "p_mean": kw["p_mean"],
                "n_draw": kw["n_draw"], "n_iter": N_ITER,
            })
            if not res["mw"].empty:
                for _, row in res["mw"].iterrows():
                    paper_rows.append({
                        "Analysis": f"MW {pfx} {row['Comparison']}",
                        "Cohen_d": row["Cohen_d_mean"],
                        "p_mean":  row["p_mean"],
                    })
        pd.DataFrame(paper_rows).to_excel(
            w, sheet_name="Summary_for_paper", index=False)

    log.info("Excel saved: %s", out_xl)

    # ── FINAL SUMMARY ──
    log.info("\n" + "=" * 60)
    log.info("FINAL SUMMARY  |  threshold=%.0f µmol/L", thr)
    log.info("=" * 60)
    log.info("Active motifs:      %d / %d", len(active), len(motif_mic))
    log.info("2cat combinations:  %d → %d", len(comb_2), len(comb_2_f))
    log.info("3cat combinations:  %d → %d", len(comb_3), len(comb_3_f))
    cols = [c for c in ["Motif_1","Motif_2","Combination_type",
                         "MIC_combined","MIC_delta"]
            if c in res_2["top20"].columns]
    log.info("\nTop 5 combinations (2cat filtered):\n%s",
             res_2["top20"][cols].head(5).to_string(index=False))
    log.info("\nOutputs: %s", out_dir)


if __name__ == "__main__":
    main()