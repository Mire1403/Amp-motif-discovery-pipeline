"""
Usage:
    python 07_fimo_enrichment.py
    python 07_fimo_enrichment.py --alpha 0.01
    python 07_fimo_enrichment.py --min_amp_hits 5
    python 07_fimo_enrichment.py --test
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

# =====================================================
# PROJECT ROOT
# =====================================================

def _find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find project root.")

PROJECT_ROOT = _find_project_root()
FIMO_BASE    = PROJECT_ROOT / "results" / "05_motif_scanning"
CLUSTER_DIR  = PROJECT_ROOT / "results" / "02_redundancy_reduction"
BG_DIR       = PROJECT_ROOT / "data"    / "intermediate" / "background"
LOGS_DIR     = PROJECT_ROOT / "logs"

FASTA_TOTALS = {
    "cdhit_amps":   CLUSTER_DIR / "AMP_MASTER_cdhit.fasta",
    "mmseq_amps":  CLUSTER_DIR / "AMP_MASTER_mmseqs_rep_seq.fasta",
    "cdhit_nonamp": BG_DIR      / "background_cdhit_10x.fasta",
    "mmseq_nonamp":BG_DIR      / "background_mmseqs_10x.fasta",
}

STRUCTURE = {
    ("cdhit",  "meme"):   {
        "amps":   FIMO_BASE / "cdhit"  / "meme"   / "fimo_amps"    / "fimo.tsv",
        "nonamp": FIMO_BASE / "cdhit"  / "meme"   / "fimo_nonamps" / "fimo.tsv",
    },
    ("cdhit",  "streme"): {
        "amps":   FIMO_BASE / "cdhit"  / "streme" / "fimo_amps"    / "fimo.tsv",
        "nonamp": FIMO_BASE / "cdhit"  / "streme" / "fimo_nonamps" / "fimo.tsv",
    },
    ("mmseq", "meme"):   {
        "amps":   FIMO_BASE / "mmseq" / "meme"   / "fimo_amps"    / "fimo.tsv",
        "nonamp": FIMO_BASE / "mmseq" / "meme"   / "fimo_nonamps" / "fimo.tsv",
    },
    ("mmseq", "streme"): {
        "amps":   FIMO_BASE / "mmseq" / "streme" / "fimo_amps"    / "fimo.tsv",
        "nonamp": FIMO_BASE / "mmseq" / "streme" / "fimo_nonamps" / "fimo.tsv",
    },
}

PALETTE = {
    "cdhit_meme":    "#E07B54",
    "cdhit_streme":  "#C0504D",
    "mmseq_meme":   "#4C72B0",
    "mmseq_streme": "#2E75B6",
}
DPI = 300

# =====================================================
# LOGGING
# =====================================================

def setup_logging(out_dir: Path) -> None:
    for d in [out_dir, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "07_fimo_enrichment.log"
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", font_scale=1.05)

# =====================================================
# HELPERS
# =====================================================

def clean_motif_name(x: str) -> str:
    """Remove numeric prefixes from STREME: '24-DEGEMTEEEKK' -> 'DEGEMTEEEKK'"""
    x = str(x).strip()
    x = re.sub(r"^\d+[-_]", "", x)
    return x.strip()


def count_fasta(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(f"FASTA not found: {path}")
    return sum(1 for l in open(path, encoding="utf-8", errors="ignore")
               if l.startswith(">"))


def safe_read_fimo(path: Path) -> tuple[pd.DataFrame, str]:
    """Read FIMO TSV robustly. Returns (df, status_string)."""
    if not path.exists():
        return pd.DataFrame(columns=["motif_name","sequence_name"]), "missing_file"
    try:
        df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    except Exception as e:
        return pd.DataFrame(columns=["motif_name","sequence_name"]), f"read_error:{e}"
    if df.empty:
        return pd.DataFrame(columns=["motif_name","sequence_name"]), "empty_table"
    if "sequence_name" not in df.columns:
        return pd.DataFrame(columns=["motif_name","sequence_name"]), "missing_sequence_name"

    motif_col = None
    if "motif_id" in df.columns:
        if df["motif_id"].dropna().astype(str).str.contains(r"[A-Za-z]").any():
            motif_col = "motif_id"
    if motif_col is None and "motif_alt_id" in df.columns:
        motif_col = "motif_alt_id"
    if motif_col is None:
        return pd.DataFrame(columns=["motif_name","sequence_name"]), "missing_motif_columns"

    out = df[[motif_col,"sequence_name"]].copy()
    out.columns = ["motif_raw","sequence_name"]
    out["motif_name"] = out["motif_raw"].astype(str).map(clean_motif_name)
    out["sequence_name"] = out["sequence_name"].astype(str).str.strip()
    return out[["motif_name","sequence_name"]].dropna(), f"ok_using_{motif_col}"


def safe_fisher(amp_hit, amp_total, non_hit, non_total):
    table = [[amp_hit, amp_total - amp_hit],
             [non_hit, non_total - non_hit]]
    try:
        or_, p = fisher_exact(table)
        return float(or_), float(p)
    except Exception:
        return np.nan, np.nan

# =====================================================
# ENRICHMENT PER PIPELINE
# =====================================================

def compute_enrichment(
        clustering: str,
        tool:       str,
        files:      dict,
        totals:     dict,
        min_amp_hits: int,
) -> tuple[pd.DataFrame, dict]:

    df_amp, amp_status = safe_read_fimo(files["amps"])
    df_non, non_status = safe_read_fimo(files["nonamp"])

    total_amp = totals[f"{clustering}_amps"]
    total_non = totals[f"{clustering}_nonamp"]

    amp_motifs  = set(df_amp["motif_name"].dropna().astype(str))
    non_motifs  = set(df_non["motif_name"].dropna().astype(str))
    shared      = amp_motifs & non_motifs

    diag = {
        "Clustering":     clustering,
        "Tool":           tool,
        "AMP_status":     amp_status,
        "nonAMP_status":  non_status,
        "AMP_rows":       len(df_amp),
        "nonAMP_rows":    len(df_non),
        "AMP_motifs":     len(amp_motifs),
        "nonAMP_motifs":  len(non_motifs),
        "Shared_motifs":  len(shared),
        "Comment":        "ok",
    }

    logger.info(
        "  %s-%s: AMP=%d rows  nonAMP=%d rows  "
        "AMP_motifs=%d  shared=%d",
        clustering, tool, len(df_amp), len(df_non),
        len(amp_motifs), len(shared),
    )

    rows = []
    for motif in sorted(amp_motifs | non_motifs):
        amp_hit = int(df_amp.loc[df_amp["motif_name"]==motif,
                                  "sequence_name"].nunique())
        non_hit = int(df_non.loc[df_non["motif_name"]==motif,
                                  "sequence_name"].nunique()) \
                  if not df_non.empty else 0
        amp_hit = min(amp_hit, total_amp)
        non_hit = min(non_hit, total_non)

        if amp_hit < min_amp_hits:
            continue

        amp_rate = amp_hit / total_amp
        non_rate = non_hit / total_non
        er       = np.inf if non_rate == 0 else amp_rate / non_rate
        or_, p   = safe_fisher(amp_hit, total_amp, non_hit, total_non)

        rows.append({
            "Clustering":               clustering,
            "Tool":                     tool,
            "Motif":                    motif,
            "AMP_hits":                 amp_hit,
            "nonAMP_hits":              non_hit,
            "Total_AMP":                total_amp,
            "Total_nonAMP":             total_non,
            "AMP_hit_rate":             round(amp_rate, 6),
            "nonAMP_hit_rate":          round(non_rate, 6),
            "Enrichment_ratio":         er,
            "log2_ER":                  np.log2(er) if er not in (0, np.inf) else np.nan,
            "ER_infinite":              bool(non_rate == 0 and amp_hit > 0),
            "Fisher_OR":                or_,
            "Fisher_p":                 p,
        })

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if not df.empty and df["Fisher_p"].notna().any():
        valid = df["Fisher_p"].notna()
        df.loc[valid, "FDR"] = multipletests(
            df.loc[valid, "Fisher_p"].values, method="fdr_bh")[1]
        df["Sig_FDR"] = (df["FDR"] < 0.05) & (df["Enrichment_ratio"] > 1)
    elif not df.empty:
        df["FDR"] = np.nan
        df["Sig_FDR"] = False

    if df.empty:
        diag["Comment"] = "no motifs passed filters"

    return df, diag

# =====================================================
# FIGURES
# =====================================================

def fig_volcano(results_df: pd.DataFrame, out_dir: Path) -> None:
    """Volcano plot: log2(ER) vs -log10(Fisher_p) per pipeline."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#fafafa")

    for ax, ((cl, tl), grp) in zip(
            axes.flat,
            results_df.groupby(["Clustering","Tool"])):

        ax.set_facecolor("#fafafa")
        key = f"{cl}_{tl}"
        color = PALETTE.get(key, "#888")

        plot = grp.copy()
        plot = plot[plot["log2_ER"].notna() & plot["Fisher_p"].notna()]
        plot["neg_log10_p"] = -np.log10(plot["Fisher_p"].clip(lower=1e-300))

        sig   = plot[plot["Sig_FDR"]]
        nonsig= plot[~plot["Sig_FDR"]]

        ax.scatter(nonsig["log2_ER"], nonsig["neg_log10_p"],
                   s=25, alpha=0.4, color="#cccccc", linewidths=0)
        ax.scatter(sig["log2_ER"],    sig["neg_log10_p"],
                   s=40, alpha=0.85, color=color, linewidths=0)

        # label top 5
        top5 = sig.nlargest(5, "neg_log10_p")
        for _, r in top5.iterrows():
            ax.annotate(r["Motif"], (r["log2_ER"], r["neg_log10_p"]),
                        fontsize=6, ha="left",
                        xytext=(3,3), textcoords="offset points")

        ax.axhline(-np.log10(0.05), color="grey", lw=0.8, ls="--", alpha=0.6)
        ax.axvline(0, color="grey", lw=0.8, ls="--", alpha=0.4)
        ax.set_xlabel("log₂ Enrichment Ratio", fontsize=9)
        ax.set_ylabel("-log₁₀(p)", fontsize=9)
        ax.set_title(f"{cl} — {tl}  (sig={len(sig)})",
                     fontweight="bold", fontsize=10)
        ax.spines[["top","right"]].set_visible(False)

    fig.suptitle("Volcano plots — Motif enrichment in AMPs vs background",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = out_dir / "01_volcano_plot.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("Fig 1 saved: %s", out)


def fig_top_motifs(df_sig: pd.DataFrame, out_dir: Path, top_n: int = 25) -> None:
    """Horizontal bar chart — top enriched motifs by ER (finite only)."""
    plot = df_sig[df_sig["log2_ER"].notna()].copy()
    plot = plot.sort_values("log2_ER", ascending=False).head(top_n)

    if plot.empty:
        logger.warning("No finite ER motifs for top motifs bar chart.")
        return

    fig, ax = plt.subplots(figsize=(10, max(5, len(plot)*0.35)))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    colors = [PALETTE.get(f"{r['Clustering']}_{r['Tool']}", "#888")
              for _, r in plot.iterrows()]
    labels = [f"{r['Motif']} ({r['Clustering']}-{r['Tool']})"
              for _, r in plot.iterrows()]

    ax.barh(range(len(plot)), plot["log2_ER"].values,
            color=colors, alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(plot)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("log₂ Enrichment Ratio", fontsize=10)
    ax.set_title(f"Top {top_n} enriched motifs (FDR < 0.05)",
                 fontweight="bold")

    patches = [mpatches.Patch(color=v, label=k)
               for k,v in PALETTE.items()]
    ax.legend(handles=patches, fontsize=8, loc="lower right")
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()

    out = out_dir / "02_top_motifs_barplot.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("Fig 2 saved: %s", out)


def fig_heatmap(results_df: pd.DataFrame, out_dir: Path) -> None:
    """Heatmap: AMP hit rate per motif × pipeline condition."""
    pivot = results_df[results_df["Sig_FDR_global"]].copy()
    if pivot.empty:
        logger.warning("No significant motifs for heatmap.")
        return

    pivot["condition"] = pivot["Clustering"] + "_" + pivot["Tool"]
    heat = pivot.pivot_table(
        index="Motif", columns="condition",
        values="AMP_hit_rate", aggfunc="mean"
    ).fillna(0)

    # keep top 30 by mean
    heat = heat.loc[heat.mean(axis=1).nlargest(30).index]

    fig, ax = plt.subplots(figsize=(10, max(6, len(heat)*0.35)))
    fig.patch.set_facecolor("#fafafa")
    sns.heatmap(heat, cmap="YlOrRd", linewidths=0.3, linecolor="#eeeeee",
                annot=len(heat) <= 20, fmt=".3f",
                annot_kws={"size":7}, ax=ax,
                cbar_kws={"label":"AMP hit rate"})
    ax.set_title("AMP hit rate — top 30 significant motifs × condition",
                 fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel("")
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(fontsize=7)
    fig.tight_layout()

    out = out_dir / "03_heatmap_amp_hit_rate.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("Fig 3 saved: %s", out)


def fig_boxplot_hit_rates(df_sig: pd.DataFrame, out_dir: Path) -> None:
    """Boxplot: AMP hit rate by clustering and by motif tool."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#fafafa")

    for ax, col, title in zip(
            axes,
            ["Clustering", "Tool"],
            ["By clustering method", "By motif discovery tool"]):
        ax.set_facecolor("#fafafa")
        groups = {k: v for k, v in df_sig.groupby(col)["AMP_hit_rate"]
                  if len(v) > 0}
        if not groups:
            continue
        bp = ax.boxplot(
            list(groups.values()),
            labels=list(groups.keys()),
            patch_artist=True,
            showfliers=False,
            medianprops={"color":"black","lw":1.5},
        )
        colors = ["#E07B54","#4C72B0","#3BAF77","#9B59B6"]
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.75)
        ax.set_ylabel("AMP hit rate", fontsize=10)
        ax.set_title(title, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)

    fig.suptitle("AMP hit rate — globally significant motifs",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = out_dir / "04_boxplot_hit_rates.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("Fig 4 saved: %s", out)

# =====================================================
# CLI
# =====================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute FIMO motif enrichment with Fisher test and BH-FDR.")
    parser.add_argument("--alpha",        type=float, default=0.05)
    parser.add_argument("--min_amp_hits", type=int,   default=1)
    parser.add_argument("--test",         action="store_true",
                        help="Output to results/06_motif_statistics_test/")
    return parser.parse_args()

# =====================================================
# MAIN
# =====================================================

def main() -> None:
    args = parse_args()

    out_dir = PROJECT_ROOT / (
        "results/06_motif_statistics_test/01_fimo_enrichment"
        if args.test else
        "results/06_motif_statistics/01_fimo_enrichment"
    )
    setup_logging(out_dir)
    logger.info("FIMO enrichment analysis started.")
    logger.info("alpha=%.3f  min_amp_hits=%d", args.alpha, args.min_amp_hits)

    # precompute FASTA totals
    totals = {}
    for k, path in FASTA_TOTALS.items():
        try:
            totals[k] = count_fasta(path)
            logger.info("Total sequences %-22s: %d", k, totals[k])
        except FileNotFoundError as e:
            logger.error("%s", e); sys.exit(1)

    # run per-pipeline enrichment
    all_dfs:   list[pd.DataFrame] = []
    diag_rows: list[dict]         = []

    for (cl, tl), files in STRUCTURE.items():
        logger.info("=== %s - %s ===", cl, tl)
        df, diag = compute_enrichment(cl, tl, files, totals, args.min_amp_hits)
        diag_rows.append(diag)

        if df.empty:
            logger.warning("No results for %s-%s", cl, tl)
            continue

        all_dfs.append(df)
        tag = f"{cl}_{tl}"

        # per-pipeline Excel
        out_xl = out_dir / f"enrichment_{tag}.xlsx"
        with pd.ExcelWriter(out_xl, engine="openpyxl") as w:
            df.to_excel(w, sheet_name="All",         index=False)
            df[df["Sig_FDR"]].to_excel(w, sheet_name="Significant", index=False)
        logger.info("Saved: %s", out_xl.name)

    if not all_dfs:
        logger.error("No enrichment tables produced. Check FIMO outputs.")
        sys.exit(1)

    # global FDR across all pipelines
    results_df = pd.concat(all_dfs, ignore_index=True)
    valid = results_df["Fisher_p"].notna()
    results_df["FDR_global"] = np.nan
    if valid.any():
        results_df.loc[valid, "FDR_global"] = multipletests(
            results_df.loc[valid, "Fisher_p"].values, method="fdr_bh")[1]
    results_df["Sig_FDR_global"] = (
        (results_df["FDR_global"] < args.alpha) &
        (results_df["Enrichment_ratio"] > 1)
    )
    results_df = results_df.sort_values(
        ["Sig_FDR_global","ER_infinite","Enrichment_ratio","FDR_global"],
        ascending=[False,False,False,True])

    df_sig = results_df[results_df["Sig_FDR_global"]].copy()
    logger.info(
        "Global: %d motifs total  %d significant (FDR<%.2f)",
        len(results_df), len(df_sig), args.alpha)

    # summary table
    summary = (
        results_df.groupby(["Clustering","Tool"])
        .agg(
            n_motifs=("Motif","count"),
            n_sig_fdr=("Sig_FDR","sum"),
            n_sig_global=("Sig_FDR_global","sum"),
            mean_log2ER=("log2_ER", lambda x: x.replace([np.inf,-np.inf],np.nan).mean()),
            n_infinite_ER=("ER_infinite","sum"),
        ).reset_index()
    )
    logger.info("\n=== Summary ===\n%s", summary.to_string(index=False))

    # diagnostics
    diag_df = pd.DataFrame(diag_rows)

    # save master Excel
    out_master = out_dir / "fimo_enrichment_master.xlsx"
    with pd.ExcelWriter(out_master, engine="openpyxl") as w:
        results_df.to_excel(w, sheet_name="All_results",    index=False)
        df_sig.to_excel(w,     sheet_name="Significant",    index=False)
        summary.to_excel(w,    sheet_name="Summary",        index=False)
        diag_df.to_excel(w,    sheet_name="Diagnostics",    index=False)

        # GraphPad sheets
        for col in ["AMP_hit_rate","nonAMP_hit_rate","log2_ER"]:
            pivot = results_df[results_df["Sig_FDR_global"]].pivot_table(
                index="Motif", columns=["Clustering","Tool"],
                values=col, aggfunc="mean")
            pivot.to_excel(w, sheet_name=f"GP_{col[:12]}")
    logger.info("Master Excel: %s", out_master)

    # figures
    fig_volcano(results_df, out_dir)
    fig_top_motifs(df_sig, out_dir)
    fig_heatmap(results_df, out_dir)
    fig_boxplot_hit_rates(df_sig, out_dir)

    logger.info("FIMO enrichment analysis completed successfully.")
    logger.info("Outputs: %s", out_dir)


if __name__ == "__main__":
    main()