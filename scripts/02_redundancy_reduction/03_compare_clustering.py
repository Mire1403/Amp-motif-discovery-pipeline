from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# suppress matplotlib font debug noise
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

PROJECT_ROOT  = _find_project_root()
CLUSTER_DIR   = PROJECT_ROOT / "results" / "02_redundancy_reduction"
DATA_FINAL    = PROJECT_ROOT / "data" / "final"
DATA_INTERM   = PROJECT_ROOT / "data" / "intermediate"
OUT_DIR       = PROJECT_ROOT / "results" / "02_redundancy_reduction" / "statistics"
LOGS_DIR      = PROJECT_ROOT / "logs"

ORIG_AMP_FASTA    = DATA_FINAL  / "AMP_MASTER.fasta"
ORIG_NONAMP_FASTA = DATA_INTERM / "nonamp_clean.fasta"

FILES = {
    "cdhit_amp":     "AMP_MASTER_cdhit.fasta",
    "mmseqs_amp":    "AMP_MASTER_mmseqs_rep_seq.fasta",
    "cdhit_nonamp":  "NONAMP_cdhit.fasta",
    "mmseqs_nonamp": "NONAMP_mmseqs_rep_seq.fasta",
}

PALETTE = {
    "cdhit_amp":     "#E07B54",
    "mmseqs_amp":    "#C0504D",
    "cdhit_nonamp":  "#4C72B0",
    "mmseqs_nonamp": "#2E75B6",
}
DPI = 300

# =====================================================
# LOGGING
# =====================================================

def setup_logging() -> None:
    for d in [OUT_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "03_analyze_cluster_representatives.log"
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
# FASTA HELPERS
# =====================================================

def read_fasta_lengths(path: Path) -> list[int]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    lengths: list[int] = []
    seq_len = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_len > 0:
                    lengths.append(seq_len)
                seq_len = 0
            else:
                seq_len += len(line)
        if seq_len > 0:
            lengths.append(seq_len)
    return lengths


def read_fasta_sequences(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    seqs: set[str] = set()
    parts: list[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if parts:
                    seqs.add("".join(parts).upper())
                    parts = []
            else:
                parts.append(line)
        if parts:
            seqs.add("".join(parts).upper())
    return seqs


def compute_stats(lengths: list[int], label: str = "") -> dict:
    if not lengths:
        return {"condition": label, "count": 0}
    arr = np.asarray(lengths, dtype=float)
    return {
        "condition": label,
        "count":     int(arr.size),
        "min":       int(arr.min()),
        "q25":       float(np.percentile(arr, 25)),
        "median":    float(np.median(arr)),
        "mean":      round(float(arr.mean()), 2),
        "q75":       float(np.percentile(arr, 75)),
        "max":       int(arr.max()),
        "std":       round(float(arr.std()), 2),
    }

# =====================================================
# ANALYSIS 1: LENGTH STATS
# =====================================================

def analysis_length_stats(all_lengths: dict[str, list[int]]) -> pd.DataFrame:
    rows = [compute_stats(lengths, label)
            for label, lengths in all_lengths.items()]
    df = pd.DataFrame(rows)
    logger.info("\n=== Length statistics per condition ===")
    logger.info("\n%s", df.to_string(index=False))
    return df

# =====================================================
# ANALYSIS 2: OVERLAP CD-HIT vs MMseqs2
# =====================================================

def analysis_overlap(all_seqs: dict[str, set[str]]) -> pd.DataFrame:
    logger.info("\n=== Overlap CD-HIT vs MMseqs2 ===")
    rows = []
    for dataset in ("amp", "nonamp"):
        cdhit  = all_seqs.get(f"cdhit_{dataset}",  set())
        mmseqs = all_seqs.get(f"mmseqs_{dataset}", set())
        inter  = cdhit & mmseqs
        union  = cdhit | mmseqs
        only_cdhit  = cdhit - mmseqs
        only_mmseqs = mmseqs - cdhit
        jaccard = len(inter) / len(union) if union else 0
        logger.info(
            "  %s: CD-HIT=%d  MMseqs2=%d  shared=%d  "
            "only_cdhit=%d  only_mmseqs=%d  Jaccard=%.3f",
            dataset, len(cdhit), len(mmseqs), len(inter),
            len(only_cdhit), len(only_mmseqs), jaccard,
        )
        rows.append({
            "dataset":         dataset,
            "n_cdhit":         len(cdhit),
            "n_mmseqs":        len(mmseqs),
            "n_shared":        len(inter),
            "n_only_cdhit":    len(only_cdhit),
            "n_only_mmseqs":   len(only_mmseqs),
            "jaccard":         round(jaccard, 4),
            "pct_shared_cdhit":  round(100 * len(inter) / len(cdhit),  1) if cdhit  else 0,
            "pct_shared_mmseqs": round(100 * len(inter) / len(mmseqs), 1) if mmseqs else 0,
        })
    return pd.DataFrame(rows)

# =====================================================
# ANALYSIS 4: STATISTICAL COMPARISON CD-HIT vs MMseqs2
# Separate for AMP and non-AMP
# =====================================================

def analysis_stats_cdhit_vs_mmseqs(
        all_lengths: dict[str, list[int]],
        all_seqs:    dict[str, set[str]],
        n_orig_amp:  int,
        n_orig_nonamp: int,
) -> pd.DataFrame:
    """
    For each dataset (AMP, non-AMP):
      1. Z-test of proportions: is retention rate significantly different?
      2. Mann-Whitney: do representative lengths differ between tools?
      3. Cohen's Kappa: concordance of sequence selection (1=selected, 0=not)
    """
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.proportion import proportions_ztest
    from sklearn.metrics import cohen_kappa_score

    logger.info("\n=== Statistical comparison: CD-HIT vs MMseqs2 ===")
    rows = []

    for dataset, n_orig in [("amp", n_orig_amp), ("nonamp", n_orig_nonamp)]:
        label = dataset.upper()
        cdhit_len  = all_lengths.get(f"cdhit_{dataset}",  [])
        mmseqs_len = all_lengths.get(f"mmseqs_{dataset}", [])
        cdhit_seqs  = all_seqs.get(f"cdhit_{dataset}",  set())
        mmseqs_seqs = all_seqs.get(f"mmseqs_{dataset}", set())

        if not cdhit_len or not mmseqs_len or n_orig == 0:
            continue

        logger.info("\n  --- %s ---", label)

        # --------------------------------------------------
        # 1. Z-test of proportions
        # H0: retention rate CD-HIT == retention rate MMseqs2
        # --------------------------------------------------
        n_cdhit  = len(cdhit_seqs)
        n_mmseqs = len(mmseqs_seqs)
        counts   = np.array([n_cdhit, n_mmseqs])
        nobs     = np.array([n_orig,  n_orig])
        z_stat, p_ztest = proportions_ztest(counts, nobs)
        pct_cdhit  = 100 * n_cdhit  / n_orig
        pct_mmseqs = 100 * n_mmseqs / n_orig
        logger.info(
            "  Z-test retention: CD-HIT=%.1f%%  MMseqs2=%.1f%%  "
            "z=%.3f  p=%.4e",
            pct_cdhit, pct_mmseqs, z_stat, p_ztest,
        )

        # --------------------------------------------------
        # 2. Mann-Whitney: representative lengths
        # H0: length distributions are equal
        # --------------------------------------------------
        u_stat, p_mw = mannwhitneyu(cdhit_len, mmseqs_len,
                                     alternative="two-sided")
        d_cohen = (np.mean(cdhit_len) - np.mean(mmseqs_len)) / np.sqrt(
            (np.std(cdhit_len)**2 + np.std(mmseqs_len)**2) / 2
        )
        logger.info(
            "  Mann-Whitney lengths: CD-HIT mean=%.1f  MMseqs2 mean=%.1f  "
            "p=%.4e  Cohen_d=%.3f",
            np.mean(cdhit_len), np.mean(mmseqs_len), p_mw, d_cohen,
        )

        # --------------------------------------------------
        # 3. Cohen's Kappa: concordance of selection
        # Build binary vectors over the union of all sequences
        # 1 = selected as representative, 0 = not selected
        # --------------------------------------------------
        all_union = cdhit_seqs | mmseqs_seqs
        y_cdhit  = [1 if s in cdhit_seqs  else 0 for s in all_union]
        y_mmseqs = [1 if s in mmseqs_seqs else 0 for s in all_union]
        kappa = cohen_kappa_score(y_cdhit, y_mmseqs)
        logger.info("  Cohen's Kappa (selection concordance): %.4f", kappa)

        rows.append({
            "dataset":          label,
            "n_orig":           n_orig,
            "n_cdhit":          n_cdhit,
            "n_mmseqs":         n_mmseqs,
            "pct_retained_cdhit":  round(pct_cdhit, 2),
            "pct_retained_mmseqs": round(pct_mmseqs, 2),
            "z_stat":           round(z_stat, 4),
            "p_ztest":          round(p_ztest, 8),
            "mean_len_cdhit":   round(float(np.mean(cdhit_len)), 2),
            "mean_len_mmseqs":  round(float(np.mean(mmseqs_len)), 2),
            "p_mannwhitney":    round(p_mw, 8),
            "Cohen_d_length":   round(d_cohen, 4),
            "Cohen_kappa_selection": round(kappa, 4),
        })

    df = pd.DataFrame(rows)
    logger.info("\n%s", df.to_string(index=False))
    return df

# =====================================================
# FIGURE 4: Length KDE — CD-HIT vs MMseqs2 per dataset
# =====================================================

def fig_kde_cdhit_vs_mmseqs(all_lengths: dict[str, list[int]]) -> None:
    """KDE overlapping CD-HIT and MMseqs2 for AMP and non-AMP separately."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#fafafa")

    for ax, dataset in zip(axes, ("amp", "nonamp")):
        ax.set_facecolor("#fafafa")
        cdhit  = all_lengths.get(f"cdhit_{dataset}",  [])
        mmseqs = all_lengths.get(f"mmseqs_{dataset}", [])
        label  = dataset.upper()
        if cdhit:
            sns.kdeplot(cdhit,  fill=True, alpha=0.5,
                        color=PALETTE[f"cdhit_{dataset}"],
                        label=f"CD-HIT (n={len(cdhit):,}  mean={np.mean(cdhit):.1f})",
                        ax=ax)
        if mmseqs:
            sns.kdeplot(mmseqs, fill=True, alpha=0.5,
                        color=PALETTE[f"mmseqs_{dataset}"],
                        label=f"MMseqs2 (n={len(mmseqs):,}  mean={np.mean(mmseqs):.1f})",
                        ax=ax)
        ax.set_xlabel("Sequence length (aa)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"{label} — CD-HIT vs MMseqs2", fontweight="bold")
        ax.legend(fontsize=8.5)
        ax.spines[["top","right"]].set_visible(False)

    fig.suptitle("Representative sequence length: CD-HIT vs MMseqs2",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "04_kde_cdhit_vs_mmseqs.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("Fig 4 saved: %s", out)

# =====================================================
# ANALYSIS 3: REDUNDANCY REDUCTION SUMMARY
# =====================================================

def analysis_redundancy_summary(
        all_lengths: dict[str, list[int]],
        n_orig_amp: int,
        n_orig_nonamp: int,
) -> pd.DataFrame:
    logger.info("\n=== Redundancy reduction summary ===")
    rows = []
    for label, lengths in all_lengths.items():
        n_orig  = n_orig_amp if "nonamp" not in label else n_orig_nonamp
        n_after = len(lengths)
        removed = n_orig - n_after
        pct     = 100 * removed / n_orig if n_orig > 0 else 0
        logger.info(
            "  %-22s  before=%7d  after=%7d  removed=%7d  (%.1f%%)",
            label, n_orig, n_after, removed, pct,
        )
        rows.append({
            "condition":   label,
            "n_before":    n_orig,
            "n_after":     n_after,
            "n_removed":   removed,
            "pct_removed": round(pct, 1),
        })
    return pd.DataFrame(rows)

# =====================================================
# FIGURES
# =====================================================

def fig_length_distributions(all_lengths: dict[str, list[int]]) -> None:
    """2×2 histogram grid — one panel per condition."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor("#fafafa")

    for ax, (label, lengths) in zip(axes.flat, all_lengths.items()):
        ax.set_facecolor("#fafafa")
        if not lengths:
            ax.set_title(label); continue
        ax.hist(lengths, bins=40, color=PALETTE.get(label, "#888"),
                alpha=0.85, edgecolor="white")
        ax.axvline(float(np.median(lengths)), color="black",
                   lw=1.5, ls="--", label=f"median={np.median(lengths):.0f}")
        ax.set_xlabel("Sequence length (aa)", fontsize=9)
        ax.set_ylabel("Frequency", fontsize=9)
        ax.set_title(f"{label}  (n={len(lengths):,})",
                     fontweight="bold", fontsize=9)
        ax.legend(fontsize=8)
        ax.spines[["top","right"]].set_visible(False)

    fig.suptitle("Length distributions — cluster representative sequences",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "01_length_distributions_2x2.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("Fig 1 saved: %s", out)


def fig_overlap(overlap_df: pd.DataFrame) -> None:
    """Stacked bar — shared / only CD-HIT / only MMseqs2 per dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#fafafa")

    for ax, (_, row) in zip(axes, overlap_df.iterrows()):
        ax.set_facecolor("#fafafa")
        cats   = ["Shared", "Only CD-HIT", "Only MMseqs2"]
        vals   = [row["n_shared"], row["n_only_cdhit"], row["n_only_mmseqs"]]
        colors = ["#5B8DB8", "#E07B54", "#3BAF77"]
        bars   = ax.bar(cats, vals, color=colors, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.01,
                    f"{val:,}", ha="center", fontsize=9, fontweight="bold")
        ax.set_title(
            f"{row['dataset'].upper()}  —  Jaccard={row['jaccard']:.3f}",
            fontweight="bold"
        )
        ax.set_ylabel("Number of sequences", fontsize=10)
        ax.spines[["top","right"]].set_visible(False)

    fig.suptitle("Overlap between CD-HIT and MMseqs2 representative sequences",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "02_overlap_cdhit_vs_mmseqs.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("Fig 2 saved: %s", out)


def fig_redundancy_summary(redund_df: pd.DataFrame) -> None:
    """Grouped bar — sequences before/after clustering per condition."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    x     = np.arange(len(redund_df))
    width = 0.35
    ax.bar(x - width/2, redund_df["n_before"], width,
           color="#CCCCCC", alpha=0.9, edgecolor="white", label="Before")
    bars = ax.bar(x + width/2, redund_df["n_after"], width,
                  color=[PALETTE.get(c, "#888") for c in redund_df["condition"]],
                  alpha=0.9, edgecolor="white", label="After")

    for bar, pct in zip(bars, redund_df["pct_removed"]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + redund_df["n_before"].max() * 0.01,
                f"-{pct:.0f}%", ha="center", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(redund_df["condition"],
                       rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Number of sequences", fontsize=10)
    ax.set_title("Redundancy reduction — before vs after clustering",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()

    out = OUT_DIR / "03_redundancy_summary.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("Fig 3 saved: %s", out)

# =====================================================
# MAIN
# =====================================================

def main() -> None:
    setup_logging()
    logger.info("Cluster representative analysis started.")

    all_lengths: dict[str, list[int]] = {}
    all_seqs:    dict[str, set[str]]  = {}

    for label, fname in FILES.items():
        path = CLUSTER_DIR / fname
        try:
            all_lengths[label] = read_fasta_lengths(path)
            all_seqs[label]    = read_fasta_sequences(path)
            logger.info("Loaded %-22s  %d sequences",
                        label, len(all_lengths[label]))
        except FileNotFoundError as e:
            logger.error("%s", e)

    n_orig_amp = sum(
        1 for l in open(ORIG_AMP_FASTA) if l.startswith(">")
    ) if ORIG_AMP_FASTA.exists() else 0
    n_orig_nonamp = sum(
        1 for l in open(ORIG_NONAMP_FASTA) if l.startswith(">")
    ) if ORIG_NONAMP_FASTA.exists() else 0

    stats_df   = analysis_length_stats(all_lengths)
    overlap_df = analysis_overlap(all_seqs)
    stats_tools_df = analysis_stats_cdhit_vs_mmseqs(
        all_lengths, all_seqs, n_orig_amp, n_orig_nonamp)
    redund_df  = analysis_redundancy_summary(
        all_lengths, n_orig_amp, n_orig_nonamp)

    out_excel = OUT_DIR / "cluster_analysis.xlsx"
    with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
        stats_df.to_excel(writer,       sheet_name="Length_stats",         index=False)
        overlap_df.to_excel(writer,     sheet_name="Overlap_CDHIT_MMseqs", index=False)
        stats_tools_df.to_excel(writer, sheet_name="Stats_CDHITvMMseqs",   index=False)
        redund_df.to_excel(writer,      sheet_name="Redundancy_summary",   index=False)
    logger.info("Excel saved: %s", out_excel)

    # ---- GraphPad Excel — wide format, one sheet per figure ----
    def to_wide(d: dict) -> pd.DataFrame:
        return pd.DataFrame({k: pd.Series(v).reset_index(drop=True)
                             for k, v in d.items()})

    gp_excel = OUT_DIR / "cluster_analysis_graphpad.xlsx"
    with pd.ExcelWriter(gp_excel, engine="openpyxl") as writer:

        # GP Fig 1 — length distributions (one column per condition)
        to_wide({label: lengths
                 for label, lengths in all_lengths.items()}).to_excel(
            writer, sheet_name="GP_Fig1_lengths", index=False)

        # GP Fig 2 — overlap (shared / only_cdhit / only_mmseqs per dataset)
        overlap_df[["dataset","n_shared","n_only_cdhit",
                    "n_only_mmseqs","jaccard"]].to_excel(
            writer, sheet_name="GP_Fig2_overlap", index=False)

        # GP Fig 3 — redundancy before/after
        redund_df[["condition","n_before","n_after",
                   "pct_removed"]].to_excel(
            writer, sheet_name="GP_Fig3_redundancy", index=False)

        # GP Fig 4 — KDE data (CD-HIT vs MMseqs2 per dataset)
        for dataset in ("amp", "nonamp"):
            to_wide({
                "cdhit":  all_lengths.get(f"cdhit_{dataset}",  []),
                "mmseqs": all_lengths.get(f"mmseqs_{dataset}", []),
            }).to_excel(writer, sheet_name=f"GP_Fig4_{dataset}", index=False)

        # GP Stats — z-test and Mann-Whitney results for annotation
        stats_tools_df.to_excel(writer, sheet_name="GP_Stats", index=False)

    logger.info("GraphPad Excel saved: %s", gp_excel)

    fig_length_distributions(all_lengths)
    fig_overlap(overlap_df)
    fig_redundancy_summary(redund_df)
    fig_kde_cdhit_vs_mmseqs(all_lengths)

    logger.info("Cluster representative analysis completed successfully.")


if __name__ == "__main__":
    main()