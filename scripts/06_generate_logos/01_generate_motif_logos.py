"""
11_motif_logos.py
==================
Generates sequence logos for all significant AMP motifs
organized by family. Each motif gets an individual PNG logo
with ER, FDR, and family metadata. Each family gets a
multi-panel PDF grid with all its motifs.

Usage:
    python 11_motif_logos.py
    python 11_motif_logos.py --top_families 10   # only top N families by ER
    python 11_motif_logos.py --test
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import logomaker

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
MOTIF_DISC    = PROJECT_ROOT / "results" / "04_motif_discovery"
FAMILIES_DIR  = PROJECT_ROOT / "results" / "07_motif_families" / "01_tomtom"
REPORTING_DIR = PROJECT_ROOT / "results" / "06_motif_statistics" / "02_fimo_reporting"
LOGS_DIR      = PROJECT_ROOT / "logs"

MOTIF_FILES = {
    ("cdhit",  "meme"):   MOTIF_DISC / "meme_cdhit"    / "meme.txt",
    ("mmseqs", "meme"):   MOTIF_DISC / "meme_mmseq"    / "meme.txt",
    ("cdhit",  "streme"): MOTIF_DISC / "streme_cdhit"  / "streme.txt",
    ("mmseqs", "streme"): MOTIF_DISC / "streme_mmseq"  / "streme.txt",
}

VALID_AA = list("ACDEFGHIKLMNPQRSTVWY")

# AA color scheme — chemical property groups
AA_COLORS = {
    # hydrophobic — blue
    "A": "#4C72B0", "V": "#4C72B0", "I": "#4C72B0", "L": "#4C72B0",
    "M": "#4C72B0", "F": "#4C72B0", "W": "#4C72B0", "P": "#4C72B0",
    # polar — green
    "S": "#3BAF77", "T": "#3BAF77", "C": "#3BAF77", "Y": "#3BAF77",
    "N": "#3BAF77", "Q": "#3BAF77", "G": "#3BAF77",
    # positive — orange/red
    "K": "#E07B54", "R": "#E07B54", "H": "#E07B54",
    # negative — purple
    "D": "#9B59B6", "E": "#9B59B6",
}

LEGEND_GROUPS = [
    ("Hydrophobic", "#4C72B0"),
    ("Polar",       "#3BAF77"),
    ("Positive",    "#E07B54"),
    ("Negative",    "#9B59B6"),
]

DPI         = 300
LOGO_W      = 7      # inches per logo
LOGO_H      = 4      # inches per logo — increased to prevent cutoff
GRID_COLS   = 3      # logos per row in family PDF

# =====================================================
# LOGGING
# =====================================================

def setup_logging(out_dir: Path) -> None:
    for d in [out_dir, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "11_motif_logos.log"
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

# =====================================================
# HELPERS
# =====================================================

def clean_motif(x: str) -> str:
    return re.sub(r"^\d+[-_]", "", str(x).strip()).strip()


def safe_filename(x: str) -> str:
    return re.sub(r"[^\w\-\.]+", "_", str(x))[:100]


def infer_alphabet(lines: list[str]) -> str:
    for ln in lines[:200]:
        if ln.startswith("ALPHABET="):
            alph = ln.split("=", 1)[1].strip().replace(" ", "")
            if alph: return alph
    return "ACDEFGHIKLMNPQRSTVWY"


def prob_to_ic(df: pd.DataFrame) -> pd.DataFrame:
    """Convert probability matrix to information content (bits)."""
    bg  = 1.0 / len(df.columns)
    out = df.copy().astype(float)
    for c in out.columns:
        p       = out[c].clip(lower=1e-10)
        out[c]  = (p * np.log2(p / bg)).clip(lower=0)
    return out


def consensus_str(df: pd.DataFrame) -> str:
    """Return consensus sequence (most probable AA per position)."""
    return "".join(df.columns[row.values.argmax()] for _, row in df.iterrows())

# =====================================================
# MEME PARSER
# =====================================================

def parse_meme_motifs(path: Path) -> dict[str, pd.DataFrame]:
    """Parse MEME/STREME text file → {motif_name: probability_matrix}."""
    if not path.exists():
        logger.warning("Motif file not found: %s", path)
        return {}

    lines    = path.read_text(errors="ignore").splitlines(True)
    alphabet = infer_alphabet(lines)
    result   = {}
    i        = 0

    while i < len(lines):
        if lines[i].startswith("MOTIF"):
            parts = lines[i].strip().split()
            name  = clean_motif(parts[1]) if len(parts) >= 2 else ""
            i    += 1
            while i < len(lines) and "letter-probability matrix" not in lines[i]:
                i += 1
            if i >= len(lines): break
            w_match = re.search(r"w=\s*(\d+)", lines[i])
            if not w_match: i += 1; continue
            width = int(w_match.group(1)); i += 1
            matrix = []
            for _ in range(width):
                if i >= len(lines): break
                try:
                    row = list(map(float, lines[i].strip().split()[:len(alphabet)]))
                    if len(row) == len(alphabet):
                        matrix.append(row)
                except ValueError:
                    pass
                i += 1
            if matrix:
                df = pd.DataFrame(matrix, columns=list(alphabet))
                df = df[[c for c in df.columns if c in VALID_AA]]
                result[name] = df
        else:
            i += 1

    logger.debug("Parsed %d motifs from %s", len(result), path.name)
    return result

# =====================================================
# INDIVIDUAL LOGO
# =====================================================

def draw_logo(
        pwm:        pd.DataFrame,
        motif:      str,
        family_id:  str,
        clustering: str,
        tool:       str,
        er:         float,
        fdr:        float,
        out_path:   Path,
) -> None:
    """
    Draw a single motif logo with:
    - IC-weighted logo (top)
    - Y axis in bits
    - Consensus string below
    - ER and FDR as subtitle
    - Chemical property legend
    """
    ic = prob_to_ic(pwm)
    cons = consensus_str(pwm)
    max_bits = np.log2(len(pwm.columns))

    fig = plt.figure(figsize=(LOGO_W, LOGO_H), facecolor="white")
    gs  = gridspec.GridSpec(
        3, 1,
        height_ratios=[4.0, 0.9, 0.5],
        hspace=0.15,
        figure=fig,
    )

    # ---- logo panel ----
    ax_logo = fig.add_subplot(gs[0])
    try:
        logomaker.Logo(
            ic,
            ax=ax_logo,
            color_scheme=AA_COLORS,
            show_spines=True,
            font_name="DejaVu Sans",
        )
    except Exception as e:
        logger.warning("logomaker failed for %s: %s", motif, e)
        ax_logo.text(0.5, 0.5, motif, ha="center", va="center",
                     fontsize=12, transform=ax_logo.transAxes)

    ax_logo.set_ylabel("Bits", fontsize=8)
    ax_logo.set_ylim(0, max_bits * 1.05)
    ax_logo.set_xticks(range(len(ic)))
    ax_logo.set_xticklabels(range(1, len(ic)+1), fontsize=7)
    ax_logo.yaxis.set_tick_params(labelsize=7)
    ax_logo.spines[["top","right"]].set_visible(False)

    # title with metadata
    er_str  = "∞" if np.isinf(er) else f"{er:.1f}"
    fdr_str = f"{fdr:.2e}" if pd.notna(fdr) else "n/a"
    ax_logo.set_title(
        f"{motif}  |  {family_id}  |  {clustering}-{tool}\n"
        f"ER={er_str}   FDR={fdr_str}   len={len(ic)}aa   consensus: {cons}",
        fontsize=8, fontweight="bold", pad=4,
    )

    # ---- dominant AA bar ----
    ax_cons = fig.add_subplot(gs[1])
    ax_cons.set_xlim(0, len(pwm))
    ax_cons.set_ylim(0, 1)
    ax_cons.axis("off")

    for j, (_, pos_row) in enumerate(pwm.iterrows()):
        aa   = pos_row.idxmax()
        prob = float(pos_row.max())
        color = AA_COLORS.get(aa, "#888888")
        # color intensity proportional to probability
        alpha = 0.4 + 0.6 * prob
        ax_cons.add_patch(
            plt.Rectangle(
                (j + 0.05, 0.15), 0.9, 0.7,
                facecolor=color, edgecolor="white",
                linewidth=0.5, alpha=alpha, clip_on=False,
            )
        )
        # AA letter
        ax_cons.text(j + 0.5, 0.62, aa,
                     ha="center", va="center",
                     fontsize=8, fontweight="bold", color="white")
        # probability below
        ax_cons.text(j + 0.5, 0.22, f"{prob:.2f}",
                     ha="center", va="center",
                     fontsize=5.5, color="white")

    # ---- legend ----
    ax_leg = fig.add_subplot(gs[2])
    ax_leg.axis("off")
    handles = [mpatches.Patch(color=c, label=l, alpha=0.85)
               for l, c in LEGEND_GROUPS]
    ax_leg.legend(handles=handles, loc="center", ncol=4,
                  frameon=False, fontsize=6.5, handlelength=1.2)

    plt.savefig(out_path, dpi=DPI, bbox_inches="tight",
                facecolor="white")
    plt.close()

# =====================================================
# FAMILY GRID PDF
# =====================================================

def draw_family_pdf(
        motifs_data: list[dict],
        family_id:   str,
        out_path:    Path,
) -> None:
    """
    Multi-panel PDF with all motifs of a family in a grid layout.
    Each panel is a full logo with metadata.
    """
    n     = len(motifs_data)
    if n == 0: return
    ncols = min(GRID_COLS, n)
    nrows = (n + ncols - 1) // ncols

    fig = plt.figure(
        figsize=(LOGO_W * ncols, LOGO_H * nrows),
        facecolor="white"
    )
    fig.suptitle(
        f"Family {family_id} — {n} motif(s)",
        fontsize=14, fontweight="bold", y=1.01
    )

    for idx, md in enumerate(motifs_data):
        pwm    = md["pwm"]
        motif  = md["motif"]
        ic     = prob_to_ic(pwm)
        cons   = consensus_str(pwm)
        er     = md.get("er", np.nan)
        fdr    = md.get("fdr", np.nan)
        cl     = md.get("clustering", "")
        tl     = md.get("tool", "")

        ax = fig.add_subplot(nrows, ncols, idx + 1)
        try:
            logomaker.Logo(ic, ax=ax, color_scheme=AA_COLORS,
                           show_spines=True, font_name="DejaVu Sans")
        except Exception:
            ax.text(0.5, 0.5, motif, ha="center", va="center",
                    fontsize=10, transform=ax.transAxes)

        er_str  = "∞" if np.isinf(er) else f"{er:.1f}"
        fdr_str = f"{fdr:.2e}" if pd.notna(fdr) else "n/a"
        max_bits = np.log2(len(pwm.columns))

        ax.set_ylim(0, max_bits * 1.05)
        ax.set_ylabel("Bits", fontsize=7)
        ax.set_xticks(range(len(ic)))
        ax.set_xticklabels(range(1, len(ic)+1), fontsize=6)
        ax.yaxis.set_tick_params(labelsize=6)
        ax.spines[["top","right"]].set_visible(False)
        ax.set_title(
            f"{motif}  ({cl}-{tl})\n"
            f"ER={er_str}  FDR={fdr_str}  {cons}",
            fontsize=7, fontweight="bold", pad=3,
        )

        # dominant AA bar below as inset
        ax_bar = ax.inset_axes([0, -0.32, 1, 0.25])
        ax_bar.set_xlim(0, len(pwm))
        ax_bar.set_ylim(0, 1)
        ax_bar.axis("off")
        for j, (_, pos_row) in enumerate(pwm.iterrows()):
            aa   = pos_row.idxmax()
            prob = float(pos_row.max())
            color = AA_COLORS.get(aa, "#888")
            alpha = 0.4 + 0.6 * prob
            ax_bar.add_patch(plt.Rectangle(
                (j+0.05, 0.15), 0.9, 0.7,
                facecolor=color, edgecolor="white",
                linewidth=0.4, alpha=alpha))
            ax_bar.text(j+0.5, 0.62, aa,
                        ha="center", va="center",
                        fontsize=6, fontweight="bold", color="white")
            ax_bar.text(j+0.5, 0.22, f"{prob:.2f}",
                        ha="center", va="center",
                        fontsize=4.5, color="white")

    fig.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight",
                facecolor="white")
    plt.close()
    logger.info("Family PDF: %s  (%d motifs)", out_path.name, n)

# =====================================================
# CLI
# =====================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate sequence logos for AMP motifs organized by family.")
    parser.add_argument("--top_families", type=int, default=None,
                        help="Only generate logos for top N families by median ER")
    parser.add_argument("--test", action="store_true",
                        help="Output to results/08_motif_logos_test/")
    return parser.parse_args()

# =====================================================
# MAIN
# =====================================================

def main() -> None:
    args    = parse_args()
    out_dir = PROJECT_ROOT / (
        "results/08_motif_logos_test"
        if args.test else
        "results/08_motif_logos"
    )
    setup_logging(out_dir)
    logger.info("Motif logo generation started.")

    # ---- load data ----
    master_fam = FAMILIES_DIR / "motif_families_master.xlsx"
    master_rep = REPORTING_DIR / "fimo_reporting_master.xlsx"

    if not master_fam.exists():
        logger.error("Not found: %s — run 09_motif_families_tomtom.py first.", master_fam)
        sys.exit(1)
    if not master_rep.exists():
        logger.error("Not found: %s — run 08_fimo_reporting.py first.", master_rep)
        sys.exit(1)

    df_fam = pd.read_excel(master_fam, sheet_name="All_motifs_with_family",
                            engine="openpyxl")
    df_sig = pd.read_excel(master_rep, sheet_name="Significant",
                            engine="openpyxl")

    # normalize
    for df in [df_fam, df_sig]:
        for col in ["Clustering","Tool","Motif"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()
        if "Motif" in df.columns:
            df["Motif"] = df["Motif"].map(clean_motif).str.upper()
        if "Clustering" in df.columns:
            df["Clustering"] = df["Clustering"].str.replace(
                r"^mmseq$","mmseqs", regex=True)

    # merge family assignments into sig table
    fam_cols = ["Clustering","Tool","Motif","Family_ID"]
    fam_cols = [c for c in fam_cols if c in df_fam.columns]
    df_sig = df_sig.merge(
        df_fam[fam_cols].drop_duplicates(),
        on=["Clustering","Tool","Motif"], how="left"
    )

    # ER / FDR columns
    er_col  = next((c for c in ["Enrichment_ratio","ER"] if c in df_sig.columns), None)
    fdr_col = next((c for c in ["FDR_report","FDR_corrected_global","FDR_corrected"]
                    if c in df_sig.columns), None)
    if er_col:
        df_sig[er_col] = pd.to_numeric(df_sig[er_col], errors="coerce")
    if fdr_col:
        df_sig[fdr_col] = pd.to_numeric(df_sig[fdr_col], errors="coerce")

    logger.info("Loaded %d significant motifs  |  families: %s",
                len(df_sig), df_sig["Family_ID"].nunique()
                if "Family_ID" in df_sig.columns else "unknown")

    # ---- parse PWMs ----
    parsed: dict[tuple, dict[str, pd.DataFrame]] = {}
    for key, path in MOTIF_FILES.items():
        parsed[key] = parse_meme_motifs(path)
        logger.info("Loaded PWMs %-20s: %d motifs", str(key), len(parsed[key]))

    # ---- filter to top families if requested ----
    families = df_sig["Family_ID"].dropna().unique().tolist() \
               if "Family_ID" in df_sig.columns else []

    if args.top_families and "Family_ID" in df_sig.columns and er_col:
        fam_er = (df_sig.groupby("Family_ID")[er_col]
                  .apply(lambda x: x.replace([np.inf,-np.inf], np.nan).median())
                  .nlargest(args.top_families).index.tolist())
        families = [f for f in families if f in fam_er]
        logger.info("Filtered to top %d families by median ER", args.top_families)

    # ---- generate logos ----
    n_logos = 0
    n_skip  = 0

    for family_id in sorted(families):
        fam_str  = str(family_id)
        fam_dir  = out_dir / safe_filename(fam_str)
        fam_dir.mkdir(parents=True, exist_ok=True)

        sub = df_sig[df_sig["Family_ID"].astype(str) == fam_str]
        family_data: list[dict] = []

        for _, row in sub.iterrows():
            motif  = str(row["Motif"])
            cl     = str(row["Clustering"])
            tl     = str(row["Tool"])
            # normalize clustering key for lookup
            cl_key = "mmseqs" if cl == "mmseqs" else cl
            key    = (cl_key, tl)
            pwm    = parsed.get(key, {}).get(motif)

            if pwm is None:
                n_skip += 1
                continue

            er  = float(row[er_col])  if er_col  and pd.notna(row[er_col])  else np.nan
            fdr = float(row[fdr_col]) if fdr_col and pd.notna(row[fdr_col]) else np.nan

            # individual PNG
            fname   = safe_filename(f"{cl}_{tl}__{motif}.png")
            out_png = fam_dir / fname
            draw_logo(pwm, motif, fam_str, cl, tl, er, fdr, out_png)
            n_logos += 1

            family_data.append({
                "pwm": pwm, "motif": motif,
                "clustering": cl, "tool": tl,
                "er": er, "fdr": fdr,
            })

        # family PDF grid
        if family_data:
            pdf_path = fam_dir / f"{safe_filename(fam_str)}_grid.pdf"
            draw_family_pdf(family_data, fam_str, pdf_path)

    logger.info("Logos generated: %d  |  Skipped (PWM not found): %d",
                n_logos, n_skip)
    logger.info("Motif logo generation completed.")
    logger.info("Outputs: %s", out_dir)


if __name__ == "__main__":
    main()