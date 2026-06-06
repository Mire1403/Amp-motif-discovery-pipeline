"""
Mechanism thresholds (Shai 1999, Brogden 2005, Wimley 2010):
  Carpet        : tilt > 70°
  Barrel_stave  : tilt < 35°, depth > 5 Å
  Toroidal_pore : tilt 35–70°, depth 2–25 Å
  Ambiguous     : everything else

Usage:
    python 15_fmap_classify.py
    python 15_fmap_classify.py --optimize
    python 15_fmap_classify.py --test
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logging.getLogger("matplotlib").setLevel(logging.WARNING)

# =====================================================
# PROJECT ROOT
# =====================================================

def _find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find project root.")

PROJECT_ROOT    = _find_project_root()
FMAP_PARSED_DIR = PROJECT_ROOT / "results" / "10_motif_mechanism_output" / "01_fmap_output" / "results2"
LOGS_DIR        = PROJECT_ROOT / "logs"

MECHANISMS = ["Carpet", "Barrel_stave", "Toroidal_pore", "Ambiguous"]

MECHANISM_COLORS = {
    "Carpet":        "#457b9d",
    "Barrel_stave":  "#f4a261",
    "Toroidal_pore": "#2a9d8f",
    "Ambiguous":     "#adb5bd",
}

DPI = 300

# =====================================================
# LOGGING
# =====================================================

def setup_logging(out_dir: Path) -> None:
    for d in [out_dir, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "15_fmap_classify.log"
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

def classify_mechanism(row,
                        carpet_tilt:   float = 70.0,
                        barrel_tilt:   float = 35.0,
                        barrel_depth:  float = 5.0,
                        toroidal_lo:   float = 2.0,
                        toroidal_hi:   float = 25.0) -> str:
    tilt  = row["Tilt_Angle"]
    depth = row["Depth_Thickness"]
    if pd.isna(tilt) or pd.isna(depth):
        return "Ambiguous"
    if tilt > carpet_tilt:
        return "Carpet"
    if tilt < barrel_tilt and depth > barrel_depth:
        return "Barrel_stave"
    if barrel_tilt <= tilt <= carpet_tilt and toroidal_lo <= depth <= toroidal_hi:
        return "Toroidal_pore"
    return "Ambiguous"


def classify_insertion_strength(energy: float) -> str:
    if pd.isna(energy):
        return "Unknown"
    if energy < -12: return "Strong   (< -12)"
    if energy <  -8: return "Moderate (-12 to -8)"
    if energy <  -5: return "Weak     (-8 to -5)"
    if energy <  -2: return "Marginal (-5 to -2)"
    return "None     (> -2)"


def select_primary_helix(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_priority"] = df["Is_False"].astype(int)
    return (
        df.sort_values(["_priority", "Transfer_Energy"])
        .groupby("Peptide_ID", as_index=False)
        .first()
        .drop(columns=["_priority"])
    )


def add_legend(ax: plt.Axes, mechanisms: list[str]) -> None:
    patches = [mpatches.Patch(color=MECHANISM_COLORS[m], label=m)
               for m in mechanisms]
    ax.legend(handles=patches, title="Mechanism", fontsize=8,
              title_fontsize=8, loc="best", framealpha=0.8)

# =====================================================
# THRESHOLD OPTIMIZER
# =====================================================

def run_optimizer(primary: pd.DataFrame) -> pd.DataFrame:
    logger.info("Running threshold optimizer...")
    carpet_tilts  = [65, 70, 75, 80]
    barrel_depths = [4, 5, 6, 7, 8]
    barrel_tilts  = [20, 25, 30, 35]
    toroidal_lo   = [2, 3, 4]
    toroidal_hi   = [10, 15, 20, 25]
    rows = []
    n = len(primary)

    for ct, bd, bt, tlo, thi in itertools.product(
            carpet_tilts, barrel_depths, barrel_tilts,
            toroidal_lo, toroidal_hi):
        if tlo >= thi or bt >= ct: continue

        counts = primary.apply(
            classify_mechanism, axis=1,
            carpet_tilt=ct, barrel_tilt=bt, barrel_depth=bd,
            toroidal_lo=tlo, toroidal_hi=thi
        ).value_counts()

        n_car = counts.get("Carpet", 0)
        n_tor = counts.get("Toroidal_pore", 0)
        n_amb = counts.get("Ambiguous", 0)

        if any(x / n < 0.05 for x in [n_car, n_tor]): continue

        rows.append({
            "carpet_tilt":    ct,
            "barrel_tilt":    bt,
            "barrel_depth":   bd,
            "toroidal_depth": f"{tlo}–{thi}",
            "Carpet_%":       round(100 * n_car / n, 1),
            "Barrel_%":       round(100 * counts.get("Barrel_stave", 0) / n, 1),
            "Toroidal_%":     round(100 * n_tor / n, 1),
            "Ambiguous_%":    round(100 * n_amb / n, 1),
            "Ambiguous_n":    n_amb,
        })

    opt_df = pd.DataFrame(rows).sort_values("Ambiguous_n")
    logger.info("Optimizer top 10:\n%s", opt_df.head(10).to_string(index=False))
    return opt_df

# =====================================================
# FIGURES
# =====================================================

def fig_tilt_vs_depth(primary: pd.DataFrame, out_dir: Path) -> None:
    colors = primary["Mechanism"].map(MECHANISM_COLORS).fillna("#cccccc")
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
    ax.scatter(primary["Tilt_Angle"], primary["Depth_Thickness"],
               c=colors, alpha=0.65, s=45, edgecolors="white", linewidths=0.3)
    for x in [35, 70]:
        ax.axvline(x, color="gray", lw=0.8, ls="--", alpha=0.5)
    for y in [2, 25]:
        ax.axhline(y, color="gray", lw=0.8, ls=":", alpha=0.5)
    for x, y, label in [(15,15,"Barrel\nstave"),(52,14,"Toroidal\npore"),
                         (80,10,"Carpet"),(52,1,"Ambiguous")]:
        ax.text(x, y, label, fontsize=7, color="gray",
                ha="center", va="center", style="italic")
    ax.set_xlabel("Tilt angle (°)", fontsize=10)
    ax.set_ylabel("Immersion depth / thickness (Å)", fontsize=10)
    ax.set_title("FMAP classification space: Tilt vs Depth", fontweight="bold")
    add_legend(ax, [m for m in MECHANISMS if m in primary["Mechanism"].values])
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir/"01_tilt_vs_depth.png", dpi=DPI, bbox_inches="tight")
    plt.close(); logger.info("Fig 1 saved.")


def fig_tilt_vs_energy(primary: pd.DataFrame, out_dir: Path) -> None:
    colors = primary["Mechanism"].map(MECHANISM_COLORS).fillna("#cccccc")
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
    ax.scatter(primary["Tilt_Angle"], primary["Membrane_Binding_Energy"],
               c=colors, alpha=0.65, s=45, edgecolors="white", linewidths=0.3)
    for y in [-2, -8]:
        ax.axhline(y, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Tilt angle (°)", fontsize=10)
    ax.set_ylabel("Membrane binding energy (kcal/mol)", fontsize=10)
    ax.set_title("Membrane binding energy vs tilt angle", fontweight="bold")
    add_legend(ax, [m for m in MECHANISMS if m in primary["Mechanism"].values])
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir/"02_tilt_vs_energy.png", dpi=DPI, bbox_inches="tight")
    plt.close(); logger.info("Fig 2 saved.")


def fig_energy_distribution(primary: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
    for mech in MECHANISMS:
        sub = primary[primary["Mechanism"]==mech]["Membrane_Binding_Energy"].dropna()
        if sub.empty: continue
        ax.hist(sub, bins=25, alpha=0.6, color=MECHANISM_COLORS[mech],
                label=f"{mech} (n={len(sub)})", edgecolor="none")
    ax.axvline(-2, color="black", lw=1, ls="--", alpha=0.6)
    ax.set_xlabel("Membrane binding energy (kcal/mol)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Membrane binding energy by mechanism", fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.8)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir/"03_energy_distribution.png", dpi=DPI, bbox_inches="tight")
    plt.close(); logger.info("Fig 3 saved.")


def fig_depth_boxplot(primary: pd.DataFrame, out_dir: Path) -> None:
    mech_order = [m for m in MECHANISMS if m in primary["Mechanism"].values]
    data_bp    = [primary[primary["Mechanism"]==m]["Depth_Thickness"].dropna().values
                  for m in mech_order]
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
    bp = ax.boxplot(data_bp, patch_artist=True,
                    tick_labels=mech_order,
                    medianprops=dict(color="black", lw=1.5),
                    whiskerprops=dict(lw=1), capprops=dict(lw=1),
                    flierprops=dict(marker=".", markersize=4, alpha=0.5))
    for patch, mech in zip(bp["boxes"], mech_order):
        patch.set_facecolor(MECHANISM_COLORS[mech]); patch.set_alpha(0.75)
    ax.set_ylabel("Immersion depth / thickness (Å)", fontsize=10)
    ax.set_title("Insertion depth by mechanism", fontweight="bold")
    plt.xticks(rotation=20, ha="right", fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir/"04_depth_boxplot.png", dpi=DPI, bbox_inches="tight")
    plt.close(); logger.info("Fig 4 saved.")

# =====================================================
# CLI
# =====================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FMAP mechanism classification and visualization.")
    parser.add_argument("--input",      type=Path, default=None,
                        help="Path to fmap_parsed.csv")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--optimize",   action="store_true",
                        help="Run threshold optimizer (slow)")
    parser.add_argument("--test",       action="store_true")
    return parser.parse_args()

# =====================================================
# MAIN
# =====================================================

def main() -> None:
    args    = parse_args()
    out_dir = args.output_dir or PROJECT_ROOT / (
        "results/11_mechanism_analysis/1_motif_activity_analysis/fmap2_test"
        if args.test else
        "results/11_mechanism_analysis/1_motif_activity_analysis/fmap3"
    )
    setup_logging(out_dir)
    logger.info("FMAP classification started.")

    # load
    input_csv = args.input or FMAP_PARSED_DIR / "fmap_parsed.csv"
    if not input_csv.exists():
        logger.error("Input not found: %s", input_csv)
        sys.exit(1)

    df = pd.read_csv(input_csv)
    logger.info("Loaded %d helix rows, %d unique peptides",
                len(df), df["Peptide_ID"].nunique())

    # resolve duplicate columns from merge
    for col in ["Sequence", "Motif"]:
        if col not in df.columns:
            for cand in [f"{col}_x", f"{col}_y"]:
                if cand in df.columns:
                    df[col] = df[cand]; break
            else:
                df[col] = None

    # numeric cleaning
    for col in ["Membrane_Binding_Energy","Transfer_Energy","Tilt_Angle",
                "Depth_Thickness","Helix_Start","Helix_End"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Is_False"]    = df["Is_False_Helix"].astype(str).str.upper() == "TRUE"
    df["Helix_Length"] = df["Helix_End"] - df["Helix_Start"] + 1

    # select primary helix
    primary = select_primary_helix(
        df.dropna(subset=["Transfer_Energy","Tilt_Angle","Depth_Thickness"]))
    logger.info("Primary helices selected: %d  (real: %d, false fallback: %d)",
                len(primary), (~primary["Is_False"]).sum(),
                primary["Is_False"].sum())

    # classify
    primary["Mechanism"]          = primary.apply(classify_mechanism, axis=1)
    primary["Insertion_Strength"] = primary["Membrane_Binding_Energy"].apply(
        classify_insertion_strength)

    # log distribution
    mech_counts = primary["Mechanism"].value_counts()
    logger.info("\n=== Mechanism distribution ===")
    for m in MECHANISMS:
        n = mech_counts.get(m, 0)
        logger.info("  %-20s %4d  (%.1f%%)", m, n, 100*n/len(primary))

    logger.info("\n=== Insertion strength ===\n%s",
                primary["Insertion_Strength"].value_counts().to_string())

    # ambiguous breakdown
    amb = primary[primary["Mechanism"]=="Ambiguous"]
    logger.info("\n=== Ambiguous: %d ===", len(amb))
    if not amb.empty:
        logger.info("\n%s", amb[["Tilt_Angle","Depth_Thickness"]].describe().round(2).to_string())

    # optimizer
    opt_df = run_optimizer(primary) if args.optimize else pd.DataFrame()

    # save Excel
    out_xlsx = out_dir / "fmap_classified.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        primary.to_excel(w,    sheet_name="Classified",      index=False)
        amb.to_excel(w,        sheet_name="Ambiguous",       index=False)
        mech_counts.reset_index().to_excel(
            w, sheet_name="Mechanism_counts", index=False)
        primary["Insertion_Strength"].value_counts().reset_index().to_excel(
            w, sheet_name="Insertion_strength", index=False)
        if not opt_df.empty:
            opt_df.to_excel(w, sheet_name="Threshold_optimizer", index=False)
        # GraphPad
        gp = primary[["Peptide_ID","Mechanism","Tilt_Angle",
                       "Depth_Thickness","Membrane_Binding_Energy"]].copy()
        gp.to_excel(w, sheet_name="GP_classified", index=False)

    primary.to_csv(out_dir / "fmap_classified.csv", index=False)
    logger.info("Saved: %s", out_xlsx)

    # figures
    fig_tilt_vs_depth(primary, out_dir)
    fig_tilt_vs_energy(primary, out_dir)
    fig_energy_distribution(primary, out_dir)
    fig_depth_boxplot(primary, out_dir)

    logger.info("FMAP classification completed successfully.")
    logger.info("Outputs: %s", out_dir)


if __name__ == "__main__":
    main()