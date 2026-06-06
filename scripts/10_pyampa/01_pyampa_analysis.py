"""
PyAMPA analysis — Top E individual candidates + Top 10 combinations (no linker)
Uses PyAMPA models directly (without GUI) via pickle.

Usage:
    conda activate pyampa_env
    python scripts/10_pyampa/01_pyampa_analysis.py

Requirements:
    - PyAMPA repository cloned at /mnt/c/Users/mirei/Desktop/PyAMPA
      (or set PYAMPA_DIR below)
    - pyampa_env conda environment with scikit-learn==1.2.2

Output:
    results/20_pyampa_analysis/
        pyampa_results.csv
        pyampa_results.xlsx  (sheets: All, individual, nolinker)
        fig1_individual.png
        fig2_nolinker.png
        fig3_group_comparison.png
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =====================================================
# CONFIG
# =====================================================

def _find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find project root.")

PROJECT_ROOT = _find_project_root()
OUT_DIR      = PROJECT_ROOT / "results/20_pyampa_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300

# PyAMPA directory — change if needed
DEFAULT_PYAMPA_DIR = Path("/mnt/c/Users/mirei/Desktop/PyAMPA")

# =====================================================
# LOGGING
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOGS_DIR / "01_pyampa_analysis.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# =====================================================
# CANDIDATES
# =====================================================

# Top E — 10 diverse candidates (1 per hierarchical cluster, no family restriction)
CANDIDATES_IND = [
    {"Motif": "KLLKKLLKLL",    "Seq": "KLLKKLLKLL",    "MIC_ind": 17.58, "Family": "FAM_101", "Mech": "Carpet", "Group": "individual"},
    {"Motif": "LKKLLKKLKK",    "Seq": "LKKLLKKLKK",    "MIC_ind": 23.48, "Family": "FAM_101", "Mech": "Carpet", "Group": "individual"},
    {"Motif": "ARLLRRLAR",     "Seq": "ARLLRRLAR",     "MIC_ind": 43.03, "Family": "FAM_009", "Mech": "Carpet", "Group": "individual"},
    {"Motif": "FFKKIKKKIKKIGK","Seq": "FFKKIKKKIKKIGK","MIC_ind": 36.04, "Family": "FAM_202", "Mech": "Carpet", "Group": "individual"},
    {"Motif": "RKKWFW",        "Seq": "RKKWFW",        "MIC_ind": 69.44, "Family": "FAM_121", "Mech": "Pore",   "Group": "individual"},
    {"Motif": "WRWWRRW",       "Seq": "WRWWRRW",       "MIC_ind": 70.83, "Family": "FAM_137", "Mech": "Carpet", "Group": "individual"},
    {"Motif": "FLPILKGLLKGL",  "Seq": "FLPILKGLLKGL",  "MIC_ind": 63.84, "Family": "FAM_102", "Mech": "Carpet", "Group": "individual"},
    {"Motif": "IRVAVIR",       "Seq": "IRVAVIR",       "MIC_ind": 69.77, "Family": "FAM_078", "Mech": "Carpet", "Group": "individual"},
    {"Motif": "KHFIHRF",       "Seq": "KHFIHRF",       "MIC_ind": 43.42, "Family": "FAM_049", "Mech": "Carpet", "Group": "individual"},
    {"Motif": "NLKAJAA",       "Seq": "NLKALAA",       "MIC_ind": 47.36, "Family": "FAM_110", "Mech": "Carpet", "Group": "individual"},
]

# Top 10 combinations — no linker (lowest MIC predicted by APEX)
CANDIDATES_COMB = [
    {"Motif": "KLLKKLLK+KLLKKLLKLL",   "Seq": "KLLKKLLKKLLKKLLKLL",   "MIC_ind":  5.04, "Family": "FAM_101", "Mech": "Carpet+Carpet", "Group": "nolinker"},
    {"Motif": "KLLKKLLK+LGLLGKLL",     "Seq": "KLLKKLLKLGLLGKLL",     "MIC_ind":  5.23, "Family": "FAM_101", "Mech": "Carpet+Pore",   "Group": "nolinker"},
    {"Motif": "KLLKKLLKLL+LGLLGKLL",   "Seq": "KLLKKLLKLLLGLLGKLL",   "MIC_ind":  5.26, "Family": "FAM_101", "Mech": "Carpet+Pore",   "Group": "nolinker"},
    {"Motif": "KLLKKLLKLL+ARLLRRLAR",  "Seq": "KLLKKLLKLLARLLRRLAR",  "MIC_ind":  5.42, "Family": "FAM_101", "Mech": "Carpet+Carpet", "Group": "nolinker"},
    {"Motif": "KLLKKLLKLL+KLLKKLLKLL", "Seq": "KLLKKLLKLLKLLKKLLKLL", "MIC_ind":  6.18, "Family": "FAM_101", "Mech": "Carpet+Carpet", "Group": "nolinker"},
    {"Motif": "ARLLRRLAR+LKLLLKL",     "Seq": "ARLLRRLARLKLLLKL",     "MIC_ind":  6.20, "Family": "FAM_009", "Mech": "Carpet+Carpet", "Group": "nolinker"},
    {"Motif": "KLLKKLLKLL+KIRVRL",     "Seq": "KLLKKLLKLLKIRVRL",     "MIC_ind":  6.26, "Family": "FAM_101", "Mech": "Carpet+Carpet", "Group": "nolinker"},
    {"Motif": "KLLKKLLKLL+LKLLLKL",    "Seq": "KLLKKLLKLLLKLLLKL",    "MIC_ind":  6.27, "Family": "FAM_101", "Mech": "Carpet+Carpet", "Group": "nolinker"},
    {"Motif": "ARLLRRLAR+KLLKKLLKLL",  "Seq": "ARLLRRLARKLLKKLLKLL",  "MIC_ind":  6.53, "Family": "FAM_009", "Mech": "Carpet+Carpet", "Group": "nolinker"},
    {"Motif": "RIRVAVIRA+LKLLLKL",     "Seq": "RIRVAVIRALKLLLKL",     "MIC_ind":  6.56, "Family": "FAM_078", "Mech": "Carpet+Carpet", "Group": "nolinker"},
]

CANDIDATES = CANDIDATES_IND + CANDIDATES_COMB

# =====================================================
# HELPERS
# =====================================================

AA_SUB = {"J": "L", "B": "N", "Z": "E", "U": "C", "X": "A"}

def clean_seq(seq: str) -> str:
    return "".join(AA_SUB.get(aa, aa) for aa in seq.upper()
                   if aa.upper() in "ACDEFGHIKLMNPQRSTVWY" + "".join(AA_SUB.keys()))

def to_bigrams(seq: str) -> str:
    s = seq.lower()
    return " ".join(s[i:i+2] for i in range(len(s) - 1))

def predict_classifier(seq, vec, model):
    X = vec.transform([to_bigrams(seq)])
    pred  = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    pos_idx = list(model.classes_).index(1) if 1 in model.classes_ else -1
    return int(pred), round(float(proba[pos_idx]), 3)

def predict_regressor(seq, vec, model):
    from scipy.sparse import hstack, csr_matrix
    X = vec.transform([to_bigrams(seq)])
    n_exp = model.n_features_in_
    if X.shape[1] < n_exp:
        X = hstack([X, csr_matrix((X.shape[0], n_exp - X.shape[1]))])
    elif X.shape[1] > n_exp:
        X = X[:, :n_exp]
    return float(model.predict(X)[0])

# =====================================================
# PALETTE
# =====================================================

PAL = {
    "Carpet":         "#E07B54",
    "Pore":           "#3BAF77",
    "Carpet+Carpet":  "#E07B54",
    "Carpet+Pore":    "#5B8DB8",
    "Pore+Pore":      "#3BAF77",
}

# =====================================================
# FIGURES
# =====================================================

def plot_group(sub: pd.DataFrame, title: str, fname: str) -> None:
    if sub.empty:
        return
    colors = [PAL.get(m, "#888") for m in sub["Mechanism"]]
    fig, axes = plt.subplots(1, 3, figsize=(16, max(4, len(sub) * 0.5)),
                             facecolor="white")
    for ax, col, ttl, thresh, tcol in [
        (axes[0], "P_AMP",        "P(AMP)\nhigher = more antimicrobial", 0.5, "green"),
        (axes[1], "P_Hemolytic",  "P(Hemolytic)\nlower = safer",          0.5, "red"),
        (axes[2], "Safety_score", "Safety score\n(1-P_Hemo)×P_AMP",      None, None),
    ]:
        bars = ax.barh(range(len(sub)), sub[col], color=colors,
                       alpha=0.85, edgecolor="white")
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels([r["Motif"][:30] for _, r in sub.iterrows()], fontsize=8)
        if thresh:
            ax.axvline(thresh, color=tcol, lw=1.2, ls="--", alpha=0.6)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Score", fontsize=9)
        ax.set_title(ttl, fontweight="bold", fontsize=9)
        for bar, val in zip(bars, sub[col]):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)
    patches = [mpatches.Patch(color=c, label=m)
               for m, c in PAL.items() if m in sub["Mechanism"].values]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"PyAMPA — {title}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("Figure saved: %s", fname)

# =====================================================
# MAIN
# =====================================================

def parse_args():
    p = argparse.ArgumentParser(description="PyAMPA analysis — Top E candidates")
    p.add_argument("--pyampa_dir", type=Path, default=DEFAULT_PYAMPA_DIR,
                   help=f"Path to PyAMPA directory (default: {DEFAULT_PYAMPA_DIR})")
    return p.parse_args()


def main():
    args = parse_args()
    pyampa_dir = args.pyampa_dir

    if not pyampa_dir.exists():
        logger.error("PyAMPA directory not found: %s", pyampa_dir)
        logger.error("Clone from https://github.com/SysBioUAB/PyAMPA and set --pyampa_dir")
        sys.exit(1)

    logger.info("Loading PyAMPA models from %s", pyampa_dir)
    models = {}
    for name, fname in [
        ("hemo_model",   "hemolysis_model.pkl"),
        ("hemo_vec",     "hemolysis_vectorizer.pkl"),
        ("amp_model",    "AMPValidate.pkl"),
        ("amp_vec",      "amp_validate_vectorizer.pkl"),
        ("act_model",    "activities_model.pkl"),
        ("act_vec",      "activities_vectorizer.pkl"),
        ("tox_model",    "tox_model.pkl"),
        ("tox_vec",      "tox_vectorizer.pkl"),
        ("cpp_model",    "cpp_model.pkl"),
        ("cpp_vec",      "cpp_vectorizer.pkl"),
    ]:
        path = pyampa_dir / fname
        if not path.exists():
            logger.error("Model file not found: %s", path)
            sys.exit(1)
        with open(path, "rb") as f:
            models[name] = pickle.load(f)
        logger.info("  Loaded %-30s -> %s", fname, type(models[name]).__name__)

    # run predictions
    rows = []
    for c in CANDIDATES:
        seq = clean_seq(c["Seq"])
        if len(seq) < 3:
            logger.warning("Skipping %s — sequence too short after cleaning", c["Motif"])
            continue

        amp_pred,  p_amp  = predict_classifier(seq, models["amp_vec"],  models["amp_model"])
        hemo_pred, p_hemo = predict_classifier(seq, models["hemo_vec"], models["hemo_model"])
        tox_pred,  p_tox  = predict_classifier(seq, models["tox_vec"],  models["tox_model"])
        cpp_pred,  p_cpp  = predict_classifier(seq, models["cpp_vec"],  models["cpp_model"])
        act_val           = predict_regressor( seq, models["act_vec"],  models["act_model"])

        row = {
            "Motif":          c["Motif"],
            "Sequence":       seq,
            "Group":          c["Group"],
            "MIC_ind":        c["MIC_ind"],
            "Family":         c["Family"],
            "Mechanism":      c["Mech"],
            "P_AMP":          p_amp,
            "P_Hemolytic":    p_hemo,
            "P_Toxic":        p_tox,
            "P_CPP":          p_cpp,
            "Activity_score": round(act_val, 3),
        }
        rows.append(row)
        logger.info("[%-10s] %-30s AMP=%.2f  Hemo=%.2f  Tox=%.2f",
                    c["Group"], c["Motif"], p_amp, p_hemo, p_tox)

    df = pd.DataFrame(rows)
    df["Safety_score"] = ((1 - df["P_Hemolytic"]) * df["P_AMP"]).round(3)

    # print summary
    for grp in ["individual", "nolinker"]:
        sub = df[df["Group"] == grp]
        if sub.empty:
            continue
        logger.info("\n=== %s ===", grp.upper())
        logger.info("\n%s", sub[["Motif", "MIC_ind", "P_AMP",
                                  "P_Hemolytic", "P_Toxic",
                                  "Safety_score"]].to_string(index=False))

    # save
    df.to_csv(OUT_DIR / "pyampa_results.csv", index=False)
    with pd.ExcelWriter(OUT_DIR / "pyampa_results.xlsx", engine="openpyxl") as w:
        df.to_excel(w, sheet_name="All", index=False)
        for grp in ["individual", "nolinker"]:
            sub = df[df["Group"] == grp]
            if not sub.empty:
                sub.to_excel(w, sheet_name=grp, index=False)
    logger.info("Results saved to %s", OUT_DIR)

    # figures
    plot_group(df[df["Group"] == "individual"],
               "Individual motifs — Top E candidates",
               "fig1_individual.png")
    plot_group(df[df["Group"] == "nolinker"],
               "Combinations — No linker (top 10)",
               "fig2_nolinker.png")

    # group comparison
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    grp_means = df.groupby("Group")[["P_AMP", "P_Hemolytic", "Safety_score"]].mean()
    grp_means = grp_means.reindex(["individual", "nolinker"])
    x = np.arange(len(grp_means)); w = 0.25
    ax.bar(x - w, grp_means["P_AMP"],       width=w, color="#3BAF77",
           alpha=0.85, edgecolor="white", label="P(AMP)")
    ax.bar(x,     grp_means["P_Hemolytic"], width=w, color="#E07B54",
           alpha=0.85, edgecolor="white", label="P(Hemolytic)")
    ax.bar(x + w, grp_means["Safety_score"],width=w, color="#4C72B0",
           alpha=0.85, edgecolor="white", label="Safety score")
    for i, (_, row) in enumerate(grp_means.iterrows()):
        ax.text(i + w, row["Safety_score"] + 0.01,
                f"{row['Safety_score']:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Individual", "No linker"], fontsize=11)
    ax.set_ylabel("Mean score", fontsize=10)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="grey", lw=0.8, ls="--", alpha=0.4)
    ax.set_title("PyAMPA — Mean scores per group\n"
                 "P(AMP) high + P(Hemolytic) low = best therapeutic profile",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_group_comparison.png",
                dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("Figure saved: fig3_group_comparison.png")
    logger.info("PyAMPA analysis completed. Outputs: %s", OUT_DIR)


if __name__ == "__main__":
    main()