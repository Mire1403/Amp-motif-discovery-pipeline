"""
ColabFold analysis — best combination per Top E candidate.
Generates pLDDT per residue panel (2×5) and summary figures.

Usage:
    conda activate amp_meme
    python scripts/colabfold/02_analyze_best_combos.py

Output:
    results/19_colabfold_analysis_motifs/best_combos/
        figures/
            01_plddt_per_residue.png
            02_plddt_mean_barplot.png
        colabfold_best_combos.xlsx
"""

from __future__ import annotations
import json
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =====================================================
# CONFIG
# =====================================================

def _find_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / ".git").exists() or (p / "pyproject.toml").exists():
            return p
    raise RuntimeError("Project root not found.")

ROOT    = _find_root()
BASE    = ROOT / "data/raw/colabfold_best_combos"
OUT_DIR = ROOT / "results/19_colabfold_analysis_motifs/best_combos"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

DPI = 300

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "02_analyze_best_combos.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Best combination per Top E candidate
MAPPING = {
    1:  ("KLLKKLLK",    "KLLKKLLKLL",    5.04,  "Carpet+Carpet"),
    2:  ("LKKLLKKLKK",  "LGLLGKLL",      6.57,  "Carpet+Pore"),
    3:  ("KLLKKLLKLL",  "ARLLRRLAR",     5.42,  "Carpet+Carpet"),
    4:  ("RRWWCL",      "FFKKIKKKIKKIGK",16.74, "Carpet+Carpet"),
    5:  ("RKKWFW",      "LKLLLKL",       8.20,  "Pore+Carpet"),
    6:  ("KLLKKLLKLL",  "WRWWRRW",       7.71,  "Carpet+Carpet"),
    7:  ("KLLKKLLK",    "FLPILKGLLKGL",  11.41, "Carpet+Carpet"),
    8:  ("IRVAVIR",     "KLLKKLLKLL",    6.70,  "Carpet+Carpet"),
    9:  ("KLLKKLLKLL",  "KHFIHRF",       14.74, "Carpet+Carpet"),
    10: ("NLKALAA",     "KLLKKLLKLL",    10.58, "Carpet+Carpet"),
}

PAL_MECH = {"Carpet+Carpet": "#E07B54", "Carpet+Pore": "#5B8DB8",
            "Pore+Carpet": "#5B8DB8"}

# =====================================================
# HELPERS
# =====================================================

def read_plddt_json(json_path: Path) -> np.ndarray:
    with open(json_path) as f:
        data = json.load(f)
    return np.array(data["plddt"])

def find_rank1_json(folder: Path) -> Path | None:
    hits = sorted(folder.glob("*scores_rank_001*.json"))
    return hits[0] if hits else None

def plddt_color(v: float) -> str:
    if v >= 90: return "#0053D6"   # blau fosc — molt alt
    if v >= 70: return "#65CBF3"   # cian — confident
    if v >= 50: return "#FFDB13"   # groc — baix
    return "#FF7D45"               # taronja — molt baix

# =====================================================
# LOAD DATA
# =====================================================

rows = []
plddt_data = {}

for i, (m1, m2, mic, mech) in MAPPING.items():
    folders = sorted(BASE.glob(f"best_comb{i}_*"))
    if not folders:
        logger.warning("NOT FOUND: best_comb%d", i)
        continue

    nested = folders[0] / folders[0].name
    json_path = find_rank1_json(nested)
    if json_path is None:
        logger.warning("NO JSON: comb%d", i)
        continue

    plddt = read_plddt_json(json_path)
    plddt_data[i] = plddt

    rows.append({
        "ID":             i,
        "Motif_1":        m1,
        "Motif_2":        m2,
        "Mechanism":      mech,
        "MIC_predicted":  mic,
        "n_residues":     len(plddt),
        "pLDDT_mean":     round(float(plddt.mean()), 1),
        "pLDDT_pct_90":   round(float((plddt >= 90).mean() * 100), 1),
        "pLDDT_pct_70":   round(float((plddt >= 70).mean() * 100), 1),
        "pLDDT_min":      round(float(plddt.min()), 1),
        "pLDDT_max":      round(float(plddt.max()), 1),
    })
    logger.info("comb%2d: %s+%s  MIC=%.2f  pLDDT=%.1f  >70=%.0f%%",
                i, m1, m2, mic, plddt.mean(), (plddt>=70).mean()*100)

df = pd.DataFrame(rows)

# =====================================================
# FIG 1: pLDDT per residu — panel 2×5
# =====================================================

fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=True)
fig.patch.set_facecolor("white")
axes = axes.flatten()

for idx, i in enumerate(range(1, 11)):
    ax = axes[idx]
    ax.set_facecolor("white")

    if i not in plddt_data:
        ax.set_title(f"comb{i:02d}\n(no data)", fontsize=9)
        continue

    arr = plddt_data[i]
    colors = [plddt_color(v) for v in arr]
    ax.bar(range(len(arr)), arr, color=colors, alpha=0.95,
           edgecolor="none", width=0.85)

    ax.axhline(90, color="#0053D6", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(70, color="#65CBF3", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(50, color="#FFDB13", lw=0.8, ls="--", alpha=0.4)

    m1, m2, mic, mech = MAPPING[i]
    ax.set_title(
        f"comb{i:02d}: {m1[:8]}+{m2[:8]}\n"
        f"MIC={mic:.1f} µmol/L  ⌀={arr.mean():.0f}",
        fontsize=8, fontweight="bold"
    )
    ax.set_xlabel("Posició (residu)", fontsize=7)
    if idx % 5 == 0:
        ax.set_ylabel("pLDDT", fontsize=8)
    ax.set_ylim(0, 105)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)

# llegenda colors
patches = [
    mpatches.Patch(color="#0053D6", label="pLDDT ≥ 90 (molt alt)"),
    mpatches.Patch(color="#65CBF3", label="70–90 (confident)"),
    mpatches.Patch(color="#FFDB13", label="50–70 (baix)"),
    mpatches.Patch(color="#FF7D45", label="< 50 (molt baix)"),
]
fig.legend(handles=patches, fontsize=9, loc="lower center",
           ncol=4, bbox_to_anchor=(0.5, -0.02))
fig.suptitle(
    "pLDDT per posició de residu — millor combinació dels 10 candidats Top E (No_linker)\n"
    "AlphaFold2 via ColabFold (mode single_sequence)",
    fontsize=12, fontweight="bold", y=1.01
)
fig.tight_layout()
fig.savefig(FIG_DIR / "01_plddt_per_residue.png",
            dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close()
logger.info("Figure saved: 01_plddt_per_residue.png")

# =====================================================
# FIG 2: pLDDT mean barplot
# =====================================================

fig, ax = plt.subplots(figsize=(11, 5), facecolor="white")
ax.set_facecolor("white")

colors_bar = [PAL_MECH.get(MAPPING[i][3], "#888") for i in df["ID"]]
bars = ax.bar(range(len(df)), df["pLDDT_mean"],
              color=colors_bar, alpha=0.85, edgecolor="white", width=0.7)

for bar, val, pct in zip(bars, df["pLDDT_mean"], df["pLDDT_pct_70"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.axhline(90, color="#0053D6", lw=1.2, ls="--", alpha=0.6, label="pLDDT=90")
ax.axhline(70, color="#65CBF3", lw=1.2, ls="--", alpha=0.6, label="pLDDT=70")
ax.set_xticks(range(len(df)))
ax.set_xticklabels(
    [f"c{r.ID:02d}\n{r.Motif_1[:7]}\n+{r.Motif_2[:7]}"
     for _, r in df.iterrows()],
    fontsize=7.5
)
ax.set_ylabel("pLDDT mitjà", fontsize=10)
ax.set_ylim(0, 105)

patches_mech = [
    mpatches.Patch(color="#E07B54", label="Carpet+Carpet"),
    mpatches.Patch(color="#5B8DB8", label="Carpet+Pore / Pore+Carpet"),
]
ax.legend(handles=patches_mech + [
    mpatches.Patch(color="none", label=""),
    plt.Line2D([0],[0], color="#0053D6", ls="--", lw=1.2, label="pLDDT=90"),
    plt.Line2D([0],[0], color="#65CBF3", ls="--", lw=1.2, label="pLDDT=70"),
], fontsize=8, framealpha=0.8)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "02_plddt_mean_barplot.png",
            dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close()
logger.info("Figure saved: 02_plddt_mean_barplot.png")

# =====================================================
# SAVE EXCEL
# =====================================================

out_xl = OUT_DIR / "colabfold_best_combos.xlsx"
with pd.ExcelWriter(out_xl, engine="openpyxl") as w:
    df.to_excel(w, sheet_name="Summary", index=False)
    pd.DataFrame([{
        "mean_pLDDT_all":    df["pLDDT_mean"].mean().round(1),
        "n_above90_mean":    (df["pLDDT_mean"] >= 90).sum(),
        "n_above70_mean":    (df["pLDDT_mean"] >= 70).sum(),
        "min_pLDDT":         df["pLDDT_mean"].min(),
        "max_pLDDT":         df["pLDDT_mean"].max(),
    }]).to_excel(w, sheet_name="Stats", index=False)

logger.info("Excel saved: %s", out_xl)
logger.info("Done. Outputs: %s", OUT_DIR)

# print summary
print("\n=== RESUM ===")
print(df[["ID","Motif_1","Motif_2","MIC_predicted",
          "pLDDT_mean","pLDDT_pct_70"]].to_string(index=False))
print(f"\nMean pLDDT global: {df['pLDDT_mean'].mean():.1f}")
print(f"Combinacions amb pLDDT_mean > 90: {(df['pLDDT_mean']>=90).sum()}/10")
print(f"Combinacions amb pLDDT_mean > 70: {(df['pLDDT_mean']>=70).sum()}/10")