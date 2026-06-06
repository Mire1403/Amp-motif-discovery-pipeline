"""
PyMOL panel — best combination per Top E candidate.
Standard AlphaFold pLDDT colors.

Usage:
    conda activate amp_meme
    cd /mnt/c/Users/mirei/Desktop/AMP-Comparative-Clustering-and-Motif-Analysis
    python scripts/colabfold/pymol_panel.py
"""

import os, subprocess, tempfile
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

ROOT    = Path(__file__).resolve().parents[2]
BASE    = ROOT / "data/raw/colabfold_best_combos"
FIG_DIR = ROOT / "results/19_colabfold_analysis_motifs/best_combos/figures"
TMP_DIR = ROOT / "results/19_colabfold_analysis_motifs/best_combos/tmp_pymol"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

MAPPING = {
    1:  ("KLLKKLLK",     "KLLKKLLKLL",     5.04),
    2:  ("LKKLLKKLKK",   "LGLLGKLL",       6.57),
    3:  ("KLLKKLLKLL",   "ARLLRRLAR",      5.42),
    4:  ("RRWWCL",       "FFKKIKKKIKKIGK", 16.74),
    5:  ("RKKWFW",       "LKLLLKL",        8.20),
    6:  ("KLLKKLLKLL",   "WRWWRRW",        7.71),
    7:  ("KLLKKLLK",     "FLPILKGLLKGL",   11.41),
    8:  ("IRVAVIR",      "KLLKKLLKLL",     6.70),
    9:  ("KLLKKLLKLL",   "KHFIHRF",        14.74),
    10: ("NLKALAA",      "KLLKKLLKLL",     10.58),
}

PLDDT_MEANS = {
    1: 96.4, 2: 91.2, 3: 88.2, 4: 93.3, 5: 75.0,
    6: 82.5, 7: 91.4, 8: 90.3, 9: 93.3, 10: 91.0,
}

def find_rank1_pdb(folder):
    hits = sorted(folder.glob("*rank_001*.pdb"))
    return hits[0] if hits else None

def read_pdb_bfactor(pdb_path):
    residues = {}
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                res_num = int(line[22:26].strip())
                bfactor = float(line[60:66].strip())
                residues[res_num] = bfactor
    if not residues:
        return np.array([])
    return np.array([residues[r] for r in sorted(residues)])

def crop_whitespace(img_array, padding=25):
    bg   = np.array([255, 255, 255])
    mask = np.abs(img_array[:,:,:3].astype(int) - bg).sum(axis=2) > 15
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return img_array
    r0, r1 = np.where(rows)[0][[0,-1]]
    c0, c1 = np.where(cols)[0][[0,-1]]
    r0 = max(0, r0-padding); r1 = min(img_array.shape[0], r1+padding)
    c0 = max(0, c0-padding); c1 = min(img_array.shape[1], c1+padding)
    return img_array[r0:r1, c0:c1]

PYMOL_TPL = """from pymol import cmd

cmd.load('{pdb}', '{name}')
cmd.remove('solvent')
cmd.show_as('cartoon', '{name}')
cmd.bg_color('white')

cmd.set_color('af_blue',   [0.000, 0.325, 0.839])
cmd.set_color('af_cyan',   [0.396, 0.796, 0.953])
cmd.set_color('af_yellow', [1.000, 0.859, 0.071])
cmd.set_color('af_orange', [1.000, 0.490, 0.271])

cmd.select('s1', '{name} and b > 89.9')
cmd.select('s2', '{name} and b > 69.9 and b < 90.0')
cmd.select('s3', '{name} and b > 49.9 and b < 70.0')
cmd.select('s4', '{name} and b < 50.0')

cmd.color('af_blue',   's1')
cmd.color('af_cyan',   's2')
cmd.color('af_yellow', 's3')
cmd.color('af_orange', 's4')
cmd.deselect()

cmd.set('cartoon_fancy_helices', 1)
cmd.set('cartoon_tube_radius', 0.35)
cmd.set('ray_shadows', 0)
cmd.set('antialias', 2)
cmd.set('ray_opaque_background', 1)
cmd.orient('{name}')
cmd.zoom('{name}', 5)
cmd.png('{out}', width=1000, height=1000, dpi=300, ray=1)
cmd.quit()
"""

# ── render ────────────────────────────────────────────────────────────────────
img_paths = {}

for i in range(1, 11):
    folders = sorted(BASE.glob(f"best_comb{i}_*"))
    if not folders:
        print(f"NOT FOUND: comb{i}"); continue
    nested   = folders[0] / folders[0].name
    pdb_path = find_rank1_pdb(nested)
    if pdb_path is None:
        print(f"NO PDB: comb{i}"); continue

    plddt = read_pdb_bfactor(str(pdb_path))
    print(f"  comb{i:02d}: n_res={len(plddt)}  mean={plddt.mean():.1f}")

    out_png = str(TMP_DIR / f"comb{i:02d}.png")
    script  = PYMOL_TPL.format(
        pdb=str(pdb_path), name=f"comb{i:02d}", out=out_png)

    tf = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w")
    tf.write(script); tf.close()

    result = subprocess.run(["pymol", "-cq", tf.name],
                            capture_output=True, text=True, timeout=180)
    os.unlink(tf.name)

    if os.path.exists(out_png):
        print(f"  OK comb{i:02d}")
        img_paths[i] = out_png
    else:
        print(f"  ERROR comb{i}: {result.stderr[:400]}")

# ── panel 2×5 ─────────────────────────────────────────────────────────────────
panels = "abcdefghij"

# Use gridspec with extra space at top of each cell for letter
# and extra space at bottom for labels
fig = plt.figure(figsize=(26, 12), facecolor="white")
from matplotlib.gridspec import GridSpec
gs = GridSpec(2, 5, figure=fig,
              hspace=0.02, wspace=0.04,
              top=0.93, bottom=0.12,
              left=0.01, right=0.99)

for idx in range(10):
    row = idx // 5
    col = idx % 5
    ax  = fig.add_subplot(gs[row, col])
    i   = idx + 1

    ax.set_facecolor("white")
    ax.axis("off")

    if i in img_paths:
        img_arr = np.array(Image.open(img_paths[i]))
        img_arr = crop_whitespace(img_arr, padding=25)
        ax.imshow(img_arr, aspect="equal")

    m1, m2, mic = MAPPING[i]

    # lletra panel — dalt esquerra, AMB espai sobre la figura
    ax.text(0.01, 1.10, panels[idx],
            transform=ax.transAxes,
            fontsize=28, fontweight="bold",
            va="bottom", ha="left", color="black")

    # pLDDT — dalt dreta, dins de la figura
    ax.text(0.99, 0.99, f"pLDDT = {PLDDT_MEANS[i]:.0f}",
            transform=ax.transAxes,
            fontsize=13, va="top", ha="right", color="#111",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="#bbb", alpha=0.92))

    # seqüència — sota, línia 1
    ax.text(0.5, -0.04,
            f"{m1}+{m2}",
            transform=ax.transAxes,
            fontsize=12, va="top", ha="center",
            color="#222", fontweight="bold")

    # MIC — sota, línia 2
    ax.text(0.5, -0.11,
            f"MIC = {mic:.1f} µmol/L",
            transform=ax.transAxes,
            fontsize=12, va="top", ha="center",
            color="#444")

# llegenda
legend_patches = [
    mpatches.Patch(color="#0053D6", label="pLDDT ≥ 90"),
    mpatches.Patch(color="#65CBF3", label="70–90"),
    mpatches.Patch(color="#FFDB13", label="50–70"),
    mpatches.Patch(color="#FF7D45", label="< 50"),
]
fig.legend(handles=legend_patches, fontsize=15,
           loc="lower center", ncol=4,
           bbox_to_anchor=(0.5, 0.02),
           frameon=True, edgecolor="#ccc",
           handlelength=1.8, handleheight=1.4)

out_panel = FIG_DIR / "01_pymol_panel.png"
fig.savefig(out_panel, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nPanel saved: {out_panel}")