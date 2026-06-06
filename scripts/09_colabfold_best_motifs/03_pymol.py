from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/raw/colabfold_motifs"
OUT  = ROOT / "results/19_colabfold_analysis_motifs/figures/pymol"
OUT.mkdir(parents=True, exist_ok=True)

MAPPING = {
    1:  ("KLLKKLLK",   "KLLKKLLKLL",  5.04, 96.4),
    2:  ("KLLKKLLK",   "LGLLGKLL",    5.23, 87.9),
    3:  ("KLLKKLLKLL", "LGLLGKLL",    5.26, 93.0),
    4:  ("KLLKKLLKLL", "ARLLRRLAR",   5.42, 88.2),
    5:  ("KLLKKLLKLL", "KLLKKLLKLL",  6.18, 96.8),
    6:  ("ARLLRRLAR",  "LKLLLKL",     6.20, 76.5),
    7:  ("KLLKKLLKLL", "KIRVRL",      6.26, 88.5),
    8:  ("KLLKKLLKLL", "LKLLLKL",     6.27, 95.2),
    9:  ("ARLLRRLAR",  "KLLKKLLKLL",  6.53, 89.1),
    10: ("RIRVAVIRA",  "LKLLLKL",     6.56, 82.8),
}

CONDITIONS = {
    "nolinker": BASE / "single_motif",
    "aaa":      BASE / "comb_aaa",
    "ggg":      BASE / "comb_ggg",
}

COND_LABELS = {"nolinker": "No linker", "aaa": "AAA linker", "ggg": "GGG linker"}
COND_COLORS = {"nolinker": "#888888",   "aaa": "#E07B54",    "ggg": "#4C72B0"}

def get_rank1_pdb(folder_path):
    pdbs = sorted(Path(folder_path).glob("*rank_001*.pdb"))
    if pdbs:
        return pdbs[0]
    # check double-nested
    for sub in Path(folder_path).iterdir():
        if sub.is_dir():
            pdbs2 = sorted(sub.glob("*rank_001*.pdb"))
            if pdbs2:
                return pdbs2[0]
    return None

def find_folder(cond_dir, cond_key, i):
    if cond_key == "nolinker":
        p = cond_dir / f"single_motif{i}"
        return p if p.exists() else None
    matches = sorted(cond_dir.glob(f"comb_{cond_key}{i}_*"))
    matches = [m for m in matches if m.is_dir() and not m.name.endswith(".zip")]
    return matches[0] if matches else None

def read_bfactors(pdb_path):
    residues = {}
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                res_num = int(line[22:26].strip())
                residues[res_num] = float(line[60:66].strip())
    if not residues:
        return np.array([])
    return np.array([residues[r] for r in sorted(residues)])

def plddt_color(val):
    if val >= 90:   return "#0057B7"
    elif val >= 70: return "#65CBF3"
    elif val >= 50: return "#F0AB00"
    else:           return "#FF4040"

LEGEND_PATCHES = [
    mpatches.Patch(color="#0057B7", label="Very high ≥90"),
    mpatches.Patch(color="#65CBF3", label="Confident 70-90"),
    mpatches.Patch(color="#F0AB00", label="Low 50-70"),
    mpatches.Patch(color="#FF4040", label="Very low <50"),
]

# load all pLDDT data
plddt_all = {}
for cond_key, cond_dir in CONDITIONS.items():
    plddt_all[cond_key] = {}
    for i in range(1, 11):
        folder = find_folder(cond_dir, cond_key, i)
        if folder is None:
            print(f"  NOT FOUND: {cond_key} {i}")
            continue
        pdb = get_rank1_pdb(folder)
        if pdb is None:
            print(f"  NO PDB: {cond_key} {i}")
            continue
        arr = read_bfactors(pdb)
        plddt_all[cond_key][i] = {"arr": arr, "pdb": pdb,
                                   "mean": arr.mean() if len(arr)>0 else 0}
        print(f"  {cond_key} {i:2d}: n_res={len(arr)}  pLDDT={arr.mean():.1f}")

# ── Fig 1: pLDDT per residue — 3 conditions × 10 combinations ──
fig = plt.figure(figsize=(22, 14), facecolor="white")
gs  = gridspec.GridSpec(3, 10, figure=fig, hspace=0.6, wspace=0.35)

for ci, cond_key in enumerate(["nolinker","aaa","ggg"]):
    for idx, i in enumerate(range(1, 11)):
        ax  = fig.add_subplot(gs[ci, idx])
        d   = plddt_all[cond_key].get(i)
        m1, m2, mic, _ = MAPPING[i]
        if d is None or len(d["arr"]) == 0:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
            ax.set_title(f"c{i}\n{COND_LABELS[cond_key]}", fontsize=6)
            continue
        arr = d["arr"]
        ax.bar(range(1, len(arr)+1), arr,
               color=[plddt_color(v) for v in arr], edgecolor="none", width=1.0)
        ax.axhline(90, color="#0057B7", lw=0.6, ls="--", alpha=0.5)
        ax.axhline(70, color="#F0AB00", lw=0.6, ls="--", alpha=0.5)
        ax.set_ylim(0, 105)
        ax.set_xlim(0.5, len(arr)+0.5)
        ax.set_title(f"c{i}: {m1[:6]}+{m2[:6]}\n{COND_LABELS[cond_key]}\n"
                     f"MIC={mic:.2f} | pLDDT={d['mean']:.0f}",
                     fontsize=6.5, fontweight="bold")
        ax.tick_params(labelsize=5.5)
        if idx == 0:
            ax.set_ylabel("pLDDT", fontsize=7)
        ax.spines[["top","right"]].set_visible(False)

fig.legend(handles=LEGEND_PATCHES, loc="lower center", ncol=4,
           fontsize=9, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("AlphaFold2 (ColabFold) — pLDDT per residue\n"
             "Top 10 combinations × 3 linker conditions",
             fontsize=14, fontweight="bold")
fig.savefig(OUT/"01_plddt_all_conditions.png", dpi=300,
            bbox_inches="tight", facecolor="white")
plt.close()
print("Saved: 01_plddt_all_conditions.png")

# ── Fig 2: Top 5 no-linker panel (for memory) ──
TOP5 = [5, 1, 8, 3, 4]
fig, axes = plt.subplots(1, 5, figsize=(22, 4), facecolor="white")
for ax, i in zip(axes, TOP5):
    m1, m2, mic, plddt_mean = MAPPING[i]
    d   = plddt_all["nolinker"].get(i)
    arr = d["arr"] if d else np.array([])
    if len(arr) == 0:
        continue
    ax.bar(range(1, len(arr)+1), arr,
           color=[plddt_color(v) for v in arr], edgecolor="none", width=1.0)
    ax.axhline(90, color="#0057B7", lw=1, ls="--", alpha=0.5)
    ax.axhline(70, color="#F0AB00", lw=1, ls="--", alpha=0.5)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Residue", fontsize=9)
    ax.set_ylabel("pLDDT" if ax == axes[0] else "", fontsize=9)
    ax.set_title(f"comb{i:02d}: {m1}+{m2}\nMIC={mic:.2f} µmol/L | pLDDT={plddt_mean:.1f}",
                 fontsize=8.5, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.spines[["top","right"]].set_visible(False)
fig.legend(handles=LEGEND_PATCHES, loc="lower center", ncol=4,
           fontsize=9, bbox_to_anchor=(0.5, -0.08))
fig.suptitle("Top 5 combinations — AlphaFold2 pLDDT (no linker)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT/"02_plddt_top5_nolinker.png", dpi=300,
            bbox_inches="tight", facecolor="white")
plt.close()
print("Saved: 02_plddt_top5_nolinker.png")

# ── Fig 3: Mean pLDDT comparison 3 conditions ──
fig, ax = plt.subplots(figsize=(13, 5), facecolor="white")
x = np.arange(10)
w = 0.26
for ci, cond_key in enumerate(["nolinker","aaa","ggg"]):
    means = [plddt_all[cond_key].get(i, {}).get("mean", np.nan)
             for i in range(1, 11)]
    bars = ax.bar(x + ci*w, means, width=w,
                  color=COND_COLORS[cond_key], alpha=0.85,
                  edgecolor="white", label=COND_LABELS[cond_key])
    for bar, val in zip(bars, means):
        if not np.isnan(val):
            ax.text(bar.get_x()+bar.get_width()/2, val+0.3,
                    f"{val:.0f}", ha="center", fontsize=6.5, fontweight="bold")
ax.axhline(90, color="#0057B7", lw=1, ls="--", alpha=0.5)
ax.axhline(70, color="#F0AB00", lw=1, ls="--", alpha=0.5)
ax.set_xticks(x + w)
ax.set_xticklabels([f"c{i}\n{MAPPING[i][0][:6]}+\n{MAPPING[i][1][:6]}"
                    for i in range(1, 11)], fontsize=7.5)
ax.set_ylabel("Mean pLDDT", fontsize=11)
ax.set_ylim(0, 110)
ax.set_title("Mean pLDDT — top 10 combinations × linker condition",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT/"03_plddt_mean_comparison.png", dpi=300,
            bbox_inches="tight", facecolor="white")
plt.close()
print("Saved: 03_plddt_mean_comparison.png")

# ── Fig 4: Heatmap residue × combination (no linker) ──
max_len = max((len(d["arr"]) for d in plddt_all["nolinker"].values()
               if len(d["arr"]) > 0), default=0)
hmap = np.full((10, max_len), np.nan)
for idx, i in enumerate(range(1, 11)):
    d = plddt_all["nolinker"].get(i)
    if d and len(d["arr"]) > 0:
        hmap[idx, :len(d["arr"])] = d["arr"]

cmap = LinearSegmentedColormap.from_list(
    "plddt", ["#FF4040","#F0AB00","#65CBF3","#0057B7"])
fig, ax = plt.subplots(figsize=(14, 6), facecolor="white")
im = ax.imshow(hmap, aspect="auto", cmap=cmap, vmin=30, vmax=100)
plt.colorbar(im, ax=ax, label="pLDDT", shrink=0.8)
ax.set_yticks(range(10))
ax.set_yticklabels([f"c{i}: {MAPPING[i][0]}+{MAPPING[i][1]}"
                    for i in range(1, 11)], fontsize=9)
ax.set_xlabel("Residue position", fontsize=10)
ax.set_title("pLDDT heatmap — top 10 combinations (no linker)",
             fontsize=12, fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT/"04_plddt_heatmap.png", dpi=300,
            bbox_inches="tight", facecolor="white")
plt.close()
print("Saved: 04_plddt_heatmap.png")

# ── PyMOL 3D figures ──
try:
    import pymol
    from pymol import cmd as pymol_cmd
    pymol.finish_launching(["pymol", "-qc"])

    pymol_cmd.bg_color("white")
    pymol_cmd.set("ray_shadows", 0)
    pymol_cmd.set("antialias", 2)
    pymol_cmd.set("cartoon_fancy_helices", 1)
    pymol_cmd.set("ray_opaque_background", 0)

    for cond_key, cond_dir in CONDITIONS.items():
        for i in range(1, 11):
            d = plddt_all[cond_key].get(i)
            if d is None:
                continue
            obj = f"comb{i:02d}_{cond_key}"
            pymol_cmd.load(str(d["pdb"]), obj)
            pymol_cmd.show("cartoon", obj)
            pymol_cmd.hide("lines", obj)
            pymol_cmd.spectrum("b", "red_orange_yellow_cyan_blue",
                                obj, minimum=30, maximum=100)

    for cond_key in ["nolinker","aaa","ggg"]:
        for i in range(1, 11):
            obj = f"comb{i:02d}_{cond_key}"
            if obj not in pymol_cmd.get_object_list():
                continue
            pymol_cmd.hide("everything")
            pymol_cmd.show("cartoon", obj)
            pymol_cmd.orient(obj)
            pymol_cmd.zoom(obj, buffer=3)
            pymol_cmd.ray(700, 550)
            m1, m2, mic, _ = MAPPING[i]
            out_path = str(OUT / f"3D_{obj}_{m1}_{m2}.png")
            pymol_cmd.png(out_path, dpi=300)
            print(f"Saved: 3D_{obj}")

    pymol_cmd.quit()
    print("PyMOL 3D figures done.")

except Exception as e:
    print(f"PyMOL error: {e}")
    print("Install: conda install -c conda-forge pymol-open-source")

print(f"\n✅ All figures saved to: {OUT}")