from pathlib import Path
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

ROOT    = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data/raw/colabfold_motifs"
OUT_DIR = ROOT / "results/19_colabfold_analysis_motifs"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.05)
DPI = 300

MAPPING = {
    1:  {"Motif_1":"KLLKKLLK",  "Motif_2":"KLLKKLLKLL", "Seq":"KLLKKLLKKLLKKLLKLL",        "MIC":5.04},
    2:  {"Motif_1":"KLLKKLLK",  "Motif_2":"LGLLGKLL",   "Seq":"KLLKKLLKLGLLGKLL",           "MIC":5.23},
    3:  {"Motif_1":"KLLKKLLKLL","Motif_2":"LGLLGKLL",   "Seq":"KLLKKLLKLLLGLLGKLL",         "MIC":5.26},
    4:  {"Motif_1":"KLLKKLLKLL","Motif_2":"ARLLRRLAR",  "Seq":"KLLKKLLKLLARLLRRLAR",        "MIC":5.42},
    5:  {"Motif_1":"KLLKKLLKLL","Motif_2":"KLLKKLLKLL", "Seq":"KLLKKLLKLLKLLKKLLKLL",       "MIC":6.18},
    6:  {"Motif_1":"ARLLRRLAR", "Motif_2":"LKLLLKL",    "Seq":"ARLLRRLARLKLLLKL",           "MIC":6.20},
    7:  {"Motif_1":"KLLKKLLKLL","Motif_2":"KIRVRL",     "Seq":"KLLKKLLKLLKIRVRL",           "MIC":6.26},
    8:  {"Motif_1":"KLLKKLLKLL","Motif_2":"LKLLLKL",    "Seq":"KLLKKLLKLLLKLLLKL",          "MIC":6.27},
    9:  {"Motif_1":"ARLLRRLAR", "Motif_2":"KLLKKLLKLL", "Seq":"ARLLRRLARKLLKKLLKLL",        "MIC":6.53},
    10: {"Motif_1":"RIRVAVIRA", "Motif_2":"LKLLLKL",    "Seq":"RIRVAVIRALKLLLKL",           "MIC":6.56},
}

CONDITIONS = {
    "nolinker": ("single_motif", "single_motif{i}"),
    "aaa":      ("comb_aaa",     "comb_aaa{i}_*"),
    "ggg":      ("comb_ggg",     "comb_ggg{i}_*"),
}

AA3 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
       "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
       "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}

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
    resnums = sorted(residues)
    return np.array([residues[r] for r in resnums])

def read_pdb_sequence(pdb_path):
    residues = {}
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                res_num = int(line[22:26].strip())
                res_name = line[17:20].strip()
                residues[res_num] = AA3.get(res_name, "X")
    if not residues:
        return ""
    return "".join(residues[r] for r in sorted(residues))

def assign_ss_simple(pdb_path):
    helix_res = set()
    sheet_res = set()
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("HELIX"):
                try:
                    start = int(line[21:25].strip())
                    end   = int(line[33:37].strip())
                    helix_res.update(range(start, end+1))
                except: pass
            elif line.startswith("SHEET"):
                try:
                    start = int(line[22:26].strip())
                    end   = int(line[33:37].strip())
                    sheet_res.update(range(start, end+1))
                except: pass
    return helix_res, sheet_res

def read_scores_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    plddt = data.get("plddt", [])
    ptm   = data.get("ptm", None)
    return np.array(plddt), ptm

def find_rank1_files(folder):
    pdb_files = sorted(folder.glob("*rank_001*.pdb"))
    jsn_files = sorted(folder.glob("*scores_rank_001*.json"))
    return (pdb_files[0] if pdb_files else None,
            jsn_files[0] if jsn_files else None)

rows = []
plddt_per_comb = {}

for cond_key, (subdir, pattern) in CONDITIONS.items():
    cond_dir = RAW_DIR / subdir
    for i in range(1, 11):
        if "{i}" in pattern:
            folder_name = pattern.replace("{i}", str(i))
        else:
            matches = list(cond_dir.glob(pattern.replace("{i}", str(i))))
            folder_name = matches[0].name if matches else None

        if folder_name is None:
            continue
        folder = cond_dir / folder_name
        if not folder.exists():
            candidates = list(cond_dir.glob(f"*{cond_key.replace('nolinker','')}{i}_*"))
            if candidates:
                folder = candidates[0]
            else:
                print(f"  NOT FOUND: {cond_key} {i}")
                continue

        if not list(folder.glob("*.pdb")):
            nested = folder / folder.name
            if nested.exists() and list(nested.glob("*.pdb")):
                folder = nested
        pdb_path, json_path = find_rank1_files(folder)
        if pdb_path is None:
            print(f"  NO PDB: {folder}")
            continue

        plddt_arr = read_pdb_bfactor(pdb_path)
        if len(plddt_arr) == 0 and json_path:
            plddt_arr, _ = read_scores_json(json_path)

        ptm = None
        if json_path:
            _, ptm = read_scores_json(json_path)

        helix_res, sheet_res = assign_ss_simple(pdb_path)
        seq_from_pdb = read_pdb_sequence(pdb_path)
        n_res = len(plddt_arr) if len(plddt_arr) > 0 else len(seq_from_pdb)

        plddt_mean = float(np.mean(plddt_arr)) if len(plddt_arr) > 0 else np.nan
        plddt_high = float(np.mean(plddt_arr > 70)) * 100 if len(plddt_arr) > 0 else np.nan

        # helix content from SS records or from pLDDT heuristic
        helix_pct = 100 * len(helix_res) / n_res if n_res > 0 and helix_res else np.nan

        # check if helix is continuous (no gap > 2 residues)
        helix_continuous = False
        if helix_res and n_res > 0:
            h_sorted = sorted(helix_res)
            gaps = [h_sorted[j+1]-h_sorted[j] for j in range(len(h_sorted)-1)]
            max_gap = max(gaps) if gaps else 0
            helix_continuous = (max_gap <= 2 and
                                 len(helix_res) >= 0.5 * n_res)

        m = MAPPING[i]
        label = f"comb{i:02d}_{m['Motif_1']}+{m['Motif_2']}"

        rows.append({
            "ID": i,
            "Condition": cond_key,
            "Label": label,
            "Motif_1": m["Motif_1"],
            "Motif_2": m["Motif_2"],
            "Sequence": m["Seq"],
            "n_residues": n_res,
            "MIC_predicted": m["MIC"],
            "pLDDT_mean": round(plddt_mean, 1),
            "pLDDT_pct_above70": round(plddt_high, 1),
            "ptm": round(float(ptm), 3) if ptm is not None else np.nan,
            "helix_pct": round(helix_pct, 1) if not np.isnan(helix_pct) else np.nan,
            "helix_continuous": helix_continuous,
            "pdb_path": str(pdb_path),
        })

        if cond_key == "nolinker":
            plddt_per_comb[i] = plddt_arr

        print(f"  {cond_key} {i:2d}: pLDDT={plddt_mean:.1f}  "
              f"helix={helix_pct:.0f}%  "
              f"seq={seq_from_pdb[:20]}...")

df = pd.DataFrame(rows)
print(f"\nLoaded {len(df)} structures")
print(df[df["Condition"]=="nolinker"][[
    "ID","Label","pLDDT_mean","pLDDT_pct_above70","helix_pct","helix_continuous"
]].to_string(index=False))

# ── Fig 1: pLDDT mean per combination × condition ──
fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
pal = {"nolinker":"#888888","aaa":"#E07B54","ggg":"#4C72B0"}
x   = np.arange(10)
w   = 0.25
for idx, cond in enumerate(["nolinker","aaa","ggg"]):
    sub = df[df["Condition"]==cond].sort_values("ID")
    vals = sub["pLDDT_mean"].values if len(sub)==10 else [np.nan]*10
    ax.bar(x + idx*w, vals, width=w, color=pal[cond],
           alpha=0.85, edgecolor="white", label=cond)
ax.axhline(70, color="red", lw=1, ls="--", alpha=0.6, label="pLDDT=70 (threshold)")
ax.axhline(50, color="orange", lw=1, ls="--", alpha=0.4, label="pLDDT=50")
ax.set_xticks(x + w)
labels_x = [f"c{i}\n{MAPPING[i]['Motif_1'][:6]}+\n{MAPPING[i]['Motif_2'][:6]}"
             for i in range(1,11)]
ax.set_xticklabels(labels_x, fontsize=7.5)
ax.set_ylabel("Mean pLDDT", fontsize=10)
ax.set_title("ColabFold — Mean pLDDT per combination × linker condition\n"
             "(>70 = confident, <50 = disordered)", fontweight="bold")
ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)
ax.set_ylim(0, 105)
fig.tight_layout()
fig.savefig(FIG_DIR/"01_plddt_mean_comparison.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("  Fig 01: plddt_mean_comparison")

# ── Fig 2: pLDDT per posició (no linker, tots 10) ──
n_with_data = sum(1 for v in plddt_per_comb.values() if len(v) > 0)
if n_with_data > 0:
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharey=True)
    fig.patch.set_facecolor("#fafafa")
    axes = axes.flatten()
    for idx, i in enumerate(range(1, 11)):
        ax = axes[idx]; ax.set_facecolor("#fafafa")
        arr = plddt_per_comb.get(i, np.array([]))
        if len(arr) == 0:
            ax.set_title(f"comb{i:02d}\n(no data)", fontsize=8)
            continue
        colors = ["#3BAF77" if v >= 70 else "#E07B54" if v >= 50 else "#C0392B"
                  for v in arr]
        ax.bar(range(len(arr)), arr, color=colors, alpha=0.85, edgecolor="none")
        ax.axhline(70, color="red", lw=0.8, ls="--", alpha=0.5)
        ax.axhline(50, color="orange", lw=0.8, ls="--", alpha=0.4)
        m = MAPPING[i]
        ax.set_title(f"c{i}: {m['Motif_1'][:8]}+{m['Motif_2'][:8]}\n"
                     f"MIC={m['MIC']:.1f}  pLDDT={np.mean(arr):.0f}",
                     fontsize=7.5, fontweight="bold")
        ax.set_xlabel("Residue", fontsize=7)
        ax.set_ylabel("pLDDT" if idx % 5 == 0 else "", fontsize=7)
        ax.set_ylim(0, 105); ax.spines[["top","right"]].set_visible(False)
    patches = [mpatches.Patch(color="#3BAF77",label="≥70 (confident)"),
               mpatches.Patch(color="#E07B54",label="50-70 (low)"),
               mpatches.Patch(color="#C0392B",label="<50 (disordered)")]
    fig.legend(handles=patches, fontsize=9, loc="lower center",
               ncol=3, bbox_to_anchor=(0.5,-0.02))
    fig.suptitle("pLDDT per residue — top 10 combinations (no linker)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR/"02_plddt_per_residue.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  Fig 02: plddt_per_residue")

# ── Fig 3: pLDDT mean vs MIC_predicted ──
nl = df[df["Condition"]=="nolinker"].dropna(subset=["pLDDT_mean","MIC_predicted"])
if len(nl) >= 3:
    from scipy.stats import spearmanr
    rho, p = spearmanr(nl["pLDDT_mean"], nl["MIC_predicted"])
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
    ax.scatter(nl["MIC_predicted"], nl["pLDDT_mean"],
               c="#4C72B0", s=80, alpha=0.8, edgecolors="white")
    for _, row in nl.iterrows():
        ax.annotate(f"c{int(row['ID'])}", (row["MIC_predicted"], row["pLDDT_mean"]),
                    fontsize=8, xytext=(3,3), textcoords="offset points")
    ax.annotate(f"ρ={rho:.3f}  p={p:.3f}",
                xy=(0.05,0.92), xycoords="axes fraction", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3",fc="white",ec="#ddd"))
    ax.set_xlabel("MIC predicted (µmol/L)", fontsize=10)
    ax.set_ylabel("Mean pLDDT", fontsize=10)
    ax.set_title("pLDDT vs predicted MIC\n(does lower MIC = better structure?)",
                 fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR/"03_plddt_vs_mic.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Fig 03: plddt_vs_mic  ρ={rho:.3f} p={p:.3f}")

# ── Fig 4: helix content comparison (if available) ──
has_helix = df["helix_pct"].notna().sum() > 0
if has_helix:
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
    for idx, cond in enumerate(["nolinker","aaa","ggg"]):
        sub = df[df["Condition"]==cond].sort_values("ID")
        vals = sub["helix_pct"].fillna(0).values
        ax.bar(x + idx*w, vals, width=w, color=pal[cond],
               alpha=0.85, edgecolor="white", label=cond)
    ax.set_xticks(x + w)
    ax.set_xticklabels(labels_x, fontsize=7.5)
    ax.set_ylabel("Helix content (%)", fontsize=10)
    ax.set_title("Alpha-helix content per combination × linker condition",
                 fontweight="bold")
    ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR/"04_helix_content.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  Fig 04: helix_content")

# ── Fig 5: Summary heatmap pLDDT ──
pivot = df.pivot_table(index="ID", columns="Condition",
                        values="pLDDT_mean", aggfunc="mean")
if not pivot.empty:
    fig, ax = plt.subplots(figsize=(6, 7))
    fig.patch.set_facecolor("#fafafa")
    ytick_labels = [f"c{i}: {MAPPING[i]['Motif_1'][:7]}+{MAPPING[i]['Motif_2'][:7]}"
                    for i in range(1,11) if i in pivot.index]
    sns.heatmap(pivot.loc[[i for i in range(1,11) if i in pivot.index]],
                annot=True, fmt=".0f", cmap="RdYlGn",
                vmin=30, vmax=90, linewidths=0.5, ax=ax,
                cbar_kws={"label":"Mean pLDDT"},
                yticklabels=ytick_labels)
    ax.set_title("pLDDT summary — combinations × conditions",
                 fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIG_DIR/"05_plddt_heatmap.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  Fig 05: plddt_heatmap")

# ── Save Excel ──
out_xl = OUT_DIR / "colabfold_analysis.xlsx"
with pd.ExcelWriter(out_xl, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="All_structures", index=False)
    df[df["Condition"]=="nolinker"].to_excel(
        writer, sheet_name="NoLinker", index=False)
    df[df["Condition"]=="aaa"].to_excel(
        writer, sheet_name="AAA", index=False)
    df[df["Condition"]=="ggg"].to_excel(
        writer, sheet_name="GGG", index=False)
    pivot.to_excel(writer, sheet_name="pLDDT_pivot")
    pd.DataFrame([{
        "n_structures": len(df),
        "mean_pLDDT_nolinker": df[df["Condition"]=="nolinker"]["pLDDT_mean"].mean(),
        "mean_pLDDT_aaa": df[df["Condition"]=="aaa"]["pLDDT_mean"].mean(),
        "mean_pLDDT_ggg": df[df["Condition"]=="ggg"]["pLDDT_mean"].mean(),
        "pct_above70_nolinker": df[df["Condition"]=="nolinker"]["pLDDT_pct_above70"].mean(),
    }]).to_excel(writer, sheet_name="Summary", index=False)

print(f"\n✅ Excel: {out_xl}")
print(f"✅ Figures: {FIG_DIR}")
print(f"\n=== FINAL SUMMARY ===")
for cond in ["nolinker","aaa","ggg"]:
    sub = df[df["Condition"]==cond]
    print(f"  {cond:<10} mean_pLDDT={sub['pLDDT_mean'].mean():.1f}  "
          f"n_above70={( sub['pLDDT_mean']>70).sum()}/10")