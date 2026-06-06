from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import sys

# =========================
# CONFIG
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "results/14_final_candidates"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

DPI = 300
sns.set_theme(style="whitegrid", font_scale=1.1)

CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")

# tee log
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files: f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()

_log = open(OUT_DIR / "final_candidates_log.txt", "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, _log)
print(f"Run: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

PALETTE = {"Carpet":"#E07B54", "Pore":"#3BAF77",
           "No_linker":"#888888", "GGG":"#4C72B0", "AAA":"#E07B54"}
PAL_PIPELINE = {
    "cdhit_meme":    "#4C72B0",
    "cdhit_streme":  "#E07B54",
    "mmseqs_meme":   "#3BAF77",
    "mmseqs_streme": "#9B59B6",
}
PAL_FAM = {}  # filled dynamically

# =========================
# LOAD ALL SOURCES
# =========================

motif_mic = pd.read_csv(
    PROJECT_ROOT / "results/12_apex_prediction/02_results/02_motif_mic_summary_merged.csv"
)
comb_2cat = pd.read_csv(
    PROJECT_ROOT / "results/13_motif_combinations/01_combinations/04_merged_predicted_MICs.csv"
)
comb_3cat = pd.read_csv(
    PROJECT_ROOT / "results/13_motif_combinations/01b_combinations_3cat/04_merged_predicted_MICs.csv"
)
linker_2cat = pd.read_excel(
    PROJECT_ROOT / "results/13_motif_combinations/04_linker_analysis_2cat"
    / "01_linker_analysis_2cat.xlsx",
    sheet_name="All_data"
)
order_pattern = pd.read_excel(
    PROJECT_ROOT / "results/13_motif_combinations/02_analysis_2cat/01_analysis_2cat.xlsx",
    sheet_name="Order_pattern"
)
sig_order = order_pattern[order_pattern["p_mw"] < 0.05]
motif_profile_2cat = pd.read_excel(
    PROJECT_ROOT / "results/13_motif_combinations/02_analysis_2cat/01_analysis_2cat.xlsx",
    sheet_name="Motif_profile"
)
score_linker = pd.read_excel(
    PROJECT_ROOT / "results/13_motif_combinations/04_linker_analysis_2cat"
    / "01_linker_analysis_2cat.xlsx",
    sheet_name="Score_per_condition"
)

# pipeline origin data
fimo_master = pd.read_excel(
    PROJECT_ROOT / "results/06_motif_statistics/02_fimo_reporting"
    / "fimo_reporting_master.xlsx",
    sheet_name="Significant"
)
fimo_master["Motif"] = fimo_master["Motif"].astype(str).str.strip().str.upper()
fimo_master["Pipeline"] = (
    fimo_master["Clustering"].str.strip().str.lower() + "_" +
    fimo_master["Tool"].str.strip().str.lower()
)

print(f"Loaded {len(motif_mic)} motifs, {len(comb_2cat)} 2cat combos, "
      f"{len(comb_3cat)} 3cat combos")
print(f"FIMO master: {len(fimo_master)} rows, "
      f"{fimo_master['Motif'].nunique()} unique motifs")

# =========================
# BUILD MASTER TABLE (same as v1)
# =========================

print(f"\n{'='*60}")
print("BUILDING MASTER CANDIDATE TABLE")
print(f"{'='*60}")

master_rows = []
for _, row in motif_mic.iterrows():
    motif    = row["Motif_representative"]
    mic_ind  = row.get("MIC_mean", np.nan)
    mech_2cat= row.get("Dominant_reduced", "")
    mech_3cat= row.get("Dominant_mechanism", mech_2cat)
    family   = row.get("Family_ID", "")

    mask_2cat = (comb_2cat["Motif_1"]==motif) | (comb_2cat["Motif_2"]==motif)
    comb_sub  = comb_2cat[mask_2cat]["MIC_combined"].dropna()
    best_comb_mic = comb_sub.min() if len(comb_sub) > 0 else np.nan
    best_comb_row = comb_2cat[mask_2cat & (comb_2cat["MIC_combined"]==best_comb_mic)]
    best_partner = ""; best_comb_delta = np.nan; best_comb_seq = ""
    if len(best_comb_row) > 0:
        r = best_comb_row.iloc[0]
        best_partner    = r["Motif_2"] if r["Motif_1"]==motif else r["Motif_1"]
        best_comb_delta = r.get("MIC_delta", np.nan)
        best_comb_seq   = r.get("Sequence", "")

    prof_row     = motif_profile_2cat[motif_profile_2cat["Motif"]==motif]
    score_nolink = prof_row["Score_combined"].values[0] if len(prof_row)>0 else np.nan
    cv_nolink    = prof_row["CV"].values[0]             if len(prof_row)>0 else np.nan
    mic_mean_comb= prof_row["MIC_mean_in_comb"].values[0] if len(prof_row)>0 else np.nan

    sc_row    = score_linker[score_linker["Motif"]==motif] \
                if "Motif" in score_linker.columns else pd.DataFrame()
    score_ggg = sc_row["GGG"].values[0] if len(sc_row)>0 and "GGG" in sc_row.columns else np.nan
    score_aaa = sc_row["AAA"].values[0] if len(sc_row)>0 and "AAA" in sc_row.columns else np.nan
    scores    = {"No_linker": score_nolink, "GGG": score_ggg, "AAA": score_aaa}
    valid_scores = {k:v for k,v in scores.items() if not pd.isna(v)}
    best_linker  = min(valid_scores, key=valid_scores.get) if valid_scores else ""

    ord_row    = sig_order[sig_order["Motif"]==motif]
    order_pref = ord_row["Prefers"].values[0] if len(ord_row)>0 else "No_preference"
    order_delta= ord_row["Delta_pos1_minus_pos2"].values[0] if len(ord_row)>0 else np.nan

    master_rows.append({
        "Motif":               motif,
        "Mechanism_2cat":      mech_2cat,
        "Mechanism_3cat":      mech_3cat,
        "Family_ID":           family,
        "MIC_individual":      round(float(mic_ind),2) if not pd.isna(mic_ind) else np.nan,
        "MIC_mean_in_comb":    round(float(mic_mean_comb),2) if not pd.isna(mic_mean_comb) else np.nan,
        "CV_in_comb":          round(float(cv_nolink),1) if not pd.isna(cv_nolink) else np.nan,
        "Score_nolinker":      round(float(score_nolink),2) if not pd.isna(score_nolink) else np.nan,
        "Score_GGG":           round(float(score_ggg),2) if not pd.isna(score_ggg) else np.nan,
        "Score_AAA":           round(float(score_aaa),2) if not pd.isna(score_aaa) else np.nan,
        "Best_linker":         best_linker,
        "Best_partner_motif":  best_partner,
        "Best_combo_MIC":      round(float(best_comb_mic),2) if not pd.isna(best_comb_mic) else np.nan,
        "Best_combo_delta":    round(float(best_comb_delta),2) if not pd.isna(best_comb_delta) else np.nan,
        "Best_combo_sequence": best_comb_seq,
        "Order_preference":    order_pref,
        "Order_delta":         round(float(order_delta),2) if not pd.isna(order_delta) else np.nan,
    })

master_df = pd.DataFrame(master_rows)
for col in ["MIC_individual","Score_nolinker","Best_combo_MIC"]:
    master_df[f"Rank_{col}"] = master_df[col].rank(ascending=True, na_option="bottom")
rank_cols = [c for c in master_df.columns if c.startswith("Rank_")]
master_df["Composite_rank"] = master_df[rank_cols].mean(axis=1).round(2)
master_df = master_df.sort_values("Composite_rank")

print(f"\nTop 20 candidates (composite rank):")
print(master_df[["Motif","Mechanism_2cat","Family_ID","MIC_individual",
                  "Score_nolinker","Best_combo_MIC","Best_linker",
                  "Composite_rank"]].head(20).to_string(index=False))

# =========================
# VALIDATION CROSS-CHECK (same as v1)
# =========================

print(f"\n{'='*60}")
print("VALIDATION CROSS-CHECK")
print(f"{'='*60}")

top20_score = set(master_df.nsmallest(20,"Score_nolinker")["Motif"])
top20_combo = set(master_df.nsmallest(20,"Best_combo_MIC")["Motif"])
top20_indiv = set(master_df.nsmallest(20,"MIC_individual")["Motif"])

overlap_sc = len(top20_score & top20_combo)
overlap_si = len(top20_score & top20_indiv)
overlap_ir = len(top20_indiv & set(master_df.nsmallest(20,"Composite_rank")["Motif"]))
triple     = top20_score & top20_combo & top20_indiv

print(f"  Top20 score ∩ top20 best_combo:    {overlap_sc}/20")
print(f"  Top20 score ∩ top20 individual:    {overlap_si}/20")
print(f"  Top20 individual ∩ composite rank: {overlap_ir}/20")
print(f"\n  Motifs in ALL 3 top20 lists:")
for m in sorted(triple):
    r = master_df[master_df["Motif"]==m].iloc[0]
    print(f"    {m:<25} MIC_ind={r['MIC_individual']:.1f}  "
          f"Score={r['Score_nolinker']:.1f}  BestCombo={r['Best_combo_MIC']:.1f}")

valid = master_df.dropna(subset=["Score_nolinker","Best_combo_MIC","MIC_individual"])
rho_sc, _ = spearmanr(valid["Score_nolinker"], valid["Best_combo_MIC"])
rho_si, _ = spearmanr(valid["Score_nolinker"], valid["MIC_individual"])
rho_ic, _ = spearmanr(valid["MIC_individual"], valid["Best_combo_MIC"])
print(f"\n  Spearman correlations:")
print(f"    Score vs BestCombo:  ρ={rho_sc:.3f}")
print(f"    Score vs Individual: ρ={rho_si:.3f}")
print(f"    Individual vs Combo: ρ={rho_ic:.3f}")

cross_check_df = pd.DataFrame([
    {"Comparison":"Score vs BestCombo",  "Spearman_rho": round(rho_sc,3)},
    {"Comparison":"Score vs Individual", "Spearman_rho": round(rho_si,3)},
    {"Comparison":"Individual vs Combo", "Spearman_rho": round(rho_ic,3)},
])

# =========================
# D. PIPELINE ORIGIN ANALYSIS
# =========================

print(f"\n{'='*60}")
print("D. PIPELINE ORIGIN ANALYSIS")
print(f"{'='*60}")

# map each motif to the pipelines that discovered it
motif_pipelines = (
    fimo_master.groupby("Motif")["Pipeline"]
    .apply(lambda x: sorted(set(x)))
    .reset_index()
    .rename(columns={"Pipeline": "Pipelines_origin"})
)
motif_pipelines["n_pipelines"] = motif_pipelines["Pipelines_origin"].apply(len)
motif_pipelines["Pipelines_str"] = motif_pipelines["Pipelines_origin"].apply(
    lambda x: ";".join(x))

# individual pipeline flags
for pipe in PAL_PIPELINE:
    motif_pipelines[f"in_{pipe}"] = motif_pipelines["Pipelines_origin"].apply(
        lambda x: pipe in x)

# merge into master
master_df = master_df.merge(
    motif_pipelines[["Motif","n_pipelines","Pipelines_str"] +
                     [f"in_{p}" for p in PAL_PIPELINE]],
    on="Motif", how="left"
)

# top 30 pipeline origin
top30 = master_df.head(30).copy()
print(f"\n  Pipeline origin of top 30 motifs (composite rank):")
print(top30[["Motif","n_pipelines","Pipelines_str",
             "MIC_individual","Composite_rank"]].to_string(index=False))

# pipeline distribution: full dataset vs top 30
print(f"\n  Pipeline frequency — full dataset vs top30:")
pipe_rows = []
for pipe in PAL_PIPELINE:
    col  = f"in_{pipe}"
    n_all  = master_df[col].sum()   if col in master_df.columns else 0
    n_top  = top30[col].sum()       if col in top30.columns else 0
    pct_all= 100*n_all/len(master_df)
    pct_top= 100*n_top/len(top30)
    print(f"    {pipe:<20} all={n_all:3d} ({pct_all:.0f}%)  "
          f"top30={n_top:3d} ({pct_top:.0f}%)")
    pipe_rows.append({"Pipeline": pipe,
                       "n_all": int(n_all), "pct_all": round(pct_all,1),
                       "n_top30": int(n_top), "pct_top30": round(pct_top,1)})
pipeline_summary = pd.DataFrame(pipe_rows)

# robustness in top motifs
print(f"\n  n_pipelines distribution in top 30:")
print(top30["n_pipelines"].value_counts().sort_index().to_string())

# Fig D1: Pipeline origin heatmap — top 30 motifs
fig, ax = plt.subplots(figsize=(8, max(6, len(top30)*0.35)))
fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
heat_data = top30[[f"in_{p}" for p in PAL_PIPELINE]].astype(int)
heat_data.columns = [p.replace("_"," ").title() for p in PAL_PIPELINE]
heat_data.index = top30["Motif"].values
sns.heatmap(heat_data, cmap="Blues", linewidths=0.5, ax=ax,
            cbar=False, annot=True, fmt="d",
            annot_kws={"size":9, "weight":"bold"})
ax.set_title("D1. Pipeline origin — top 30 motifs\n(1 = discovered by pipeline)",
             fontweight="bold")
ax.set_xlabel(""); ax.set_ylabel("")
ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right", fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR/"D1_pipeline_origin_heatmap.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("\n  Fig D1: D1_pipeline_origin_heatmap.png")

# Fig D2: Grouped bar — pipeline frequency all vs top30
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
x = np.arange(len(pipeline_summary))
w = 0.35
ax.bar(x - w/2, pipeline_summary["pct_all"],   width=w,
       color="#AAAAAA", alpha=0.85, edgecolor="white", label="All motifs")
ax.bar(x + w/2, pipeline_summary["pct_top30"], width=w,
       color="#4C72B0", alpha=0.85, edgecolor="white", label="Top 30 (composite rank)")
ax.set_xticks(x)
ax.set_xticklabels([p.replace("_","\n") for p in pipeline_summary["Pipeline"]],
                    fontsize=9)
ax.set_ylabel("% motifs discovered by pipeline", fontsize=10)
ax.set_title("D2. Pipeline origin — all motifs vs top 30",
             fontweight="bold")
ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR/"D2_pipeline_frequency.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("  Fig D2: D2_pipeline_frequency.png")

# Fig D3: Scatter MIC_individual vs n_pipelines
fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
for mech, color in [("Carpet","#E07B54"),("Pore","#3BAF77")]:
    sub = master_df[master_df["Mechanism_2cat"]==mech].dropna(
        subset=["n_pipelines","MIC_individual"])
    ax.scatter(sub["n_pipelines"] + np.random.default_rng(42).uniform(
               -0.1,0.1,len(sub)), sub["MIC_individual"],
               c=color, alpha=0.7, s=50, edgecolors="white",
               linewidths=0.4, label=mech)
rho_p, p_p = spearmanr(
    pd.to_numeric(master_df.dropna(subset=["n_pipelines","MIC_individual"])["n_pipelines"], errors="coerce"),
    pd.to_numeric(master_df.dropna(subset=["n_pipelines","MIC_individual"])["MIC_individual"], errors="coerce")
)
ax.annotate(f"ρ={rho_p:.3f}  p={p_p:.4f}",
            xy=(0.05,0.92), xycoords="axes fraction", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3",fc="white",ec="#ddd"))
ax.set_xlabel("Number of pipelines discovering the motif", fontsize=10)
ax.set_ylabel("MIC individual (µmol/L)", fontsize=10)
ax.set_title("D3. Robustness (n pipelines) vs activity\n"
             "Does more pipelines = more active?", fontweight="bold")
ax.set_xticks([1,2,3,4])
ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR/"D3_npipelines_vs_mic.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("  Fig D3: D3_npipelines_vs_mic.png")
print(f"  Spearman n_pipelines vs MIC: ρ={rho_p:.3f}  p={p_p:.4f}")

# =========================
# E. STRUCTURAL DIVERSITY — HIERARCHICAL CLUSTERING
# =========================

print(f"\n{'='*60}")
print("E. STRUCTURAL DIVERSITY — HIERARCHICAL CLUSTERING")
print(f"{'='*60}")

# ── clean sequences (remove non-canonical AA) ──
def clean_seq(s):
    return "".join(aa for aa in str(s).upper() if aa in CANONICAL_AA)

top_n_cluster = min(50, len(master_df))
top_df = master_df.head(top_n_cluster).copy()
top_df["Seq_clean"] = top_df["Motif"].apply(clean_seq)
top_df = top_df[top_df["Seq_clean"].str.len() >= 4].copy()
print(f"\n  Sequences for clustering: {len(top_df)}")

# ── pairwise distance matrix (normalized edit distance) ──
def norm_edit_distance(s1, s2):
    """Normalized Levenshtein distance [0,1]."""
    m, n = len(s1), len(s2)
    if m == 0 and n == 0: return 0.0
    dp = np.zeros((m+1, n+1), dtype=int)
    for i in range(m+1): dp[i,0] = i
    for j in range(n+1): dp[0,j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if s1[i-1]==s2[j-1] else 1
            dp[i,j] = min(dp[i-1,j]+1, dp[i,j-1]+1, dp[i-1,j-1]+cost)
    return dp[m,n] / max(m, n)

seqs  = top_df["Seq_clean"].tolist()
names = top_df["Motif"].tolist()
n_seq = len(seqs)

print(f"  Computing {n_seq}×{n_seq} distance matrix...")
dist_mat = np.zeros((n_seq, n_seq))
for i in range(n_seq):
    for j in range(i+1, n_seq):
        d = norm_edit_distance(seqs[i], seqs[j])
        dist_mat[i,j] = d
        dist_mat[j,i] = d

dist_condensed = squareform(dist_mat)

# ── hierarchical clustering (Ward) ──
Z = linkage(dist_condensed, method="ward")

# choose n_clusters to get 10 diverse candidates
N_CLUSTERS = 10
clusters = fcluster(Z, N_CLUSTERS, criterion="maxclust")
top_df["Cluster"] = clusters

print(f"\n  Cluster distribution (n_clusters={N_CLUSTERS}):")
for c in sorted(top_df["Cluster"].unique()):
    members = top_df[top_df["Cluster"]==c]["Motif"].tolist()
    best    = top_df[top_df["Cluster"]==c].nsmallest(1,"Composite_rank").iloc[0]
    print(f"    Cluster {c:2d} (n={len(members):2d}): "
          f"best={best['Motif']:<25} MIC={best['MIC_individual']:.1f}  "
          f"rank={best['Composite_rank']:.1f}")

# ── select representative per cluster ──
cluster_reps = (
    top_df.groupby("Cluster", group_keys=False)
    .apply(lambda g: g.nsmallest(1, "Composite_rank"))
    .reset_index(drop=True)
    .sort_values("Composite_rank")
)
# ensure Cluster column is present after groupby reset
if "Cluster" not in cluster_reps.columns:
    cluster_reps = cluster_reps.copy()
    cluster_reps["Cluster"] = cluster_reps.index + 1
print(f"\n  === FINAL DIVERSE CANDIDATES (1 per cluster) ===")
print(cluster_reps[["Motif","Mechanism_2cat","Family_ID","MIC_individual",
                      "Score_nolinker","Best_combo_MIC","n_pipelines",
                      "Cluster","Composite_rank"]].to_string(index=False))

# ── Fig E1: Dendrogram ──
fig, ax = plt.subplots(figsize=(14, max(8, n_seq*0.28)))
fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")

# color map for families
fam_list  = top_df["Family_ID"].fillna("Unknown").unique()
import matplotlib as _mpl
cmap_fam  = _mpl.colormaps["tab20"].resampled(len(fam_list))
fam_color = {f: mcolors.to_hex(cmap_fam(i)) for i,f in enumerate(fam_list)}
leaf_colors = {
    i: fam_color.get(top_df.iloc[i]["Family_ID"], "#AAAAAA")
    for i in range(len(top_df))
}

def leaf_color_func(leaf_id):
    return leaf_colors.get(leaf_id, "#AAAAAA")

dend = dendrogram(
    Z,
    labels=names,
    orientation="left",
    ax=ax,
    leaf_font_size=8,
    color_threshold=Z[-N_CLUSTERS+1,2] if N_CLUSTERS > 1 else 0,
    above_threshold_color="#AAAAAA",
    leaf_rotation=0,
)
ax.set_xlabel("Ward distance", fontsize=10)
ax.set_title(f"E1. Hierarchical clustering — top {n_seq} motifs\n"
             f"(normalized edit distance, Ward linkage)",
             fontweight="bold")

# mark cluster representatives
rep_names = set(cluster_reps["Motif"])
for label in ax.get_yticklabels():
    if label.get_text() in rep_names:
        label.set_fontweight("bold")
        label.set_color("#C0392B")
        label.set_fontsize(9)

# family legend
patches_fam = [mpatches.Patch(color=c, label=f, alpha=0.8)
               for f, c in fam_color.items() if f in top_df["Family_ID"].values]
ax.legend(handles=patches_fam[:12], title="Family",
          fontsize=7, title_fontsize=8,
          loc="lower right", bbox_to_anchor=(1.0, 0.0))
ax.annotate("● bold red = cluster representative",
            xy=(0.02, 0.02), xycoords="axes fraction",
            fontsize=8, color="#C0392B")
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR/"E1_dendrogram_diversity.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("\n  Fig E1: E1_dendrogram_diversity.png")

# ── Fig E2: Distance heatmap ──
fig, ax = plt.subplots(figsize=(max(8, n_seq*0.28), max(7, n_seq*0.28)))
fig.patch.set_facecolor("#fafafa")
dist_df = pd.DataFrame(dist_mat, index=names, columns=names)
# reorder by dendrogram leaf order
leaf_order = dend["leaves"]
dist_reord = dist_df.iloc[leaf_order, leaf_order]
sns.heatmap(dist_reord, cmap="YlOrRd", linewidths=0.2, ax=ax,
            cbar_kws={"label":"Normalized edit distance","shrink":0.5},
            xticklabels=True, yticklabels=True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=6)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=6)
ax.set_title(f"E2. Pairwise distance matrix — top {n_seq} motifs\n"
             "(reordered by dendrogram)", fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR/"E2_distance_heatmap.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("  Fig E2: E2_distance_heatmap.png")

# ── Fig E3: Summary — diverse candidates ──
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.patch.set_facecolor("#fafafa")

# left: MIC individual per cluster rep
ax = axes[0]; ax.set_facecolor("#fafafa")
colors_rep = ["#E07B54" if m=="Carpet" else "#3BAF77"
              for m in cluster_reps["Mechanism_2cat"]]
bars = ax.barh(range(len(cluster_reps)), cluster_reps["MIC_individual"],
               color=colors_rep, alpha=0.85, edgecolor="white")
ax.set_yticks(range(len(cluster_reps)))
ax.set_yticklabels([f"C{r['Cluster']}: {r['Motif']}"
                    for _, r in cluster_reps.iterrows()], fontsize=9)
for bar, row in zip(bars, cluster_reps.itertuples()):
    ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
            f"combo={row.Best_combo_MIC:.1f}",
            va="center", fontsize=7.5, color="#555")
ax.set_xlabel("MIC individual (µmol/L)", fontsize=10)
ax.set_title("E3a. Diverse candidates (1 per cluster)\nMIC individual",
             fontweight="bold")
patches_mech = [mpatches.Patch(color="#E07B54",label="Carpet"),
                mpatches.Patch(color="#3BAF77",label="Pore")]
ax.legend(handles=patches_mech, fontsize=9)
ax.spines[["top","right"]].set_visible(False)

# right: scatter MIC individual vs Best_combo_MIC all top_df
ax = axes[1]; ax.set_facecolor("#fafafa")
for mech, color in [("Carpet","#E07B54"),("Pore","#3BAF77")]:
    sub = top_df[top_df["Mechanism_2cat"]==mech].dropna(
        subset=["MIC_individual","Best_combo_MIC"])
    ax.scatter(sub["MIC_individual"], sub["Best_combo_MIC"],
               c=color, alpha=0.6, s=40, edgecolors="white",
               linewidths=0.3, label=mech)
# highlight cluster reps
for _, r in cluster_reps.iterrows():
    if pd.notna(r["MIC_individual"]) and pd.notna(r["Best_combo_MIC"]):
        ax.scatter(r["MIC_individual"], r["Best_combo_MIC"],
                   c="black", s=120, zorder=5, marker="*")
        ax.annotate(f"C{r['Cluster']}", (r["MIC_individual"], r["Best_combo_MIC"]),
                    fontsize=7, xytext=(3,3), textcoords="offset points",
                    color="#C0392B", fontweight="bold")
ax.set_xlabel("MIC individual (µmol/L)", fontsize=10)
ax.set_ylabel("Best combination MIC (µmol/L)", fontsize=10)
ax.set_title("E3b. Individual vs best combination MIC\n★ = cluster representative",
             fontweight="bold")
ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR/"E3_diverse_candidates.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("  Fig E3: E3_diverse_candidates.png")

# =========================
# SAVE EXCEL (extended)
# =========================

out_excel = OUT_DIR / "01_master_candidates.xlsx"
with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
    master_df.to_excel(writer,         sheet_name="All_motifs_ranked",   index=False)
    master_df.head(30).to_excel(writer,sheet_name="Top30_candidates",    index=False)
    master_df[master_df["Mechanism_2cat"]=="Carpet"].head(20).to_excel(
        writer, sheet_name="Top20_Carpet", index=False)
    master_df[master_df["Mechanism_2cat"]=="Pore"].head(20).to_excel(
        writer, sheet_name="Top20_Pore", index=False)
    cross_check_df.to_excel(writer,    sheet_name="Validation_crosscheck",index=False)

    # pipeline
    pipeline_summary.to_excel(writer,  sheet_name="Pipeline_summary",    index=False)
    top_df[["Motif","n_pipelines","Pipelines_str",
             "MIC_individual","Composite_rank"]].to_excel(
        writer, sheet_name="Pipeline_top30", index=False)

    # clustering
    top_df[["Motif","Cluster","Seq_clean","Mechanism_2cat","Family_ID",
             "MIC_individual","Score_nolinker","Best_combo_MIC",
             "n_pipelines","Composite_rank"]].to_excel(
        writer, sheet_name="Clustering_all",  index=False)
    cluster_reps[["Motif","Cluster","Mechanism_2cat","Family_ID",
                   "MIC_individual","Score_nolinker","Best_combo_MIC",
                   "n_pipelines","Best_linker","Order_preference",
                   "Pipelines_str","Composite_rank"]].to_excel(
        writer, sheet_name="Diverse_candidates", index=False)

    # distance matrix
    pd.DataFrame(dist_mat, index=names, columns=names).to_excel(
        writer, sheet_name="Distance_matrix")

    # GraphPad
    master_df[["Motif","Mechanism_2cat","MIC_individual",
               "Score_nolinker","Best_combo_MIC","n_pipelines",
               "Composite_rank"]].to_excel(
        writer, sheet_name="GP_master", index=False)
    cluster_reps[["Motif","Cluster","MIC_individual","Best_combo_MIC"]].to_excel(
        writer, sheet_name="GP_diverse", index=False)
    pipeline_summary.to_excel(writer,  sheet_name="GP_pipeline", index=False)

print(f"\n✅ Master table: {out_excel}")

# =========================
# SUPPLEMENTARY CSV
# =========================

supp_cols = ["Motif","Mechanism_2cat","Mechanism_3cat","Family_ID",
             "MIC_individual","MIC_mean_in_comb","CV_in_comb",
             "Score_nolinker","Score_GGG","Score_AAA","Best_linker",
             "Best_partner_motif","Best_combo_MIC","Best_combo_delta",
             "Best_combo_sequence","Order_preference","n_pipelines",
             "Pipelines_str","Composite_rank"]
master_df[[c for c in supp_cols if c in master_df.columns]].to_csv(
    OUT_DIR / "02_master_candidates_supplementary.csv", index=False)
cluster_reps.to_csv(OUT_DIR / "03_diverse_candidates.csv", index=False)

print(f"✅ Supplementary CSV: 02_master_candidates_supplementary.csv")
print(f"✅ Diverse candidates: 03_diverse_candidates.csv")

# =========================
# FINAL SUMMARY
# =========================

print(f"\n{'='*60}")
print("TOP 10 FINAL CANDIDATES (composite rank)")
print(f"{'='*60}")
print(master_df[["Motif","Mechanism_2cat","Family_ID","MIC_individual",
                  "Score_nolinker","Best_combo_MIC","n_pipelines",
                  "Best_linker","Composite_rank"]].head(10).to_string(index=False))

print(f"\n{'='*60}")
print(f"DIVERSE CANDIDATES ({N_CLUSTERS} clusters)")
print(f"{'='*60}")
print(cluster_reps[["Motif","Cluster","Mechanism_2cat","Family_ID",
                     "MIC_individual","Best_combo_MIC","n_pipelines",
                     "Composite_rank"]].to_string(index=False))

print(f"\n✅ All outputs saved to: {OUT_DIR}")
sys.stdout = sys.__stdout__
_log.close()

# =========================
# E2. CONSTRAINED DIVERSITY — max 1 representative from FAM_047
# =========================

print(f"\n{'='*60}")
print("E2. CONSTRAINED DIVERSITY — max 1 representative per family")
print(f"{'='*60}")

TARGET_N = 10  # desired number of final candidates

def select_constrained_family(df_in, Z_mat, n_clust):
    """
    Select 1 representative per cluster AND max 1 per family.
    Processes clusters in order of their best motif's global rank,
    so the best motif overall always gets priority.
    """
    clust = fcluster(Z_mat, n_clust, criterion="maxclust")
    df    = df_in.copy()
    df["Cluster_c"] = clust
    df_sorted = df.sort_values("Composite_rank")

    # order clusters by the rank of their best motif
    cluster_order = (
        df_sorted.groupby("Cluster_c")["Composite_rank"]
        .min()
        .sort_values()
        .index.tolist()
    )

    selected  = []
    used_fams = set()

    for c in cluster_order:
        cluster_members = df_sorted[df_sorted["Cluster_c"] == c]
        for _, row in cluster_members.iterrows():
            fam = row["Family_ID"]
            if pd.isna(fam) or fam not in used_fams:
                selected.append(row)
                if pd.notna(fam):
                    used_fams.add(fam)
                break  # one per cluster

    result = pd.DataFrame(selected).sort_values("Composite_rank").reset_index(drop=True)
    return result

# find n_clusters that gives TARGET_N candidates with 1-per-cluster AND 1-per-family
best_reps_c = None
best_nc     = TARGET_N
for nc in range(TARGET_N, TARGET_N + 15):
    reps_c = select_constrained_family(top_df, Z, nc)
    if len(reps_c) >= TARGET_N:
        best_reps_c = reps_c.head(TARGET_N)
        best_nc     = nc
        break
if best_reps_c is None:
    best_reps_c = reps_c

print(f"\n  n_clusters used: {best_nc}  →  {len(best_reps_c)} candidates")
print(f"  (FAM_047 limited to max 1 representative)")
print(f"\n  === CONSTRAINED DIVERSE CANDIDATES ===")
print(best_reps_c[["Motif","Mechanism_2cat","Family_ID","MIC_individual",
                    "Score_nolinker","Best_combo_MIC","n_pipelines",
                    "Pipelines_str","Composite_rank"]].to_string(index=False))

# Fig E4
fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
colors_c = ["#E07B54" if m=="Carpet" else "#3BAF77"
            for m in best_reps_c["Mechanism_2cat"]]
ax.barh(range(len(best_reps_c)), best_reps_c["MIC_individual"],
        color=colors_c, alpha=0.85, edgecolor="white")
ax.set_yticks(range(len(best_reps_c)))
ax.set_yticklabels([f"{r['Motif']}  [{r['Family_ID']}]"
                    for _, r in best_reps_c.iterrows()], fontsize=9)
for i, row in enumerate(best_reps_c.itertuples()):
    ax.text(row.MIC_individual + 0.5, i,
            f"combo={row.Best_combo_MIC:.1f} | {row.Pipelines_str}",
            va="center", fontsize=7.5, color="#555")
ax.set_xlabel("MIC individual (µmol/L)", fontsize=10)
ax.set_title("E4. Constrained diverse candidates\n"
             "(max 1 per family / FAM_047 limited to 1)",
             fontweight="bold")
ax.legend(handles=[mpatches.Patch(color="#E07B54",label="Carpet"),
                   mpatches.Patch(color="#3BAF77",label="Pore")], fontsize=9)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR/"E4_constrained_diverse.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("  Fig E4: E4_constrained_diverse.png")

best_reps_c.to_csv(OUT_DIR / "04_constrained_diverse_candidates.csv", index=False)
with pd.ExcelWriter(OUT_DIR/"01_master_candidates.xlsx", engine="openpyxl",
                    mode="a", if_sheet_exists="replace") as writer:
    best_reps_c.to_excel(writer, sheet_name="Constrained_diverse", index=False)
print("  Saved: 04_constrained_diverse_candidates.csv")
print(f"\n✅ All constrained outputs saved.")