"""
Comparative analysis: no-linker vs GGG linker vs AAA linker

For each combination type (Carpet+Carpet, Carpet+Pore, Pore+Pore):
  1. Compare MIC distributions: no-linker vs GGG vs AAA
  2. Paired comparison: same motif pair across the 3 conditions
  3. Test hypothesis: GGG better for Pore (independent domains)
                      AAA better for Carpet (continuous helix)
  4. ΔMIC per condition vs mean individual
  5. Order effect per linker

Runs separately for 2cat and 3cat.
"""

from pathlib import Path
from itertools import combinations as _comb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu, wilcoxon, spearmanr, friedmanchisquare

# =========================
# CONFIG
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# no-linker (already merged)
NO_LINKER_2CAT = (PROJECT_ROOT / "results/13_motif_combinations/01_combinations"
                  / "04_merged_predicted_MICs.csv")
NO_LINKER_3CAT = (PROJECT_ROOT / "results/13_motif_combinations/01b_combinations_3cat"
                  / "04_merged_predicted_MICs.csv")

# linker predictions
LINKER_DIR = PROJECT_ROOT / "results/13_motif_combinations/03_combinations_linker"

PRED_2CAT_GGG = LINKER_DIR / "Predicted_MICs_2cat_GGG.csv"
PRED_2CAT_AAA = LINKER_DIR / "Predicted_MICs_2cat_AAA.csv"
PRED_3CAT_GGG = LINKER_DIR / "Predicted_MICs_3cat_GGG.csv"
PRED_3CAT_AAA = LINKER_DIR / "Predicted_MICs_3cat_AAA.csv"

MAPPING_2CAT_GGG = LINKER_DIR / "09_apex_mapping_2cat_GGG.csv"
MAPPING_2CAT_AAA = LINKER_DIR / "10_apex_mapping_2cat_AAA.csv"
MAPPING_3CAT_GGG = LINKER_DIR / "11_apex_mapping_3cat_GGG.csv"
MAPPING_3CAT_AAA = LINKER_DIR / "12_apex_mapping_3cat_AAA.csv"

OUT_DIR_2CAT = PROJECT_ROOT / "results/13_motif_combinations/04_linker_analysis_2cat"
OUT_DIR_3CAT = PROJECT_ROOT / "results/13_motif_combinations/04_linker_analysis_3cat"
OUT_DIR_2CAT.mkdir(parents=True, exist_ok=True)
OUT_DIR_3CAT.mkdir(parents=True, exist_ok=True)
(OUT_DIR_2CAT / "figures").mkdir(exist_ok=True)
(OUT_DIR_3CAT / "figures").mkdir(exist_ok=True)

TARGET_COL = "E. coli ATCC11775"
DPI        = 300

PALETTE_COND = {
    "No_linker": "#888888",
    "GGG":       "#4C72B0",
    "AAA":       "#E07B54",
}
CTYPES_2CAT = ["Carpet+Carpet", "Carpet+Pore", "Pore+Pore"]

sns.set_theme(style="whitegrid", font_scale=1.1)

# =========================
# HELPERS
# =========================

def load_linker_pred(pred_file, mapping_file):
    """Merge APEX predictions with mapping using row order."""
    pred = pd.read_csv(pred_file)
    if "Sequence" not in pred.columns:
        pred = pred.rename(columns={pred.columns[0]: "Sequence"})
    pred["APEX_row"] = pred.index
    mapping = pd.read_csv(mapping_file)
    df = mapping.merge(pred[["APEX_row", TARGET_COL]], on="APEX_row", how="left")
    df = df.rename(columns={TARGET_COL: "MIC_combined"})
    df["MIC_combined"] = pd.to_numeric(df["MIC_combined"], errors="coerce")
    return df

def add_delta(df, motif_mic):
    mic_map = motif_mic.set_index("Motif_representative")["MIC_mean"].to_dict()
    df["MIC_motif1"] = df["Motif_1"].map(mic_map)
    df["MIC_motif2"] = df["Motif_2"].map(mic_map)
    df["MIC_mean_individual"] = (df["MIC_motif1"] + df["MIC_motif2"]) / 2
    df["MIC_delta"] = df["MIC_combined"] - df["MIC_mean_individual"]
    return df

motif_mic = pd.read_csv(
    PROJECT_ROOT / "results/12_apex_prediction/02_results/02_motif_mic_summary_merged.csv"
)

# =========================
# TEE: print to terminal AND log file simultaneously
# =========================

import sys

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

LOG_DIR = PROJECT_ROOT / "results/13_motif_combinations"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "linker_analysis_log.txt"
_log_fh  = open(LOG_FILE, "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, _log_fh)

print(f"Log file: {LOG_FILE}")
print(f"Run date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

# =========================
# FUNCTION: FULL ANALYSIS PER CAT
# =========================

def run_analysis(cat_label, no_linker_file, pred_ggg, map_ggg,
                 pred_aaa, map_aaa, out_dir, ctypes):

    FIG_DIR = out_dir / "figures"
    print(f"\n{'='*60}")
    print(f"LINKER ANALYSIS — {cat_label}")
    print(f"{'='*60}")

    # load
    df_none = pd.read_csv(no_linker_file)
    df_none["MIC_combined"] = pd.to_numeric(df_none["MIC_combined"], errors="coerce")
    df_none["Condition"] = "No_linker"
    df_none = add_delta(df_none, motif_mic)

    df_ggg = load_linker_pred(pred_ggg, map_ggg)
    df_ggg["Condition"] = "GGG"
    df_ggg = add_delta(df_ggg, motif_mic)

    df_aaa = load_linker_pred(pred_aaa, map_aaa)
    df_aaa["Condition"] = "AAA"
    df_aaa = add_delta(df_aaa, motif_mic)

    # use Combination_type from no-linker (same pairs, same order)
    ctype_col = "Combination_type" if "Combination_type" in df_none.columns \
                else "Comb3_type"
    for df in [df_ggg, df_aaa]:
        if "Combination_type" not in df.columns and "Comb3_type" not in df.columns:
            # recover from no-linker via Motif_1 + Motif_2
            ctype_map = df_none.set_index(
                ["Motif_1","Motif_2"])[ctype_col].to_dict()
            df["Combination_type"] = df.apply(
                lambda r: ctype_map.get((r["Motif_1"],r["Motif_2"]), "Unknown"),
                axis=1)
        elif "Comb3_type" in df.columns:
            df["Combination_type"] = df["Comb3_type"]

    all_df = pd.concat([df_none, df_ggg, df_aaa], ignore_index=True)

    # =========================
    # ANALYSIS 1: Overall MIC per condition
    # =========================
    print(f"\n=== Overall MIC per condition ===")
    summary_rows = []
    for cond in ["No_linker","GGG","AAA"]:
        sub = all_df[all_df["Condition"]==cond]["MIC_combined"].dropna()
        print(f"  {cond:<12} mean={sub.mean():.1f}  median={sub.median():.1f}  "
              f"std={sub.std():.1f}  n={len(sub)}")
        summary_rows.append({"Condition": cond, "mean": round(sub.mean(),2),
                              "median": round(sub.median(),2),
                              "std": round(sub.std(),2), "n": len(sub)})
    summary_df = pd.DataFrame(summary_rows)

    # Kruskal-Wallis overall
    groups = [all_df[all_df["Condition"]==c]["MIC_combined"].dropna()
              for c in ["No_linker","GGG","AAA"]]
    h, p_kw = kruskal(*groups)
    print(f"\n  Kruskal-Wallis (3 conditions): H={h:.3f}  p={p_kw:.4f}")

    # pairwise MW
    mw_rows = []
    for c1, c2 in _comb(["No_linker","GGG","AAA"], 2):
        g1 = all_df[all_df["Condition"]==c1]["MIC_combined"].dropna()
        g2 = all_df[all_df["Condition"]==c2]["MIC_combined"].dropna()
        u, p = mannwhitneyu(g1, g2, alternative="two-sided")
        d = (g1.mean()-g2.mean()) / np.sqrt((g1.std()**2+g2.std()**2)/2)
        print(f"  {c1} vs {c2}: p={p:.4f}  d={d:.3f}  "
              f"({'GGG better' if c1=='GGG' and g1.mean()<g2.mean() else 'AAA better' if c1=='AAA' and g1.mean()<g2.mean() else 'no-linker better' if c2!='No_linker' else ''})")
        mw_rows.append({"Comparison": f"{c1} vs {c2}", "p": round(p,6),
                        "Cohen_d": round(d,3),
                        "mean_c1": round(g1.mean(),2), "mean_c2": round(g2.mean(),2)})
    mw_df = pd.DataFrame(mw_rows)

    # =========================
    # ANALYSIS 2: Per combination type per condition
    # =========================
    print(f"\n=== MIC per combination type × condition ===")
    type_cond_rows = []
    for ct in ctypes:
        for cond in ["No_linker","GGG","AAA"]:
            sub = all_df[(all_df["Combination_type"]==ct) &
                         (all_df["Condition"]==cond)]["MIC_combined"].dropna()
            if len(sub) == 0: continue
            print(f"  {ct:<25} {cond:<12} mean={sub.mean():.1f}  n={len(sub)}")
            type_cond_rows.append({
                "Combination_type": ct, "Condition": cond,
                "mean_MIC": round(sub.mean(),2),
                "median_MIC": round(sub.median(),2),
                "n": len(sub)
            })
    type_cond_df = pd.DataFrame(type_cond_rows)

    # =========================
    # ANALYSIS 3: PAIRED comparison (same pair, 3 conditions)
    # Wilcoxon signed-rank on matched pairs
    # =========================
    print(f"\n=== Paired comparison: No_linker vs GGG vs AAA ===")
    pair_key = ["Motif_1","Motif_2"]
    merged_pairs = df_none[pair_key + ["MIC_combined","Combination_type"]].merge(
        df_ggg[pair_key + ["MIC_combined"]].rename(columns={"MIC_combined":"MIC_GGG"}),
        on=pair_key, how="inner"
    ).merge(
        df_aaa[pair_key + ["MIC_combined"]].rename(columns={"MIC_combined":"MIC_AAA"}),
        on=pair_key, how="inner"
    ).rename(columns={"MIC_combined":"MIC_None"})

    print(f"  Matched pairs: {len(merged_pairs)}")
    paired_rows = []
    for cond_a, cond_b, col_a, col_b in [
        ("No_linker","GGG","MIC_None","MIC_GGG"),
        ("No_linker","AAA","MIC_None","MIC_AAA"),
        ("GGG","AAA","MIC_GGG","MIC_AAA"),
    ]:
        valid = merged_pairs[[col_a,col_b]].dropna()
        if len(valid) < 5: continue
        stat, p = wilcoxon(valid[col_a], valid[col_b])
        delta = (valid[col_b] - valid[col_a]).mean()
        better = cond_b if delta < 0 else cond_a
        print(f"  {cond_a} vs {cond_b}: Wilcoxon p={p:.4f}  "
              f"mean_Δ={delta:.2f}  better={better}")
        paired_rows.append({
            "Comparison": f"{cond_a} vs {cond_b}",
            "Wilcoxon_stat": round(stat,1), "p": round(p,6),
            "mean_delta": round(delta,3), "better": better,
            "n_pairs": len(valid)
        })
    paired_df = pd.DataFrame(paired_rows)

    # per combination type
    print(f"\n  Paired by combination type:")
    paired_ct_rows = []
    for ct in ctypes:
        sub_pairs = merged_pairs[merged_pairs["Combination_type"]==ct].dropna(
            subset=["MIC_None","MIC_GGG","MIC_AAA"])
        if len(sub_pairs) < 5: continue
        for col_a, col_b, label in [
            ("MIC_None","MIC_GGG","None vs GGG"),
            ("MIC_None","MIC_AAA","None vs AAA"),
            ("MIC_GGG","MIC_AAA","GGG vs AAA"),
        ]:
            stat, p = wilcoxon(sub_pairs[col_a], sub_pairs[col_b])
            delta = (sub_pairs[col_b] - sub_pairs[col_a]).mean()
            better = col_b.replace("MIC_","") if delta < 0 \
                     else col_a.replace("MIC_","")
            print(f"    {ct:<25} {label}: p={p:.4f}  Δ={delta:.2f}  better={better}")
            paired_ct_rows.append({
                "Combination_type": ct, "Comparison": label,
                "p": round(p,6), "mean_delta": round(delta,3),
                "better": better, "n_pairs": len(sub_pairs)
            })
    paired_ct_df = pd.DataFrame(paired_ct_rows)

    # =========================
    # ANALYSIS 4: ΔMIC per condition
    # =========================
    print(f"\n=== ΔMIC (combined - mean_individual) per condition ===")
    delta_rows = []
    for cond in ["No_linker","GGG","AAA"]:
        sub = all_df[all_df["Condition"]==cond]["MIC_delta"].dropna()
        print(f"  {cond:<12} mean_Δ={sub.mean():.2f}  "
              f"synergy={100*(sub<0).mean():.1f}%")
        delta_rows.append({"Condition": cond,
                           "mean_delta": round(sub.mean(),3),
                           "pct_synergy": round(100*(sub<0).mean(),1)})
    delta_summary_df = pd.DataFrame(delta_rows)

    # =========================
    # ANALYSIS 5: CORRELATION MIC_combined vs mean_individual per condition
    # =========================
    print(f"\n=== Correlation: MIC_combined vs mean_individual per condition ===")
    corr_cond_rows = []
    for cond in ["No_linker","GGG","AAA"]:
        sub = all_df[all_df["Condition"]==cond].dropna(
            subset=["MIC_combined","MIC_mean_individual"])
        rho, p = spearmanr(sub["MIC_combined"], sub["MIC_mean_individual"])
        print(f"  {cond:<12} ρ={rho:.3f}  p={p:.4f}  n={len(sub)}")
        corr_cond_rows.append({"Condition": cond,
                               "Spearman_rho": round(rho,3),
                               "p": round(p,6), "n": len(sub)})
    corr_cond_df = pd.DataFrame(corr_cond_rows)

    # per combination type
    print(f"\n  Per combination type:")
    corr_ct_rows = []
    for ct in ctypes:
        for cond in ["No_linker","GGG","AAA"]:
            sub = all_df[(all_df["Condition"]==cond) &
                         (all_df["Combination_type"]==ct)].dropna(
                             subset=["MIC_combined","MIC_mean_individual"])
            if len(sub) < 5: continue
            rho, p = spearmanr(sub["MIC_combined"], sub["MIC_mean_individual"])
            print(f"    {ct:<25} {cond:<12} ρ={rho:.3f}  p={p:.4f}")
            corr_ct_rows.append({"Combination_type": ct, "Condition": cond,
                                  "Spearman_rho": round(rho,3),
                                  "p": round(p,6), "n": len(sub)})
    corr_ct_df = pd.DataFrame(corr_ct_rows)

    # =========================
    # ANALYSIS 6: ORDER EFFECT per linker condition
    # =========================
    print(f"\n=== Order effect per condition ===")
    order_cond_rows = []
    for cond, df_cond in [("No_linker",df_none),("GGG",df_ggg),("AAA",df_aaa)]:
        df_cond = df_cond.copy()
        df_cond["Pair_key"] = df_cond.apply(
            lambda r: "_".join(sorted([str(r["Motif_1"]),str(r["Motif_2"])])), axis=1)
        pairs_cond = df_cond.groupby("Pair_key").filter(lambda g: len(g)==2)
        if len(pairs_cond) < 10:
            print(f"  {cond}: not enough paired data")
            continue
        pa = pairs_cond.groupby("Pair_key").apply(
            lambda g: g.sort_values("Combination_ID").iloc[0]["MIC_combined"])
        pb = pairs_cond.groupby("Pair_key").apply(
            lambda g: g.sort_values("Combination_ID").iloc[1]["MIC_combined"])
        valid = pa.notna() & pb.notna()
        from scipy.stats import ttest_rel
        t, p_ord = ttest_rel(pa[valid], pb[valid])
        diff = (pa[valid] - pb[valid]).abs().mean()
        print(f"  {cond:<12} t={t:.3f}  p={p_ord:.4f}  mean|Δ|={diff:.2f}")
        order_cond_rows.append({"Condition": cond, "t_stat": round(t,3),
                                 "p": round(p_ord,6),
                                 "mean_abs_delta": round(diff,3),
                                 "n_pairs": valid.sum()})
    order_cond_df = pd.DataFrame(order_cond_rows)

    # =========================
    # ANALYSIS 7: TOP 20 per condition
    # =========================
    top20_per_cond = {}
    for cond, df_cond in [("No_linker",df_none),("GGG",df_ggg),("AAA",df_aaa)]:
        top20_per_cond[cond] = df_cond.nsmallest(20, "MIC_combined")[[
            "Motif_1","Motif_2","Combination_type",
            "Sequence","MIC_combined","MIC_mean_individual","MIC_delta"
        ]]

    # =========================
    # ANALYSIS 8: FRIEDMAN TEST (3 paired conditions)
    # Correct test for 3 related conditions on same pairs
    # =========================
    print(f"\n=== Friedman test (No_linker vs GGG vs AAA, paired) ===")
    valid_friedman = merged_pairs[["MIC_None","MIC_GGG","MIC_AAA"]].dropna()
    friedman_rows = []
    if len(valid_friedman) >= 5:
        stat_f, p_f = friedmanchisquare(
            valid_friedman["MIC_None"],
            valid_friedman["MIC_GGG"],
            valid_friedman["MIC_AAA"]
        )
        print(f"  Overall: χ²={stat_f:.3f}  p={p_f:.4f}  n={len(valid_friedman)}")
        friedman_rows.append({"Comparison": "Overall", "chi2": round(stat_f,3),
                               "p": round(p_f,6), "n": len(valid_friedman)})
        # per combination type
        for ct in ctypes:
            sub_f = merged_pairs[merged_pairs["Combination_type"]==ct][
                ["MIC_None","MIC_GGG","MIC_AAA"]].dropna()
            if len(sub_f) < 5: continue
            stat_f2, p_f2 = friedmanchisquare(
                sub_f["MIC_None"], sub_f["MIC_GGG"], sub_f["MIC_AAA"])
            print(f"  {ct:<30} χ²={stat_f2:.3f}  p={p_f2:.4f}  n={len(sub_f)}")
            friedman_rows.append({"Comparison": ct, "chi2": round(stat_f2,3),
                                   "p": round(p_f2,6), "n": len(sub_f)})
    friedman_df = pd.DataFrame(friedman_rows)

    # =========================
    # ANALYSIS 9: SCORE COMBINADO per condition
    # =========================
    print(f"\n=== Score combinado per condition ===")
    score_rows = []
    for cond, df_cond in [("No_linker",df_none),("GGG",df_ggg),("AAA",df_aaa)]:
        motif_stats = []
        all_motifs = pd.concat([df_cond["Motif_1"],df_cond["Motif_2"]]).unique()
        for motif in all_motifs:
            mask = (df_cond["Motif_1"]==motif) | (df_cond["Motif_2"]==motif)
            sub  = df_cond[mask]["MIC_combined"].dropna()
            if len(sub) < 3: continue
            mean_mic = sub.mean()
            cv       = sub.std()/mean_mic*100 if mean_mic > 0 else np.nan
            score    = mean_mic * (1 + cv/100) if not np.isnan(cv) else np.nan
            motif_stats.append({"Motif": motif, "Condition": cond,
                                 "MIC_mean_in_comb": round(mean_mic,2),
                                 "CV": round(cv,1), "Score": round(score,2)})
        score_rows.extend(motif_stats)
    score_df = pd.DataFrame(score_rows)

    # pivot: compare score per motif across conditions
    score_pivot = score_df.pivot_table(
        index="Motif", columns="Condition", values="Score"
    ).reset_index()
    print(f"  Top 10 motifs by Score (No_linker):")
    top_scores = score_pivot.dropna().nsmallest(10, "No_linker")
    print(top_scores[["Motif","No_linker","GGG","AAA"]].to_string(index=False))

    # =========================
    # ANALYSIS 10: SAME-FAMILY vs CROSS-FAMILY per condition
    # =========================
    print(f"\n=== Same-family vs cross-family per condition ===")
    family_map_sf = motif_mic.set_index("Motif_representative")["Family_ID"].to_dict() \
                   if "Family_ID" in motif_mic.columns else {}
    sf_rows = []
    for cond, df_cond in [("No_linker",df_none),("GGG",df_ggg),("AAA",df_aaa)]:
        df_cond = df_cond.copy()
        df_cond["Fam1"] = df_cond["Motif_1"].map(family_map_sf)
        df_cond["Fam2"] = df_cond["Motif_2"].map(family_map_sf)
        df_cond["Same_fam"] = (df_cond["Fam1"]==df_cond["Fam2"]) & df_cond["Fam1"].notna()
        same  = df_cond[df_cond["Same_fam"]]["MIC_combined"].dropna()
        cross = df_cond[~df_cond["Same_fam"]]["MIC_combined"].dropna()
        if len(same) < 5 or len(cross) < 5: continue
        u, p_sf = mannwhitneyu(same, cross, alternative="two-sided")
        d_sf = (same.mean()-cross.mean())/np.sqrt((same.std()**2+cross.std()**2)/2)
        print(f"  {cond:<12} same={same.mean():.1f}  cross={cross.mean():.1f}  "
              f"p={p_sf:.4f}  d={d_sf:.3f}")
        sf_rows.append({"Condition": cond,
                         "mean_same": round(same.mean(),2),
                         "mean_cross": round(cross.mean(),2),
                         "p_mw": round(p_sf,6), "Cohen_d": round(d_sf,3),
                         "n_same": len(same), "n_cross": len(cross)})
    sf_linker_df = pd.DataFrame(sf_rows)

    # =========================
    # ANALYSIS 11: TOP CROSS-FAMILY EXCLUDING DOMINANT per condition
    # =========================
    top_nodom_per_cond = {}
    for cond, df_cond in [("No_linker",df_none),("GGG",df_ggg),("AAA",df_aaa)]:
        df_cond = df_cond.copy()
        df_cond["Fam1"] = df_cond["Motif_1"].map(family_map_sf)
        df_cond["Fam2"] = df_cond["Motif_2"].map(family_map_sf)
        df_cond["Same_fam"] = (df_cond["Fam1"]==df_cond["Fam2"]) & df_cond["Fam1"].notna()
        top_all = df_cond.nsmallest(20,"MIC_combined")
        dom = pd.concat([top_all["Motif_1"],top_all["Motif_2"]]).value_counts().index[0]
        top_nodom = df_cond[
            ~df_cond["Same_fam"] &
            (df_cond["Motif_1"]!=dom) &
            (df_cond["Motif_2"]!=dom)
        ].nsmallest(20,"MIC_combined")[[
            "Motif_1","Motif_2","Combination_type",
            "Sequence","MIC_combined","MIC_delta"]]
        top_nodom_per_cond[cond] = (dom, top_nodom)

    # =========================
    # SUMMARY TABLE for paper
    # =========================
    summary_table = pd.DataFrame([
        {"Analysis": "KW 3 conditions", "Statistic": "H",
         "Value": round(h,3), "p": round(p_kw,4)},
        *[{"Analysis": f"MW {r['Comparison']}", "Statistic": "Cohen_d",
           "Value": r["Cohen_d"], "p": r["p"]} for _, r in mw_df.iterrows()],
        *[{"Analysis": f"Wilcoxon {r['Comparison']}", "Statistic": "stat",
           "Value": r["Wilcoxon_stat"], "p": r["p"]} for _, r in paired_df.iterrows()],
        *[{"Analysis": f"Friedman {r['Comparison']}", "Statistic": "chi2",
           "Value": r["chi2"], "p": r["p"]} for _, r in friedman_df.iterrows()],
        *[{"Analysis": f"Order {r['Condition']}", "Statistic": "t",
           "Value": r["t_stat"], "p": r["p"]} for _, r in order_cond_df.iterrows()],
        *[{"Analysis": f"SameFam {r['Condition']}", "Statistic": "Cohen_d",
           "Value": r["Cohen_d"], "p": r["p_mw"]} for _, r in sf_linker_df.iterrows()],
    ])

    # =========================
    # SAVE EXCEL
    # =========================
    out_excel = out_dir / f"01_linker_analysis_{cat_label}.xlsx"
    gp_excel  = out_dir / f"02_graphpad_linker_{cat_label}.xlsx"

    with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
        summary_df.to_excel(writer,       sheet_name="Overall_summary",    index=False)
        mw_df.to_excel(writer,            sheet_name="MannWhitney",         index=False)
        type_cond_df.to_excel(writer,     sheet_name="ByType_condition",    index=False)
        paired_df.to_excel(writer,        sheet_name="Paired_overall",      index=False)
        paired_ct_df.to_excel(writer,     sheet_name="Paired_by_type",      index=False)
        delta_summary_df.to_excel(writer, sheet_name="Delta_MIC_summary",   index=False)
        corr_cond_df.to_excel(writer,     sheet_name="Corr_per_condition",  index=False)
        corr_ct_df.to_excel(writer,       sheet_name="Corr_per_type",       index=False)
        order_cond_df.to_excel(writer,    sheet_name="Order_per_condition", index=False)
        friedman_df.to_excel(writer,      sheet_name="Friedman_test",       index=False)
        score_pivot.to_excel(writer,      sheet_name="Score_per_condition", index=False)
        sf_linker_df.to_excel(writer,     sheet_name="SameVsCross_linker",  index=False)
        merged_pairs.to_excel(writer,     sheet_name="Matched_pairs",       index=False)
        all_df.to_excel(writer,           sheet_name="All_data",            index=False)
        summary_table.to_excel(writer,    sheet_name="Summary_for_paper",   index=False)
        for cond, top20 in top20_per_cond.items():
            top20.to_excel(writer, sheet_name=f"Top20_{cond}", index=False)
        for cond, (dom, top_nd) in top_nodom_per_cond.items():
            top_nd.to_excel(writer, sheet_name=f"Top20_nodom_{cond[:7]}", index=False)
    print(f"\n✅ Excel: {out_excel}")

    # GraphPad
    def to_wide(d):
        return pd.DataFrame({k: pd.Series(v).reset_index(drop=True)
                              for k,v in d.items()})

    with pd.ExcelWriter(gp_excel, engine="openpyxl") as writer:
        # violin overall
        to_wide({c: all_df[all_df["Condition"]==c]["MIC_combined"].dropna().values
                 for c in ["No_linker","GGG","AAA"]}).to_excel(
            writer, sheet_name="GP_Fig1_violin_overall", index=False)
        # violin per type
        for ct in ctypes:
            safe = ct.replace("+","_")[:20]
            to_wide({c: all_df[(all_df["Condition"]==c) &
                               (all_df["Combination_type"]==ct)
                               ]["MIC_combined"].dropna().values
                     for c in ["No_linker","GGG","AAA"]}).to_excel(
                writer, sheet_name=f"GP_{safe}", index=False)
        # paired data
        merged_pairs[["MIC_None","MIC_GGG","MIC_AAA",
                       "Combination_type"]].to_excel(
            writer, sheet_name="GP_Fig2_paired", index=False)
        # delta
        to_wide({c: all_df[all_df["Condition"]==c]["MIC_delta"].dropna().values
                 for c in ["No_linker","GGG","AAA"]}).to_excel(
            writer, sheet_name="GP_Fig3_delta", index=False)
        # stats
        paired_ct_df.to_excel(writer, sheet_name="GP_Stats_paired",  index=False)
        mw_df.to_excel(writer,        sheet_name="GP_Stats_MW",       index=False)
        corr_cond_df.to_excel(writer, sheet_name="GP_Stats_corr",     index=False)
        friedman_df.to_excel(writer,  sheet_name="GP_Stats_Friedman", index=False)
        sf_linker_df.to_excel(writer, sheet_name="GP_SameVsCross",    index=False)
        summary_table.to_excel(writer,sheet_name="GP_Summary_paper",  index=False)
        # score pivot
        score_pivot.to_excel(writer,  sheet_name="GP_Score_conditions",index=False)
        # top20 per condition
        for cond, top20 in top20_per_cond.items():
            top20.to_excel(writer, sheet_name=f"GP_Top20_{cond[:7]}", index=False)
        # top cross-family no-dominant per condition
        for cond, (dom, top_nd) in top_nodom_per_cond.items():
            top_nd.to_excel(writer, sheet_name=f"GP_nodom_{cond[:7]}", index=False)
        # scatter data: MIC_combined vs mean_individual per condition
        for cond in ["No_linker","GGG","AAA"]:
            sub = all_df[all_df["Condition"]==cond].dropna(
                subset=["MIC_combined","MIC_mean_individual"])
            pd.DataFrame({
                "MIC_mean_individual": sub["MIC_mean_individual"].values,
                "MIC_combined":        sub["MIC_combined"].values,
                "Combination_type":    sub["Combination_type"].values,
            }).to_excel(writer, sheet_name=f"GP_corr_{cond[:7]}", index=False)
    print(f"✅ GraphPad: {gp_excel}")

    # =========================
    # FIGURES
    # =========================

    # Fig 1: Violin overall — 3 conditions
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
    plot_rows = []
    for c in ["No_linker","GGG","AAA"]:
        for v in all_df[all_df["Condition"]==c]["MIC_combined"].dropna():
            plot_rows.append({"Condition": c, "MIC": v})
    plot_df = pd.DataFrame(plot_rows)
    sns.violinplot(data=plot_df, x="Condition", y="MIC", hue="Condition",
                   palette=PALETTE_COND, inner="quart",
                   linewidth=0.9, legend=False, ax=ax)
    ax.set_xlabel(""); ax.set_ylabel("MIC E. coli ATCC11775 (µmol/L)", fontsize=11)
    ax.set_title(f"MIC: no linker vs GGG vs AAA — {cat_label}",
                 fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR/f"01_violin_conditions_{cat_label}.png",
                dpi=DPI, bbox_inches="tight"); plt.close()

    # Fig 2: Violin per combination type × condition
    fig, axes = plt.subplots(1, len(ctypes), figsize=(6*len(ctypes), 5),
                             sharey=True)
    if len(ctypes) == 1: axes = [axes]
    fig.patch.set_facecolor("#fafafa")
    for ax, ct in zip(axes, ctypes):
        ax.set_facecolor("#fafafa")
        rows = []
        for c in ["No_linker","GGG","AAA"]:
            for v in all_df[(all_df["Condition"]==c) &
                            (all_df["Combination_type"]==ct)]["MIC_combined"].dropna():
                rows.append({"Condition":c,"MIC":v})
        if not rows: continue
        sns.violinplot(data=pd.DataFrame(rows), x="Condition", y="MIC",
                       hue="Condition", palette=PALETTE_COND,
                       inner="quart", linewidth=0.9, legend=False, ax=ax)
        ax.set_title(ct, fontweight="bold", fontsize=9)
        ax.set_xlabel(""); ax.set_ylabel("MIC (µmol/L)" if ax==axes[0] else "")
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle(f"MIC per combination type × linker condition — {cat_label}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR/f"02_violin_by_type_{cat_label}.png",
                dpi=DPI, bbox_inches="tight"); plt.close()

    # Fig 3: Paired scatter — No_linker vs GGG and No_linker vs AAA
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#fafafa")
    for ax, (col_x, col_y, label) in zip(axes, [
        ("MIC_None","MIC_GGG","No linker vs GGG"),
        ("MIC_None","MIC_AAA","No linker vs AAA"),
    ]):
        ax.set_facecolor("#fafafa")
        valid = merged_pairs[[col_x,col_y,"Combination_type"]].dropna()
        for ct, color in zip(ctypes, ["#E07B54","#5B8DB8","#3BAF77"]):
            sub = valid[valid["Combination_type"]==ct]
            ax.scatter(sub[col_x], sub[col_y], c=color, alpha=0.3,
                       s=10, edgecolors="none", label=ct)
        lims = [valid[[col_x,col_y]].min().min()-2,
                valid[[col_x,col_y]].max().max()+2]
        ax.plot(lims,lims,"k--",lw=1,alpha=0.4,label="y=x (no effect)")
        rho, _ = spearmanr(valid[col_x], valid[col_y])
        ax.annotate(f"ρ={rho:.3f}", xy=(0.05,0.92), xycoords="axes fraction",
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ddd"))
        ax.set_xlabel(col_x.replace("MIC_","MIC "), fontsize=10)
        ax.set_ylabel(col_y.replace("MIC_","MIC "), fontsize=10)
        ax.set_title(label, fontweight="bold")
        ax.legend(fontsize=7, framealpha=0.8)
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle(f"Paired MIC: linker effect on same pairs — {cat_label}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR/f"03_paired_scatter_{cat_label}.png",
                dpi=DPI, bbox_inches="tight"); plt.close()

    # Fig 4: ΔMIC KDE per condition
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
    for c, color in PALETTE_COND.items():
        sub = all_df[all_df["Condition"]==c]["MIC_delta"].dropna()
        sns.kdeplot(sub, fill=True, alpha=0.4, color=color,
                    label=f"{c} (mean_Δ={sub.mean():.1f})", ax=ax)
    ax.axvline(0, color="black", lw=1.2, ls="--", alpha=0.6)
    ax.set_xlabel("ΔMIC = combined − mean(individual)", fontsize=10)
    ax.set_ylabel("Density"); ax.legend(fontsize=9)
    ax.set_title(f"ΔMIC distribution per condition — {cat_label}", fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR/f"04_delta_MIC_{cat_label}.png",
                dpi=DPI, bbox_inches="tight"); plt.close()

    # Fig 5: Heatmap mean MIC — combination type × condition
    hmap_data = type_cond_df.pivot(index="Combination_type",
                                   columns="Condition", values="mean_MIC")
    hmap_data = hmap_data.reindex(columns=["No_linker","GGG","AAA"])
    fig, ax = plt.subplots(figsize=(7, max(3, len(ctypes)*0.8)))
    fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
    sns.heatmap(hmap_data.astype(float), annot=True, fmt=".1f",
                cmap="YlOrRd_r", linewidths=0.5, ax=ax,
                annot_kws={"size":11,"weight":"bold"})
    ax.set_title(f"Mean MIC (µmol/L) — combination type × linker\n{cat_label}",
                 fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIG_DIR/f"05_heatmap_type_condition_{cat_label}.png",
                dpi=DPI, bbox_inches="tight"); plt.close()

    print(f"✅ Figures saved: {FIG_DIR}")

    # Fig 6: Correlation MIC_combined vs mean_individual per condition
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.patch.set_facecolor("#fafafa")
    for ax, cond in zip(axes, ["No_linker","GGG","AAA"]):
        ax.set_facecolor("#fafafa")
        sub = all_df[all_df["Condition"]==cond].dropna(
            subset=["MIC_combined","MIC_mean_individual"])
        for ct, color in zip(ctypes, ["#E07B54","#5B8DB8","#3BAF77",
                                       "#9B59B6","#E67E22","#1ABC9C"]):
            s = sub[sub["Combination_type"]==ct]
            ax.scatter(s["MIC_mean_individual"], s["MIC_combined"],
                       c=color, alpha=0.25, s=10, edgecolors="none", label=ct)
        lims = [sub[["MIC_mean_individual","MIC_combined"]].min().min()-2,
                sub[["MIC_mean_individual","MIC_combined"]].max().max()+2]
        ax.plot(lims, lims, "k--", lw=1, alpha=0.4)
        rho = corr_cond_df.loc[corr_cond_df["Condition"]==cond,"Spearman_rho"].values
        rho_val = rho[0] if len(rho) > 0 else 0
        ax.annotate(f"ρ={rho_val:.3f}",
                    xy=(0.05,0.92), xycoords="axes fraction", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ddd"))
        ax.set_title(cond, fontweight="bold")
        ax.set_xlabel("Mean MIC individual (µmol/L)", fontsize=9)
        ax.set_ylabel("MIC combined (µmol/L)" if ax==axes[0] else "", fontsize=9)
        ax.legend(fontsize=6, framealpha=0.7)
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle(f"Correlation: MIC combined vs mean individual — per linker condition ({cat_label})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR/f"06_corr_combined_vs_individual_{cat_label}.png",
                dpi=DPI, bbox_inches="tight"); plt.close()

    # Fig 7: ρ comparison bar — how correlation changes with linker
    if len(corr_cond_df) > 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
        x = np.arange(len(corr_cond_df))
        bars = ax.bar(x, corr_cond_df["Spearman_rho"],
                      color=[PALETTE_COND[c] for c in corr_cond_df["Condition"]],
                      alpha=0.85, edgecolor="white", width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(corr_cond_df["Condition"], fontsize=11)
        ax.set_ylabel("Spearman ρ", fontsize=11)
        ax.set_ylim(0, 1)
        ax.axhline(0.7, color="grey", lw=0.8, ls="--", alpha=0.5)
        for bar, val in zip(bars, corr_cond_df["Spearman_rho"]):
            ax.text(bar.get_x()+bar.get_width()/2, val+0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=11,
                    fontweight="bold")
        ax.set_title(f"Spearman ρ (combined vs individual) per condition — {cat_label}",
                     fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIG_DIR/f"07_rho_per_condition_{cat_label}.png",
                    dpi=DPI, bbox_inches="tight"); plt.close()

    print(f"✅ All figures saved: {FIG_DIR}")

    # Fig 8: Score combinado per condition — top 20 motifs
    if not score_pivot.empty and "No_linker" in score_pivot.columns:
        top_sc = score_pivot.dropna().nsmallest(20, "No_linker")
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
        x = np.arange(len(top_sc))
        w = 0.25
        for i, (cond, color) in enumerate(PALETTE_COND.items()):
            if cond not in top_sc.columns: continue
            ax.bar(x + i*w, top_sc[cond], width=w, color=color,
                   alpha=0.85, edgecolor="white", label=cond)
        ax.set_xticks(x + w)
        ax.set_xticklabels(top_sc["Motif"], rotation=45, ha="right", fontsize=7.5)
        ax.set_ylabel("Score combinado (lower = better)", fontsize=10)
        ax.set_title(f"Score combinado per condition — top 20 motifs ({cat_label})",
                     fontweight="bold")
        ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIG_DIR/f"08_score_per_condition_{cat_label}.png",
                    dpi=DPI, bbox_inches="tight"); plt.close()

    # Fig 9: Same-family vs cross-family per condition — grouped bars
    if len(sf_linker_df) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.patch.set_facecolor("#fafafa")
        ax = axes[0]; ax.set_facecolor("#fafafa")
        x = np.arange(len(sf_linker_df))
        w = 0.35
        ax.bar(x - w/2, sf_linker_df["mean_same"],  width=w,
               color="#CCCCCC", alpha=0.85, edgecolor="white", label="Same family")
        ax.bar(x + w/2, sf_linker_df["mean_cross"], width=w,
               color="#4C72B0", alpha=0.85, edgecolor="white", label="Cross family")
        ax.set_xticks(x)
        ax.set_xticklabels(sf_linker_df["Condition"], fontsize=10)
        ax.set_ylabel("Mean MIC (µmol/L)", fontsize=10)
        ax.set_title("Same-family vs cross-family\nper linker condition",
                     fontweight="bold")
        ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)
        # Cohen's d per condition
        ax2 = axes[1]; ax2.set_facecolor("#fafafa")
        cols = [PALETTE_COND.get(c,"#888") for c in sf_linker_df["Condition"]]
        bars = ax2.bar(range(len(sf_linker_df)), sf_linker_df["Cohen_d"].abs(),
                       color=cols, alpha=0.85, edgecolor="white")
        ax2.set_xticks(range(len(sf_linker_df)))
        ax2.set_xticklabels(sf_linker_df["Condition"], fontsize=10)
        ax2.set_ylabel("|Cohen's d|", fontsize=10)
        ax2.set_title("Effect size (same vs cross-family)\nper linker condition",
                      fontweight="bold")
        for bar, p_val in zip(bars, sf_linker_df["p_mw"]):
            sig = "***" if p_val<0.001 else "**" if p_val<0.01 else "*" if p_val<0.05 else "ns"
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                     sig, ha="center", fontsize=12, fontweight="bold")
        ax2.spines[["top","right"]].set_visible(False)
        fig.suptitle(f"Same-family vs cross-family analysis per condition — {cat_label}",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(FIG_DIR/f"09_samefam_per_condition_{cat_label}.png",
                    dpi=DPI, bbox_inches="tight"); plt.close()

    # Fig 10: Top cross-family no-dominant per condition — side by side bars
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    fig.patch.set_facecolor("#fafafa")
    for ax, (cond, (dom, top_nd)) in zip(axes, top_nodom_per_cond.items()):
        ax.set_facecolor("#fafafa")
        if top_nd.empty: continue
        colors_ct = [{"Carpet+Carpet":"#E07B54","Carpet+Pore":"#5B8DB8",
                      "Pore+Pore":"#3BAF77"}.get(ct,"#888")
                     for ct in top_nd["Combination_type"]]
        ax.barh(range(len(top_nd)), top_nd["MIC_combined"],
                color=colors_ct, alpha=0.85, edgecolor="white")
        ax.set_yticks(range(len(top_nd)))
        ax.set_yticklabels([f"{r['Motif_1']}+{r['Motif_2']}"
                            for _, r in top_nd.iterrows()], fontsize=7)
        ax.set_xlabel("MIC combined (µmol/L)", fontsize=9)
        ax.set_title(f"{cond}\n(excl. {dom})", fontweight="bold", fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle(f"Top cross-family combinations excl. dominant — {cat_label}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR/f"10_nodom_per_condition_{cat_label}.png",
                dpi=DPI, bbox_inches="tight"); plt.close()

    print(f"✅ Figures 8-10 saved")
    return all_df, merged_pairs

# =========================
# RUN 2CAT
# =========================

all_2cat, pairs_2cat = run_analysis(
    "2cat", NO_LINKER_2CAT,
    PRED_2CAT_GGG, MAPPING_2CAT_GGG,
    PRED_2CAT_AAA, MAPPING_2CAT_AAA,
    OUT_DIR_2CAT, CTYPES_2CAT
)

# =========================
# RUN 3CAT
# =========================

CTYPES_3CAT = ["Barrel_stave+Barrel_stave","Barrel_stave+Carpet",
               "Barrel_stave+Toroidal_pore","Carpet+Carpet",
               "Carpet+Toroidal_pore","Toroidal_pore+Toroidal_pore"]

all_3cat, pairs_3cat = run_analysis(
    "3cat", NO_LINKER_3CAT,
    PRED_3CAT_GGG, MAPPING_3CAT_GGG,
    PRED_3CAT_AAA, MAPPING_3CAT_AAA,
    OUT_DIR_3CAT, CTYPES_3CAT
)

print(f"\n✅ All linker analyses complete.")
print(f"✅ Log saved: {LOG_FILE}")

# restore stdout and close log
sys.stdout = sys.__stdout__
_log_fh.close()