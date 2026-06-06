"""
GRAMPA units: log10(µmol/L) → converted to µmol/L (same as APEX output).
Filters to E. coli entries for maximum comparability with APEX target
(E. coli ATCC11775).

"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

sns.set_theme(style="whitegrid", font_scale=1.05)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PEPTIDE_FILE = (PROJECT_ROOT / "results/12_apex_prediction/02_results"
                / "01_peptide_mic_merged.csv")
GRAMPA_FILE  = PROJECT_ROOT / "data/raw/grampa.csv"
OUT_DIR      = PROJECT_ROOT / "results/12_apex_prediction/03_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300

# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────

print("Loading data...")
df     = pd.read_csv(PEPTIDE_FILE)
grampa = pd.read_csv(GRAMPA_FILE)

df["MIC_ecoli"] = pd.to_numeric(df["MIC_ecoli"], errors="coerce")
df["Sequence_window"] = df["Sequence_window"].str.upper().str.strip()

# convert GRAMPA log10(µmol/L) → µmol/L
grampa["sequence"]    = grampa["sequence"].str.upper().str.strip()
grampa["MIC_umol"]    = 10 ** pd.to_numeric(grampa["value"], errors="coerce")

# keep only E. coli entries
grampa_ecoli = grampa[
    grampa["bacterium"].str.contains("coli", case=False, na=False)
].copy()

print(f"Peptides to validate:    {len(df)}")
print(f"GRAMPA total entries:    {len(grampa)}")
print(f"GRAMPA E. coli entries:  {len(grampa_ecoli)}")
print(f"GRAMPA E. coli unique seq: {grampa_ecoli['sequence'].nunique()}")

# build lookup set for fast exact matching
grampa_seq_set = set(grampa_ecoli["sequence"])

# ─────────────────────────────────────────────
# MATCHING
# ─────────────────────────────────────────────

print("\nSearching for matches...")

match_rows = []

for _, pep in df.iterrows():
    win = pep["Sequence_window"]
    if not isinstance(win, str) or len(win) < 4:
        continue

    # exact match
    exact_hits = grampa_ecoli[grampa_ecoli["sequence"] == win].copy()
    exact_hits["Match_type"] = "exact"

    # window contained in GRAMPA sequence (longer GRAMPA seq contains motif)
    in_db_hits = grampa_ecoli[
        grampa_ecoli["sequence"].str.contains(win, regex=False, na=False) &
        (grampa_ecoli["sequence"] != win)
    ].copy()
    in_db_hits["Match_type"] = "window_in_DB"

    # GRAMPA sequence contained in window (shorter GRAMPA seq inside motif)
    db_in_win_hits = grampa_ecoli[
        grampa_ecoli["sequence"].apply(
            lambda s: isinstance(s, str) and s in win and s != win
        )
    ].copy()
    db_in_win_hits["Match_type"] = "DB_in_window"

    all_hits = pd.concat([exact_hits, in_db_hits, db_in_win_hits],
                          ignore_index=True)
    # remove duplicate GRAMPA sequences (keep best match type: exact > window_in_DB > DB_in_window)
    type_order = {"exact": 0, "window_in_DB": 1, "DB_in_window": 2}
    if not all_hits.empty:
        all_hits["_order"] = all_hits["Match_type"].map(type_order)
        all_hits = (all_hits.sort_values("_order")
                             .drop_duplicates("sequence")
                             .drop(columns=["_order"]))

    base = {
        "Peptide_ID":               pep["Peptide_ID"],
        "Sequence_window":          win,
        "Motif_representative":     pep.get("Motif_representative"),
        "MIC_ecoli_predicted_umol": pep.get("MIC_ecoli"),
        "Mechanism_weighted":       pep.get("Mechanism_weighted"),
    }

    if all_hits.empty:
        match_rows.append({**base,
                           "DB_match": False,
                           "DB_sequence": None,
                           "DB_MIC_umol": None,
                           "DB_bacterium": None,
                           "DB_strain": None,
                           "Match_type": None})
    else:
        for _, hit in all_hits.iterrows():
            match_rows.append({**base,
                                "DB_match":    True,
                                "DB_sequence": hit["sequence"],
                                "DB_MIC_umol": hit["MIC_umol"],
                                "DB_bacterium":hit.get("bacterium"),
                                "DB_strain":   hit.get("strain"),
                                "Match_type":  hit["Match_type"]})

results = pd.DataFrame(match_rows)
results["DB_MIC_umol"]               = pd.to_numeric(results["DB_MIC_umol"], errors="coerce")
results["MIC_ecoli_predicted_umol"]  = pd.to_numeric(results["MIC_ecoli_predicted_umol"], errors="coerce")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────

n_total   = df["Peptide_ID"].nunique()
n_matched = results[results["DB_match"]]["Peptide_ID"].nunique()
n_mic     = results[results["DB_MIC_umol"].notna()]["Peptide_ID"].nunique()

print(f"\n=== Match summary ===")
print(f"  Peptides with any match:        {n_matched}/{n_total} ({100*n_matched/n_total:.1f}%)")
print(f"  Peptides with experimental MIC: {n_mic}/{n_total} ({100*n_mic/n_total:.1f}%)")

# by match type
if n_matched > 0:
    mt = (results[results["DB_match"]]
          .drop_duplicates("Peptide_ID")
          ["Match_type"].value_counts())
    print("\n  Matches by type (unique peptides):")
    for t, c in mt.items():
        print(f"    {t:<20} {c}")

# correlation
if n_mic >= 5:
    # filter to APEX-predicted active peptides (MIC <= 128 µmol/L)
    # consistent with APEX's own activity threshold
    mic_comp = (results[
                    results["DB_MIC_umol"].notna() &
                    (results["MIC_ecoli_predicted_umol"] <= 128) &
                    (results["DB_MIC_umol"] <= 128)
                ]
                .drop_duplicates("Peptide_ID")
                .dropna(subset=["MIC_ecoli_predicted_umol","DB_MIC_umol"]))
    print(f"\n  Filtered to APEX && GRAMPA MIC ≤ 128 µmol/L: {len(mic_comp)} peptides")

    print(f"\n  Predicted  MIC (APEX, µmol/L): "
          f"mean={mic_comp['MIC_ecoli_predicted_umol'].mean():.1f}  "
          f"median={mic_comp['MIC_ecoli_predicted_umol'].median():.1f}")
    print(f"  Experimental MIC (GRAMPA, µmol/L): "
          f"mean={mic_comp['DB_MIC_umol'].mean():.1f}  "
          f"median={mic_comp['DB_MIC_umol'].median():.1f}")

    rho, p_rho = spearmanr(mic_comp["MIC_ecoli_predicted_umol"],
                            mic_comp["DB_MIC_umol"])
    r,   p_r   = pearsonr(mic_comp["MIC_ecoli_predicted_umol"],
                           mic_comp["DB_MIC_umol"])
    print(f"\n  Correlation (n={len(mic_comp)}):")
    print(f"    Spearman ρ = {rho:.3f}  (p={p_rho:.4f})")
    print(f"    Pearson  r = {r:.3f}  (p={p_r:.4f})")

    # by match type
    print("\n  Correlation by match type:")
    corr_rows = []
    for mt_val in ["exact", "window_in_DB", "DB_in_window"]:
        sub = mic_comp[mic_comp["Match_type"] == mt_val].dropna(
            subset=["MIC_ecoli_predicted_umol","DB_MIC_umol"])
        if len(sub) < 5:
            print(f"    {mt_val:<20} n={len(sub)} (too few)")
            corr_rows.append({"Match_type": mt_val, "n": len(sub),
                               "Spearman_rho": np.nan, "p": np.nan})
            continue
        rho_t, p_t = spearmanr(sub["MIC_ecoli_predicted_umol"],
                                sub["DB_MIC_umol"])
        print(f"    {mt_val:<20} n={len(sub)}  ρ={rho_t:.3f}  p={p_t:.4f}")
        corr_rows.append({"Match_type": mt_val, "n": len(sub),
                           "Spearman_rho": round(rho_t,3), "p": round(p_t,6)})
    corr_by_type = pd.DataFrame(corr_rows)

    # ─────────────────────────────────────────────
    # FIGURES
    # ─────────────────────────────────────────────

    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Fig 1: scatter predicted vs experimental — all match types
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    fig.patch.set_facecolor("#fafafa")
    colors_mt = {"exact": "#E07B54",
                 "window_in_DB": "#5B8DB8",
                 "DB_in_window": "#3BAF77"}

    for ax, mt_val in zip(axes, ["exact","window_in_DB","DB_in_window"]):
        ax.set_facecolor("#fafafa")
        sub = mic_comp[mic_comp["Match_type"] == mt_val].dropna(
            subset=["MIC_ecoli_predicted_umol","DB_MIC_umol"])
        if sub.empty:
            ax.set_title(f"{mt_val}\n(no data)", fontsize=9)
            continue
        ax.scatter(sub["DB_MIC_umol"],
                   sub["MIC_ecoli_predicted_umol"],
                   c=colors_mt[mt_val], alpha=0.7, s=50,
                   edgecolors="white", linewidths=0.4)
        if len(sub) >= 5:
            rho_t, p_t = spearmanr(sub["MIC_ecoli_predicted_umol"],
                                    sub["DB_MIC_umol"])
            ax.annotate(f"ρ={rho_t:.3f}  p={p_t:.4f}\nn={len(sub)}",
                        xy=(0.05,0.92), xycoords="axes fraction", fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3",fc="white",ec="#ddd"))
        ax.set_xlabel("MIC experimental (GRAMPA, µmol/L)", fontsize=9)
        ax.set_ylabel("MIC predicta (APEX, µmol/L)", fontsize=9)
        ax.set_title(f"Match type: {mt_val}\n(n={len(sub)})",
                     fontweight="bold", fontsize=9)
        ax.spines[["top","right"]].set_visible(False)

    fig.suptitle("Validació APEX vs GRAMPA (E. coli)\n"
                 "Predicted µmol/L vs Experimental µmol/L",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir/"grampa_validation_scatter.png",
                dpi=DPI, bbox_inches="tight")
    plt.close()

    # Fig 2: scatter total (tots els match types junts)
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
    for mt_val, color in colors_mt.items():
        sub = mic_comp[mic_comp["Match_type"] == mt_val].dropna(
            subset=["MIC_ecoli_predicted_umol","DB_MIC_umol"])
        if sub.empty: continue
        ax.scatter(sub["DB_MIC_umol"],
                   sub["MIC_ecoli_predicted_umol"],
                   c=color, alpha=0.7, s=50,
                   edgecolors="white", linewidths=0.4,
                   label=f"{mt_val} (n={len(sub)})")
    ax.annotate(f"ρ={rho:.3f}  p={p_rho:.4f}\nn={len(mic_comp)}",
                xy=(0.05,0.92), xycoords="axes fraction", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3",fc="white",ec="#ddd"))
    ax.set_xlabel("MIC experimental (GRAMPA, µmol/L)", fontsize=11)
    ax.set_ylabel("MIC predicta (APEX, µmol/L)", fontsize=11)
    ax.set_title("APEX vs GRAMPA — tots els matches\n(E. coli, mateixa unitat)",
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir/"grampa_validation_total.png",
                dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"\n  Figures saved: {fig_dir}")

else:
    print(f"\n  Not enough matches for correlation (n={n_mic} < 5)")
    mic_comp       = pd.DataFrame()
    corr_by_type   = pd.DataFrame()

# ─────────────────────────────────────────────
# SAVE (same filenames as original script)
# ─────────────────────────────────────────────

out_all   = OUT_DIR / "01_db_matches_all.csv"
out_mic   = OUT_DIR / "02_db_matches_with_mic.csv"
out_excel = OUT_DIR / "03_db_validation.xlsx"

results.to_csv(out_all, index=False)
results[results["DB_MIC_umol"].notna()].to_csv(out_mic, index=False)

with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
    results.to_excel(writer, sheet_name="All_matches", index=False)
    results[results["DB_match"]].to_excel(
        writer, sheet_name="Matched_only", index=False)
    results[results["DB_MIC_umol"].notna()].to_excel(
        writer, sheet_name="With_experimental_MIC", index=False)
    results[~results["DB_match"]][
        ["Peptide_ID","Sequence_window","Motif_representative",
         "MIC_ecoli_predicted_umol"]
    ].drop_duplicates().to_excel(
        writer, sheet_name="No_DB_match", index=False)

    # exact matches only (most reliable)
    results[results["Match_type"]=="exact"].to_excel(
        writer, sheet_name="Exact_matches", index=False)

    # correlation summary
    if not mic_comp.empty:
        pd.DataFrame([{
            "n_total_peptides":    n_total,
            "n_matched":           n_matched,
            "n_with_MIC":          n_mic,
            "Spearman_rho_all":    round(rho, 3),
            "Spearman_p_all":      round(p_rho, 6),
            "Pearson_r_all":       round(r, 3),
            "Pearson_p_all":       round(p_r, 6),
            "note": ("Comparison APEX (µmol/L predicted) vs GRAMPA (µmol/L "
                     "experimental, E. coli). Exact = same sequence; "
                     "window_in_DB = motif contained in longer GRAMPA peptide; "
                     "DB_in_window = shorter GRAMPA peptide contained in motif.")
        }]).to_excel(writer, sheet_name="Correlation_summary", index=False)

        if not corr_by_type.empty:
            corr_by_type.to_excel(writer,
                                   sheet_name="Corr_by_match_type", index=False)

        # GraphPad — scatter data
        pd.DataFrame({
            "MIC_experimental_umol": mic_comp["DB_MIC_umol"].values,
            "MIC_predicted_umol":    mic_comp["MIC_ecoli_predicted_umol"].values,
            "Match_type":            mic_comp["Match_type"].values,
        }).to_excel(writer, sheet_name="GP_scatter", index=False)

print(f"\n✅ Saved:")
print(f"  {out_all}")
print(f"  {out_mic}")
print(f"  {out_excel}")