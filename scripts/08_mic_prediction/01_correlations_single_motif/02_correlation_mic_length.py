from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "results/12_apex_prediction/03_validation/02_db_matches_with_mic.csv"
OUT_DIR    = PROJECT_ROOT / "results/12_apex_prediction/03_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

# columna correcta amb les noves unitats (µmol/L)
df["DB_MIC_umol"]    = pd.to_numeric(df["DB_MIC_umol"], errors="coerce")
df["Sequence_window"] = df["Sequence_window"].astype(str)
df["seq_len"]         = df["Sequence_window"].str.len()

df = df[df["DB_MIC_umol"].notna()].copy()

# ── summary per motiu ──
grouped = df.groupby("Motif_representative").agg(
    count      = ("DB_MIC_umol", "count"),
    mic_median = ("DB_MIC_umol", "median"),
    mic_mean   = ("DB_MIC_umol", "mean"),
    mic_min    = ("DB_MIC_umol", "min"),
    mic_max    = ("DB_MIC_umol", "max"),
    mic_std    = ("DB_MIC_umol", "std"),
    len_mean   = ("seq_len", "mean"),
    len_min    = ("seq_len", "min"),
    len_max    = ("seq_len", "max"),
).reset_index()

# correlació longitud vs MIC
corr_spearman = df["seq_len"].corr(df["DB_MIC_umol"], method="spearman")
corr_pearson  = df["seq_len"].corr(df["DB_MIC_umol"], method="pearson")

print("\n=== Length vs MIC correlation ===")
print(f"Spearman: {corr_spearman:.3f}")
print(f"Pearson:  {corr_pearson:.3f}")

grouped["log_mic"] = np.log10(grouped["mic_median"].clip(lower=1e-6))

out_summary = OUT_DIR / "04_motif_mic_summary.csv"
out_full    = OUT_DIR / "05_length_vs_mic_full.csv"

grouped.to_csv(out_summary, index=False)
df.to_csv(out_full, index=False)

print(f"\nSaved:")
print(f"  {out_summary}")
print(f"  {out_full}")

# ── llegeix el full per als plots ──
df = pd.read_csv(out_full)
df = df[df["DB_MIC_umol"].notna()].copy()
df["log_mic"] = np.log10(df["DB_MIC_umol"].clip(lower=1e-6))

# ── scatter: longitud vs MIC (log) ──
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
ax.scatter(df["seq_len"], df["log_mic"], alpha=0.4, s=20,
           color="#5B8DB8", edgecolors="none")
ax.set_xlabel("Sequence length (aa)", fontsize=10)
ax.set_ylabel("log₁₀(MIC µmol/L)", fontsize=10)
ax.set_title(f"Length vs MIC (log scale)\nSpearman ρ={corr_spearman:.3f}",
             fontweight="bold")
ax.annotate(f"n={len(df)}", xy=(0.95, 0.05), xycoords="axes fraction",
            ha="right", fontsize=9)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
out1 = OUT_DIR / "length_vs_mic_scatter.png"
fig.savefig(out1, dpi=300, bbox_inches="tight"); plt.close()

# ── boxplot per bins de longitud ──
df["len_bin"] = pd.cut(df["seq_len"], bins=[0, 10, 15, 20, 25, 30, 100],
                        labels=["1-10","11-15","16-20","21-25","26-30",">30"])
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
groups  = [df[df["len_bin"]==b]["log_mic"].dropna().values
           for b in df["len_bin"].cat.categories]
labels  = [str(b) for b in df["len_bin"].cat.categories]
bp = ax.boxplot(groups, tick_labels=labels, patch_artist=True,
                medianprops=dict(color="black", lw=1.5))
for patch in bp["boxes"]:
    patch.set_facecolor("#5B8DB8"); patch.set_alpha(0.7)
ax.set_xlabel("Sequence length bin (aa)", fontsize=10)
ax.set_ylabel("log₁₀(MIC µmol/L)", fontsize=10)
ax.set_title("MIC distribution by length bins", fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
out2 = OUT_DIR / "length_vs_mic_boxplot.png"
fig.savefig(out2, dpi=300, bbox_inches="tight"); plt.close()

# ── línia: MIC mitjà per longitud ──
mean_by_len = df.groupby("seq_len")["log_mic"].mean()
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
ax.plot(mean_by_len.index, mean_by_len.values,
        color="#E07B54", lw=1.5, marker="o", markersize=4)
ax.set_xlabel("Sequence length (aa)", fontsize=10)
ax.set_ylabel("Mean log₁₀(MIC µmol/L)", fontsize=10)
ax.set_title("Mean MIC vs sequence length", fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
out3 = OUT_DIR / "length_vs_mic_mean.png"
fig.savefig(out3, dpi=300, bbox_inches="tight"); plt.close()

print("\nSaved plots:")
print(out1)
print(out2)
print(out3)