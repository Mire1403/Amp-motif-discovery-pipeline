from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT  = ROOT / "results/colabfold_input"
OUT.mkdir(parents=True, exist_ok=True)

COMB_FILE   = ROOT / "results/13_motif_combinations/01_combinations/04_merged_predicted_MICs.csv"
LINKER_FILE = ROOT / "results/13_motif_combinations/04_linker_analysis_2cat/01_linker_analysis_2cat.xlsx"

CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")

def is_valid(seq):
    return all(aa in CANONICAL for aa in str(seq).upper())

def clean(seq):
    return "".join(aa for aa in str(seq).upper() if aa in CANONICAL)

comb = pd.read_csv(COMB_FILE)
comb["MIC_combined"] = pd.to_numeric(comb["MIC_combined"], errors="coerce")
comb["MIC_delta"]    = pd.to_numeric(comb["MIC_delta"],    errors="coerce")

top10 = comb.nsmallest(10, "MIC_combined").copy()
top10["Seq_clean"] = top10["Sequence"].apply(clean)
top10 = top10[top10["Seq_clean"].apply(is_valid)].head(10).reset_index(drop=True)

print("Top 10 combinations (no linker):")
print(top10[["Motif_1","Motif_2","Seq_clean","MIC_combined","MIC_delta"]].to_string(index=False))

linker_data = pd.read_excel(LINKER_FILE, sheet_name="All_data")
linker_data["MIC_combined"] = pd.to_numeric(linker_data["MIC_combined"], errors="coerce")

def get_linker_seq(motif1, motif2, condition, linker_df):
    mask = ((linker_df["Motif_1"]==motif1) & (linker_df["Motif_2"]==motif2) &
            (linker_df["Condition"]==condition))
    sub = linker_df[mask]
    if len(sub) == 0:
        mask2 = ((linker_df["Motif_1"]==motif2) & (linker_df["Motif_2"]==motif1) &
                 (linker_df["Condition"]==condition))
        sub = linker_df[mask2]
    if len(sub) > 0 and "Sequence" in sub.columns:
        return clean(sub.iloc[0]["Sequence"])
    linker = "AAA" if condition=="AAA" else "GGG"
    return clean(motif1 + linker + motif2)

rows = []
for cond in ["No_linker", "AAA", "GGG"]:
    fname = OUT / f"colabfold_top10_{cond.lower().replace('_','')}.fasta"
    with open(fname, "w") as f:
        for i, row in top10.iterrows():
            m1, m2 = row["Motif_1"], row["Motif_2"]
            if cond == "No_linker":
                seq = row["Seq_clean"]
                mic = row["MIC_combined"]
            else:
                seq = get_linker_seq(m1, m2, cond, linker_data)
                mask = ((linker_data["Motif_1"]==m1) &
                        (linker_data["Motif_2"]==m2) &
                        (linker_data["Condition"]==cond))
                mic_row = linker_data[mask]["MIC_combined"]
                mic = float(mic_row.values[0]) if len(mic_row)>0 else float("nan")

            if not seq or not is_valid(seq):
                print(f"  SKIP {m1}+{m2} ({cond}): invalid seq '{seq}'")
                continue

            header = f">comb{i+1:02d}_{cond}_{m1}_{m2}_MIC{mic:.1f}"
            f.write(header + "\n")
            f.write(seq + "\n")
            rows.append({
                "ID": f"comb{i+1:02d}",
                "Condition": cond,
                "Motif_1": m1,
                "Motif_2": m2,
                "Sequence": seq,
                "Length": len(seq),
                "MIC_predicted": round(mic, 2),
                "MIC_delta": round(float(row["MIC_delta"]), 2) if cond=="No_linker" else float("nan"),
                "FASTA_header": header,
            })
    print(f"\n  {fname.name} — {sum(1 for r in rows if r['Condition']==cond)} seqs")

mapping = pd.DataFrame(rows)
mapping.to_csv(OUT / "colabfold_mapping.csv", index=False)
print(f"\n  colabfold_mapping.csv — {len(mapping)} rows")

print("\n=== Sequences ===")
print(mapping[mapping["Condition"]=="No_linker"][
    ["ID","Motif_1","Motif_2","Sequence","Length","MIC_predicted"]
].to_string(index=False))
print(f"\nOutput: {OUT}")