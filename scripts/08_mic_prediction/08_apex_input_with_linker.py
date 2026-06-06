"""
Generate APEX inputs for motif combinations WITH linker.

4 inputs:
  2cat + GGG  (Carpet/Pore × GGG linker)
  2cat + AAA  (Carpet/Pore × AAA linker)
  3cat + GGG  (Carpet/Toroidal/Barrel × GGG linker)
  3cat + AAA  (Carpet/Toroidal/Barrel × AAA linker)
"""

from pathlib import Path
from itertools import combinations_with_replacement
import pandas as pd

# =========================
# CONFIG
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONSENSUS_2CAT = (
    PROJECT_ROOT / "results/13_motif_combinations/00_consensus_motifs"
    / "02_consensus_motifs_reduced.xlsx"
)
CONSENSUS_3CAT = (
    PROJECT_ROOT / "results/13_motif_combinations/00_consensus_motifs"
    / "01_consensus_motifs_3cat.xlsx"
)

OUT_DIR = PROJECT_ROOT / "results/13_motif_combinations/03_combinations_linker"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LEN  = 50
LINKERS  = {"GGG": "GGG", "AAA": "AAA"}
valid_aa = set("ACDEFGHIKLMNPQRSTVWY")

# =========================
# HELPERS
# =========================

def clean_seq(seq):
    return "".join(aa for aa in str(seq).upper() if aa in valid_aa)

def load_motifs(consensus_file):
    df = pd.read_excel(consensus_file, sheet_name="Motif_summary")
    df = df.rename(columns={"Motif_representative": "Motif"})
    df["Sequence"] = df["Motif"].apply(clean_seq)
    df = df[df["Sequence"].str.len() >= 4].reset_index(drop=True)
    return df

def generate_combinations(motif_df, linker_seq, linker_name, cat_label):
    mech_map = dict(zip(motif_df["Motif"], motif_df["Mechanism"]))
    seq_map  = dict(zip(motif_df["Motif"], motif_df["Sequence"]))
    mic_map  = dict(zip(motif_df["Motif"], motif_df["MIC_mean"]))
    motifs   = motif_df["Motif"].tolist()

    rows      = []
    apex_seqs = []
    comb_id   = 0

    for m1, m2 in combinations_with_replacement(motifs, 2):
        s1, s2 = seq_map[m1], seq_map[m2]
        mech1, mech2 = mech_map[m1], mech_map[m2]
        comb_type = "+".join(sorted([mech1, mech2]))

        for seq, m_a, m_b, mech_a, mech_b, order_label in [
            (s1 + linker_seq + s2, m1, m2, mech1, mech2, "A"),
            (s2 + linker_seq + s1, m2, m1, mech2, mech1, "B"),
        ]:
            if order_label == "B" and m1 == m2:
                continue
            if len(seq) > MAX_LEN:
                continue
            comb_id += 1
            cid = f"{cat_label}{linker_name}{comb_id:05d}{order_label}"
            rows.append({
                "Combination_ID":   cid,
                "Motif_1":          m_a,
                "Mechanism_1":      mech_a,
                "Motif_2":          m_b,
                "Mechanism_2":      mech_b,
                "Combination_type": comb_type,
                "Order":            f"{m_a}+{linker_name}+{m_b}",
                "Linker":           linker_name,
                "Sequence":         seq,
                "Seq_len":          len(seq),
                "MIC_motif1":       mic_map.get(m_a),
                "MIC_motif2":       mic_map.get(m_b),
            })
            apex_seqs.append(seq)

    return pd.DataFrame(rows), apex_seqs

# =========================
# LOAD BOTH CONSENSUS FILES
# =========================

motifs_2cat = load_motifs(CONSENSUS_2CAT)
motifs_3cat = load_motifs(CONSENSUS_3CAT)

print(f"2cat motifs: {len(motifs_2cat)}")
print(motifs_2cat["Mechanism"].value_counts().to_string())
print(f"\n3cat motifs: {len(motifs_3cat)}")
print(motifs_3cat["Mechanism"].value_counts().to_string())

# =========================
# GENERATE ALL 4 COMBINATIONS
# =========================

configs = [
    ("2cat", motifs_2cat, "GGG"),
    ("2cat", motifs_2cat, "AAA"),
    ("3cat", motifs_3cat, "GGG"),
    ("3cat", motifs_3cat, "AAA"),
]

for idx, (cat_label, motif_df, linker_name) in enumerate(configs, start=1):
    linker_seq = LINKERS[linker_name]
    label      = f"{cat_label}_{linker_name}"

    print(f"\n=== {cat_label} + {linker_name} ===")
    df_comb, apex_seqs = generate_combinations(
        motif_df, linker_seq, linker_name, cat_label
    )

    print(f"  Total combinations: {len(df_comb)}")
    for ctype in sorted(df_comb["Combination_type"].unique()):
        n = (df_comb["Combination_type"]==ctype).sum()
        print(f"    {ctype:<35} n={n}")
    print(f"  Length: mean={df_comb['Seq_len'].mean():.1f}  "
          f"min={df_comb['Seq_len'].min()}  max={df_comb['Seq_len'].max()}")

    # Excel
    out_excel = OUT_DIR / f"{idx:02d}_combinations_{label}.xlsx"
    with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
        df_comb.to_excel(writer, sheet_name="All_combinations", index=False)
        for ctype in sorted(df_comb["Combination_type"].unique()):
            sub   = df_comb[df_comb["Combination_type"]==ctype]
            sheet = ctype.replace("+","_")[:31]
            sub.to_excel(writer, sheet_name=sheet, index=False)
    print(f"  ✅ Excel: {out_excel}")

    # APEX input
    apex_txt = OUT_DIR / f"{idx+4:02d}_apex_input_{label}.txt"
    with open(apex_txt, "w") as f:
        for seq in apex_seqs:
            f.write(seq + "\n")
    print(f"  ✅ APEX input: {apex_txt}  ({len(apex_seqs)} sequences)")

    # Mapping
    df_comb["APEX_row"] = range(len(df_comb))
    mapping_file = OUT_DIR / f"{idx+8:02d}_apex_mapping_{label}.csv"
    df_comb.to_csv(mapping_file, index=False)
    print(f"  ✅ Mapping: {mapping_file}")

# =========================
# PRINT NEXT STEPS
# =========================

print(f"\n=== Next steps (run in order) ===")
print(f"cd ~/apex_local/apex")
for idx, (cat_label, _, linker_name) in enumerate(configs, start=1):
    label    = f"{cat_label}_{linker_name}"
    apex_txt = OUT_DIR / f"{idx+4:02d}_apex_input_{label}.txt"
    out_csv  = OUT_DIR / f"Predicted_MICs_{label}.csv"
    print(f"\npython predict.py {apex_txt}")
    print(f"mv Predicted_MICs.csv {out_csv}")