"""
Generate all pairwise combinations of consensus motifs for APEX prediction.

3-category version:
    - Carpet
    - Barrel_stave
    - Toroidal_pore

For each pair (Motif1, Motif2) — including same motif with itself:
    Sequence_A = Motif1 + Motif2
    Sequence_B = Motif2 + Motif1  (only if Motif1 != Motif2)

Filters:
    - Combined length <= 50 aa (APEX hard limit)
    - Uses Motif_representative sequence directly

Output:
    - 01_combinations.xlsx
    - 02_apex_input_combinations.txt
    - 03_apex_mapping_combinations.csv
"""

from pathlib import Path
from itertools import combinations_with_replacement
import pandas as pd

# =========================
# CONFIG
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONSENSUS_FILE = (
    PROJECT_ROOT
    / "results/13_motif_combinations/00_consensus_motifs"
    / "01_consensus_motifs_3cat.xlsx"
)

OUT_DIR = (
    PROJECT_ROOT
    / "results/13_motif_combinations/01b_combinations_3cat"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LEN = 50  # APEX hard limit

# =========================
# LOAD CONSENSUS MOTIFS
# =========================

motif_df = pd.read_excel(
    CONSENSUS_FILE,
    sheet_name="Motif_summary"
)

# use motif representative directly
motif_df = motif_df.rename(
    columns={"Motif_representative": "Motif"}
)

# clean sequences
motif_df["Sequence"] = (
    motif_df["Motif"]
    .astype(str)
    .str.upper()
    .str.strip()
)

valid_aa = set("ACDEFGHIKLMNPQRSTVWY")

def clean_seq(seq):
    return "".join(
        aa for aa in str(seq)
        if aa in valid_aa
    )

motif_df["Sequence"] = motif_df["Sequence"].apply(clean_seq)

# remove tiny motifs
motif_df = motif_df[
    motif_df["Sequence"].str.len() >= 4
].reset_index(drop=True)

print("=" * 60)
print("CONSENSUS MOTIFS")
print("=" * 60)

print(f"Loaded motifs: {len(motif_df)}")

print("\nMechanism distribution:")
print(
    motif_df["Mechanism"]
    .value_counts()
    .to_string()
)

print("\nMotif length distribution:")
print(
    motif_df["Sequence"]
    .str.len()
    .describe()
    .round(1)
    .to_string()
)

# =========================
# LOOKUP DICTS
# =========================

mech_map = dict(
    zip(motif_df["Motif"], motif_df["Mechanism"])
)

seq_map = dict(
    zip(motif_df["Motif"], motif_df["Sequence"])
)

mic_map = dict(
    zip(motif_df["Motif"], motif_df["MIC_mean"])
)

motifs = motif_df["Motif"].tolist()

# =========================
# GENERATE COMBINATIONS
# =========================

print("\n" + "=" * 60)
print("GENERATING COMBINATIONS")
print("=" * 60)

rows = []
apex_seqs = []

comb_id = 0
excluded = 0

for m1, m2 in combinations_with_replacement(motifs, 2):

    s1 = seq_map[m1]
    s2 = seq_map[m2]

    mech1 = mech_map[m1]
    mech2 = mech_map[m2]

    # alphabetical standardization
    comb_type = "+".join(
        sorted([mech1, mech2])
    )

    # =====================================
    # ORDER A
    # =====================================

    seq_a = s1 + s2

    if len(seq_a) <= MAX_LEN:

        comb_id += 1
        cid_a = f"COMB{comb_id:05d}A"

        rows.append({
            "Combination_ID": cid_a,
            "Motif_1": m1,
            "Mechanism_1": mech1,
            "Motif_2": m2,
            "Mechanism_2": mech2,
            "Combination_type": comb_type,
            "Order": f"{m1}+{m2}",
            "Sequence": seq_a,
            "Seq_len": len(seq_a),
            "MIC_motif1": mic_map.get(m1),
            "MIC_motif2": mic_map.get(m2),
        })

        apex_seqs.append((cid_a, seq_a))

    else:
        excluded += 1

    # =====================================
    # ORDER B
    # =====================================

    if m1 != m2:

        seq_b = s2 + s1

        if len(seq_b) <= MAX_LEN:

            comb_id += 1
            cid_b = f"COMB{comb_id:05d}B"

            rows.append({
                "Combination_ID": cid_b,
                "Motif_1": m2,
                "Mechanism_1": mech2,
                "Motif_2": m1,
                "Mechanism_2": mech1,
                "Combination_type": comb_type,
                "Order": f"{m2}+{m1}",
                "Sequence": seq_b,
                "Seq_len": len(seq_b),
                "MIC_motif1": mic_map.get(m2),
                "MIC_motif2": mic_map.get(m1),
            })

            apex_seqs.append((cid_b, seq_b))

        else:
            excluded += 1

# =========================
# FINAL DATAFRAME
# =========================

df_comb = pd.DataFrame(rows)

print(f"\nTotal combinations generated: {len(df_comb)}")
print(f"Excluded (>50 aa): {excluded}")

print("\nCombination types:")
print(
    df_comb["Combination_type"]
    .value_counts()
    .to_string()
)

print("\nSequence length distribution:")
print(
    df_comb["Seq_len"]
    .describe()
    .round(1)
    .to_string()
)

# =========================
# SAVE EXCEL
# =========================

print("\n" + "=" * 60)
print("SAVING OUTPUTS")
print("=" * 60)

out_excel = OUT_DIR / "01_combinations.xlsx"

with pd.ExcelWriter(
    out_excel,
    engine="openpyxl"
) as writer:

    # full table
    df_comb.to_excel(
        writer,
        sheet_name="All_combinations",
        index=False
    )

    # separate sheets by combination type
    for ctype in sorted(
        df_comb["Combination_type"].unique()
    ):

        sub = df_comb[
            df_comb["Combination_type"] == ctype
        ]

        if not sub.empty:

            sheet_name = (
                ctype
                .replace("+", "_")
                [:31]
            )

            sub.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False
            )

print(f"\n✅ Excel saved:")
print(out_excel)

# =========================
# SAVE APEX INPUT
# =========================

apex_input_file = (
    OUT_DIR
    / "02_apex_input_combinations.txt"
)

with open(apex_input_file, "w") as f:

    for cid, seq in apex_seqs:
        f.write(seq + "\n")

print(f"\n✅ APEX input saved:")
print(apex_input_file)

print(f"Sequences written: {len(apex_seqs)}")

# =========================
# SAVE MAPPING
# =========================

mapping_df = df_comb.copy()

mapping_df["APEX_row"] = range(
    len(mapping_df)
)

mapping_file = (
    OUT_DIR
    / "03_apex_mapping_combinations.csv"
)

mapping_df.to_csv(
    mapping_file,
    index=False
)

print(f"\n✅ Mapping saved:")
print(mapping_file)

# =========================
# DONE
# =========================

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

print("\nNext step:")
print(
    "python scripts/08_mic_prediction/07b_batches.py"
)