"""
Generate APEX input file from motif sequence windows.

Takes the sequences already used for FMAP and PPM (Sequence_window, max 35 aa)
and writes them to a plain text file for APEX prediction.

Output: one sequence per line, as required by APEX predict.py
Also saves a mapping CSV (Peptide_ID → Sequence_window → Motif info)
so results can be merged back with mechanism labels.
"""

from pathlib import Path
import pandas as pd

# =========================
# CONFIG
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MOTIF_FILE = (
    PROJECT_ROOT / "results/09_motif_mechanism_input/01_fmap_input"
    / "motif_selected_sequences.csv"
)

OUT_DIR  = PROJECT_ROOT / "results/12_apex_prediction/01_input"
OUT_DIR.mkdir(parents=True, exist_ok=True)

APEX_INPUT   = OUT_DIR / "apex_sequences.txt"
MAPPING_FILE = OUT_DIR / "apex_sequence_mapping.csv"

MAX_LEN = 50  # APEX hard limit

# =========================
# LOAD
# =========================

df = pd.read_csv(MOTIF_FILE)
print(f"Loaded {len(df)} sequences from {df['Motif_representative'].nunique()} motifs")

# =========================
# FILTER
# =========================

# remove sequences longer than APEX limit
df["seq_len"] = df["Sequence_window"].str.len()
too_long = df[df["seq_len"] > MAX_LEN]
if len(too_long) > 0:
    print(f"WARNING: {len(too_long)} sequences exceed {MAX_LEN} aa — excluded")
    print(too_long[["Peptide_ID", "Sequence_window", "seq_len"]].to_string())

df = df[df["seq_len"] <= MAX_LEN].copy()

# remove sequences with invalid amino acids
valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
def is_valid(seq):
    return all(aa in valid_aa for aa in str(seq).upper())

df["valid"] = df["Sequence_window"].apply(is_valid)
invalid = df[~df["valid"]]
if len(invalid) > 0:
    print(f"WARNING: {len(invalid)} sequences contain invalid amino acids — excluded")

df = df[df["valid"]].copy()
df["Sequence_window"] = df["Sequence_window"].str.upper()

print(f"\nSequences after filtering: {len(df)}")
print(f"Motifs represented:        {df['Motif_representative'].nunique()}")
print(f"Families represented:      {df['Family_ID'].nunique()}")
print(f"\nSequence length stats:")
print(df["seq_len"].describe().round(1).to_string())

# =========================
# WRITE APEX INPUT
# one sequence per line, no headers, no IDs
# =========================

with open(APEX_INPUT, "w") as f:
    for seq in df["Sequence_window"]:
        f.write(seq + "\n")

print(f"\n✅ APEX input written: {APEX_INPUT}")
print(f"   {len(df)} sequences")

# =========================
# SAVE MAPPING
# APEX output will be in the same row order as the input,
# so we need this to merge predictions back
# =========================

mapping = df[[
    "Peptide_ID", "Family_ID", "Motif_representative",
    "Motif_variant", "Sequence_window", "seq_len",
    "Start", "End", "p-value", "Specificity"
]].reset_index(drop=True)
mapping.index.name = "APEX_row"
mapping = mapping.reset_index()  # APEX_row = 0-based row in input file

mapping.to_csv(MAPPING_FILE, index=False)
print(f"✅ Mapping saved:     {MAPPING_FILE}")

# =========================
# SUMMARY PER MOTIF
# =========================

print(f"\nSequences per motif (top 10):")
print(df.groupby("Motif_representative").size()
      .sort_values(ascending=False).head(10).to_string())