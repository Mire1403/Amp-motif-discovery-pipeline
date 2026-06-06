"""
Process new FMAP and PPM results (234 consensus motif sequences).

Steps:
  1. Load new FMAP CSV → renumber IDs from P0631
  2. Load new PPM CSV  → renumber IDs + rename old-format columns
  3. Classify mechanism (same thresholds as before)
  4. Save classified CSVs
  5. Generate APEX input for the 234 new sequences

ID offset: existing data has P0001–P0630 (630 peptides)
           new data starts at P0631
"""

from pathlib import Path
import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# new raw files (old-format IDs, P0001-P0234)
NEW_FMAP_FILE = (
    PROJECT_ROOT / "results/10_motif_mechanism_output"
    / "01_fmap_output" / "fmap_final_results.csv"
)
NEW_PPM_FILE = (
    PROJECT_ROOT / "results/10_motif_mechanism_output"
    / "02_ppm_output" / "ppm_results_clean.csv"
)

# existing classified files (for reference / append)
EXISTING_FMAP = (
    PROJECT_ROOT / "results/11_mechanism_analysis"
    / "1_motif_activity_analysis/fmap2/01_fmap_classified.csv"
)
EXISTING_PPM = (
    PROJECT_ROOT / "results/11_mechanism_analysis"
    / "1_motif_activity_analysis/ppm2/01_ppm_classified.csv"
)

OUT_DIR_FMAP = (
    PROJECT_ROOT / "results/11_mechanism_analysis"
    / "1_motif_activity_analysis/fmap2"
)
OUT_DIR_PPM = (
    PROJECT_ROOT / "results/11_mechanism_analysis"
    / "1_motif_activity_analysis/ppm2"
)
APEX_OUT_DIR = PROJECT_ROOT / "results/12_apex_prediction/01b_input"

OUT_DIR_FMAP.mkdir(parents=True, exist_ok=True)
OUT_DIR_PPM.mkdir(parents=True, exist_ok=True)
APEX_OUT_DIR.mkdir(parents=True, exist_ok=True)

ID_OFFSET = 630

# =========================
# MECHANISM THRESHOLDS
# (same as existing pipeline)
# =========================

def classify_mechanism_fmap(row):
    tilt  = row["Tilt_Angle"]
    depth = row["Depth_Thickness"]
    if pd.isna(tilt) or pd.isna(depth):
        return "Ambiguous"
    if tilt > 70.0:
        return "Carpet"
    if tilt < 35.0 and depth > 5.0:
        return "Barrel_stave"
    if 35.0 <= tilt <= 70.0 and 2.0 <= depth <= 25.0:
        return "Toroidal_pore"
    return "Ambiguous"

def classify_mechanism_ppm(row):
    tilt  = row["Tilt_deg"]
    depth = row["Thickness_A"]
    if pd.isna(tilt) or pd.isna(depth):
        return "Ambiguous"
    if tilt > 70.0:
        return "Carpet"
    if tilt < 35.0 and depth > 6.0:
        return "Barrel_stave"
    if 35.0 <= tilt <= 70.0 and 3.0 <= depth <= 25.0:
        return "Toroidal_pore"
    return "Ambiguous"

def renumber_ids(df, offset):
    """P0001 → P0632 if offset=631, etc."""
    df = df.copy()
    df["Peptide_ID"] = df["Peptide_ID"].apply(
        lambda pid: f"P{int(str(pid).replace('P','')) + offset:04d}"
    )
    return df

# =========================
# LOAD & PROCESS FMAP
# =========================

print("=== Processing new FMAP ===")
fmap_raw = pd.read_csv(NEW_FMAP_FILE)
print(f"  Loaded {len(fmap_raw)} rows, {fmap_raw['Peptide_ID'].nunique()} peptides")

fmap_raw = renumber_ids(fmap_raw, ID_OFFSET)
print(f"  IDs renumbered: {fmap_raw['Peptide_ID'].min()} – {fmap_raw['Peptide_ID'].max()}")

# resolve Sequence_x/y from merge if present
for col in ["Sequence", "Motif"]:
    if col not in fmap_raw.columns:
        for candidate in [f"{col}_x", f"{col}_y"]:
            if candidate in fmap_raw.columns:
                fmap_raw[col] = fmap_raw[candidate]
                break

# numeric cleaning
for col in ["Transfer_Energy", "Tilt_Angle", "Depth_Thickness",
            "Membrane_Binding_Energy", "Helix_Start", "Helix_End"]:
    if col in fmap_raw.columns:
        fmap_raw[col] = pd.to_numeric(fmap_raw[col], errors="coerce")

fmap_raw["Is_False"] = fmap_raw["Is_False_Helix"].astype(str).str.upper() == "TRUE"

# select primary helix per peptide
fmap_clean = fmap_raw.dropna(subset=["Transfer_Energy", "Tilt_Angle", "Depth_Thickness"]).copy()
fmap_clean["_priority"] = fmap_clean["Is_False"].astype(int)
fmap_primary = (
    fmap_clean
    .sort_values(["_priority", "Transfer_Energy"])
    .groupby("Peptide_ID", as_index=False)
    .first()
    .drop(columns=["_priority"])
)

fmap_primary["Mechanism"] = fmap_primary.apply(classify_mechanism_fmap, axis=1)

print(f"  Primary helices: {len(fmap_primary)}")
print(f"  Mechanism distribution:")
print(fmap_primary["Mechanism"].value_counts().to_string())

# =========================
# LOAD & PROCESS PPM
# =========================

print("\n=== Processing new PPM ===")
ppm_raw = pd.read_csv(NEW_PPM_FILE)
print(f"  Loaded {len(ppm_raw)} rows, {ppm_raw['Peptide_ID'].nunique()} peptides")

ppm_raw = renumber_ids(ppm_raw, ID_OFFSET)

def unify_ppm_row(row):
    """Return unified Thickness_A, Tilt_deg, Best_Energy, Membrane_Type."""
    # flat (datapar1): Depth=Thickness_A, Angle=Tilt_deg, DeltaG1=Transfer_Energy
    if pd.notna(row.get("Depth")) and pd.notna(row.get("Angle")):
        return pd.Series({
            "Thickness_A":     row["Depth"],
            "Tilt_deg":        row["Angle"],
            "Transfer_Energy": row.get("DeltaG1"),
            "Thickness_SD":    row.get("Tilt"),
            "Best_Energy":     row.get("DeltaG1"),
            "Membrane_Type":   "flat",
            "Radius_Curv_A":   np.nan,
            "Energy_Curved":   np.nan,
            "Energy_Flat":     np.nan,
        })
    # curved (datapar2): Param1=Thickness_A, Param3=Tilt_deg, Energy1=Energy_Curved
    elif pd.notna(row.get("Param1")) and pd.notna(row.get("Param3")):
        return pd.Series({
            "Thickness_A":     row["Param1"],
            "Tilt_deg":        row["Param3"],
            "Transfer_Energy": np.nan,
            "Thickness_SD":    np.nan,
            "Best_Energy":     row.get("Energy1"),
            "Membrane_Type":   "curved",
            "Radius_Curv_A":   row.get("Param2"),
            "Energy_Curved":   row.get("Energy1"),
            "Energy_Flat":     row.get("Energy2"),
        })
    else:
        return pd.Series({
            "Thickness_A": np.nan, "Tilt_deg": np.nan,
            "Transfer_Energy": np.nan, "Thickness_SD": np.nan,
            "Best_Energy": np.nan, "Membrane_Type": "unknown",
            "Radius_Curv_A": np.nan, "Energy_Curved": np.nan,
            "Energy_Flat": np.nan,
        })

unified_cols = ppm_raw.apply(unify_ppm_row, axis=1)
ppm_unified  = pd.concat([ppm_raw[["Peptide_ID"]].reset_index(drop=True),
                           unified_cols.reset_index(drop=True)], axis=1)

# keep one row per peptide (take the one with best energy if duplicates)
ppm_unified["Best_Energy"] = pd.to_numeric(ppm_unified["Best_Energy"], errors="coerce")
ppm_primary = (
    ppm_unified.sort_values("Best_Energy")
    .groupby("Peptide_ID", as_index=False)
    .first()
)

ppm_primary["Mechanism"] = ppm_primary.apply(classify_mechanism_ppm, axis=1)

print(f"  Primary rows: {len(ppm_primary)}")
print(f"  Mechanism distribution:")
print(ppm_primary["Mechanism"].value_counts().to_string())

# =========================
# APPEND TO EXISTING CLASSIFIED FILES
# =========================

fmap_existing = pd.read_csv(EXISTING_FMAP)
ppm_existing  = pd.read_csv(EXISTING_PPM)

fmap_combined = pd.concat([fmap_existing, fmap_primary], ignore_index=True)
ppm_combined  = pd.concat([ppm_existing,  ppm_primary],  ignore_index=True)

print(f"\nCombined FMAP: {len(fmap_existing)} + {len(fmap_primary)} = {len(fmap_combined)}")
print(f"Combined PPM:  {len(ppm_existing)} + {len(ppm_primary)} = {len(ppm_combined)}")

fmap_combined.to_csv(OUT_DIR_FMAP / "01_fmap_classified_extended.csv", index=False)
ppm_combined.to_csv(OUT_DIR_PPM   / "01_ppm_classified_extended.csv",  index=False)
print(f"\n✅ Extended classified files saved")

# =========================
# GENERATE APEX INPUT
# sequences come from Sequence_x in the FMAP file
# =========================

print("\n=== Generating APEX input for new sequences ===")

MAX_LEN  = 50
valid_aa = set("ACDEFGHIKLMNPQRSTVWY")

# get one sequence per peptide from FMAP raw
seq_col  = "Sequence_x" if "Sequence_x" in fmap_raw.columns else "Sequence"
motif_col = "Motif" if "Motif" in fmap_raw.columns else None

keep_cols = ["Peptide_ID", seq_col]
if motif_col:
    keep_cols.append(motif_col)

seq_df = fmap_raw[keep_cols].drop_duplicates("Peptide_ID").copy()
seq_df = seq_df.rename(columns={seq_col: "Sequence"})

def is_valid(seq):
    return all(aa in valid_aa for aa in seq)

seq_df["Sequence"] = seq_df["Sequence"].astype(str).str.upper()

seq_df = seq_df[
    seq_df["Sequence"].str.len().le(MAX_LEN) &
    seq_df["Sequence"].apply(is_valid)
].reset_index(drop=True)

seq_df.index.name = "APEX_row_new"
seq_df = seq_df.reset_index()

apex_input_file  = APEX_OUT_DIR / "apex_sequences_new.txt"
mapping_file_new = APEX_OUT_DIR / "apex_sequence_mapping_new.csv"

with open(apex_input_file, "w") as f:
    for seq in seq_df["Sequence"]:
        f.write(seq + "\n")

seq_df.to_csv(mapping_file_new, index=False)

print(f"  {len(seq_df)} sequences written")
print(f"  Length: mean={seq_df['Sequence'].str.len().mean():.1f} "
      f"min={seq_df['Sequence'].str.len().min()} "
      f"max={seq_df['Sequence'].str.len().max()}")
print(f"\n✅ APEX input: {apex_input_file}")
print(f"✅ Mapping:    {mapping_file_new}")
print(f"\nNext step:")
print(f"  cd ~/apex_local/apex")
print(f"  python predict.py {apex_input_file}")