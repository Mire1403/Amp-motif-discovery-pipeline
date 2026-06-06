import pandas as pd
import re
import os

# =========================
# CONFIG
# =========================

BASE_DIR = "results/10_motif_mechanism_output/02_ppm_output/results2"

DATAPAR1 = f"{BASE_DIR}/datapar1"
DATAPAR2 = f"{BASE_DIR}/datapar2"
DATASUB1 = f"{BASE_DIR}/datasub1"

OUT_CSV  = f"{BASE_DIR}/ppm_results_clean.csv"
OUT_XLSX = f"{BASE_DIR}/ppm_results_clean.xlsx"

# =========================
# NORMALIZE ID
# =========================

def normalize_id(filename):
    """
    Convert: 15.pdb, 015.pdb, 00015.pdb, P0015.pdb → P0015
    """
    name = filename.replace(".pdb", "").replace("P", "")
    return f"P{int(name):04d}"

# =========================
# PARSE DATAPAR1
# flat membrane proteins
# format: fname;thickness;thickness_sd;tilt;tilt_sd;transfer_energy;
# =========================

def parse_datapar1(file):
    rows = []
    with open(file) as f:
        for line in f:
            parts = line.strip().split(";")
            if len(parts) < 6:
                continue
            try:
                pid = normalize_id(parts[0].strip())
                rows.append({
                    "Peptide_ID":      pid,
                    "Thickness_A":     float(parts[1]),
                    "Thickness_SD":    float(parts[2]),
                    "Tilt_deg":        float(parts[3]),
                    "Tilt_SD":         float(parts[4]),
                    "Transfer_Energy": float(parts[5]),
                    "Membrane_Type":   "flat",
                })
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(rows)  # keep all rows — multi-membrane proteins have multiple lines

# =========================
# PARSE DATAPAR2
# curved membrane proteins
# format: fname;thickness;radius_curvature;tilt;energy_curved;energy_flat;
# =========================

def parse_datapar2(file):
    rows = []
    with open(file) as f:
        for line in f:
            parts = line.strip().split(";")
            if len(parts) < 6:
                continue
            try:
                pid = normalize_id(parts[0].strip())
                rows.append({
                    "Peptide_ID":      pid,
                    "Thickness_A":     float(parts[1]),
                    "Radius_Curv_A":   float(parts[2]),
                    "Tilt_deg":        float(parts[3]),
                    "Energy_Curved":   float(parts[4]),
                    "Energy_Flat":     float(parts[5]),
                    "Membrane_Type":   "curved",
                })
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(rows)

# =========================
# PARSE DATASUB1
# TM proteins only; one row per (protein, chain, TM segment)
# format: fname;chain;tilt; 1(start-end), 2(start-end), ...
# =========================

def parse_datasub1(file):
    rows = []
    with open(file) as f:
        for line in f:
            parts = line.strip().split(";")
            if len(parts) < 4:
                continue
            try:
                pid   = normalize_id(parts[0].strip())
                chain = parts[1].strip()
                tilt  = int(parts[2].strip())
                seg_field = parts[3]

                # extract ALL segments: 1(35-60), 2(72-99), ...
                segments = re.findall(r"(\d+)\(\s*(\d+)\s*-\s*(\d+)\s*\)", seg_field)
                for seg_num, start, end in segments:
                    rows.append({
                        "Peptide_ID":  pid,
                        "Chain":       chain,
                        "Tilt_deg":    tilt,
                        "Segment_Num": int(seg_num),
                        "TM_Start":    int(start),
                        "TM_End":      int(end),
                    })
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(rows)

# =========================
# MAIN
# =========================

if __name__ == "__main__":

    print("Parsing PPM output files...")

    df1 = parse_datapar1(DATAPAR1)
    df2 = parse_datapar2(DATAPAR2)
    df3 = parse_datasub1(DATASUB1)

    print(f"  datapar1 (flat):    {len(df1)} entries")
    print(f"  datapar2 (curved):  {len(df2)} entries")
    print(f"  datasub1 (TM segs): {len(df3)} entries")

    # --------------------------------------------------
    # Combine flat + curved geometry params
    # --------------------------------------------------
    df_geom = pd.concat([df1, df2], ignore_index=True, sort=False)

    # Merge geometry with TM segments
    # outer join: keeps proteins with no TM segments too
    df = df_geom.merge(df3, on="Peptide_ID", how="outer", suffixes=("_geom", "_tm"))

    # --------------------------------------------------
    # Reconcile duplicate Tilt_deg columns that arise
    # from the merge (geom has it, datasub1 has it too)
    # Prefer the TM tilt when available, fall back to geom
    # --------------------------------------------------
    if "Tilt_deg_tm" in df.columns and "Tilt_deg_geom" in df.columns:
        df["Tilt_deg"] = df["Tilt_deg_tm"].combine_first(df["Tilt_deg_geom"])
        df = df.drop(columns=["Tilt_deg_geom", "Tilt_deg_tm"])
    # if only one exists (e.g. datasub1 was empty), rename it cleanly
    elif "Tilt_deg_geom" in df.columns:
        df = df.rename(columns={"Tilt_deg_geom": "Tilt_deg"})
    elif "Tilt_deg_tm" in df.columns:
        df = df.rename(columns={"Tilt_deg_tm": "Tilt_deg"})

    # --------------------------------------------------
    # Sanity check: how decisive are curved assignments?
    # Energy_Delta = Energy_Curved - Energy_Flat
    # Values close to 0 mean borderline assignment
    # --------------------------------------------------
    curved = df[df["Membrane_Type"] == "curved"].copy()
    if not curved.empty and {"Energy_Curved", "Energy_Flat"}.issubset(curved.columns):
        df.loc[df["Membrane_Type"] == "curved", "Energy_Delta"] = (
            curved["Energy_Curved"] - curved["Energy_Flat"]
        )
        print("\n--- Curved assignment confidence (Energy_Curved - Energy_Flat, kcal/mol) ---")
        print(curved["Energy_Curved"].sub(curved["Energy_Flat"]).describe().round(2).to_string())
        n_borderline = (curved["Energy_Curved"].sub(curved["Energy_Flat"]).abs() < 1.0).sum()
        print(f"\nBorderline assignments (|delta| < 1 kcal/mol): {n_borderline} / {len(curved)}")

    # --------------------------------------------------
    # Reorder columns for readability
    # --------------------------------------------------
    preferred_order = [
        "Peptide_ID", "Membrane_Type",
        "Thickness_A", "Thickness_SD",
        "Tilt_deg", "Tilt_SD",
        "Transfer_Energy",
        "Radius_Curv_A", "Energy_Curved", "Energy_Flat", "Energy_Delta",
        "Chain", "Segment_Num", "TM_Start", "TM_End",
    ]
    cols = [c for c in preferred_order if c in df.columns]
    extra = [c for c in df.columns if c not in cols]
    df = df[cols + extra]

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------
    print(f"\nTotal rows after merge:  {len(df)}")
    print(f"Unique peptides:         {df['Peptide_ID'].nunique()}")
    print(f"Flat:    {(df['Membrane_Type'] == 'flat').sum()} rows")
    print(f"Curved:  {(df['Membrane_Type'] == 'curved').sum()} rows")
    print(f"TM segs: {df['Segment_Num'].notna().sum()} rows with segment info")

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    os.makedirs(BASE_DIR, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    df.to_excel(OUT_XLSX, index=False)

    print(f"\n✅ Saved:\n  {OUT_CSV}\n  {OUT_XLSX}")
