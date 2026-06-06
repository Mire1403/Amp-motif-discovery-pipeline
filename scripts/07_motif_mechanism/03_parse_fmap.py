"""
Usage:
    python 14_fmap_parse.py
    python 14_fmap_parse.py --test
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd

# =====================================================
# PROJECT ROOT
# =====================================================

def _find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find project root.")

PROJECT_ROOT  = _find_project_root()
FMAP_INPUT_DIR  = PROJECT_ROOT / "results" / "09_motif_mechanism_input"  / "01_fmap_input"
FMAP_OUTPUT_DIR = PROJECT_ROOT / "results" / "10_motif_mechanism_output" / "01_fmap_parsed"
LOGS_DIR        = PROJECT_ROOT / "logs"

# =====================================================
# LOGGING
# =====================================================

def setup_logging(out_dir: Path) -> None:
    for d in [out_dir, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "14_fmap_parse.log"
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

logger = logging.getLogger(__name__)

# =====================================================
# PARSER
# =====================================================

def build_row(peptide_id, sequence, membrane_energy, fpen, h):
    return {
        "Peptide_ID":              peptide_id,
        "Sequence":                sequence,
        "Membrane_Binding_Energy": membrane_energy,
        "Helix_Start":             h.get("start"),
        "Helix_End":               h.get("end"),
        "Helix_length":            (h.get("end") - h.get("start") + 1)
                                   if h.get("start") and h.get("end") else None,
        "Stability_Water":         h.get("stability_water"),
        "Stability_Bound_Coil":    h.get("stability_bound"),
        "Transfer_Energy":         h.get("transfer_energy"),
        "Tilt_Angle":              h.get("tilt_angle"),
        "Depth_Thickness":         h.get("depth_thickness"),
        "Fpen":                    fpen,
        "Is_False_Helix":          h.get("is_false", "FALSE"),
        "Is_Recalculated":         h.get("recalculated", "FALSE"),
    }


def parse_fmap_output(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data, no_helix = [], []

    peptide_id, sequence        = None, None
    membrane_energy, fpen       = None, None
    helices: dict               = {}
    current_helix_idx           = None
    recalculated_helices: set   = set()

    def flush(pid, seq, me, fp, hx, recalc):
        if pid is None:
            return
        if hx:
            for idx, h in hx.items():
                h["recalculated"] = "TRUE" if idx in recalc else "FALSE"
                data.append(build_row(pid, seq, me, fp, h))
        else:
            no_helix.append({
                "Peptide_ID":              pid,
                "Sequence":                seq,
                "Membrane_Binding_Energy": me,
                "Fpen":                    fp,
                "Note":                    "No solution NMR detectable helix",
            })

    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            # new peptide header
            m = re.match(r"^\d+\s+[\d.]+\s+\w+\s+(P\d+)\s*$", line)
            if m:
                flush(peptide_id, sequence, membrane_energy, fpen,
                      helices, recalculated_helices)
                peptide_id           = m.group(1)
                sequence             = None
                membrane_energy      = None
                fpen                 = None
                helices              = {}
                current_helix_idx    = None
                recalculated_helices = set()
                continue

            # sequence
            if peptide_id and re.fullmatch(r"[A-Z]{4,}", line):
                sequence = line
                continue

            # membrane binding energy
            if membrane_energy is None:
                m = re.search(r"binding energy of peptide:\s*(-?\d+\.?\d*)", line)
                if m:
                    membrane_energy = float(m.group(1))
                    continue

            # fpen
            m = re.match(r"^fpen=\s*(-?\d+\.?\d*)", line)
            if m:
                fpen = float(m.group(1))
                continue

            # solution NMR detectable helix
            m = re.match(
                r"Solution NMR detectable helix\s+(\d+):\s+(\d+)\s*-\s*(\d+)"
                r".*?(-?\d+\.?\d*)\s+and\s+(-?\d+\.?\d*)", line)
            if m:
                idx = int(m.group(1))
                helices[idx] = {
                    "start":           int(m.group(2)),
                    "end":             int(m.group(3)),
                    "stability_water": float(m.group(4)),
                    "stability_bound": float(m.group(5)),
                    "transfer_energy": None,
                    "tilt_angle":      None,
                    "depth_thickness": None,
                    "is_false":        "FALSE",
                    "recalculated":    "FALSE",
                }
                current_helix_idx = idx
                continue

            # false helix flag
            if "false helix" in line.lower():
                if current_helix_idx in helices:
                    helices[current_helix_idx]["is_false"] = "TRUE"
                continue

            # transfer energy
            m = re.search(
                r"Transfer energy of alpha-helix\s+(\d+).*?:\s*(-?\d+\.?\d*)"
                r".*?tilt angle:\s*(-?\d+\.?\d*)\.?,"
                r".*?depth/thickness:\s*(-?\d+\.?\d*)", line)
            if m:
                idx = int(m.group(1))
                if idx in helices:
                    already_set = helices[idx]["transfer_energy"] is not None
                    helices[idx]["transfer_energy"] = float(m.group(2))
                    helices[idx]["tilt_angle"]       = float(m.group(3))
                    helices[idx]["depth_thickness"]  = float(m.group(4))
                    if already_set:
                        recalculated_helices.add(idx)
                        helices[idx]["recalculated"] = "TRUE"
                continue

    flush(peptide_id, sequence, membrane_energy, fpen,
          helices, recalculated_helices)

    return pd.DataFrame(data), pd.DataFrame(no_helix)

# =====================================================
# CLI
# =====================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse FMAP output and merge with sequence metadata.")
    parser.add_argument("--fmap_output", type=Path, default=None,
                        help="Path to fmap_output.txt (default: auto-detect)")
    parser.add_argument("--metadata", type=Path, default=None,
                        help="Path to motif_selected_sequences.csv")
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()

# =====================================================
# MAIN
# =====================================================

def main() -> None:
    args    = parse_args()
    out_dir = PROJECT_ROOT / (
        "results/10_motif_mechanism_output_test/01_fmap_parsed"
        if args.test else
        "results/10_motif_mechanism_output/02_fmap_parsed"
    )
    setup_logging(out_dir)
    logger.info("FMAP output parsing started.")

    # resolve paths
    fmap_path = args.fmap_output or \
                PROJECT_ROOT / "results" / "10_motif_mechanism_output" / "01_fmap_output" / "fmap_output.txt"
    meta_path = args.metadata or \
                FMAP_INPUT_DIR / "motif_selected_sequences.csv"

    if not fmap_path.exists():
        logger.error("FMAP output not found: %s", fmap_path)
        sys.exit(1)
    if not meta_path.exists():
        logger.error("Metadata not found: %s", meta_path)
        sys.exit(1)

    # parse
    logger.info("Parsing: %s", fmap_path.name)
    df, df_nohelix = parse_fmap_output(fmap_path)

    logger.info("Helices parsed:              %d", len(df))
    logger.info("Unique peptides with helices:%d",
                df["Peptide_ID"].nunique() if not df.empty else 0)
    logger.info("False helices:               %d",
                (df["Is_False_Helix"] == "TRUE").sum() if not df.empty else 0)
    logger.info("Recalculated helices:        %d",
                (df["Is_Recalculated"] == "TRUE").sum() if not df.empty else 0)
    logger.info("Peptides with no helix:      %d", len(df_nohelix))

    # merge metadata
    meta_df  = pd.read_csv(meta_path)
    logger.info("Metadata loaded: %d rows", len(meta_df))

    final_df = df.merge(meta_df, on="Peptide_ID", how="left") \
               if not df.empty else df

    # save
    with pd.ExcelWriter(out_dir / "fmap_parsed.xlsx",
                        engine="openpyxl") as w:
        final_df.to_excel(w,    sheet_name="Helices",      index=False)
        df_nohelix.to_excel(w,  sheet_name="No_helix",     index=False)
        # summary per peptide
        if not final_df.empty:
            pep_summary = (
                final_df.groupby("Peptide_ID")
                .agg(
                    n_helices=       ("Helix_Start","count"),
                    membrane_energy= ("Membrane_Binding_Energy","first"),
                    min_transfer_E=  ("Transfer_Energy","min"),
                    mean_tilt=       ("Tilt_Angle","mean"),
                    n_false=         ("Is_False_Helix",
                                      lambda x: (x=="TRUE").sum()),
                ).reset_index()
            )
            pep_summary.to_excel(w, sheet_name="Per_peptide", index=False)

    # also save CSV for downstream scripts
    final_df.to_csv(out_dir / "fmap_parsed.csv", index=False)
    if not df_nohelix.empty:
        df_nohelix.to_csv(out_dir / "fmap_no_helix.csv", index=False)

    logger.info("Saved: %s", out_dir / "fmap_parsed.xlsx")
    logger.info("Total rows: %d", len(final_df))
    logger.info("FMAP parsing completed successfully.")


if __name__ == "__main__":
    main()