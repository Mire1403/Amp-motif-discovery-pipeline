"""
Usage:
    python 13_colabfold_input.py
    python 13_colabfold_input.py --min_len 6 --max_len 50
    python 13_colabfold_input.py --test
"""

from __future__ import annotations

import argparse
import logging
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

PROJECT_ROOT = _find_project_root()
FMAP_DIR     = PROJECT_ROOT / "results" / "09_motif_mechanism_input" / "01_fmap_input"
LOGS_DIR     = PROJECT_ROOT / "logs"

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# =====================================================
# LOGGING
# =====================================================

def setup_logging(out_dir: Path) -> None:
    for d in [out_dir, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "13_colabfold_input.log"
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
# HELPERS
# =====================================================

def clean_sequence(seq: str) -> str:
    """Replace non-standard AAs with L (leucine) — consistent with script 12."""
    return "".join(aa if aa in VALID_AA else "L" for aa in str(seq).upper())

# =====================================================
# CLI
# =====================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ColabFold FASTA from FMAP sequence selection.")
    parser.add_argument("--min_len", type=int, default=6,
                        help="Minimum sequence length to include (default 6)")
    parser.add_argument("--max_len", type=int, default=50,
                        help="Maximum sequence length to include (default 50)")
    parser.add_argument("--dedup",   action="store_true", default=True,
                        help="Remove duplicate sequences (default True)")
    parser.add_argument("--test",    action="store_true",
                        help="Output to results/09_fmap_input_test/02_colabfold_input/")
    return parser.parse_args()

# =====================================================
# MAIN
# =====================================================

def main() -> None:
    args    = parse_args()
    out_dir = PROJECT_ROOT / (
        "results/09_motif_mechanism_input_test/02_colabfold_input"
        if args.test else
        "results/09_motif_mechanism_input/02_colabfold_input"
    )
    setup_logging(out_dir)
    logger.info("ColabFold FASTA generation started.")
    logger.info("min_len=%d  max_len=%d  dedup=%s",
                args.min_len, args.max_len, args.dedup)

    # ---- load input ----
    input_csv = FMAP_DIR / "motif_selected_sequences.csv"
    if not input_csv.exists():
        logger.error("Input not found: %s — run 12_fmap_input.py first.",
                     input_csv)
        sys.exit(1)

    df = pd.read_csv(input_csv)
    logger.info("Loaded %d sequences from %s", len(df), input_csv.name)

    # ---- validate columns ----
    required = ["Peptide_ID", "Sequence_window"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        logger.error("Missing columns: %s", missing)
        sys.exit(1)

    # ---- clean sequences ----
    df["seq_clean"] = df["Sequence_window"].map(clean_sequence)

    # ---- filter by length ----
    n_before = len(df)
    df = df[df["seq_clean"].str.len().between(args.min_len, args.max_len)]
    n_filtered = n_before - len(df)
    if n_filtered > 0:
        logger.info("Removed %d sequences outside length [%d, %d]",
                    n_filtered, args.min_len, args.max_len)

    # ---- deduplicate ----
    if args.dedup:
        n_before = len(df)
        df = df.drop_duplicates(subset=["seq_clean"])
        n_dup = n_before - len(df)
        if n_dup > 0:
            logger.info("Removed %d duplicate sequences", n_dup)

    logger.info("Final sequences for ColabFold: %d", len(df))

    # ---- write FASTA ----
    # header format: >PeptideID|FamilyID|Motif
    fasta_path = out_dir / "colabfold_input.fasta"
    n_written  = 0

    with open(fasta_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            pid    = str(row["Peptide_ID"])
            seq    = row["seq_clean"]
            family = str(row["Family_ID"]) \
                     if "Family_ID" in row and pd.notna(row["Family_ID"]) else "NA"
            motif  = str(row["Motif"]) \
                     if "Motif" in row and pd.notna(row["Motif"]) else "NA"
            f.write(f">{pid}|{family}|{motif}\n{seq}\n\n")
            n_written += 1

    logger.info("FASTA written: %s  (%d sequences)", fasta_path.name, n_written)

    # ---- save metadata table ----
    meta_path = out_dir / "colabfold_sequence_metadata.csv"
    df[["Peptide_ID","Family_ID","Motif","seq_clean","Specificity","ER"]].to_csv(
        meta_path, index=False)
    logger.info("Metadata saved: %s", meta_path.name)

    # ---- summary ----
    logger.info("\n=== Summary ===")
    logger.info("  Input sequences:    %d", n_before if not args.dedup else len(df) + n_dup if 'n_dup' in dir() else len(df))
    logger.info("  Output sequences:   %d", n_written)
    logger.info("  Length range:       %d - %d aa",
                df["seq_clean"].str.len().min(),
                df["seq_clean"].str.len().max())
    logger.info("  Mean length:        %.1f aa",
                df["seq_clean"].str.len().mean())
    if "Family_ID" in df.columns:
        logger.info("  Families covered:   %d", df["Family_ID"].nunique())

    logger.info("ColabFold FASTA generation completed.")
    logger.info("Output: %s", out_dir)


if __name__ == "__main__":
    main()