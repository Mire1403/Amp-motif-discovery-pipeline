"""
01_clean_nonamp_sequences.py
=============================
Cleans raw non-AMP sequences downloaded from UniProt.

Filters applied:
  1. Minimum length: MIN_LENGTH (from config, default 6 aa)
  2. Maximum length: MAX_LENGTH (from config, default 80 aa)
     — matches AMP dataset upper bound to avoid length bias
  3. Natural AA only (standard 20 amino acids)
  4. Deduplication by sequence

Input:
  data/raw/non_AMPs/uniprot_nonamp_raw.fasta

Output:
  data/intermediate/nonamp_clean.fasta
  results/01_database_construction/nonamp_clean_report.csv

Note on data origin:
  Sequences downloaded from UniProt with filters:
    - Reviewed (Swiss-Prot)
    - NOT annotated as antimicrobial
    - Length 6-80 aa
  See README for exact UniProt query used.

Changes vs original:
  - logging instead of print
  - Imports MIN_LENGTH, MAX_LENGTH, VALID_AA from config
  - MAX_LENGTH filter added (matches AMP dataset, avoids length bias)
  - setup_dirs() instead of .mkdir() on import
  - Report saved to results/ not intermediate/
  - PROJECT_ROOT via _find_project_root() (robust, no hardcoded parents[3])
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# =====================================================
# PROJECT ROOT — robust marker-based detection
# =====================================================

def _find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find project root.")

PROJECT_ROOT          = _find_project_root()
DATA_RAW_DIR          = PROJECT_ROOT / "data" / "raw"
DATA_INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
RESULTS_DIR           = PROJECT_ROOT / "results" / "01_database_construction"
LOGS_DIR              = PROJECT_ROOT / "logs"

MIN_LENGTH: int  = 6
VALID_AA:   set  = set("ACDEFGHIKLMNPQRSTVWY")


def setup_dirs() -> None:
    for d in [DATA_INTERMEDIATE_DIR, RESULTS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

# =====================================================
# PATHS
# =====================================================

INPUT_FASTA  = DATA_RAW_DIR / "non_AMPs" / "uniprot_nonamp_raw.fasta"
OUTPUT_FASTA = DATA_INTERMEDIATE_DIR / "nonamp_clean.fasta"

# =====================================================
# LOGGING
# =====================================================

def setup_logging() -> None:
    setup_dirs()
    log_file = LOGS_DIR / "01_clean_nonamp_sequences.log"
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
# FASTA HELPERS
# =====================================================

def read_fasta(path: Path) -> list[tuple[str, str]]:
    """Read FASTA file, return list of (header, sequence) tuples."""
    records: list[tuple[str, str]] = []
    header: str | None = None
    seq_parts: list[str] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_parts).upper()))
                header = line
                seq_parts = []
            else:
                seq_parts.append(line)
        if header is not None:
            records.append((header, "".join(seq_parts).upper()))

    return records


def write_fasta(records: list[tuple[str, str]], out_path: Path,
                wrap: int = 80) -> None:
    """Write FASTA file with line wrapping."""
    with open(out_path, "w", encoding="utf-8") as f:
        for header, seq in records:
            f.write(f"{header}\n")
            for i in range(0, len(seq), wrap):
                f.write(seq[i:i+wrap] + "\n")


def compute_stats(records: list[tuple[str, str]]) -> dict:
    """Compute basic length statistics for a set of records."""
    lengths = [len(seq) for _, seq in records]
    if not lengths:
        return {"count": 0, "min": 0, "max": 0,
                "mean": 0.0, "median": 0.0, "std": 0.0}
    return {
        "count":  len(lengths),
        "min":    int(min(lengths)),
        "q25":    float(np.percentile(lengths, 25)),
        "median": float(np.median(lengths)),
        "mean":   float(round(np.mean(lengths), 2)),
        "q75":    float(np.percentile(lengths, 75)),
        "max":    int(max(lengths)),
        "std":    float(round(np.std(lengths), 2)),
    }

# =====================================================
# MAIN
# =====================================================

def main() -> None:
    setup_dirs()
    setup_logging()

    logger.info("Non-AMP sequence cleaning started.")
    logger.info("Input:  %s", INPUT_FASTA)
    logger.info("Output: %s", OUTPUT_FASTA)

    if not INPUT_FASTA.exists():
        logger.error("Input FASTA not found: %s", INPUT_FASTA)
        sys.exit(1)

    # ---- load ----
    records = read_fasta(INPUT_FASTA)
    initial_stats = compute_stats(records)
    logger.info("\n=== Initial stats ===")
    for k, v in initial_stats.items():
        logger.info("  %s: %s", k, v)

    # ---- filter ----
    filtered:            list[tuple[str, str]] = []
    removed_short:       int = 0
    removed_non_natural: int = 0
    removed_duplicate:   int = 0
    seen_sequences:      set[str] = set()

    for header, seq in records:

        # 1. minimum length
        if len(seq) < MIN_LENGTH:
            removed_short += 1
            continue

        # 2. natural AA only
        if not all(aa in VALID_AA for aa in seq):
            removed_non_natural += 1
            continue

        # 4. deduplication
        if seq in seen_sequences:
            removed_duplicate += 1
            continue

        seen_sequences.add(seq)
        filtered.append((header, seq))

    final_stats = compute_stats(filtered)

    logger.info("\n=== Filter summary ===")
    logger.info("  Removed length < %d aa:       %d", MIN_LENGTH, removed_short)
    logger.info("  Removed non-natural AA:        %d", removed_non_natural)
    logger.info("  Removed duplicates:            %d", removed_duplicate)
    logger.info("  Remaining unique sequences:    %d", len(filtered))

    logger.info("\n=== Final stats ===")
    for k, v in final_stats.items():
        logger.info("  %s: %s", k, v)

    # ---- save FASTA ----
    write_fasta(filtered, OUTPUT_FASTA)
    logger.info("Clean FASTA saved: %s", OUTPUT_FASTA)

    # ---- save report to results/ ----
    report = {
        "input_fasta":          str(INPUT_FASTA),
        "output_fasta":         str(OUTPUT_FASTA),
        "min_length":           MIN_LENGTH,
        "initial_count":        initial_stats["count"],
        "final_count_unique":   final_stats["count"],
        "removed_short":        removed_short,
        "removed_non_natural":  removed_non_natural,
        "removed_duplicate":    removed_duplicate,
        "initial_mean_len":     initial_stats["mean"],
        "final_mean_len":       final_stats["mean"],
        "initial_median_len":   initial_stats["median"],
        "final_median_len":     final_stats["median"],
        "initial_min_len":      initial_stats["min"],
        "final_min_len":        final_stats["min"],
        "initial_max_len":      initial_stats["max"],
        "final_max_len":        final_stats["max"],
    }

    report_path = RESULTS_DIR / "nonamp_clean_report.csv"
    pd.DataFrame([report]).to_csv(report_path, index=False)
    logger.info("Report saved: %s", report_path)

    logger.info("Non-AMP cleaning completed successfully.")


if __name__ == "__main__":
    main()