"""
Usage:
    python 05_clean_master_dataset.py
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd

from config import (
    DATA_FINAL_DIR,
    DATA_INTERMEDIATE_DIR,
    LOGS_DIR,
    RESULTS_DIR,
    setup_dirs,
)

# =====================================================
# LOGGING
# =====================================================

def setup_logging() -> None:
    setup_dirs()
    log_file = LOGS_DIR / "05_clean_master_dataset.log"
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
# NORMALIZATION HELPERS
# =====================================================

def normalize_organism(x) -> str:
    """Clean organism field: remove parenthetical notes, normalize spaces."""
    if pd.isna(x):
        return ""
    s = str(x)
    s = re.sub(r"\(.*?\)", "", s)   # remove (notes)
    s = s.replace("&&", ";")
    s = re.sub(r"\s+", " ", s).strip()
    return s.title()


def normalize_taxonomy(x) -> str:
    """Clean taxonomy field: deduplicate and sort terms."""
    if pd.isna(x):
        return ""
    s = str(x).replace("&&", ";").replace(",", ";")
    s = re.sub(r"\s+", "", s)
    parts = sorted({p for p in s.split(";") if p})
    return ";".join(parts)


def concat_unique(series: pd.Series) -> str:
    """Merge multiple values into a deduplicated pipe-separated string."""
    values = [v.strip() for v in series.dropna().astype(str) if v.strip()]
    return " | ".join(sorted(set(values))) if values else ""


# =====================================================
# EFFICIENT GROUPBY
# Applying concat_unique to every column × every row is O(n × cols).
# Instead: use first() for most columns (fast vectorized),
# concat_unique only for source_db and protein_name where merging matters.
# =====================================================

def merge_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by sequence, keeping first value for most columns
    and concatenating unique values for source_db and protein_name.
    """
    # columns where we want all unique values merged
    concat_cols = [c for c in ["source_db", "protein_name"] if c in df.columns]

    # columns where first (priority-sorted) value is enough
    first_cols = [
        c for c in df.columns
        if c not in concat_cols and c != "sequence"
    ]

    agg_dict = {c: "first" for c in first_cols}
    agg_dict.update({c: concat_unique for c in concat_cols})

    grouped = df.groupby("sequence", as_index=False).agg(agg_dict)
    logger.debug(
        "Groupby complete: %d unique sequences from %d rows",
        len(grouped), len(df),
    )
    return grouped


# =====================================================
# FASTA EXPORT — enriched headers
# =====================================================

def export_fasta(df: pd.DataFrame, out_fasta: Path) -> None:
    """
    Export sequences as FASTA with minimal headers.
    Format: >AMP_000001
    Compatible with MEME, STREME, HOMER and all standard tools.
    Full metadata is available in the parquet file.
    """
    written = 0
    with open(out_fasta, "w", encoding="utf-8") as f:
        for i, seq in enumerate(
            df["sequence"].dropna().astype(str).str.strip(), start=1
        ):
            if not seq:
                continue
            f.write(f">AMP_{i:06d}\n{seq}\n")
            written += 1

    logger.info("FASTA written: %d sequences -> %s", written, out_fasta.name)


# =====================================================
# STATS REPORT
# =====================================================

def save_stats(df: pd.DataFrame, n0: int) -> None:
    """Save JSON stats and CSV preview for the clean master dataset."""
    lengths = df["length"]

    # per-source breakdown
    per_source = df.groupby("source_db").size().to_dict() \
                 if "source_db" in df.columns else {}

    stats = {
        "initial_rows_master_04":  n0,
        "final_unique_sequences":  len(df),
        "duplicates_merged":       n0 - len(df),
        "length_min":              int(lengths.min()),
        "length_q25":              float(lengths.quantile(0.25)),
        "length_median":           float(lengths.median()),
        "length_mean":             round(float(lengths.mean()), 1),
        "length_q75":              float(lengths.quantile(0.75)),
        "length_max":              int(lengths.max()),
        "per_source_db":           per_source,
    }

    stats_path = RESULTS_DIR / "05_final_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("Stats saved: %s", stats_path)

    # log summary
    logger.info("\n=== Final dataset summary ===")
    logger.info("  Unique sequences:  %d", len(df))
    logger.info("  Merged duplicates: %d", n0 - len(df))
    logger.info(
        "  Length: min=%d  median=%.0f  mean=%.1f  max=%d",
        stats["length_min"], stats["length_median"],
        stats["length_mean"], stats["length_max"],
    )
    for src, n in sorted(per_source.items()):
        logger.info("  %-10s  %5d sequences  (%.1f%%)", src, n, 100*n/len(df))

    # preview CSV
    preview_cols = [
        c for c in ["source_db","protein_name","organism","taxonomy","sequence","length"]
        if c in df.columns
    ]
    preview_path = RESULTS_DIR / "05_final_preview.csv"
    df[preview_cols].head(20).to_csv(preview_path, index=False)
    logger.info("Preview saved: %s", preview_path)


# =====================================================
# MAIN
# =====================================================

def main() -> None:
    setup_dirs()
    setup_logging()

    in_file = DATA_INTERMEDIATE_DIR / "DB_MASTER_04.parquet"
    if not in_file.exists():
        logger.error("Input not found: %s  (run step 04 first)", in_file)
        sys.exit(1)

    df = pd.read_parquet(in_file)
    n0 = len(df)
    logger.info("Loaded: %d rows from %s", n0, in_file.name)

    # ---- normalize text fields ----
    if "organism" in df.columns:
        df["organism"] = df["organism"].apply(normalize_organism)
        logger.debug("Organism field normalized.")

    if "taxonomy" in df.columns:
        df["taxonomy"] = df["taxonomy"].apply(normalize_taxonomy)
        logger.debug("Taxonomy field normalized.")

    # ---- merge duplicate sequences ----
    logger.info("Merging metadata for duplicate sequences...")
    grouped = merge_metadata(df)
    grouped["length"] = grouped["sequence"].astype(str).str.len()
    logger.info(
        "After merge: %d unique sequences  (%d duplicates consolidated)",
        len(grouped), n0 - len(grouped),
    )

    # ---- save parquet ----
    out_parquet = DATA_FINAL_DIR / "DB_MASTER_CLEAN.parquet"
    grouped.to_parquet(out_parquet, index=False)
    logger.info("Clean parquet saved: %s", out_parquet)

    # ---- export FASTA ----
    out_fasta = DATA_FINAL_DIR / "AMP_MASTER.fasta"
    export_fasta(grouped, out_fasta)

    # ---- stats ----
    save_stats(grouped, n0)

    logger.info("Step 05 completed successfully.")


if __name__ == "__main__":
    main()