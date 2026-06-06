"""
02_filter_individual_databases.py
===================================
Applies structural filters to each standardized database:
  1. Sequence length (MIN_LENGTH – MAX_LENGTH)
  2. Natural AA only (reuses has_invalid_aa flag from step 01)
  3. Remove rows where validation_source contains "synthetic"
  4. Remove rows where protein_name matches unwanted patterns
  5. Deduplicate by sequence within each DB

Changes vs original:
  - logging instead of print (terminal + log file)
  - argparse --db to run a single database
  - DB list from DB_PATHS.keys() in config (no hardcoding)
  - Reuses has_invalid_aa flag from step 01 instead of re-scanning
  - Saves per-step filter report as CSV (useful as supplementary table)

Usage:
    python 02_filter_individual_databases.py             # all DBs
    python 02_filter_individual_databases.py --db DRAMP  # one DB
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd

from config import (
    DATA_INTERMEDIATE_DIR,
    DB_PATHS,
    LOGS_DIR,
    MIN_LENGTH,
    MAX_LENGTH,
    RESULTS_DIR,
    UNWANTED_NAME_PATTERNS,
    setup_dirs,
)

# =====================================================
# LOGGING
# =====================================================

def setup_logging() -> None:
    setup_dirs()
    log_file = LOGS_DIR / "02_filter_individual_databases.log"
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
# FILTER FUNCTION
# =====================================================

def filter_one(db_name: str) -> Path:
    """
    Apply structural filters to one standardized database.
    Returns path to filtered parquet.
    """
    in_file = DATA_INTERMEDIATE_DIR / f"{db_name}_01_standardized.parquet"
    if not in_file.exists():
        raise FileNotFoundError(
            f"Input not found: {in_file}\n"
            f"Run 01_standardize_raw_databases.py first."
        )

    df = pd.read_parquet(in_file)
    counts: dict[str, int] = {"start": len(df)}
    logger.info("=== %s ===  start: %d rows", db_name, len(df))

    # --------------------------------------------------
    # 1. LENGTH FILTER
    # --------------------------------------------------
    df["length"] = df["sequence"].astype(str).str.len()
    df = df[
        (df["length"] >= MIN_LENGTH) &
        (df["length"] <= MAX_LENGTH)
    ].copy()
    counts["after_length"] = len(df)
    logger.info(
        "  Length filter [%d–%d aa]: kept %d  (removed %d)",
        MIN_LENGTH, MAX_LENGTH, len(df), counts["start"] - len(df),
    )

    # --------------------------------------------------
    # 2. NATURAL AA FILTER
    # Reuse has_invalid_aa flag computed in step 01.
    # If column missing (e.g. old run), fall back to re-computation.
    # --------------------------------------------------
    if "has_invalid_aa" in df.columns:
        df = df[~df["has_invalid_aa"]].copy()
        logger.debug("  Using has_invalid_aa flag from step 01.")
    else:
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        df = df[
            df["sequence"].astype(str).apply(
                lambda s: all(aa in valid_aa for aa in s)
            )
        ].copy()
        logger.debug("  has_invalid_aa not found — re-scanned sequences.")
    counts["after_natural_aa"] = len(df)
    logger.info(
        "  Natural AA filter: kept %d  (removed %d)",
        len(df), counts["after_length"] - len(df),
    )

    # --------------------------------------------------
    # 3. VALIDATION SOURCE FILTER
    # DBAASP: "Synthetic" = synthesis method, not origin.
    #   → keep only Ribosomal and Nonribosomal (positive filter)
    # Others: exclude rows where validation_source contains "synthetic"
    #   (negative filter — flags designed/non-natural peptides)
    # --------------------------------------------------
    if "validation_source" in df.columns:
        if db_name == "DBAASP":
            keep_vals = ["Ribosomal", "Nonribosomal"]
            df = df[
                df["validation_source"].astype(str).isin(keep_vals)
            ].copy()
            logger.info(
                "  DBAASP positive filter (keep %s): kept %d",
                keep_vals, len(df),
            )
        else:
            mask = df["validation_source"].astype(str).str.contains(
                "synthetic", case=False, na=False
            )
            df = df[~mask].copy()
            logger.info(
                "  Synthetic filter: kept %d  (removed %d)",
                len(df), counts["after_natural_aa"] - len(df),
            )
    counts["after_no_synthetic"] = len(df)

    # --------------------------------------------------
    # 4. UNWANTED NAME PATTERNS
    # --------------------------------------------------
    if "protein_name" in df.columns and UNWANTED_NAME_PATTERNS:
        pattern = "|".join(map(re.escape, UNWANTED_NAME_PATTERNS))
        mask = df["protein_name"].astype(str).str.contains(
            pattern, case=False, na=False
        )
        n_flagged = mask.sum()
        df = df[~mask].copy()
        logger.info(
            "  Name pattern filter (%d patterns): removed %d rows",
            len(UNWANTED_NAME_PATTERNS), n_flagged,
        )
    counts["after_name_filter"] = len(df)

    # --------------------------------------------------
    # 5. DEDUPLICATE WITHIN DB
    # --------------------------------------------------
    n_before = len(df)
    df = df.drop_duplicates(subset=["sequence"]).copy()
    counts["after_dedup"] = len(df)
    logger.info(
        "  Dedup: removed %d  |  final: %d rows",
        n_before - len(df), len(df),
    )

    # --------------------------------------------------
    # SAVE FILTERED PARQUET
    # --------------------------------------------------
    out_file = DATA_INTERMEDIATE_DIR / f"{db_name}_02_filtered.parquet"
    df.to_parquet(out_file, index=False)
    logger.info("Saved: %s", out_file.name)

    return out_file, counts


# =====================================================
# QC REPORT
# =====================================================

def save_filter_report(all_counts: dict[str, dict[str, int]]) -> None:
    """
    Save per-DB filter report as CSV.
    Useful as supplementary table in the paper.
    """
    rows = []
    steps = ["start", "after_length", "after_natural_aa",
             "after_no_synthetic", "after_name_filter", "after_dedup"]
    for db_name, counts in all_counts.items():
        row = {"database": db_name}
        prev = None
        for step in steps:
            n = counts.get(step, None)
            row[step] = n
            if prev is not None and n is not None:
                row[f"removed_{step}"] = prev - n
            prev = n
        row["pct_retained"] = round(
            100 * counts.get("after_dedup", 0) / counts.get("start", 1), 1
        )
        rows.append(row)

    report_df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "02_filter_report.csv"
    report_df.to_csv(out_path, index=False)
    logger.info("Filter report saved: %s", out_path)

    # print summary to terminal
    logger.info("\n=== Filter summary ===")
    for _, row in report_df.iterrows():
        logger.info(
            "  %-10s  start=%6d  final=%6d  retained=%.1f%%",
            row["database"], row["start"], row["after_dedup"], row["pct_retained"],
        )


# =====================================================
# CLI
# =====================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply structural filters to standardized AMP databases.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 02_filter_individual_databases.py             # all DBs
  python 02_filter_individual_databases.py --db DRAMP  # one DB
        """,
    )
    parser.add_argument(
        "--db",
        action="append",
        choices=list(DB_PATHS.keys()),
        dest="databases",
        metavar="DB_NAME",
        help="Database to process. Can be repeated. Default: all.",
    )
    return parser.parse_args()


# =====================================================
# MAIN
# =====================================================

def main() -> None:
    setup_dirs()
    setup_logging()

    args   = parse_args()
    to_run = args.databases if args.databases else list(DB_PATHS.keys())
    logger.info("Databases to filter: %s", to_run)

    all_counts: dict[str, dict[str, int]] = {}
    failed: list[str] = []

    for db_name in to_run:
        try:
            _, counts = filter_one(db_name)
            all_counts[db_name] = counts
        except Exception as exc:
            logger.error("FAILED %s: %s", db_name, exc, exc_info=True)
            failed.append(db_name)

    # save QC report only if at least one DB succeeded
    if all_counts:
        save_filter_report(all_counts)

    if failed:
        logger.error("Step 02 completed WITH ERRORS: %s", failed)
        sys.exit(1)
    else:
        logger.info("Step 02 completed successfully.")


if __name__ == "__main__":
    main()