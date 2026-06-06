"""
Usage:
    python 01_standardize_raw_databases.py           # all DBs
    python 01_standardize_raw_databases.py --db DRAMP  # one DB
    python 01_standardize_raw_databases.py --db CAMP --db dbAMP3  # two DBs
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from config import (
    DB_COLUMN_MAPPINGS,
    DB_PATHS,
    DATA_INTERMEDIATE_DIR,
    LOGS_DIR,
    STANDARD_COLUMNS,
    TEXT_COLS,
    VALID_AA,
    setup_dirs,
)

# =====================================================
# LOGGING SETUP
# Creates two handlers:
#   1. StreamHandler  → terminal (INFO and above)
#   2. FileHandler    → logs/01_standardize.log (DEBUG and above)
# This replaces the manual Tee class used elsewhere.
# =====================================================

def setup_logging() -> None:
    setup_dirs()   # ensure logs/ exists before opening file
    log_file = LOGS_DIR / "01_standardize_raw_databases.log"

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
# IO HELPERS
# =====================================================

def _read_any(path: Path) -> pd.DataFrame:
    """Read CSV, Excel, or tab-separated TXT."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".txt":
        return pd.read_csv(
            path,
            sep="\t",
            encoding="latin-1",
            engine="python",
            on_bad_lines="skip",
        )
    raise ValueError(f"Unsupported file format: {path}")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = []
    for c in df.columns:
        s = str(c).strip().strip('"').strip("'")
        s = s.replace("\u2019", "'")   # right single quotation mark
        s = s.strip("'")
        cleaned.append(s)
    df = df.copy()
    df.columns = cleaned
    return df


def _force_text_types(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    for col in TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string")
    return df


def _write_table(df: pd.DataFrame, out_parquet: Path) -> Path:

    try:
        df.to_parquet(out_parquet, index=False)
        return out_parquet
    except Exception as exc:
        fallback = out_parquet.with_suffix(".csv")
        logger.warning(
            "Could not write parquet (%s: %s). Falling back to CSV: %s",
            type(exc).__name__, exc, fallback.name,
        )
        df.to_csv(fallback, index=False)
        return fallback


# =====================================================
# VALIDATION FLAG (new vs original)
# =====================================================

def _flag_invalid_aa(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df["has_invalid_aa"] = df["sequence"].apply(
        lambda s: bool(set(str(s).upper()) - VALID_AA)
        if pd.notna(s) else True
    )
    n_invalid = df["has_invalid_aa"].sum()
    if n_invalid:
        logger.debug("  Sequences with non-standard AA: %d", n_invalid)
    return df


# =====================================================
# MAIN STANDARDIZATION FUNCTION
# =====================================================

def standardize_one(db_name: str, input_path: Path) -> Path:

    if db_name not in DB_COLUMN_MAPPINGS:
        raise KeyError(f"No column mapping defined for db_name='{db_name}'")
    if not input_path.exists():
        raise FileNotFoundError(f"Missing {db_name} input: {input_path}")

    logger.info("=== %s ===", db_name)
    logger.info("Reading: %s", input_path)

    df_raw = _read_any(input_path)
    df_raw = _normalize_columns(df_raw)
    logger.info("Raw rows: %d  |  Columns: %d", len(df_raw), len(df_raw.columns))

    # --- apply column mapping ---
    mapping = DB_COLUMN_MAPPINGS[db_name]
    out = pd.DataFrame(columns=STANDARD_COLUMNS)

    missing_cols = []
    for orig_col, std_col in mapping.items():
        if orig_col in df_raw.columns:
            out[std_col] = df_raw[orig_col]
        else:
            missing_cols.append(orig_col)

    if missing_cols:
        logger.warning(
            "%s: expected columns not found (skipped): %s", db_name, missing_cols
        )

    out["source_db"] = db_name

    # --- normalize sequence ---
    if "sequence" not in out.columns:
        out["sequence"] = pd.NA

    out["sequence"] = (
        out["sequence"]
        .astype("string")
        .str.upper()
        .str.strip()
    )

    # --- drop empty sequences ---
    n_before = len(out)
    out = out[out["sequence"].notna() & (out["sequence"].str.len() > 0)].copy()
    n_dropped = n_before - len(out)
    if n_dropped:
        logger.info("Dropped empty/NA sequences: %d", n_dropped)

    # --- add invalid AA flag (new) ---
    out = _flag_invalid_aa(out)

    # --- deduplicate within DB ---
    n_before = len(out)
    out = out.drop_duplicates(subset=["sequence"]).copy()
    logger.info(
        "Dedup within %s: removed %d duplicates | remaining: %d",
        db_name, n_before - len(out), len(out),
    )

    # --- safe types for parquet ---
    out = _force_text_types(out)

    # --- save ---
    DATA_INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_INTERMEDIATE_DIR / f"{db_name}_01_standardized.parquet"
    written  = _write_table(out, out_path)

    logger.info("Saved: %s  (%d rows)", written.name, len(out))
    return written


# =====================================================
# CLI — argparse (new vs original)
# =====================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standardize raw AMP databases to a common schema.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 01_standardize_raw_databases.py              # all DBs
  python 01_standardize_raw_databases.py --db DRAMP   # one DB
  python 01_standardize_raw_databases.py --db CAMP --db DBAASP
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

    args     = parse_args()
    to_run   = args.databases if args.databases else list(DB_PATHS.keys())

    logger.info("Databases to process: %s", to_run)

    failed = []
    for db_name in to_run:
        try:
            standardize_one(db_name, DB_PATHS[db_name])
        except Exception as exc:
            logger.error("FAILED %s: %s", db_name, exc, exc_info=True)
            failed.append(db_name)

    if failed:
        logger.error("Step 01 completed WITH ERRORS: %s", failed)
        sys.exit(1)
    else:
        logger.info("Step 01 completed successfully.")


if __name__ == "__main__":
    main()