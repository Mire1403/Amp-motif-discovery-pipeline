"""
Usage:
    python 03_apply_global_mic_filter.py             # all DBs
    python 03_apply_global_mic_filter.py --db DRAMP  # one DB
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DATA_INTERMEDIATE_DIR,
    DB_PATHS,
    LOGS_DIR,
    MIC_THRESHOLD_UGML,
    RESULTS_DIR,
    setup_dirs,
)

# =====================================================
# CONSTANTS (kept in script — not touching config)
# =====================================================

AA_WEIGHTS: dict[str, float] = {
    "A": 89.1,  "R": 174.2, "N": 132.1, "D": 133.1,
    "C": 121.2, "E": 147.1, "Q": 146.2, "G": 75.1,
    "H": 155.2, "I": 131.2, "L": 131.2, "K": 146.2,
    "M": 149.2, "F": 165.2, "P": 115.1, "S": 105.1,
    "T": 119.1, "W": 204.2, "Y": 181.2, "V": 117.1,
}

UNIT_PATTERNS: dict[str, str] = {
    "uM":    r"(?:µm|μm|um|micromolar|µmol\/l|μmol\/l|umol\/l|micromol\/l)",
    "ug/mL": r"(?:µg\/ml|μg\/ml|ug\/ml|micrograms?\/ml)",
    "mg/mL": r"(?:mg\/ml)",
    "ng/mL": r"(?:ng\/ml|nanograms?\/ml)",
}

MIC_KEYWORD = r"(?:\bmic\b|\bmic50\b|\bmic90\b)"

# =====================================================
# LOGGING
# =====================================================

def setup_logging() -> None:
    setup_dirs()
    log_file = LOGS_DIR / "03_apply_global_mic_filter.log"
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
# MIC PARSING HELPERS
# =====================================================

def calculate_mw(sequence: str) -> float:
    """Approximate molecular weight from sequence (sum of residue weights)."""
    return float(sum(AA_WEIGHTS.get(aa, 0.0) for aa in str(sequence).upper().strip()))


def _norm_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).replace("≤", "<=").replace("≥", ">=").lower()


def detect_unit(text: str) -> str | None:
    """Detect unit anywhere in text."""
    for unit, pat in UNIT_PATTERNS.items():
        if re.search(pat, text, flags=re.I):
            return unit
    return None


def extract_mic(text: str) -> list[tuple[float, str | None]]:

    text = _norm_text(text)
    if not re.search(MIC_KEYWORD, text, flags=re.I):
        return []

    pat = re.compile(
        rf"{MIC_KEYWORD}[^0-9<>=]*"
        rf"(?:<=|>=|=)?\s*"
        rf"([0-9]+(?:\.[0-9]+)?)(?:\s*-\s*([0-9]+(?:\.[0-9]+)?))?"
        rf"(?:\s*([a-zµμ\/]+))?",
        flags=re.I,
    )

    out: list[tuple[float, str | None]] = []
    for m in pat.finditer(text):
        v1    = float(m.group(1))
        v2    = m.group(2)
        token = (m.group(3) or "").strip()
        unit  = None

        if token:
            for u, upat in UNIT_PATTERNS.items():
                if re.search(upat, token, flags=re.I):
                    unit = u
                    break
            if unit is None and token in {"um", "µm", "μm"}:
                unit = "uM"

        if unit is None:
            unit = detect_unit(text)

        out.append((v1, unit))
        if v2:
            out.append((float(v2), unit))

    return out


def to_ugml(value: float, unit: str | None, mw: float) -> float | None:
    """Convert MIC value to µg/mL. Returns None if unit unknown."""
    if unit == "ug/mL":
        return value
    if unit == "mg/mL":
        return value * 1000.0
    if unit == "ng/mL":
        return value / 1000.0
    if unit == "uM":
        return (value * mw) / 1000.0 if mw > 0 else None
    return None


# =====================================================
# FILTER FUNCTION
# =====================================================

def filter_one(db_name: str) -> tuple[Path, dict[str, int]]:
    in_file = DATA_INTERMEDIATE_DIR / f"{db_name}_02_filtered.parquet"

    if not in_file.exists():
        raise FileNotFoundError(
            f"Input not found: {in_file}\n"
            f"Run 02_filter_individual_databases.py first."
        )

    df = pd.read_parquet(in_file).copy()
    n0 = len(df)
    counts: dict[str, int] = {"start": n0}
    logger.info("=== %s ===  start: %d rows", db_name, n0)

    if "sequence" not in df.columns:
        raise ValueError(f"{db_name}: no 'sequence' column in input.")

    # --------------------------------------------------
    # Build combined text for MIC search
    # --------------------------------------------------
    combined = (
        df.get("activity",     pd.Series([""] * n0)).astype("string").fillna("") + " " +
        df.get("target_group", pd.Series([""] * n0)).astype("string").fillna("") + " " +
        df.get("target_object",pd.Series([""] * n0)).astype("string").fillna("")
    )

    mw_series = df["sequence"].astype("string").fillna("").apply(calculate_mw)

    # --------------------------------------------------
    # Parse MIC per row
    # --------------------------------------------------
    min_mic_list:   list[float]  = []
    unit_list:      list[str]    = []

    for txt, mw in zip(combined.tolist(), mw_series.tolist()):
        vals:  list[float] = []
        units: set[str]    = set()

        for v, u in extract_mic(txt):
            cv = to_ugml(v, u, mw)
            if cv is not None:
                vals.append(cv)
                if u:
                    units.add(u)

        min_mic_list.append(float(min(vals)) if vals else np.nan)
        unit_list.append(";".join(sorted(units)) if units else "")

    df["min_mic_ugml"]      = min_mic_list
    df["mic_unit_detected"] = unit_list

    has_mic = df["min_mic_ugml"].notna()
    counts["rows_with_mic"] = int(has_mic.sum())
    logger.info(
        "  MIC parsed: %d rows  (%.1f%%)",
        counts["rows_with_mic"], 100 * counts["rows_with_mic"] / n0,
    )

    # --------------------------------------------------
    # Apply MIC threshold filter
    # Rows with no MIC are KEPT (conservative)
    # --------------------------------------------------
    remove_mask = has_mic & (df["min_mic_ugml"] >= MIC_THRESHOLD_UGML)
    counts["removed_above_threshold"] = int(remove_mask.sum())
    logger.info(
        "  Removed (MIC >= %.1f µg/mL): %d rows",
        MIC_THRESHOLD_UGML, counts["removed_above_threshold"],
    )

    df_out = df.loc[~remove_mask].copy()
    counts["final"] = len(df_out)
    logger.info("  Final: %d rows", len(df_out))

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    out_file = DATA_INTERMEDIATE_DIR / f"{db_name}_03_mic_filtered.parquet"
    df_out.to_parquet(out_file, index=False)
    logger.info("Saved: %s", out_file.name)

    return out_file, counts


# =====================================================
# QC REPORT
# =====================================================

def save_mic_report(all_counts: dict[str, dict[str, int]]) -> None:
    rows = []
    for db_name, c in all_counts.items():
        rows.append({
            "database":               db_name,
            "start":                  c.get("start", 0),
            "rows_with_mic":          c.get("rows_with_mic", 0),
            "pct_with_mic":           round(100 * c.get("rows_with_mic", 0) /
                                            max(c.get("start", 1), 1), 1),
            "removed_above_threshold":c.get("removed_above_threshold", 0),
            "final":                  c.get("final", 0),
            "pct_retained":           round(100 * c.get("final", 0) /
                                            max(c.get("start", 1), 1), 1),
            "mic_threshold_ugml":     MIC_THRESHOLD_UGML,
        })

    report_df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "03_mic_filter_report.csv"
    report_df.to_csv(out_path, index=False)
    logger.info("MIC filter report saved: %s", out_path)

    logger.info("\n=== MIC filter summary (threshold=%.1f µg/mL) ===",
                MIC_THRESHOLD_UGML)
    for _, row in report_df.iterrows():
        logger.info(
            "  %-10s  start=%6d  mic_parsed=%5d (%4.1f%%)  "
            "removed=%5d  final=%6d  retained=%.1f%%",
            row["database"], row["start"],
            row["rows_with_mic"], row["pct_with_mic"],
            row["removed_above_threshold"],
            row["final"], row["pct_retained"],
        )


# =====================================================
# CLI
# =====================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply MIC-based filter to structurally filtered AMP databases.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 03_apply_global_mic_filter.py             # all DBs
  python 03_apply_global_mic_filter.py --db DRAMP  # one DB
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
    logger.info("Databases to MIC-filter: %s", to_run)

    all_counts: dict[str, dict[str, int]] = {}
    failed: list[str] = []

    for db_name in to_run:
        try:
            _, counts = filter_one(db_name)
            all_counts[db_name] = counts
        except Exception as exc:
            logger.error("FAILED %s: %s", db_name, exc, exc_info=True)
            failed.append(db_name)

    if all_counts:
        save_mic_report(all_counts)

    if failed:
        logger.error("Step 03 completed WITH ERRORS: %s", failed)
        sys.exit(1)
    else:
        logger.info("Step 03 completed successfully.")


if __name__ == "__main__":
    main()