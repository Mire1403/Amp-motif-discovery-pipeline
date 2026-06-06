"""
config.py — Central configuration for the AMP database construction pipeline.

Design decisions:
- Project root is found via .git or pyproject.toml marker (robust to moves)
- No directory creation on import (no side effects)
- DB mappings defined here so scripts never need editing to add a new DB
- All thresholds and constants in one place
"""

from __future__ import annotations
from pathlib import Path


# =====================================================
# PROJECT ROOT — robust marker-based detection
# =====================================================

def _find_project_root() -> Path:
    """
    Walk up from this file until we find .git or pyproject.toml.
    This works regardless of how deep config.py is nested, and
    regardless of whether the user cloned the repo to a different path.
    Raises RuntimeError if the project root cannot be found.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(
        "Could not find project root. "
        "Make sure you are inside a git repository or have a pyproject.toml."
    )

PROJECT_ROOT = _find_project_root()

# =====================================================
# DIRECTORY LAYOUT
# =====================================================
# These are Path objects — they do NOT create the directories here.
# Call setup_dirs() explicitly from main() in each script.

DATA_RAW_DIR          = PROJECT_ROOT / "data" / "raw"
DATA_INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
DATA_FINAL_DIR        = PROJECT_ROOT / "data" / "final"
RESULTS_DIR           = PROJECT_ROOT / "results" / "01_database_construction"
LOGS_DIR              = PROJECT_ROOT / "logs"


def setup_dirs() -> None:
    """
    Create all required output directories.
    Call this at the top of main() in each script, not at import time.
    """
    for d in [DATA_INTERMEDIATE_DIR, DATA_FINAL_DIR, RESULTS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# =====================================================
# RAW DATABASE PATHS
# =====================================================

DB_PATHS: dict[str, Path] = {
    "CAMP":   DATA_RAW_DIR / "CAMP"   / "CAMP.txt",
    "DBAASP": DATA_RAW_DIR / "DBAASP" / "peptides.csv",
    "dbAMP3": DATA_RAW_DIR / "dbAMP3" / "dbAMP3_pepinfo.xlsx",
    "DRAMP":  DATA_RAW_DIR / "DRAMP"  / "natural_amps.txt",
}

# =====================================================
# COLUMN MAPPINGS — raw DB columns → standard names
# Moving mappings here means scripts never need editing
# when column names change in a DB update.
# =====================================================

DB_COLUMN_MAPPINGS: dict[str, dict[str, str]] = {
    "DRAMP": {
        "Sequence":          "sequence",
        "Name":              "protein_name",
        "Source":            "organism",
        "Activity":          "activity",
        "Swiss_Prot_Entry":  "swissprot_entry",
    },
    "CAMP": {
        "Seqence":           "sequence",   # known typo in CAMP export
        "Title":             "protein_name",
        "Source_Organism":   "organism",
        "Activity":          "activity",
        "Taxonomy":          "taxonomy",
        "Validation":        "validation_source",
        "Modifications":     "modifications",
        "Target":            "target_group",
    },
    "DBAASP": {
        "COMPLEXITY":        "complexity",
        "NAME":              "protein_name",
        "SEQUENCE":          "sequence",
        "TARGET GROUP":      "target_group",
        "TARGET OBJECT":     "target_object",
        "SYNTHESIS TYPE":    "validation_source",
    },
    "dbAMP3": {
        "Seq":               "sequence",
        "Name":              "protein_name",
        "Tax":               "taxonomy",
        "Source":            "organism",
        "Uniprot":           "uniprot",
        "PDB":               "pdb",
        "Targets":           "target_group",
    },
}

# =====================================================
# STANDARD COLUMNS (shared schema across all DBs)
# =====================================================

STANDARD_COLUMNS: list[str] = [
    "source_db",
    "complexity",
    "protein_name",
    "organism",
    "taxonomy",
    "sequence",
    "activity",
    "validation_source",
    "modifications",
    "target_group",
    "target_object",
    "uniprot",
    "pdb",
    "swissprot_entry",
]

# Columns that must be string type for parquet compatibility
TEXT_COLS: list[str] = [c for c in STANDARD_COLUMNS if c != "complexity"]
TEXT_COLS.append("complexity")  # complexity also string (e.g. "linear", "cyclic")

# =====================================================
# SEQUENCE FILTERS
# =====================================================

MIN_LENGTH: int   = 6
MAX_LENGTH: int   = 80
VALID_AA:   set   = set("ACDEFGHIKLMNPQRSTVWY")

# =====================================================
# MIC FILTER
# =====================================================

MIC_THRESHOLD_UGML: float = 65.0

# =====================================================
# NAME-BASED EXCLUSION PATTERNS
# Sequences whose protein_name contains any of these
# (case-insensitive) will be flagged as non-natural.
# =====================================================

UNWANTED_NAME_PATTERNS: list[str] = [
    "synthetic",
    "designed",
    "construct",
    "analog",
    "mutant",
    "fragment",
    "truncated",
]