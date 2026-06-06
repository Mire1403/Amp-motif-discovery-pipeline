"""
06_fimo_scanning.py
====================
Runs FIMO to scan AMP and background sequences for motif occurrences
using motifs discovered by MEME and STREME.

Combinations run:
  {cdhit,mmseqs} × {meme,streme} × {amps,nonamps} = 8 FIMO runs

Usage:
    python 06_fimo_scanning.py                              # all 8 combinations
    python 06_fimo_scanning.py --tool cdhit                 # only CD-HIT
    python 06_fimo_scanning.py --motif_tool meme            # only MEME motifs
    python 06_fimo_scanning.py --thresh 1e-5                # custom threshold
    python 06_fimo_scanning.py --overwrite                  # clear output dirs
    python 06_fimo_scanning.py --test                       # test mode

Changes vs original:
  - logging instead of print
  - _find_project_root() instead of parents[2]
  - Fixed: correct paths (02_redundancy_reduction, mmseqs not mmseq)
  - Fixed: correct background FASTA names and location
  - argparse: --tool, --motif_tool, --thresh, --overwrite, --test
  - Continues with remaining combinations if one fails (no sys.exit on error)
  - --test flag: outputs to results/05_motif_scanning_test/
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

# =====================================================
# PROJECT ROOT
# =====================================================

def _find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find project root.")

PROJECT_ROOT = _find_project_root()
CLUSTER_DIR  = PROJECT_ROOT / "results" / "02_redundancy_reduction"
MOTIF_DIR    = PROJECT_ROOT / "results" / "04_motif_discovery"
BG_DIR       = PROJECT_ROOT / "data"    / "intermediate" / "background"
FIMO_DIR     = PROJECT_ROOT / "results" / "05_motif_scanning"
LOGS_DIR     = PROJECT_ROOT / "logs"

# motif files
MOTIF_FILES = {
    "cdhit_meme":    MOTIF_DIR  / "meme_cdhit"    / "meme.txt",
    "mmseqs_meme":   MOTIF_DIR  / "meme_mmseqs"   / "meme.txt",
    "cdhit_streme":  MOTIF_DIR  / "streme_cdhit"  / "streme.txt",
    "mmseqs_streme": MOTIF_DIR  / "streme_mmseqs" / "streme.txt",
}

# sequence files
SEQ_FILES = {
    "cdhit_amps":    CLUSTER_DIR / "AMP_MASTER_cdhit.fasta",
    "mmseqs_amps":   CLUSTER_DIR / "AMP_MASTER_mmseqs_rep_seq.fasta",
    "cdhit_nonamp":  BG_DIR      / "background_cdhit_10x.fasta",
    "mmseqs_nonamp": BG_DIR      / "background_mmseqs_10x.fasta",
}

# =====================================================
# LOGGING
# =====================================================

def setup_logging() -> None:
    for d in [FIMO_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "06_fimo_scanning.log"
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

def get_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        logger.error(
            "Required executable '%s' not found in PATH.\n"
            "Activate the correct conda environment or install MEME Suite.\n"
            "Check with:  which %s", name, name,
        )
        sys.exit(1)
    return path


def prepare_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
        logger.debug("Cleared: %s", path)
    path.mkdir(parents=True, exist_ok=True)


def run_fimo(
        motif_file: Path,
        seq_file:   Path,
        output_dir: Path,
        thresh:     str,
        overwrite:  bool,
        fimo_bin:   str,
) -> bool:
    """
    Run FIMO for one motif/sequence combination.
    Returns True on success, False on failure.
    """
    for f in [motif_file, seq_file]:
        if not f.exists():
            logger.error("Input not found: %s", f)
            return False

    prepare_dir(output_dir, overwrite)

    cmd = [
        fimo_bin,
        "--thresh", thresh,
        "--oc",     str(output_dir),
        str(motif_file),
        str(seq_file),
    ]
    logger.info(
        "FIMO: motifs=%s  seqs=%s  out=%s",
        motif_file.parent.name + "/" + motif_file.name,
        seq_file.name,
        output_dir.relative_to(PROJECT_ROOT),
    )
    logger.debug("Command: %s", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
        logger.info("FIMO completed: %s", output_dir)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("FIMO failed (return code %d): %s", e.returncode, output_dir)
        return False
    except FileNotFoundError:
        logger.error("Executable not found: %s", cmd[0])
        return False

# =====================================================
# BUILD COMBINATIONS
# =====================================================

def build_combinations(
        tools:        list[str],
        motif_tools:  list[str],
) -> list[tuple[Path, Path, Path]]:
    """
    Build list of (motif_file, seq_file, output_dir) tuples
    for the requested tool/motif_tool combinations.
    """
    combos = []
    for tool in tools:
        for mt in motif_tools:
            motif_key = f"{tool}_{mt}"
            motif_file = MOTIF_FILES.get(motif_key)
            if motif_file is None:
                continue
            for seq_label in [f"{tool}_amps", f"{tool}_nonamp"]:
                seq_file = SEQ_FILES.get(seq_label)
                if seq_file is None:
                    continue
                out_name = "fimo_amps" if "amps" in seq_label else "fimo_nonamps"
                out_dir  = FIMO_DIR / tool / mt / out_name
                combos.append((motif_file, seq_file, out_dir))
    return combos

# =====================================================
# CLI
# =====================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan AMP and background sequences with FIMO.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 06_fimo_scanning.py                          # all combinations
  python 06_fimo_scanning.py --tool cdhit             # only CD-HIT
  python 06_fimo_scanning.py --motif_tool meme        # only MEME motifs
  python 06_fimo_scanning.py --thresh 1e-5            # stricter threshold
  python 06_fimo_scanning.py --overwrite              # clear existing outputs
  python 06_fimo_scanning.py --test                   # test mode
        """,
    )
    parser.add_argument("--tool",       choices=["cdhit","mmseqs","both"],
                        default="both")
    parser.add_argument("--motif_tool", choices=["meme","streme","both"],
                        default="both")
    parser.add_argument("--thresh",     default="1e-4",
                        help="FIMO p-value threshold (default: 1e-4)")
    parser.add_argument("--overwrite",  action="store_true",
                        help="Clear existing output dirs before running")
    parser.add_argument("--test",       action="store_true",
                        help="Test mode: outputs to results/05_motif_scanning_test/")
    return parser.parse_args()

# =====================================================
# MAIN
# =====================================================

def main() -> None:
    setup_logging()
    args = parse_args()

    # test mode
    global FIMO_DIR
    if args.test:
        FIMO_DIR       = PROJECT_ROOT / "results" / "05_motif_scanning_test"
        args.overwrite = True
        logger.info("TEST MODE — output dir: %s", FIMO_DIR)
    FIMO_DIR.mkdir(parents=True, exist_ok=True)

    tools       = ["cdhit","mmseqs"] if args.tool       == "both" else [args.tool]
    motif_tools = ["meme","streme"]  if args.motif_tool == "both" else [args.motif_tool]

    logger.info(
        "FIMO scanning started. tools=%s  motif_tools=%s  thresh=%s",
        tools, motif_tools, args.thresh,
    )

    fimo_bin = get_binary("fimo")
    logger.info("FIMO binary: %s", fimo_bin)

    combos = build_combinations(tools, motif_tools)
    logger.info("Total combinations: %d", len(combos))

    n_ok   = 0
    failed = []

    for motif_file, seq_file, out_dir in combos:
        ok = run_fimo(
            motif_file = motif_file,
            seq_file   = seq_file,
            output_dir = out_dir,
            thresh     = args.thresh,
            overwrite  = args.overwrite,
            fimo_bin   = fimo_bin,
        )
        if ok:
            n_ok += 1
        else:
            failed.append(str(out_dir.relative_to(PROJECT_ROOT)))

    logger.info(
        "FIMO scanning finished: %d/%d successful", n_ok, len(combos)
    )

    if failed:
        logger.error("Failed runs:\n  %s", "\n  ".join(failed))
        sys.exit(1)
    else:
        logger.info("All FIMO runs completed successfully.")
        logger.info("Outputs in: %s", FIMO_DIR)


if __name__ == "__main__":
    main()