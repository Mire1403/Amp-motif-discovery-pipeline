"""
Usage:
    python 01_cluster_amp_sequences.py                    # both tools, identity=0.8
    python 01_cluster_amp_sequences.py --tool cd-hit      # only CD-HIT
    python 01_cluster_amp_sequences.py --tool mmseqs      # only MMseqs2
    python 01_cluster_amp_sequences.py --identity 0.9     # custom threshold
"""

from __future__ import annotations

import argparse
import logging
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

DATA_FINAL_DIR      = PROJECT_ROOT / "data" / "final"
RESULTS_DIR         = PROJECT_ROOT / "results" / "02_redundancy_reduction"
LOGS_DIR            = PROJECT_ROOT / "logs"

FASTA_INPUT         = DATA_FINAL_DIR / "AMP_MASTER.fasta"

# outputs
CDHIT_OUTPUT        = RESULTS_DIR / "AMP_MASTER_cdhit.fasta"
CDHIT_CLUSTER_FILE  = RESULTS_DIR / "AMP_MASTER_cdhit.fasta.clstr"
MMSEQS_PREFIX       = RESULTS_DIR / "AMP_MASTER_mmseqs"
MMSEQS_TMP          = RESULTS_DIR / "mmseqs_tmp"

# MMseqs2 easy-cluster generates:
#   {prefix}_rep_seq.fasta   — representative sequences
#   {prefix}_cluster.tsv     — cluster membership table
#   {prefix}_all_seqs.fasta  — all sequences with cluster info

# =====================================================
# LOGGING
# =====================================================

def setup_logging() -> None:
    for d in [RESULTS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "01_cluster_amp_sequences.log"
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

def count_fasta(path: Path) -> int:
    """Count sequences in a FASTA file."""
    return sum(1 for line in open(path, encoding="utf-8") if line.startswith(">"))


def run_cmd(cmd: list[str]) -> None:
    """Log and run a shell command, raising on failure."""
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=True)
    logger.debug("Return code: %d", result.returncode)


# =====================================================
# CD-HIT
# =====================================================

def run_cdhit(identity: float) -> None:
    """
    Run CD-HIT for sequence clustering.

    Parameters:
        -c   sequence identity threshold
        -n   word size (5 for threshold >= 0.7)
        -aS  alignment coverage for shorter sequence
        -T   threads
        -M   memory limit in MB
    """
    logger.info("=== CD-HIT  (identity=%.2f) ===", identity)

    if not FASTA_INPUT.exists():
        raise FileNotFoundError(f"Input FASTA not found: {FASTA_INPUT}")

    n_before = count_fasta(FASTA_INPUT)
    logger.info("Input sequences: %d", n_before)

    # word size: 5 for >=0.7, 4 for >=0.6, 3 for >=0.5
    word_size = "5" if identity >= 0.7 else "4" if identity >= 0.6 else "3"

    cmd = [
        "cd-hit",
        "-i", str(FASTA_INPUT),
        "-o", str(CDHIT_OUTPUT),
        "-c", str(identity),
        "-n", word_size,
        "-aS", "1.0",
        "-T", "4",
        "-M", "16000",
    ]
    run_cmd(cmd)

    n_after = count_fasta(CDHIT_OUTPUT)
    logger.info("CD-HIT output:  %d representative sequences", n_after)
    logger.info("Removed:        %d  (%.1f%%)",
                n_before - n_after, 100 * (n_before - n_after) / n_before)
    logger.info("Cluster file:   %s", CDHIT_CLUSTER_FILE)
    logger.info("Output FASTA:   %s", CDHIT_OUTPUT)


# =====================================================
# MMSEQS2
# =====================================================

def run_mmseqs(identity: float) -> None:
    """
    Run MMseqs2 easy-cluster for sequence clustering.

    Parameters:
        --min-seq-id   minimum sequence identity
        -c             coverage threshold
        --cov-mode 1   coverage of shorter sequence
    """
    logger.info("=== MMseqs2 easy-cluster  (identity=%.2f) ===", identity)

    if not FASTA_INPUT.exists():
        raise FileNotFoundError(f"Input FASTA not found: {FASTA_INPUT}")

    n_before = count_fasta(FASTA_INPUT)
    logger.info("Input sequences: %d", n_before)

    MMSEQS_TMP.mkdir(parents=True, exist_ok=True)

    cmd = [
        "mmseqs", "easy-cluster",
        str(FASTA_INPUT),
        str(MMSEQS_PREFIX),
        str(MMSEQS_TMP),
        "--min-seq-id", str(identity),
        "-c",           str(identity),
        "--cov-mode",   "1",
    ]
    run_cmd(cmd)

    rep_fasta   = Path(str(MMSEQS_PREFIX) + "_rep_seq.fasta")
    cluster_tsv = Path(str(MMSEQS_PREFIX) + "_cluster.tsv")

    if rep_fasta.exists():
        n_after = count_fasta(rep_fasta)
        logger.info("MMseqs2 output: %d representative sequences", n_after)
        logger.info("Removed:        %d  (%.1f%%)",
                    n_before - n_after, 100 * (n_before - n_after) / n_before)

    logger.info("Representative sequences: %s", rep_fasta)
    logger.info("Cluster membership table: %s", cluster_tsv)


# =====================================================
# CLI
# =====================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reduce AMP sequence redundancy with CD-HIT and/or MMseqs2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 01_cluster_amp_sequences.py                     # both, identity=0.8
  python 01_cluster_amp_sequences.py --tool cd-hit       # only CD-HIT
  python 01_cluster_amp_sequences.py --tool mmseqs       # only MMseqs2
  python 01_cluster_amp_sequences.py --identity 0.9      # custom threshold
        """,
    )
    parser.add_argument(
        "--tool",
        choices=["cd-hit", "mmseqs", "both"],
        default="both",
        help="Which clustering tool to run (default: both)",
    )
    parser.add_argument(
        "--identity",
        type=float,
        default=0.8,
        help="Sequence identity threshold 0.0-1.0 (default: 0.8)",
    )
    return parser.parse_args()


# =====================================================
# MAIN
# =====================================================

def main() -> None:
    setup_logging()
    args = parse_args()

    logger.info("AMP redundancy reduction started.")
    logger.info("Tool: %s  |  Identity threshold: %.2f", args.tool, args.identity)
    logger.info("Input FASTA: %s", FASTA_INPUT)

    if not FASTA_INPUT.exists():
        logger.error(
            "Input FASTA not found: %s\n"
            "Run 05_clean_master_dataset.py first.", FASTA_INPUT
        )
        sys.exit(1)

    failed = []

    if args.tool in ("cd-hit", "both"):
        try:
            run_cdhit(args.identity)
        except Exception as exc:
            logger.error("CD-HIT failed: %s", exc, exc_info=True)
            failed.append("cd-hit")

    if args.tool in ("mmseqs", "both"):
        try:
            run_mmseqs(args.identity)
        except Exception as exc:
            logger.error("MMseqs2 failed: %s", exc, exc_info=True)
            failed.append("mmseqs")

    if failed:
        logger.error("Completed WITH ERRORS: %s", failed)
        sys.exit(1)
    else:
        logger.info("AMP redundancy reduction completed successfully.")


if __name__ == "__main__":
    main()