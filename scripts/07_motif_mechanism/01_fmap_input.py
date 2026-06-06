"""
12_fmap_input.py
=================
Selects diverse AMP sequences for FMAP mechanism prediction.

For each significant motif family:
  1. Filters by FDR, ER and AMP specificity
  2. Selects up to SEQS_PER_MOTIF diverse sequences from FIMO hits
     (diversity controlled by MIN_IDENTITY_DIFF)
  3. Extracts a window around the FIMO hit in the full AMP sequence
  4. Writes FMAP input file + selection tables

Usage:
    python 12_fmap_input.py
    python 12_fmap_input.py --seqs_per_motif 3 --spec 0.7
    python 12_fmap_input.py --test
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
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
FAMILIES_DIR = PROJECT_ROOT / "results" / "07_motif_families" / "01_tomtom"
FIMO_BASE    = PROJECT_ROOT / "results" / "05_motif_scanning"
AMP_FASTA    = PROJECT_ROOT / "data"    / "final" / "AMP_MASTER.fasta"
LOGS_DIR     = PROJECT_ROOT / "logs"

# =====================================================
# LOGGING
# =====================================================

def setup_logging(out_dir: Path) -> None:
    for d in [out_dir, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "12_fmap_input.log"
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

def clean_motif(x: str) -> str:
    return re.sub(r"^\d+[-_]", "", str(x).strip()).strip().upper()


def clean_sequence(seq: str) -> str:
    """Replace non-standard AAs with L (leucine)."""
    valid = set("ACDEFGHIKLMNPQRSTVWY")
    return "".join(aa if aa in valid else "L" for aa in seq.upper())


def sequence_identity(s1: str, s2: str) -> float:
    length = min(len(s1), len(s2))
    if length == 0: return 0.0
    return sum(a == b for a, b in zip(s1[:length], s2[:length])) / length


def is_diverse(seq: str, selected: list[str],
               min_diff: float) -> bool:
    return all(sequence_identity(seq, s) <= (1 - min_diff)
               for s in selected)


def load_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, str] = {}
    header, parts = None, []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header: seqs[header] = "".join(parts)
                header = line[1:].split()[0]  # use first token as ID
                parts = []
            else:
                parts.append(line)
        if header: seqs[header] = "".join(parts)
    logger.info("FASTA loaded: %d sequences", len(seqs))
    return seqs


def load_fimo_tsv(path: Path) -> pd.DataFrame:
    """Read FIMO TSV robustly (handles comment lines)."""
    try:
        df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    except Exception as e:
        logger.warning("Could not read %s: %s", path, e)
        return pd.DataFrame()
    if df.empty: return df
    df.columns = [c.strip().lower().replace("-","_") for c in df.columns]
    # motif name column — prefer motif_id (sequence) over motif_alt_id (numeric)
    mcol = None
    if "motif_id" in df.columns:
        if df["motif_id"].dropna().astype(str).str.contains(r"[A-Za-z]").any():
            mcol = "motif_id"
    if mcol is None and "motif_alt_id" in df.columns:
        mcol = "motif_alt_id"
    if mcol:
        df["motif_clean"] = df[mcol].astype(str).map(clean_motif)
    else:
        return pd.DataFrame()
    return df


def load_all_fimo(base: Path, subdir: str) -> pd.DataFrame:
    """Load all fimo.tsv files matching **/subdir/fimo.tsv."""
    files = list(base.glob(f"*/**/{subdir}/fimo.tsv"))
    logger.info("Found %d fimo.tsv files for %s", len(files), subdir)
    dfs = [load_fimo_tsv(f) for f in files]
    dfs = [d for d in dfs if not d.empty]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def extract_window(seq: str, start: int, end: int,
                   window: int, max_len: int) -> str:
    """Extract sequence window around a FIMO hit."""
    start = max(0, int(start) - 1)   # FIMO is 1-based
    end   = min(len(seq), int(end))
    left  = max(0, start - window)
    right = min(len(seq), end + window)
    sub   = seq[left:right]
    if len(sub) > max_len:
        center = (start + end) // 2
        half   = max_len // 2
        sub    = seq[max(0, center-half): min(len(seq), center+half)]
    return sub


def precompute_specificity(fimo_amp: pd.DataFrame,
                            fimo_non: pd.DataFrame) -> dict[str, float]:
    """
    Precompute AMP specificity per motif:
    spec = n_amp_hits / (n_amp_hits + n_nonamp_hits)
    Computed once as dict for O(1) lookup.
    """
    amp_counts = fimo_amp.groupby("motif_clean").size().rename("amp")
    non_counts = fimo_non.groupby("motif_clean").size().rename("non") \
                 if not fimo_non.empty else pd.Series(dtype=int)
    merged = pd.concat([amp_counts, non_counts], axis=1).fillna(0)
    merged["spec"] = merged["amp"] / (merged["amp"] + merged["non"] + 1e-6)
    return merged["spec"].to_dict()

# =====================================================
# CLI
# =====================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select diverse AMP sequences for FMAP mechanism prediction.")
    parser.add_argument("--seqs_per_motif", type=int,   default=5)
    parser.add_argument("--max_len",        type=int,   default=35)
    parser.add_argument("--window",         type=int,   default=15)
    parser.add_argument("--min_id_diff",    type=float, default=0.4,
                        help="Min sequence diversity (default 0.4 = max 60%% identity)")
    parser.add_argument("--fdr",            type=float, default=0.01)
    parser.add_argument("--er",             type=float, default=1.5)
    parser.add_argument("--spec",           type=float, default=0.6,
                        help="Min AMP specificity (default 0.6)")
    parser.add_argument("--temp",           type=int,   default=300)
    parser.add_argument("--ph",             type=float, default=7.0)
    parser.add_argument("--sys",            default="bact")
    parser.add_argument("--test",           action="store_true",
                        help="Output to results/09_fmap_input_test/")
    return parser.parse_args()

# =====================================================
# MAIN
# =====================================================

def main() -> None:
    args    = parse_args()
    out_dir = PROJECT_ROOT / (
        "results/09_motif_mechanism_input/01_fmap_input_test"
        if args.test else
        "results/09_motif_mechanism_input/01_fmap_input"
    )
    setup_logging(out_dir)

    logger.info("FMAP input generation started.")
    logger.info(
        "seqs_per_motif=%d  max_len=%d  window=%d  "
        "min_id_diff=%.2f  fdr=%.2e  er=%.1f  spec=%.2f",
        args.seqs_per_motif, args.max_len, args.window,
        args.min_id_diff, args.fdr, args.er, args.spec,
    )

    # ---- load families ----
    master_xlsx = FAMILIES_DIR / "motif_families_master.xlsx"
    if not master_xlsx.exists():
        logger.error("Not found: %s — run 09_motif_families_tomtom.py first.",
                     master_xlsx)
        sys.exit(1)

    df_fam = pd.read_excel(master_xlsx, sheet_name="All_motifs_with_family",
                            engine="openpyxl")
    logger.info("Loaded %d motif-family assignments", len(df_fam))

    # use actual column names from the Excel
    if "FDR_report" in df_fam.columns and "FDR" not in df_fam.columns:
        df_fam = df_fam.rename(columns={"FDR_report": "FDR"})
    elif "FDR_corrected_global" in df_fam.columns and "FDR" not in df_fam.columns:
        df_fam = df_fam.rename(columns={"FDR_corrected_global": "FDR"})
    elif "FDR_corrected" in df_fam.columns and "FDR" not in df_fam.columns:
        df_fam = df_fam.rename(columns={"FDR_corrected": "FDR"})

    if "Enrichment_ratio" in df_fam.columns and "ER" not in df_fam.columns:
        df_fam = df_fam.rename(columns={"Enrichment_ratio": "ER"})

    for col in ["FDR","ER"]:
        if col in df_fam.columns:
            df_fam[col] = pd.to_numeric(df_fam[col], errors="coerce")

    if "Motif" in df_fam.columns:
        df_fam["Motif"] = df_fam["Motif"].astype(str).map(clean_motif)

    # filter by FDR and ER
    n_before = len(df_fam)
    if "FDR" in df_fam.columns:
        df_fam = df_fam[df_fam["FDR"] <= args.fdr]
    if "ER" in df_fam.columns:
        df_fam = df_fam[
            df_fam["ER"].replace([np.inf,-np.inf], np.nan).fillna(999) >= args.er
        ]
    logger.info("After FDR/ER filter: %d / %d motifs", len(df_fam), n_before)

    # ---- load FASTA ----
    if not AMP_FASTA.exists():
        logger.error("AMP FASTA not found: %s", AMP_FASTA)
        sys.exit(1)
    fasta = load_fasta(AMP_FASTA)

    # ---- load FIMO ----
    fimo_amp = load_all_fimo(FIMO_BASE, "fimo_amps")
    fimo_non = load_all_fimo(FIMO_BASE, "fimo_nonamps")

    if fimo_amp.empty:
        logger.error("No FIMO AMP results found in %s", FIMO_BASE)
        sys.exit(1)

    logger.info("FIMO AMP rows: %d  |  non-AMP rows: %d",
                len(fimo_amp), len(fimo_non))

    # ---- precompute specificity ----
    spec_dict = precompute_specificity(fimo_amp, fimo_non)
    logger.info("Specificity computed for %d motifs", len(spec_dict))

    # ---- select sequences ----
    results:      list[dict] = []
    fmap_entries: list[tuple] = []
    seq_counter = 1

    motifs = df_fam["Motif"].dropna().unique()
    logger.info("Processing %d unique motifs...", len(motifs))

    for motif in sorted(motifs):
        spec = spec_dict.get(motif, 0.0)
        if spec < args.spec:
            continue

        hits = fimo_amp[fimo_amp["motif_clean"] == motif].copy()
        if hits.empty:
            continue

        # sort by p-value if available
        if "p_value" in hits.columns:
            hits = hits.sort_values("p_value")

        # get family for this motif
        fam_rows = df_fam[df_fam["Motif"] == motif]
        family   = str(fam_rows["Family_ID"].iloc[0]) \
                   if len(fam_rows) > 0 and "Family_ID" in fam_rows.columns \
                   else "UNKNOWN"
        er_val   = float(fam_rows["ER"].iloc[0]) \
                   if len(fam_rows) > 0 and "ER" in fam_rows.columns \
                   else np.nan

        selected: list[str] = []

        for _, hit in hits.iterrows():
            if len(selected) >= args.seqs_per_motif:
                break

            prot_id = str(hit.get("sequence_name", "")).strip()
            full_seq = fasta.get(prot_id)
            if not full_seq:
                continue

            start = hit.get("start", 1)
            stop  = hit.get("stop",  len(full_seq))

            window_seq = extract_window(full_seq, start, stop,
                                         args.window, args.max_len)
            if not window_seq or len(window_seq) > args.max_len:
                continue

            window_seq = clean_sequence(window_seq)

            if not is_diverse(window_seq, selected, args.min_id_diff):
                continue

            peptide_id = f"P{seq_counter:04d}"
            selected.append(window_seq)
            seq_counter += 1

            results.append({
                "Peptide_ID":           peptide_id,
                "Family_ID":            family,
                "Motif":                motif,
                "Matched_sequence":     hit.get("matched_sequence", ""),
                "Sequence_window":      window_seq,
                "Protein":              prot_id,
                "Start":                start,
                "Stop":                 stop,
                "p_value":              hit.get("p_value", np.nan),
                "Specificity":          round(spec, 4),
                "ER":                   er_val,
            })
            fmap_entries.append((peptide_id, window_seq))

        if selected:
            logger.debug("  %-25s  spec=%.2f  selected=%d",
                         motif, spec, len(selected))

    logger.info("Total peptides selected: %d", len(fmap_entries))

    # ---- save outputs ----
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "motif_selected_sequences.csv", index=False)

    with pd.ExcelWriter(out_dir / "motif_selected_sequences.xlsx",
                        engine="openpyxl") as w:
        results_df.to_excel(w, sheet_name="All_selected", index=False)
        results_df.groupby("Family_ID").size().reset_index(
            name="n_peptides").to_excel(w, sheet_name="Per_family", index=False)
        results_df.groupby("Motif").agg(
            n=("Peptide_ID","count"),
            mean_spec=("Specificity","mean"),
            mean_ER=("ER","mean"),
        ).reset_index().to_excel(w, sheet_name="Per_motif", index=False)

    # FMAP input file
    fmap_path = out_dir / "fmap_input.txt"
    with open(fmap_path, "w", encoding="utf-8") as f:
        f.write("  1\n")
        for pid, seq in fmap_entries:
            f.write(f"{args.temp} {args.ph} {args.sys} {pid}\n")
            f.write(f"{seq}\n")
            f.write("  0    0.0\n")

    logger.info("Outputs saved to: %s", out_dir)
    logger.info("FMAP input file: %s  (%d peptides)", fmap_path.name,
                len(fmap_entries))

    # summary
    logger.info("\n=== Selection summary ===")
    logger.info("  Motifs processed:   %d", len(motifs))
    logger.info("  Peptides selected:  %d", len(fmap_entries))
    if not results_df.empty:
        logger.info("  Families covered:   %d",
                    results_df["Family_ID"].nunique())
        logger.info("  Mean specificity:   %.3f",
                    results_df["Specificity"].mean())

    logger.info("FMAP input generation completed successfully.")


if __name__ == "__main__":
    main()