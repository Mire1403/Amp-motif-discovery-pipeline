"""
04_background_sampling.py — modified validate_and_plot
Generates panel figure with pastel colors, panel letters, no titles, KS annotation.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

def _find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find project root.")

PROJECT_ROOT   = _find_project_root()
CLUSTER_DIR    = PROJECT_ROOT / "results" / "02_redundancy_reduction"
BG_FASTA_DIR   = PROJECT_ROOT / "data"    / "intermediate" / "background"
STATS_DIR      = PROJECT_ROOT / "results" / "03_background" / "statistics"
LOGS_DIR       = PROJECT_ROOT / "logs"

FASTAS = {
    "cdhit": {
        "amp":    CLUSTER_DIR / "AMP_MASTER_cdhit.fasta",
        "nonamp": CLUSTER_DIR / "NONAMP_cdhit.fasta",
    },
    "mmseqs": {
        "amp":    CLUSTER_DIR / "AMP_MASTER_mmseqs_rep_seq.fasta",
        "nonamp": CLUSTER_DIR / "NONAMP_mmseqs_rep_seq.fasta",
    },
}

MIN_LENGTH = 6
ALPHA      = 0.05

# colors
COL_AMP = '#F4A97F'   # taronja pastel
COL_BG  = '#A8C8E8'   # blau clar

def setup_logging() -> None:
    for d in [BG_FASTA_DIR, STATS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "04_background_sampling.log"
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

def percent_KR(seq: str) -> float:
    return (seq.count("K") + seq.count("R")) / len(seq)

def read_fasta(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"FASTA not found: {path}")
    sequences: list[str] = []
    parts: list[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if parts:
                    sequences.append("".join(parts).upper())
                    parts = []
            else:
                parts.append(line)
        if parts:
            sequences.append("".join(parts).upper())
    sequences = [s for s in sequences if len(s) >= MIN_LENGTH]
    return sequences

def compute_amp_stats(sequences: list[str]) -> tuple[dict, dict]:
    length_counts:  dict[int, int]        = defaultdict(int)
    kr_by_length:   dict[int, list[float]] = defaultdict(list)
    for seq in sequences:
        L = len(seq)
        length_counts[L] += 1
        kr_by_length[L].append(percent_KR(seq))
    kr_stats = {
        L: (float(np.mean(vals)), float(np.std(vals)))
        for L, vals in kr_by_length.items()
    }
    return dict(length_counts), kr_stats

def write_manifest(label: str, inputs: dict, multiplier: int, seed: int) -> Path:
    manifest = STATS_DIR / f"{label}_manifest.txt"
    with open(manifest, "w", encoding="utf-8") as f:
        f.write("BACKGROUND SAMPLING MANIFEST\n")
        f.write(f"timestamp_utc={datetime.utcnow().isoformat()}Z\n")
        f.write(f"label={label}\n")
        f.write(f"multiplier={multiplier}\n")
        f.write(f"min_length={MIN_LENGTH}\n")
        f.write(f"random_seed={seed}\n")
        f.write(f"alpha={ALPHA}\n")
        for k, v in inputs.items():
            f.write(f"{k}={v}\n")
    return manifest

def progressive_random_sampling(
        nonamp_sequences: list[str],
        target_counts:    dict[int, int],
        kr_stats:         dict[int, tuple[float, float]],
        rng:              random.Random,
) -> list[str]:
    selected_counts:    dict[int, int] = defaultdict(int)
    selected_fragments: list[str]      = []
    total_required = sum(target_counts.values())
    lengths        = list(target_counts.keys())
    nonamp_shuffled = nonamp_sequences.copy()
    rng.shuffle(nonamp_shuffled)
    log_interval = max(1, len(nonamp_shuffled) // 10)
    for i, seq in enumerate(nonamp_shuffled):
        if len(selected_fragments) >= total_required:
            break
        if i % log_interval == 0:
            logger.info(
                "  Sampling progress: %d/%d sequences scanned  fragments=%d/%d",
                i, len(nonamp_shuffled), len(selected_fragments), total_required,
            )
        seq_len = len(seq)
        lengths_shuffled = lengths.copy()
        rng.shuffle(lengths_shuffled)
        for L in lengths_shuffled:
            if selected_counts[L] >= target_counts[L]:
                continue
            if seq_len < L:
                continue
            mean_kr, std_kr = kr_stats[L]
            lower = mean_kr - std_kr
            upper = mean_kr + std_kr
            positions = list(range(seq_len - L + 1))
            rng.shuffle(positions)
            for pos in positions:
                if selected_counts[L] >= target_counts[L]:
                    break
                if len(selected_fragments) >= total_required:
                    break
                frag = seq[pos:pos + L]
                if lower <= percent_KR(frag) <= upper:
                    selected_fragments.append(frag)
                    selected_counts[L] += 1
    n_missing = total_required - len(selected_fragments)
    if n_missing > 0:
        logger.warning("  Generated %d/%d fragments (%d missing).",
                       len(selected_fragments), total_required, n_missing)
    else:
        logger.info("  All %d fragments generated successfully.", total_required)
    return selected_fragments


# ── accumulated data across both tools for combined panel ──────────────────
_panel_data: list[dict] = []

def validate_and_plot(
        amp_seqs:    list[str],
        nonamp_seqs: list[str],
        label:       str,
) -> dict:
    amp_lengths    = [len(s) for s in amp_seqs]
    nonamp_lengths = [len(s) for s in nonamp_seqs]
    amp_kr         = [percent_KR(s) for s in amp_seqs]
    nonamp_kr      = [percent_KR(s) for s in nonamp_seqs]

    ks_len = ks_2samp(amp_lengths, nonamp_lengths)
    ks_kr  = ks_2samp(amp_kr,      nonamp_kr)

    def interpret(p):
        return "NOT significantly different (good match)" \
               if p >= ALPHA else "Significantly different (poor match)"

    report_path = STATS_DIR / f"{label}_validation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"BACKGROUND VALIDATION REPORT — {label}\n\n")
        f.write(f"KS Length:  stat={ks_len.statistic:.4f}  p={ks_len.pvalue:.4e}  {interpret(ks_len.pvalue)}\n")
        f.write(f"KS K+R:     stat={ks_kr.statistic:.4f}  p={ks_kr.pvalue:.4e}  {interpret(ks_kr.pvalue)}\n")

    logger.info("  KS length:  stat=%.4f  p=%.4e  → %s",
                ks_len.statistic, ks_len.pvalue, interpret(ks_len.pvalue))
    logger.info("  KS K+R:     stat=%.4f  p=%.4e  → %s",
                ks_kr.statistic, ks_kr.pvalue, interpret(ks_kr.pvalue))

    # accumulate for combined panel
    _panel_data.append({
        "amp_lengths": amp_lengths, "nonamp_lengths": nonamp_lengths,
        "amp_kr": amp_kr, "nonamp_kr": nonamp_kr,
        "ks_len_p": ks_len.pvalue, "ks_kr_p": ks_kr.pvalue,
    })

    # if both tools done → draw combined 2×2 panel
    if len(_panel_data) == 2:
        _draw_combined_panel()

    return {
        "ks_len_stat":  round(float(ks_len.statistic), 6),
        "ks_len_p":     float(ks_len.pvalue),
        "ks_kr_stat":   round(float(ks_kr.statistic), 6),
        "ks_kr_p":      float(ks_kr.pvalue),
        "len_match_ok": bool(ks_len.pvalue >= ALPHA),
        "kr_match_ok":  bool(ks_kr.pvalue  >= ALPHA),
    }


def _draw_combined_panel() -> None:
    """2×2 panel: rows = CD-HIT / MMseqs2; cols = Length / K+R."""
    panels = "abcd"
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('white')

    configs = [
        # (tool_idx, col, xlabel, bins)
        (0, "len", "Sequence length (aa)", 30),
        (0, "kr",  "K+R proportion",       20),
        (1, "len", "Sequence length (aa)", 30),
        (1, "kr",  "K+R proportion",       20),
    ]

    for idx, (tidx, col, xlabel, bins) in enumerate(configs):
        row = idx // 2
        c   = idx % 2
        ax  = axes[row][c]
        ax.set_facecolor('white')

        d = _panel_data[tidx]
        if col == "len":
            amp_vals    = d["amp_lengths"]
            nonamp_vals = d["nonamp_lengths"]
            ks_p        = d["ks_len_p"]
        else:
            amp_vals    = d["amp_kr"]
            nonamp_vals = d["nonamp_kr"]
            ks_p        = d["ks_kr_p"]

        ax.hist(amp_vals,    bins=bins, density=True, alpha=0.75,
                color=COL_AMP, edgecolor='white', linewidth=0.4)
        ax.hist(nonamp_vals, bins=bins, density=True, alpha=0.75,
                color=COL_BG,  edgecolor='white', linewidth=0.4)

        # KS p-value annotation
        ax.text(0.97, 0.95, f"KS p = {ks_p:.2e}",
                transform=ax.transAxes, fontsize=12,
                va='top', ha='right', color='#333',
                bbox=dict(boxstyle='round,pad=0.3', fc='white',
                          ec='#ccc', alpha=0.9))

        # panel letter
        ax.text(0.03, 0.95, panels[idx],
                transform=ax.transAxes, fontsize=16,
                fontweight='bold', va='top', ha='left', color='black')

        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel("Density", fontsize=13)
        ax.tick_params(labelsize=11)
        ax.spines[['top', 'right']].set_visible(False)

    # global legend
    patches = [
        mpatches.Patch(color=COL_AMP, alpha=0.75, label='AMPs'),
        mpatches.Patch(color=COL_BG,  alpha=0.75, label='Background'),
    ]
    fig.legend(handles=patches, fontsize=14, loc='lower center',
               ncol=2, bbox_to_anchor=(0.5, 0.01),
               frameon=True, edgecolor='#ccc',
               handlelength=1.8, handleheight=1.4)

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    out = STATS_DIR / "annex_background_validation_panel.png"
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info("Combined panel saved: %s", out)


def run_one(label: str, amp_fasta: Path, nonamp_fasta: Path,
            multiplier: int, seed: int) -> dict:
    rng = random.Random(seed)
    logger.info("=== %s (multiplier=%d  seed=%d) ===", label, multiplier, seed)
    write_manifest(label, {"amp_fasta": amp_fasta, "nonamp_fasta": nonamp_fasta},
                   multiplier, seed)
    amp    = read_fasta(amp_fasta)
    nonamp = read_fasta(nonamp_fasta)
    logger.info("  AMP sequences:    %d", len(amp))
    logger.info("  Non-AMP sequences:%d", len(nonamp))
    length_counts, kr_stats = compute_amp_stats(amp)
    target_counts  = {L: count * multiplier for L, count in length_counts.items()}
    total_required = sum(target_counts.values())
    logger.info("  Target fragments: %d  (%dx AMP)", total_required, multiplier)
    fragments = progressive_random_sampling(nonamp, target_counts, kr_stats, rng)
    out_fasta = BG_FASTA_DIR / f"background_{label}_{multiplier}x.fasta"
    with open(out_fasta, "w", encoding="utf-8") as f:
        for i, frag in enumerate(fragments, start=1):
            f.write(f">BG_{label}_{i:08d}\n{frag}\n")
    logger.info("  Background FASTA: %s  (%d fragments)", out_fasta.name, len(fragments))
    logger.info("  Validating distributions...")
    val = validate_and_plot(amp, fragments, label)
    return {
        "label": label, "multiplier": multiplier, "seed": seed,
        "n_amp": len(amp), "n_nonamp_pool": len(nonamp),
        "expected": total_required, "generated": len(fragments),
        "coverage": round(len(fragments)/total_required, 4) if total_required else 0,
        "output_fasta": str(out_fasta), **val,
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", choices=["cdhit","mmseqs","both"], default="both")
    parser.add_argument("--multiplier", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main() -> None:
    setup_logging()
    args = parse_args()
    logger.info("Tool: %s  |  Multiplier: %d  |  Seed: %d",
                args.tool, args.multiplier, args.seed)
    to_run = ["cdhit","mmseqs"] if args.tool == "both" else [args.tool]
    rows: list[dict] = []
    failed: list[str] = []
    for tool in to_run:
        try:
            result = run_one(
                label=tool,
                amp_fasta=FASTAS[tool]["amp"],
                nonamp_fasta=FASTAS[tool]["nonamp"],
                multiplier=args.multiplier,
                seed=args.seed,
            )
            rows.append(result)
        except Exception as exc:
            logger.error("FAILED %s: %s", tool, exc, exc_info=True)
            failed.append(tool)
    if rows:
        summary_df = pd.DataFrame(rows)
        summary_csv = STATS_DIR / "background_sampling_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        gp_excel = STATS_DIR / "background_sampling_graphpad.xlsx"
        with pd.ExcelWriter(gp_excel, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            for row in rows:
                label    = row["label"]
                amp      = read_fasta(FASTAS[label]["amp"])
                bg_fasta = Path(row["output_fasta"])
                bg       = read_fasta(bg_fasta) if bg_fasta.exists() else []
                def to_wide(a, b, ca, cb):
                    return pd.concat([pd.Series(a,name=ca), pd.Series(b,name=cb)], axis=1)
                to_wide([len(s) for s in amp],[len(s) for s in bg],
                        "AMP_length","BG_length").to_excel(
                    writer, sheet_name=f"GP_{label}_lengths", index=False)
                to_wide([percent_KR(s) for s in amp],[percent_KR(s) for s in bg],
                        "AMP_KR","BG_KR").to_excel(
                    writer, sheet_name=f"GP_{label}_KR", index=False)
    if failed:
        logger.error("Completed WITH ERRORS: %s", failed)
        sys.exit(1)
    else:
        logger.info("Background sampling completed successfully.")

if __name__ == "__main__":
    main()