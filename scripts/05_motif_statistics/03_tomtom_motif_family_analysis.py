"""
09_motif_families_tomtom.py
============================
Groups significant AMP motifs into families using TOMTOM all-vs-all
comparison and Union-Find clustering.

Pipeline:
  1. Load significant motifs from 08_fimo_reporting master Excel
  2. Build combined MEME file from all 4 pipelines
  3. Run TOMTOM all-vs-all
  4. Filter pairs by q-value, overlap, coverage (optional reciprocal)
  5. Union-Find clustering → family assignments
  6. Family-level statistics and robustness analysis

Usage:
    python 09_motif_families_tomtom.py
    python 09_motif_families_tomtom.py --qvalue 1e-5 --min_overlap 5
    python 09_motif_families_tomtom.py --no_reciprocal
    python 09_motif_families_tomtom.py --overwrite
    python 09_motif_families_tomtom.py --test
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import shutil
import subprocess
import sys
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

# =====================================================
# PROJECT ROOT
# =====================================================

def _find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find project root.")

PROJECT_ROOT = _find_project_root()
MOTIF_DISC   = PROJECT_ROOT / "results" / "04_motif_discovery"
REPORT_DIR   = PROJECT_ROOT / "results" / "06_motif_statistics" / "02_fimo_reporting"
OUT_BASE     = PROJECT_ROOT / "results" / "07_motif_families"
LOGS_DIR     = PROJECT_ROOT / "logs"

MOTIF_FILES = {
    "cdhit_meme":    MOTIF_DISC / "meme_cdhit"    / "meme.txt",
    "cdhit_streme":  MOTIF_DISC / "streme_cdhit"  / "streme.txt",
    "mmseqs_meme":   MOTIF_DISC / "meme_mmseq"    / "meme.txt",
    "mmseqs_streme": MOTIF_DISC / "streme_mmseq"  / "streme.txt",
}

PALETTE = {
    "Cys-rich":     "#E07B54",
    "Cationic":     "#4C72B0",
    "Hydrophobic":  "#3BAF77",
    "Amphipathic":  "#2E86AB",
    "Gly-rich":     "#F0C040",
    "Anionic":      "#C0504D",
    "Pro-rich":     "#9B59B6",
    "Aromatic":     "#1ABC9C",
    "His-rich":     "#E67E22",
    "Other":        "#AAAAAA",
}
DPI = 300

# =====================================================
# LOGGING
# =====================================================

def setup_logging(out_dir: Path) -> None:
    for d in [out_dir, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "09_motif_families_tomtom.log"
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
sns.set_theme(style="whitegrid", font_scale=1.05)

# =====================================================
# HELPERS
# =====================================================

def clean_motif(x: str) -> str:
    x = str(x).strip()
    x = re.sub(r"^\d+[-_]", "", x)
    return x.strip()


def sanitize(x) -> str:
    if pd.isna(x): return ""
    return re.sub(r"\s+", " ", str(x).replace("\t"," ").replace("\n"," ")).strip()


def join_unique(values: Iterable) -> str:
    vals = [sanitize(v) for v in values if sanitize(v)]
    return ";".join(sorted(set(vals)))


def classify_motif(seq: str) -> str:
    seq = str(seq); L = len(seq)
    if L == 0: return "Other"
    fC  = seq.count("C") / L
    fG  = seq.count("G") / L
    fKR = (seq.count("K") + seq.count("R")) / L
    fDE = (seq.count("D") + seq.count("E")) / L
    fP  = seq.count("P") / L
    fH  = seq.count("H") / L
    fAr = (seq.count("W") + seq.count("F") + seq.count("Y")) / L
    fHy = sum(seq.count(x) for x in "AILMFWVY") / L
    if fC   >= 0.25: return "Cys-rich"
    if fG   >= 0.30: return "Gly-rich"
    if fH   >= 0.25: return "His-rich"
    if fKR  >= 0.35: return "Cationic"
    if fDE  >= 0.30: return "Anionic"
    if fP   >= 0.30: return "Pro-rich"
    if fAr  >= 0.30: return "Aromatic"
    if fHy  >= 0.40: return "Hydrophobic"
    if fKR  >= 0.15 and fHy >= 0.25: return "Amphipathic"
    return "Other"


def finite_agg(s: pd.Series, func: str) -> float:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return getattr(s, func)()


def canonical_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)

# =====================================================
# UNION-FIND
# =====================================================

class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

# =====================================================
# MEME PARSING
# =====================================================

def split_meme_blocks(text: str) -> tuple[list[str], list[list[str]]]:
    lines = text.splitlines()
    header, blocks, current = [], [], []
    i = 0
    while i < len(lines):
        if lines[i].startswith("MOTIF "): break
        header.append(lines[i]); i += 1
    while i < len(lines):
        if lines[i].startswith("MOTIF "):
            if current: blocks.append(current)
            current = [lines[i]]
        elif current:
            current.append(lines[i])
        i += 1
    if current: blocks.append(current)
    return header, blocks


def parse_width(block: list[str]) -> int | None:
    for line in block:
        if "letter-probability matrix" in line:
            m = re.search(r"\bw=\s*(\d+)", line)
            if m: return int(m.group(1))
    return None


def build_combined_meme(motif_files: dict[str, Path],
                         output_path: Path) -> pd.DataFrame:
    rows, all_blocks, header_out = [], [], None
    for pipeline, path in motif_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Motif file not found: {path}")
        text   = path.read_text(encoding="utf-8", errors="replace")
        header, blocks = split_meme_blocks(text)
        if header_out is None: header_out = header
        for block in blocks:
            parts = block[0].split()
            if len(parts) < 2: continue
            orig_id   = sanitize(parts[1])
            tomtom_id = f"{pipeline}__{orig_id}"
            new_block = block.copy()
            new_block[0] = " ".join([parts[0], tomtom_id] + parts[2:])
            all_blocks.append(new_block)
            rows.append({"tomtom_id": tomtom_id, "pipeline": pipeline,
                          "orig_id": orig_id, "clean_id": clean_motif(orig_id),
                          "width": parse_width(block)})

    with output_path.open("w", encoding="utf-8") as f:
        for line in (header_out or []):
            f.write(line + "\n")
        f.write("\n")
        for block in all_blocks:
            for line in block: f.write(line + "\n")
            f.write("\n")

    logger.info("Combined MEME: %d motifs -> %s", len(rows), output_path.name)
    return pd.DataFrame(rows)

# =====================================================
# TOMTOM
# =====================================================

def get_tomtom_binary() -> str:
    """Find tomtom in PATH."""
    path = shutil.which("tomtom")
    if path is None:
        raise RuntimeError(
            "tomtom not found in PATH.\n"
            "Activate the environment with MEME Suite installed.\n"
            "Check: which tomtom"
        )
    return path


def run_tomtom(combined_meme: Path, out_dir: Path,
               dist: str, min_overlap: int,
               overwrite: bool) -> Path:
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
        logger.debug("Cleared: %s", out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tomtom_bin = get_tomtom_binary()
    tsv_out    = out_dir / "tomtom.tsv"

    cmd = [
        tomtom_bin,
        "-dist",        dist,
        "-min-overlap", str(min_overlap),
        "-text",
        str(combined_meme),
        str(combined_meme),
    ]
    logger.info("Running TOMTOM:\n  %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("TOMTOM stderr:\n%s", result.stderr)
        raise RuntimeError(f"TOMTOM failed (code {result.returncode})")

    tsv_out.write_text(result.stdout, encoding="utf-8")
    logger.info("TOMTOM output: %s  (%d chars)", tsv_out.name, len(result.stdout))
    return tsv_out

# =====================================================
# PAIR FILTERING
# =====================================================

def filter_tomtom_pairs(
        tsv_path:    Path,
        meta_df:     pd.DataFrame,
        qvalue_thr:  float,
        min_overlap: int,
        min_coverage:float,
        reciprocal:  bool,
) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t", comment="#")
    if df.empty:
        logger.warning("TOMTOM TSV is empty.")
        return pd.DataFrame()

    df.columns = [c.strip() for c in df.columns]
    cols = {c.lower().replace("-","_").replace(" ","_"): c for c in df.columns}

    def pick(opts):
        for o in opts:
            if o.lower() in cols: return cols[o.lower()]
        return None

    qid_col  = pick(["query_id","query id"])
    tid_col  = pick(["target_id","target id"])
    qval_col = pick(["q_value","q-value","qvalue"])
    eval_col = pick(["e_value","e-value","evalue"])
    pval_col = pick(["p_value","p-value","pvalue"])
    ov_col   = pick(["overlap"])
    off_col  = pick(["offset"])
    ori_col  = pick(["orientation"])

    if not all([qid_col, tid_col, qval_col]):
        raise ValueError(f"Cannot find required columns. Found: {list(df.columns)}")

    meta = meta_df.set_index("tomtom_id").to_dict(orient="index")
    rows = []
    dir_set: set[tuple[str, str]] = set()

    for _, r in df.iterrows():
        qid = sanitize(r[qid_col]); tid = sanitize(r[tid_col])
        if qid == tid: continue
        qval = pd.to_numeric(r[qval_col], errors="coerce")
        if pd.isna(qval) or qval > qvalue_thr: continue
        ov = pd.to_numeric(r.get(ov_col, np.nan), errors="coerce") if ov_col else np.nan
        if pd.notna(ov) and ov < min_overlap: continue
        qm  = meta.get(qid, {}); tm = meta.get(tid, {})
        qw  = pd.to_numeric(qm.get("width"), errors="coerce")
        tw  = pd.to_numeric(tm.get("width"), errors="coerce")
        mw  = np.nanmin([qw, tw]) if not (pd.isna(qw) and pd.isna(tw)) else np.nan
        cov = ov / mw if pd.notna(ov) and pd.notna(mw) and mw > 0 else np.nan
        if pd.notna(cov) and cov < min_coverage: continue
        rows.append({"qid": qid, "tid": tid, "qval": qval,
                     "eval": pd.to_numeric(r.get(eval_col, np.nan), errors="coerce") if eval_col else np.nan,
                     "pval": pd.to_numeric(r.get(pval_col, np.nan), errors="coerce") if pval_col else np.nan,
                     "overlap": ov, "coverage": cov,
                     "offset": pd.to_numeric(r.get(off_col, np.nan), errors="coerce") if off_col else np.nan,
                     "orientation": sanitize(r[ori_col]) if ori_col else ""})
        dir_set.add((qid, tid))

    logger.info("Directional pairs after filtering: %d", len(rows))

    # build undirected edges
    dir_map = {(r["qid"], r["tid"]): r for r in rows}
    edges, seen = [], set()

    for (a, b), rab in dir_map.items():
        pair = canonical_pair(a, b)
        if pair in seen: continue
        seen.add(pair)
        rba = dir_map.get((b, a))
        is_recip = rba is not None
        if reciprocal and not is_recip: continue
        both = [x for x in [rab, rba] if x]
        edges.append({
            "Motif_A":    pair[0], "Motif_B": pair[1],
            "best_qval":  np.nanmin([x["qval"]    for x in both]),
            "best_eval":  np.nanmin([x["eval"]    for x in both if pd.notna(x["eval"])]) if any(pd.notna(x["eval"]) for x in both) else np.nan,
            "best_overlap":   np.nanmax([x["overlap"]  for x in both if pd.notna(x["overlap"])]) if any(pd.notna(x["overlap"]) for x in both) else np.nan,
            "best_coverage":  np.nanmax([x["coverage"] for x in both if pd.notna(x["coverage"])]) if any(pd.notna(x["coverage"]) for x in both) else np.nan,
            "reciprocal": is_recip,
        })

    edge_df = pd.DataFrame(edges).sort_values("best_qval") if edges else pd.DataFrame()
    logger.info("Undirected edges: %d", len(edge_df))
    return edge_df

# =====================================================
# FAMILY ASSIGNMENT
# =====================================================

def assign_families(df_sig: pd.DataFrame,
                    edges:   pd.DataFrame,
                    meta:    pd.DataFrame) -> pd.DataFrame:
    """
    Assign family IDs using Union-Find on clean motif names.
    Edges connect tomtom_ids; we map them to clean motif names,
    then union all clean names that are similar across any pipeline.
    """
    # map tomtom_id -> clean motif name
    id_to_clean = dict(zip(meta["tomtom_id"], meta["clean_id"]))

    # build clean edges — connect clean motif names
    clean_pairs = []
    if not edges.empty:
        seen = set()
        for _, r in edges.iterrows():
            a_clean = id_to_clean.get(str(r["Motif_A"]))
            b_clean = id_to_clean.get(str(r["Motif_B"]))
            if not a_clean or not b_clean or a_clean == b_clean:
                continue
            pair = canonical_pair(a_clean, b_clean)
            if pair in seen:
                continue
            seen.add(pair)
            clean_pairs.append(pair)

    logger.debug("Clean edges for Union-Find: %d", len(clean_pairs))

    # Union-Find on ALL unique clean motif names in the dataset
    # (not just the ones that appear in edges)
    all_clean = sorted(set(
        id_to_clean[tid]
        for tid in meta["tomtom_id"]
        if tid in id_to_clean
    ))
    uf = UnionFind(all_clean)
    for a, b in clean_pairs:
        if a in uf.parent and b in uf.parent:
            uf.union(a, b)

    root_to_fam: dict[str, str] = {}
    counter = 1
    fam_rows = []
    for m in all_clean:
        root = uf.find(m)
        if root not in root_to_fam:
            root_to_fam[root] = f"FAM_{counter:03d}"
            counter += 1
        fam_rows.append({"Motif": m, "Family_ID": root_to_fam[root]})

    fam_df = pd.DataFrame(fam_rows)
    logger.info("Families assigned: %d unique families from %d motifs",
                fam_df["Family_ID"].nunique(), len(fam_df))

    # merge back — normalize case before merge
    df_sig = df_sig.copy()
    df_sig["Motif"] = df_sig["Motif"].map(clean_motif).str.upper()
    fam_df["Motif"] = fam_df["Motif"].str.upper()
    return df_sig.merge(fam_df, on="Motif", how="left")

# =====================================================
# FAMILY SUMMARY
# =====================================================

def build_family_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    agg = (
        df.groupby("Family_ID")
        .agg(
            n_motifs=     ("Motif",           "nunique"),
            n_pipelines=  ("Pipeline",        "nunique"),
            n_rows=        ("Motif",           "size"),
            mean_ER=       ("Enrichment_ratio", lambda x: finite_agg(x, "mean")),
            median_ER=     ("Enrichment_ratio", lambda x: finite_agg(x, "median")),
            min_FDR=       ("FDR_report",      "min"),
            max_AMP_hits=  ("AMP_sequences_with_hit", "max"),
            Family_members=("Motif",           join_unique),
            Classes=       ("Motif_class",     join_unique),
            Pipelines=     ("Pipeline",        join_unique),
        ).reset_index()
    )

    # representative = best FDR per family
    rep = (
        df.sort_values(["Family_ID","FDR_report","Enrichment_ratio"],
                       ascending=[True, True, False])
        .groupby("Family_ID", as_index=False)
        .first()[["Family_ID","Motif","Motif_class","Pipeline",
                  "Enrichment_ratio","FDR_report"]]
        .rename(columns={"Motif": "Rep_motif", "Motif_class": "Rep_class",
                          "Pipeline": "Rep_pipeline",
                          "Enrichment_ratio": "Rep_ER",
                          "FDR_report": "Rep_FDR"})
    )

    summary = (agg.merge(rep, on="Family_ID", how="left")
                  .sort_values(["n_pipelines","median_ER"],
                                ascending=[False, False]))

    robust = summary[summary["n_pipelines"] >= 2].copy()
    logger.info("Total families: %d  |  Robust (≥2 pipelines): %d",
                len(summary), len(robust))
    return summary, robust

# =====================================================
# FIGURES
# =====================================================

def fig_family_er(robust: pd.DataFrame, out_dir: Path) -> None:
    top = robust.dropna(subset=["median_ER"]).head(20)
    if top.empty: return
    top = top.sort_values("median_ER", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(5, len(top)*0.4)))
    fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
    colors = [PALETTE.get(c, "#AAAAAA") for c in top["Rep_class"]]
    ax.barh(range(len(top)), top["median_ER"].values,
            color=colors, alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f"{r.Family_ID} — {r.Rep_motif}"
                        for _, r in top.iterrows()], fontsize=8)
    ax.set_xlabel("Median Enrichment Ratio", fontsize=10)
    ax.set_title("Top robust motif families — colored by class",
                 fontweight="bold")
    patches = [mpatches.Patch(color=v, label=k) for k,v in PALETTE.items()]
    ax.legend(handles=patches, fontsize=8, loc="lower right")
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    out = out_dir/"01_family_er_by_class.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.close()
    logger.info("Fig 1: %s", out)


def fig_pipeline_heatmap(presence: pd.DataFrame, out_dir: Path) -> None:
    if presence.empty: return
    df = presence.set_index("Family_ID") if "Family_ID" in presence.columns else presence
    fig, ax = plt.subplots(figsize=(max(6, len(df.columns)*1.5),
                                    max(4, len(df)*0.3)))
    fig.patch.set_facecolor("#fafafa")
    sns.heatmap(df.astype(float), cmap="YlOrRd", linewidths=0.3,
                linecolor="#eeeeee", vmin=0, vmax=1, ax=ax,
                cbar_kws={"label": "Present"})
    ax.set_title("Motif family presence across pipelines",
                 fontweight="bold")
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(fontsize=7)
    fig.tight_layout()
    out = out_dir/"02_family_pipeline_heatmap.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.close()
    logger.info("Fig 2: %s", out)


def fig_er_vs_hits(summary: pd.DataFrame, out_dir: Path) -> None:
    plot = summary.dropna(subset=["median_ER","max_AMP_hits"]).copy()
    if plot.empty: return
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
    colors = [PALETTE.get(c, "#AAAAAA") for c in plot["Rep_class"]]
    sizes  = (plot["n_motifs"] * 40).clip(lower=30, upper=400)
    ax.scatter(plot["median_ER"], plot["max_AMP_hits"],
               c=colors, s=sizes, alpha=0.75,
               edgecolors="white", linewidths=0.3)
    for _, r in plot.nlargest(8, "median_ER").iterrows():
        ax.annotate(str(r["Rep_motif"]),
                    (r["median_ER"], r["max_AMP_hits"]),
                    fontsize=7, xytext=(3,3),
                    textcoords="offset points")
    ax.set_xlabel("Median Enrichment Ratio", fontsize=10)
    ax.set_ylabel("Max AMP sequences with hit", fontsize=10)
    ax.set_title("Enrichment vs AMP coverage per family\n(size = n motifs)",
                 fontweight="bold")
    patches = [mpatches.Patch(color=v, label=k) for k,v in PALETTE.items()]
    ax.legend(handles=patches, fontsize=8)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    out = out_dir/"03_family_er_vs_amp_hits.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.close()
    logger.info("Fig 3: %s", out)


def fig_bubble(df: pd.DataFrame, out_dir: Path) -> None:
    plot = df.dropna(subset=["Enrichment_ratio","FDR_report"]).copy()
    finite = plot["Enrichment_ratio"].replace([np.inf,-np.inf], np.nan).dropna()
    if finite.empty: return
    plot = plot.loc[finite.index]
    plot["log2_ER"]       = np.log2(finite.clip(lower=1e-12))
    plot["neg_log10_fdr"] = -np.log10(plot["FDR_report"].clip(lower=1e-300))
    sizes = (plot.get("Total_hits", pd.Series([10]*len(plot))).fillna(10)
             .clip(lower=1) * 2).clip(upper=300)
    colors = [PALETTE.get(c, "#AAAAAA") for c in plot["Motif_class"]]
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
    ax.scatter(plot["log2_ER"], plot["neg_log10_fdr"],
               s=sizes, c=colors, alpha=0.6,
               edgecolors="white", linewidths=0.3)
    ax.set_xlabel("log₂ Enrichment Ratio", fontsize=10)
    ax.set_ylabel("-log₁₀(FDR)", fontsize=10)
    ax.set_title("Motif enrichment landscape (bubble = FIMO hits)",
                 fontweight="bold")
    patches = [mpatches.Patch(color=v, label=k) for k,v in PALETTE.items()]
    ax.legend(handles=patches, fontsize=8)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    out = out_dir/"04_bubble_er_fdr.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.close()
    logger.info("Fig 4: %s", out)

# =====================================================
# CLI
# =====================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group AMP motifs into families using TOMTOM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 09_motif_families_tomtom.py
  python 09_motif_families_tomtom.py --qvalue 1e-5
  python 09_motif_families_tomtom.py --no_reciprocal
  python 09_motif_families_tomtom.py --overwrite
  python 09_motif_families_tomtom.py --test
        """)
    parser.add_argument("--qvalue",       type=float, default=1e-5)
    parser.add_argument("--min_overlap",  type=int,   default=6)
    parser.add_argument("--min_coverage", type=float, default=0.8)
    parser.add_argument("--dist",         default="pearson",
                        choices=["pearson","ed","sandelin","kullback","allr","blic1","blic5","llr1","llr5"])
    parser.add_argument("--no_reciprocal", action="store_true",
                        help="Do not require reciprocal TOMTOM hits")
    parser.add_argument("--min_pipelines", type=int, default=2,
                        help="Min pipelines for robust families (default 2)")
    parser.add_argument("--overwrite",    action="store_true")
    parser.add_argument("--test",         action="store_true",
                        help="Output to results/07_motif_families_test/")
    return parser.parse_args()

# =====================================================
# MAIN
# =====================================================

def main() -> None:
    args = parse_args()
    out_dir = PROJECT_ROOT / (
        "results/07_motif_families_test"
        if args.test else
        "results/07_motif_families/01_tomtom"
    )
    setup_logging(out_dir)

    logger.info("Motif family analysis started.")
    logger.info("qvalue=%.2e  min_overlap=%d  min_coverage=%.2f  "
                "reciprocal=%s  min_pipelines=%d",
                args.qvalue, args.min_overlap, args.min_coverage,
                not args.no_reciprocal, args.min_pipelines)

    # ---- load significant motifs ----
    report_xlsx = REPORT_DIR / "fimo_reporting_master.xlsx"
    if not report_xlsx.exists():
        logger.error("Input not found: %s — run 08_fimo_reporting.py first.",
                     report_xlsx)
        sys.exit(1)

    df_sig = pd.read_excel(report_xlsx, sheet_name="Significant",
                            engine="openpyxl")
    logger.info("Loaded %d significant motifs from %s",
                len(df_sig), report_xlsx.name)

    # normalize
    rename = {"FDR_report": "FDR_report", "FDR_corrected_global": "FDR_report",
              "FDR_corrected": "FDR_report",
              "AMP_hits": "AMP_sequences_with_hit",
              "nonAMP_hits": "nonAMP_sequences_with_hit"}
    for old, new in rename.items():
        if old in df_sig.columns and new not in df_sig.columns:
            df_sig = df_sig.rename(columns={old: new})

    for col in ["Clustering","Tool","Motif"]:
        df_sig[col] = df_sig[col].astype(str).str.strip().str.lower()
    df_sig["Clustering"] = df_sig["Clustering"].str.replace(
        r"^mmseq$", "mmseqs", regex=True)
    df_sig["Motif"]      = df_sig["Motif"].map(clean_motif).str.upper()
    df_sig["Pipeline"]   = df_sig["Clustering"] + "_" + df_sig["Tool"]
    df_sig["Motif_class"]= df_sig["Motif"].map(classify_motif)

    for col in ["Enrichment_ratio","FDR_report",
                "AMP_sequences_with_hit","nonAMP_sequences_with_hit"]:
        if col in df_sig.columns:
            df_sig[col] = pd.to_numeric(df_sig[col], errors="coerce")

    logger.info("Unique motifs: %d  |  Pipelines: %s",
                df_sig["Motif"].nunique(),
                df_sig["Pipeline"].unique().tolist())

    # ---- build combined MEME ----
    combined_meme = out_dir / "combined_motifs.meme"
    meta_df = build_combined_meme(MOTIF_FILES, combined_meme)

    # ---- run TOMTOM ----
    tomtom_dir = out_dir / "tomtom_all_vs_all"
    tsv_path = run_tomtom(
        combined_meme, tomtom_dir,
        dist=args.dist,
        min_overlap=args.min_overlap,
        overwrite=args.overwrite,
    )

    # ---- filter pairs ----
    edges = filter_tomtom_pairs(
        tsv_path    = tsv_path,
        meta_df     = meta_df,
        qvalue_thr  = args.qvalue,
        min_overlap = args.min_overlap,
        min_coverage= args.min_coverage,
        reciprocal  = not args.no_reciprocal,
    )

    # ---- assign families ----
    df_fam = assign_families(df_sig, edges, meta_df)

    # ---- family summary ----
    summary, robust = build_family_summary(df_fam)

    # ---- presence matrix ----
    presence = (
        df_fam.assign(present=1)
        .pivot_table(index="Family_ID", columns="Pipeline",
                     values="present", aggfunc="max", fill_value=0)
        .reset_index()
    )

    # ---- save Excel ----
    out_xlsx = out_dir / "motif_families_master.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        df_fam.to_excel(w,    sheet_name="All_motifs_with_family", index=False)
        summary.to_excel(w,   sheet_name="Family_summary",         index=False)
        robust.to_excel(w,    sheet_name="Robust_families",        index=False)
        presence.to_excel(w,  sheet_name="Pipeline_presence",      index=False)
        edges.to_excel(w,     sheet_name="TOMTOM_edges",           index=False) \
            if not edges.empty else None
        meta_df.to_excel(w,   sheet_name="Motif_metadata",         index=False)
        # GraphPad
        gp = robust[["Family_ID","Rep_motif","median_ER","n_pipelines",
                      "n_motifs"]].copy()
        gp.to_excel(w, sheet_name="GP_robust_families", index=False)
    logger.info("Master Excel: %s", out_xlsx)

    # ---- figures ----
    fig_family_er(robust, out_dir)
    fig_pipeline_heatmap(presence, out_dir)
    fig_er_vs_hits(summary, out_dir)
    fig_bubble(df_fam, out_dir)

    # ---- summary log ----
    logger.info("\n=== Family summary ===")
    logger.info("Total families:              %d", summary["Family_ID"].nunique())
    logger.info("Robust families (≥%d pipelines): %d",
                args.min_pipelines, len(robust))
    logger.info("Singletons:                  %d",
                (summary["n_motifs"] == 1).sum())
    if not robust.empty:
        logger.info("\nTop 10 robust families:")
        logger.info("\n%s", robust[["Family_ID","Rep_motif","Rep_class",
                                    "n_motifs","n_pipelines","median_ER",
                                    "min_FDR"]].head(10).to_string(index=False))

    logger.info("Motif family analysis completed successfully.")
    logger.info("Outputs: %s", out_dir)


if __name__ == "__main__":
    main()