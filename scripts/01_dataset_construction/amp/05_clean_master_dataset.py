from __future__ import annotations
from pathlib import Path
import re
import json
import pandas as pd

from config import DATA_INTERMEDIATE_DIR, DATA_FINAL_DIR, RESULTS_DIR


def normalize_organism(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x)
    s = re.sub(r"\(.*?\)", "", s)
    s = s.replace("&&", ";")
    s = re.sub(r"\s+", " ", s).strip()
    return s.title()


def normalize_taxonomy(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).replace("&&", ";").replace(",", ";")
    s = re.sub(r"\s+", "", s)
    parts = sorted({p for p in s.split(";") if p})
    return ";".join(parts)


def concat_unique(series: pd.Series) -> str:
    values = series.dropna().astype(str)
    values = [v.strip() for v in values if v.strip()]
    return " | ".join(sorted(set(values))) if values else ""


def export_fasta(df: pd.DataFrame, out_fasta: Path) -> None:
    seqs = (
        df["sequence"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    seqs = [s for s in seqs.tolist() if s]

    with open(out_fasta, "w", encoding="utf-8") as f:
        for i, seq in enumerate(seqs, start=1):
            f.write(f">AMP_{i:06d}\n{seq}\n")


def main() -> None:
    in_file = DATA_INTERMEDIATE_DIR / "DB_MASTER_04.parquet"
    if not in_file.exists():
        raise FileNotFoundError(in_file)

    df = pd.read_parquet(in_file)
    n0 = len(df)

    if "organism" in df.columns:
        df["organism"] = df["organism"].apply(normalize_organism)

    if "taxonomy" in df.columns:
        df["taxonomy"] = df["taxonomy"].apply(normalize_taxonomy)

    # Global dedup by sequence (merge metadata)
    grouped = (
        df
        .groupby("sequence", as_index=False)
        .agg({col: concat_unique for col in df.columns if col != "sequence"})
    )

    grouped["length"] = grouped["sequence"].astype(str).str.len()

    # Ensure output directories exist
    DATA_FINAL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    out_parquet = DATA_FINAL_DIR / "DB_MASTER_CLEAN.parquet"
    grouped.to_parquet(out_parquet, index=False)

    out_fasta = DATA_FINAL_DIR / "AMP_MASTER.fasta"
    export_fasta(grouped, out_fasta)

    stats = {
        "initial_rows_master_04": int(n0),
        "final_unique_sequences": int(len(grouped)),
        "length_min": int(grouped["length"].min()),
        "length_mean": float(grouped["length"].mean()),
        "length_max": int(grouped["length"].max()),
    }

    (RESULTS_DIR / "final_stats.json").write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8"
    )

    preview_cols = [
        c for c in
        ["source_db", "protein_name", "organism", "taxonomy", "sequence", "length"]
        if c in grouped.columns
    ]

    grouped[preview_cols].head(20).to_csv(
        RESULTS_DIR / "final_preview.csv",
        index=False
    )

    print(f"✅ clean+export: {len(grouped)} unique seqs")
    print(f"   - {out_parquet}")
    print(f"   - {out_fasta}")
    print(f"   - {RESULTS_DIR / 'final_stats.json'}")


if __name__ == "__main__":
    main()