"""
Run APEX in batches to avoid memory issues on CPU.
Accepts --input and --output as arguments so it can be
reused for any APEX input file.

Usage:
  python 07b_batches.py --input path/to/input.txt --output path/to/output.csv
"""

import subprocess
import shutil
import argparse
import pandas as pd
from pathlib import Path

# =========================
# ARGS
# =========================

parser = argparse.ArgumentParser()
parser.add_argument("--input",  required=True, help="Path to APEX input .txt file")
parser.add_argument("--output", required=True, help="Path for final merged CSV")
parser.add_argument("--batch_size", type=int, default=500)
args = parser.parse_args()

APEX_DIR     = Path.home() / "apex_local/apex"
APEX_SCRIPT  = APEX_DIR / "predict.py"
INPUT_FILE   = Path(args.input).resolve()
FINAL_OUTPUT = Path(args.output).resolve()
BATCH_SIZE   = args.batch_size

# batch dir alongside the output file
BATCH_DIR = FINAL_OUTPUT.parent / f"batches_{INPUT_FILE.stem}"
BATCH_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# LOAD INPUT
# =========================

with open(INPUT_FILE) as f:
    sequences = [line.strip() for line in f if line.strip()]

print("=" * 60)
print(f"APEX BATCH PREDICTION")
print(f"Input:  {INPUT_FILE.name}")
print(f"Output: {FINAL_OUTPUT.name}")
print("=" * 60)
print(f"Total sequences: {len(sequences)}")
print(f"Batch size:      {BATCH_SIZE}")

# =========================
# SPLIT INTO BATCHES
# =========================

batches     = [sequences[i:i+BATCH_SIZE]
               for i in range(0, len(sequences), BATCH_SIZE)]
batch_files = []

print(f"Total batches:   {len(batches)}")

for i, batch in enumerate(batches):
    bf = BATCH_DIR / f"batch_{i:04d}.txt"
    with open(bf, "w") as f:
        f.write("\n".join(batch) + "\n")
    batch_files.append(bf)

# =========================
# RUN APEX PER BATCH
# =========================

print("\n" + "=" * 60)
print("RUNNING APEX")
print("=" * 60)

result_files = []

for i, batch_file in enumerate(batch_files):
    out_csv = BATCH_DIR / f"Predicted_MICs_batch_{i:04d}.csv"

    if out_csv.exists():
        print(f"Batch {i+1}/{len(batches)} — already done, skipping")
        result_files.append(out_csv)
        continue

    print(f"Batch {i+1}/{len(batches)} ({len(batches[i])} sequences)...",
          end=" ", flush=True)

    result = subprocess.run(
        ["python", str(APEX_SCRIPT), str(batch_file)],
        cwd=str(APEX_DIR),
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"ERROR\n{result.stderr}")
        raise RuntimeError(f"APEX failed on batch {i}")

    apex_out = APEX_DIR / "Predicted_MICs.csv"
    if apex_out.exists():
        shutil.move(apex_out, out_csv)
        result_files.append(out_csv)
        print("done ✅")
    else:
        print(f"WARNING: output not found for batch {i}")

# =========================
# MERGE
# =========================

print("\n" + "=" * 60)
print("MERGING RESULTS")
print("=" * 60)

dfs = []
for f in result_files:
    df = pd.read_csv(f, index_col=0)
    dfs.append(df)

merged = pd.concat(dfs, ignore_index=False)
merged.index.name = "Sequence"
merged = merged.reset_index()

print(f"Total rows merged: {len(merged)}")
merged.to_csv(FINAL_OUTPUT, index=False)

print(f"\n✅ Final output: {FINAL_OUTPUT}")
print("DONE")