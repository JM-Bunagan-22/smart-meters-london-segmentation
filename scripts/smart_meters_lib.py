"""
process_smart_meters.py

Converts the London Smart Meters (LCL) daily_dataset and halfhourly_dataset
CSVs into compressed, GitHub-friendly parquet files (each file kept under a
size limit, default 25MB).

Usage:
    python process_smart_meters.py --data-root "path/to/your/data/folder" --out-dir "path/to/output"

Expected input layout (standard Kaggle "Smart meters in London" structure):
    <data-root>/
        daily_dataset/
            daily_dataset.csv            <- single combined file, OR
            block_0.csv, block_1.csv...  <- per-block files (either works)
        halfhourly_dataset/
            halfhourly_dataset/
                block_0.csv, block_1.csv, ... block_111.csv
        informations_households.csv

Output:
    <out-dir>/daily/daily_part_0.parquet, daily_part_1.parquet, ...
    <out-dir>/halfhourly/block_0.parquet, block_1.parquet, ...

Each parquet file is checked after writing; if it exceeds --max-mb, it is
split further and rewritten as multiple parts.
"""

import argparse
import glob
import os
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def human_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def downcast_daily(df):
    """Downcast dtypes for the daily_dataset to shrink memory/size."""
    id_cols = [c for c in df.columns if c.lower() in ("lclid",)]
    date_cols = [c for c in df.columns if c.lower() in ("day",)]

    for c in id_cols:
        df[c] = df[c].astype("category")
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    float_cols = [
        c for c in df.columns
        if c not in id_cols + date_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    for c in float_cols:
        df[c] = pd.to_numeric(df[c], downcast="float")

    # energy_count is often an integer count of readings that day
    if "energy_count" in df.columns:
        df["energy_count"] = pd.to_numeric(df["energy_count"], downcast="integer")

    return df


def downcast_halfhourly(df):
    """Downcast dtypes for the halfhourly_dataset to shrink memory/size."""
    # Standard columns: LCLid, tstp, energy(kWh/hh)
    rename_map = {}
    for c in df.columns:
        if c.strip().lower().startswith("energy"):
            rename_map[c] = "energy_kwh_hh"
    df = df.rename(columns=rename_map)

    if "LCLid" in df.columns:
        df["LCLid"] = df["LCLid"].astype("category")
    if "tstp" in df.columns:
        df["tstp"] = pd.to_datetime(df["tstp"], errors="coerce")
    if "energy_kwh_hh" in df.columns:
        # source sometimes has "Null" strings for missing readings
        df["energy_kwh_hh"] = pd.to_numeric(df["energy_kwh_hh"], errors="coerce")
        df["energy_kwh_hh"] = pd.to_numeric(df["energy_kwh_hh"], downcast="float")

    return df


def write_parquet_capped(df, out_dir, base_name, max_mb, compression="zstd"):
    """
    Write df to parquet at out_dir/base_name.parquet. If the resulting file
    exceeds max_mb, split the dataframe into N row-chunks and rewrite as
    base_name_part0.parquet, base_name_part1.parquet, etc. until each part
    fits under the cap.
    """
    os.makedirs(out_dir, exist_ok=True)
    single_path = os.path.join(out_dir, f"{base_name}.parquet")

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, single_path, compression=compression)

    size_mb = human_mb(single_path)
    if size_mb <= max_mb:
        print(f"  wrote {single_path}  ({size_mb:.1f} MB)")
        return [single_path]

    # Too big -> remove single file, split into row chunks
    os.remove(single_path)
    n_parts = max(2, int(size_mb // max_mb) + 1)
    chunk_size = (len(df) // n_parts) + 1

    written = []
    for i in range(n_parts):
        chunk = df.iloc[i * chunk_size:(i + 1) * chunk_size]
        if chunk.empty:
            continue
        part_path = os.path.join(out_dir, f"{base_name}_part{i}.parquet")
        pq.write_table(pa.Table.from_pandas(chunk, preserve_index=False),
                        part_path, compression=compression)
        part_mb = human_mb(part_path)
        print(f"  wrote {part_path}  ({part_mb:.1f} MB)")
        written.append(part_path)

        # safety: if a part is STILL too big (unlikely), recurse on it
        if part_mb > max_mb:
            print(f"    part still over {max_mb}MB, splitting further...")
            sub_parts = write_parquet_capped(
                chunk, out_dir, f"{base_name}_part{i}", max_mb, compression
            )
            os.remove(part_path)
            written.pop()
            written.extend(sub_parts)

    return written


# ---------------------------------------------------------------------------
# Processing functions
# ---------------------------------------------------------------------------

def process_daily(data_root, out_dir, max_mb):
    print("\n=== Processing daily_dataset ===")
    daily_dir = os.path.join(data_root, "daily_dataset")
    candidates = glob.glob(os.path.join(daily_dir, "**", "*.csv"), recursive=True)

    if not candidates:
        print(f"  No CSVs found under {daily_dir} -- skipping.")
        return

    frames = []
    for f in sorted(candidates):
        print(f"  reading {os.path.basename(f)}")
        df = pd.read_csv(f, low_memory=False)
        df = downcast_daily(df)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"  combined daily rows: {len(combined):,}")

    out_subdir = os.path.join(out_dir, "daily")
    write_parquet_capped(combined, out_subdir, "daily_part_0", max_mb)


def process_halfhourly_per_block(data_root, out_dir, max_mb):
    """One parquet file per source block (original behavior)."""
    print("\n=== Processing halfhourly_dataset (per-block) ===")
    hh_dir = os.path.join(data_root, "halfhourly_dataset")
    candidates = sorted(glob.glob(os.path.join(hh_dir, "**", "*.csv"), recursive=True))

    if not candidates:
        print(f"  No CSVs found under {hh_dir} -- skipping.")
        return

    out_subdir = os.path.join(out_dir, "halfhourly")
    print(f"  found {len(candidates)} block file(s)")

    for f in candidates:
        block_name = os.path.splitext(os.path.basename(f))[0]  # e.g. block_0
        print(f"  processing {block_name}")
        df = pd.read_csv(f, low_memory=False)
        df = downcast_halfhourly(df)
        write_parquet_capped(df, out_subdir, block_name, max_mb)


def process_halfhourly(data_root, out_dir, max_mb):
    """
    Combined version: reads ALL block CSVs, concatenates them into one
    dataframe (same approach as process_daily), then splits into as few
    parquet files as needed to stay under max_mb each -- instead of one
    file per source block.
    """
    print("\n=== Processing halfhourly_dataset (combined) ===")
    hh_dir = os.path.join(data_root, "halfhourly_dataset")
    candidates = sorted(glob.glob(os.path.join(hh_dir, "**", "*.csv"), recursive=True))

    if not candidates:
        print(f"  No CSVs found under {hh_dir} -- skipping.")
        return

    print(f"  found {len(candidates)} block file(s), reading and combining...")

    frames = []
    for i, f in enumerate(candidates):
        block_name = os.path.splitext(os.path.basename(f))[0]
        print(f"  reading {block_name}  ({i + 1}/{len(candidates)})")
        df = pd.read_csv(f, low_memory=False)
        df = downcast_halfhourly(df)
        frames.append(df)

    print("  concatenating all blocks...")
    combined = pd.concat(frames, ignore_index=True)
    del frames  # free memory before writing
    print(f"  combined halfhourly rows: {len(combined):,}")

    out_subdir = os.path.join(out_dir, "halfhourly")
    write_parquet_capped(combined, out_subdir, "halfhourly_part_0", max_mb)


# ---------------------------------------------------------------------------
# Command-line entry point (only used if you run THIS file with arguments,
# e.g. from a regular terminal: python smart_meters_lib.py --data-root ...)
# This block is skipped entirely when you `import` this module from Spyder
# or another script, so it will never throw the "required: --data-root"
# error just from importing.
# ---------------------------------------------------------------------------

def _cli_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True,
                         help="Folder containing daily_dataset/, halfhourly_dataset/, etc.")
    parser.add_argument("--out-dir", required=True,
                         help="Where to write the output parquet files.")
    parser.add_argument("--max-mb", type=float, default=25.0,
                         help="Max size per parquet file in MB (default: 25).")
    parser.add_argument("--skip-daily", action="store_true")
    parser.add_argument("--skip-halfhourly", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.data_root):
        sys.exit(f"data-root not found: {args.data_root}")

    if not args.skip_daily:
        process_daily(args.data_root, args.out_dir, args.max_mb)

    if not args.skip_halfhourly:
        process_halfhourly(args.data_root, args.out_dir, args.max_mb)

    print("\nDone.")


if __name__ == "__main__":
    _cli_main()
