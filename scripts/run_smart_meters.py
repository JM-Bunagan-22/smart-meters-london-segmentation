"""
run_smart_meters.py

Just hit the green "Run" button in Spyder — no command-line arguments needed.
Edit the two paths below first, then run.

IMPORTANT: this file must sit in the SAME folder as smart_meters_lib.py
so the import below can find it. In Spyder, also make sure your working
directory (top-right file browser) is set to that same folder.
"""

import os

from smart_meters_lib import process_daily, process_halfhourly

# ---- EDIT THESE TWO PATHS ----
data_root = r"PATH/TO/YOUR/EXTRACTED/KAGGLE/DOWNLOAD"   # folder containing daily_dataset/, halfhourly_dataset/
out_dir   = r"PATH/TO/YOUR/EXTRACTED/KAGGLE/DOWNLOAD"   # where the parquet output folders will be created
# -------------------------------

if not os.path.isdir(data_root):
    raise SystemExit(
        f"data_root not found: {data_root!r}\n"
        "Edit the data_root/out_dir paths at the top of this script to point "
        "at your extracted Kaggle download folder before running."
    )

process_daily(data_root, out_dir, max_mb=10)
process_halfhourly(data_root, out_dir, max_mb=10)

print("\nAll done. Check the 'daily' and 'halfhourly' subfolders in your out_dir.")
