"""
run_smart_meters.py

Just hit the green "Run" button in Spyder — no command-line arguments needed.
Edit the two paths below first, then run.

IMPORTANT: this file must sit in the SAME folder as smart_meters_lib.py
so the import below can find it. In Spyder, also make sure your working
directory (top-right file browser) is set to that same folder.
"""

from smart_meters_lib import process_daily, process_halfhourly

# ---- EDIT THESE TWO PATHS ----
data_root = r"C:\Users\ME177034\Downloads\archive"   # folder containing daily_dataset/, halfhourly_dataset/
out_dir   = r"C:\Users\ME177034\Downloads\archive"   # where the parquet output folders will be created
# -------------------------------

process_daily(data_root, out_dir, max_mb=10)
process_halfhourly(data_root, out_dir, max_mb=10)

print("\nAll done. Check the 'daily' and 'halfhourly' subfolders in your out_dir.")
