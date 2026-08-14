"""
run_daily_segmentation.py

Full load-profiling & customer-segmentation pipeline using ONLY the daily
dataset (no halfhourly data needed). Just hit Run in Spyder -- edit the
paths below first.

Must sit in the SAME folder as daily_segmentation_lib.py and
segmentation_lib.py (reuses clustering utilities from the latter).
"""

from daily_segmentation_lib import (
    load_daily, load_household_info, build_daily_features,
    normalize_features, plot_cluster_feature_profile,
)
from segmentation_lib import (
    choose_k, plot_k_selection, run_kmeans,
    merge_household_info, acorn_crosstab,
)

# ---- EDIT THESE PATHS ----
# Relative paths work as-is in GitHub Codespaces (run from inside scripts/).
# If running locally instead, change these to full paths on your machine.
processed_dir = "../data/processed"  # has daily/ subfolder
data_root = "../data/processed"  # has informations_households.csv
out_dir = "../data/processed"  # where results get saved

K = 4  # number of clusters -- adjust after reviewing the elbow/silhouette plot
# ---------------------------

print("Step 1/5: loading daily data...")
daily_df = load_daily(processed_dir)

print("\nStep 2/5: engineering per-household features...")
features = build_daily_features(daily_df)

print("\nStep 3/5: normalizing features (column-wise z-score)...")
norm_features = normalize_features(features)

print("\nStep 4/5: scanning k=2..8 for elbow/silhouette...")
k_results = choose_k(norm_features, k_range=range(2, 9))
fig1 = plot_k_selection(k_results)
fig1.savefig(f"{out_dir}/daily_k_selection.png", dpi=150)
print(f"  saved {out_dir}/daily_k_selection.png -- review this, then adjust K above if needed")

print(f"\nStep 5/5: fitting final KMeans with k={K}, merging ACORN info, plotting...")
cluster_df, model = run_kmeans(norm_features, k=K)

info_df = load_household_info(data_root)
merged = merge_household_info(cluster_df, info_df)
crosstab = acorn_crosstab(merged)
print("\nCluster vs ACORN group (% within each cluster):")
print(crosstab)

fig2 = plot_cluster_feature_profile(features, cluster_df)
fig2.savefig(f"{out_dir}/daily_cluster_profiles.png", dpi=150)
print(f"  saved {out_dir}/daily_cluster_profiles.png")

merged.to_csv(f"{out_dir}/daily_cluster_assignments.csv", index=False)
crosstab.to_csv(f"{out_dir}/daily_cluster_acorn_crosstab.csv")
print(f"  saved {out_dir}/daily_cluster_assignments.csv")
print(f"  saved {out_dir}/daily_cluster_acorn_crosstab.csv")

print("\nDone. Open the PNGs and CSVs in your out_dir to review results.")
