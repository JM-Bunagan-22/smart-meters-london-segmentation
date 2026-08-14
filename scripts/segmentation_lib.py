"""
segmentation_lib.py

Load-profiling & customer-segmentation functions for the London Smart
Meters (LCL) project. Import-only module -- see run_segmentation.py for
the runnable entry point.

Pipeline:
    1. load_halfhourly()      -- read the combined halfhourly parquet part(s)
    2. build_load_curves()    -- pivot to one row per household, 48 half-hour
                                  columns (average kWh per slot), separately
                                  for weekday and weekend
    3. normalize_curves()     -- z-score each household's curve (so
                                  clustering captures SHAPE, not magnitude)
    4. choose_k()             -- elbow + silhouette scan over a range of k
    5. run_kmeans()           -- fit final model, return cluster labels
    6. merge_household_info() -- attach ACORN group / tariff type
    7. acorn_crosstab()       -- cluster vs ACORN distribution table
    8. plot_cluster_profiles()-- average load curve per cluster
"""

import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

def load_halfhourly(processed_dir):
    """
    Read all halfhourly_part_*.parquet files under processed_dir/halfhourly
    and return one combined dataframe with columns: LCLid, tstp, energy_kwh_hh.
    """
    pattern = os.path.join(processed_dir, "halfhourly", "*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files found at {pattern}")

    print(f"  reading {len(files)} halfhourly parquet file(s)...")
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    print(f"  total halfhourly rows: {len(df):,}")
    return df


def load_household_info(data_root):
    """
    Read informations_households.csv (ACORN group, tariff type per LCLid).
    Looks directly under data_root for the CSV.
    """
    path = os.path.join(data_root, "informations_households.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"informations_households.csv not found at {path}")
    info = pd.read_csv(path)
    return info


# ---------------------------------------------------------------------------
# 2. Build load curves
# ---------------------------------------------------------------------------

def build_load_curves(hh_df, day_type="all"):
    """
    Pivot halfhourly readings into one row per household with 48 columns
    (slot_0 ... slot_47), each the household's AVERAGE kWh in that half-hour
    slot across all days in the data.

    day_type: "all", "weekday", or "weekend" -- filters which days are
    averaged before pivoting.
    """
    df = hh_df.copy()
    df["tstp"] = pd.to_datetime(df["tstp"])
    df["slot"] = (df["tstp"].dt.hour * 2 + df["tstp"].dt.minute // 30).astype("int16")

    if day_type == "weekday":
        df = df[df["tstp"].dt.dayofweek < 5]
    elif day_type == "weekend":
        df = df[df["tstp"].dt.dayofweek >= 5]
    elif day_type != "all":
        raise ValueError("day_type must be 'all', 'weekday', or 'weekend'")

    curve = (
        df.groupby(["LCLid", "slot"], observed=True)["energy_kwh_hh"]
        .mean()
        .unstack("slot")
    )
    curve.columns = [f"slot_{int(c):02d}" for c in curve.columns]
    curve = curve.reindex(columns=[f"slot_{i:02d}" for i in range(48)])

    # drop households with too much missing data (e.g. < 80% of slots present)
    min_slots = int(48 * 0.8)
    curve = curve.dropna(thresh=min_slots)
    curve = curve.fillna(curve.mean())  # fill any remaining gaps with column mean

    print(f"  built load curves for {len(curve):,} households ({day_type})")
    return curve


# ---------------------------------------------------------------------------
# 3. Normalize
# ---------------------------------------------------------------------------

def normalize_curves(curve_df):
    """
    Z-score normalize each household's 48-slot curve (row-wise), so
    clustering groups by SHAPE of usage rather than absolute magnitude.
    Households with zero variance (flat curve) are kept at 0.
    """
    values = curve_df.values
    row_mean = values.mean(axis=1, keepdims=True)
    row_std = values.std(axis=1, keepdims=True)
    row_std[row_std == 0] = 1.0  # avoid divide-by-zero
    normalized = (values - row_mean) / row_std
    return pd.DataFrame(normalized, index=curve_df.index, columns=curve_df.columns)


# ---------------------------------------------------------------------------
# 4. Choose k
# ---------------------------------------------------------------------------

def choose_k(norm_df, k_range=range(2, 9), random_state=42):
    """
    Fit KMeans for each k in k_range, return a dataframe of
    k, inertia, silhouette_score so you can pick k via elbow/silhouette.
    """
    results = []
    X = norm_df.values
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels) if k > 1 and len(set(labels)) > 1 else np.nan
        results.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
        print(f"  k={k}: inertia={km.inertia_:.1f}  silhouette={sil:.3f}")
    return pd.DataFrame(results)


def plot_k_selection(k_results):
    """Plot elbow (inertia) and silhouette score side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(k_results["k"], k_results["inertia"], marker="o")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow method")

    axes[1].plot(k_results["k"], k_results["silhouette"], marker="o", color="darkorange")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette score")
    axes[1].set_title("Silhouette score")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Final clustering
# ---------------------------------------------------------------------------

def run_kmeans(norm_df, k, random_state=42):
    """Fit final KMeans with chosen k, return dataframe with a 'cluster' column."""
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(norm_df.values)
    result = pd.DataFrame({"LCLid": norm_df.index, "cluster": labels})
    return result, km


# ---------------------------------------------------------------------------
# 6-7. Merge with household metadata + ACORN cross-tab
# ---------------------------------------------------------------------------

def merge_household_info(cluster_df, info_df):
    """Left-join cluster assignments with ACORN group / tariff metadata."""
    keep_cols = [c for c in ["LCLid", "Acorn", "Acorn_grouped", "stdorToU"] if c in info_df.columns]
    merged = cluster_df.merge(info_df[keep_cols], on="LCLid", how="left")
    return merged


def acorn_crosstab(merged_df, acorn_col="Acorn_grouped"):
    """Cross-tab of cluster vs ACORN group, as row-normalized percentages."""
    ct = pd.crosstab(merged_df["cluster"], merged_df[acorn_col], normalize="index") * 100
    return ct.round(1)


# ---------------------------------------------------------------------------
# 8. Plotting
# ---------------------------------------------------------------------------

def plot_cluster_profiles(curve_df, cluster_df):
    """
    Plot the average (un-normalized, kWh) load curve per cluster so you can
    see actual usage shape and magnitude differences.
    """
    merged = curve_df.join(cluster_df.set_index("LCLid")["cluster"])
    profile = merged.groupby("cluster").mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(48) / 2  # convert slot index to hour of day
    for cluster_id, row in profile.iterrows():
        ax.plot(x, row.values, label=f"Cluster {cluster_id} (n={sum(merged['cluster'] == cluster_id)})")

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Average kWh per half-hour")
    ax.set_title("Average load curve by cluster")
    ax.legend()
    fig.tight_layout()
    return fig
