"""
daily_segmentation_lib.py

Load-profiling & customer-segmentation using ONLY the daily_dataset
(no halfhourly data needed) -- lighter weight, GitHub-friendly.

Since daily data has no time-of-day info, this clusters households by
consumption CHARACTER instead of load-curve SHAPE:
    - overall consumption level
    - day-to-day variability
    - "peakiness" (how spiky vs flat each day tends to be)
    - weekday vs weekend behavior

Reuses choose_k / run_kmeans / merge_household_info / acorn_crosstab from
segmentation_lib.py -- only the loading, feature engineering, and
normalization differ from the halfhourly version.

Pipeline:
    1. load_daily()           -- read processed daily parquet part(s)
    2. build_daily_features()  -- one row per household, engineered features
    3. normalize_features()    -- column-wise standard scaling
    4. (from segmentation_lib) choose_k, run_kmeans, merge_household_info,
       acorn_crosstab
    5. plot_cluster_feature_profile() -- bar chart of feature means per cluster
"""

import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

def load_daily(processed_dir):
    """
    Read all daily_part_*.parquet files under processed_dir/daily and
    return one combined dataframe.
    """
    pattern = os.path.join(processed_dir, "daily", "*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files found at {pattern}")

    print(f"  reading {len(files)} daily parquet file(s)...")
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    print(f"  total daily rows: {len(df):,}")
    return df


def load_household_info(data_root):
    """Read informations_households.csv (ACORN group, tariff type per LCLid)."""
    path = os.path.join(data_root, "informations_households.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"informations_households.csv not found at {path}")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------

def build_daily_features(daily_df):
    """
    Collapse per-household-per-day rows into one feature row per household.

    Expects columns (standard LCL daily_dataset schema): LCLid, day,
    energy_median, energy_mean, energy_max, energy_min, energy_sum,
    energy_count. Missing optional columns are skipped gracefully.
    """
    df = daily_df.copy()
    df["day"] = pd.to_datetime(df["day"])
    df["dayofweek"] = df["day"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"] >= 5

    # per-day load factor (mean/max) -- how flat vs peaky that day was
    df["load_factor"] = df["energy_mean"] / df["energy_max"].replace(0, np.nan)

    agg = df.groupby("LCLid").agg(
        avg_daily_consumption=("energy_sum", "mean"),
        std_daily_consumption=("energy_sum", "std"),
        avg_daily_mean=("energy_mean", "mean"),
        avg_daily_peak=("energy_max", "mean"),
        avg_load_factor=("load_factor", "mean"),
        n_days=("day", "nunique"),
    )

    weekday_avg = (
        df[~df["is_weekend"]].groupby("LCLid")["energy_sum"].mean().rename("weekday_avg")
    )
    weekend_avg = (
        df[df["is_weekend"]].groupby("LCLid")["energy_sum"].mean().rename("weekend_avg")
    )
    agg = agg.join(weekday_avg).join(weekend_avg)

    # ratio > 1 means household uses more on weekends than weekdays
    agg["weekend_weekday_ratio"] = agg["weekend_avg"] / agg["weekday_avg"].replace(0, np.nan)

    # coefficient of variation -- day-to-day consistency (lower = steadier user)
    agg["coef_variation"] = agg["std_daily_consumption"] / agg["avg_daily_consumption"].replace(0, np.nan)

    # drop households with too few observed days to be reliable
    agg = agg[agg["n_days"] >= 7].drop(columns=["n_days"])

    # fill any remaining gaps (e.g. household with zero weekend days) with column median
    agg = agg.fillna(agg.median(numeric_only=True))

    print(f"  built features for {len(agg):,} households")
    return agg


# ---------------------------------------------------------------------------
# 3. Normalize (column-wise, since these are heterogeneous features
#    with different units -- NOT row-wise like the load-curve version)
# ---------------------------------------------------------------------------

def normalize_features(features_df):
    """Standard-scale each feature column (mean 0, std 1) across households."""
    values = features_df.values
    col_mean = values.mean(axis=0, keepdims=True)
    col_std = values.std(axis=0, keepdims=True)
    col_std[col_std == 0] = 1.0
    normalized = (values - col_mean) / col_std
    return pd.DataFrame(normalized, index=features_df.index, columns=features_df.columns)


# ---------------------------------------------------------------------------
# 4. Plotting
# ---------------------------------------------------------------------------

def plot_cluster_feature_profile(features_df, cluster_df):
    """
    Bar chart of each feature's mean value per cluster (z-scored so
    features with different units are comparable on one chart).
    """
    norm = normalize_features(features_df)
    merged = norm.join(cluster_df.set_index("LCLid")["cluster"])
    profile = merged.groupby("cluster").mean()

    fig, ax = plt.subplots(figsize=(11, 5))
    profile.T.plot(kind="bar", ax=ax)
    ax.set_ylabel("Z-scored feature value")
    ax.set_title("Cluster feature profiles (relative to overall average)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(title="Cluster")
    fig.tight_layout()
    return fig
