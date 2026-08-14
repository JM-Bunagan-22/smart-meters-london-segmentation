# Smart Meters London — Load Profiling & Customer Segmentation

Clustering London households into electricity usage archetypes (e.g. "evening
peakers," "flat baseload," "weekend spikers") using the [Smart Meters in
London](https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london)
(LCL) dataset, and comparing cluster membership against ACORN demographic
groups.

## Project background

Half-hourly and daily electricity consumption readings from ~5,500 London
households (Nov 2011–Feb 2014), collected by UK Power Networks as part of
the Low Carbon London project. Includes household ACORN demographic
classification and tariff type (standard vs. dynamic time-of-use).

## Repo structure

```
.
├── data/
│   ├── raw/                  # raw Kaggle CSVs — NOT committed (see .gitignore)
│   └── processed/
│       ├── daily/            # daily_part_0.parquet, ... (<25MB each)
│       └── halfhourly/       # halfhourly_part_0.parquet, ... (<25MB each)
├── scripts/
│   ├── smart_meters_lib.py   # processing functions (import only)
│   └── run_smart_meters.py   # edit the two paths, then run
├── notebooks/                # EDA / clustering notebooks go here
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

1. Download the dataset from Kaggle and extract it locally (outside this
   repo — the raw CSVs are far too large for GitHub).
2. Open `scripts/run_smart_meters.py`, set `data_root` to your extracted
   Kaggle folder and `out_dir` to `data/processed` inside this repo.
3. Run it (via Spyder's Run button, or `python scripts/run_smart_meters.py`
   from the `scripts/` folder). This converts the raw CSVs into compressed,
   downcasted parquet files, automatically split to stay under 25MB each so
   they can be committed to GitHub.
4. Work through the analysis in `notebooks/` — see Project Plan below.

## Project plan

1. **Data prep** — convert raw CSVs to parquet ✅ (`scripts/`)
2. **Feature engineering** — pivot half-hourly readings into per-household
   daily load curves (48 half-hour columns); derive features like peak hour,
   load factor, weekday/weekend profile
3. **Exploratory analysis** — sanity-check load curve shapes, check ACORN
   and tariff distribution
4. **Clustering** — k-means / hierarchical clustering on normalized load
   shapes; pick cluster count via elbow/silhouette analysis
5. **Cluster interpretation** — label clusters descriptively, cross-tab
   against ACORN demographic categories
6. **Dashboard** — interactive cluster explorer (Dash or Power BI)
7. **Write-up** — methodology, findings, and how this translates to
   utility-side customer segmentation

## Data source

UK Power Networks, "SmartMeter Energy Consumption Data in London
Households," via [London Datastore](https://data.london.gov.uk/) /
[Kaggle](https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london).
